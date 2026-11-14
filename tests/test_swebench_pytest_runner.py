# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Security and protocol tests for the SWE-bench authoring runner."""

from __future__ import annotations

import io
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from crucible.products.swebench.runners import swebench_pytest as runner
from crucible.products.swebench.runners.swebench_pytest import (
    _BaselineComparison,
    _PytestOutcome,
    _VisibleTestSelection,
    _baseline_identity,
    _baseline_cache_root,
    _cached_baseline_outcome,
    _compare_pytest_outcomes,
    _identity_key,
    _is_python_source_path,
    _parse_pytest_summary,
    _pytest_outcome,
    _resolve_test_command,
    _run_clean_baseline,
    main,
)


@pytest.fixture(autouse=True)
def _unit_contained_exec_adapter(monkeypatch):
    """Route contained execution through existing subprocess fakes.

    Production uses the attested subreaper. Most tests in this module exercise
    selection and parsing with a mocked Docker CLI, so preserve that boundary
    and cover the real containment dispatch in dedicated tests below.
    """

    def fake_contained_exec(
        container_id,
        command,
        *,
        timeout,
        cleanup_timeout,
        profile,
    ):
        del timeout, cleanup_timeout, profile
        completed = subprocess.run(
            [
                "docker", "exec", "--workdir", "/testbed",
                str(container_id), "bash", "-c", command,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode, completed.stdout, completed.stderr

    monkeypatch.setattr(
        runner,
        "run_contained_docker_exec",
        fake_contained_exec,
    )


def test_default_baseline_cache_is_uid_scoped_on_data_disk(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_SWEBENCH_BASELINE_CACHE", raising=False)
    monkeypatch.setattr(os, "geteuid", lambda: 1234)

    assert _baseline_cache_root() == Path(
        "/data/.crucible-swebench-visible-baselines-1234"
    )


def test_breach_observation_does_not_rewrite_visible_suite_command() -> None:
    command = ["bash", "project-gate.sh", "--capture=fd", "-q"]

    assert runner._visible_test_command(  # noqa: SLF001
        {
            "test_command": command,
            "test_args": ["--project-flag"],
            "breach_observations": True,
        },
        {},
    ) == [*command, "--project-flag"]


def test_baseline_bind_source_requires_owned_canonical_real_directory(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    symlink = tmp_path / "source-link"
    symlink.symlink_to(source, target_is_directory=True)

    assert runner._validated_bind_source(str(source)) == str(source)
    assert runner._validated_bind_source("/") is None
    assert runner._validated_bind_source("relative/source") is None
    assert runner._validated_bind_source(str(source) + "/.") is None
    assert runner._validated_bind_source(str(source) + ",other") is None
    assert runner._validated_bind_source(str(source) + "\nother") is None
    assert runner._validated_bind_source(str(symlink)) is None


def _invoke(parameters: dict) -> tuple[int, dict]:
    stdout = io.StringIO()
    rc = main(
        stdin=io.StringIO(json.dumps({"parameters": parameters, "env": {}})),
        stdout=stdout,
    )
    return rc, json.loads(stdout.getvalue())


def test_candidate_execution_uses_attested_swebench_profile(monkeypatch):
    captured: dict[str, object] = {}

    def contained(container_id, command, **kwargs):
        captured.update({
            "container_id": container_id,
            "command": command,
            **kwargs,
        })
        return 17, "bounded stdout", "bounded stderr"

    monkeypatch.setattr(runner, "run_contained_docker_exec", contained)
    completed = runner._contained_candidate_command(
        {"container_id": "authoring-container"},
        "cd /testbed && pytest -q",
        deadline=runner.time.monotonic() + 20.0,
    )

    assert completed.returncode == 17
    assert completed.stdout == "bounded stdout"
    assert completed.stderr == "bounded stderr"
    assert captured["container_id"] == "authoring-container"
    assert captured["command"] == "cd /testbed && pytest -q"
    assert captured["profile"] is runner.SWEBENCH_EXEC_PROFILE
    assert 0.0 < captured["cleanup_timeout"] <= 5.0
    assert captured["timeout"] > captured["cleanup_timeout"]


def test_candidate_execution_rejects_expired_deadline():
    with pytest.raises(ValueError, match="deadline expired"):
        runner._contained_candidate_command(
            {"container_id": "authoring-container"},
            "true",
            deadline=runner.time.monotonic(),
        )


def test_exact_visible_observation_propagates_bounded_failure_hint(monkeypatch):
    from crucible.common.runners import observations as observation_protocol

    failure = {
        "schema": "crucible.pytest.failure.v1",
        "phase": "call",
        "test": {
            "path": "tests/test_breach.py",
            "function": "test_alias_style",
        },
        "frame": {
            "path": "tests/test_breach.py",
            "function": "test_alias_style",
            "line": 47,
        },
        "exception": {
            "type": "AttributeError",
            "subject_kind": "type",
            "subject_module": "adaptix",
            "subject_name": "NameStyle",
            "attribute": "SNAKE",
            "direct_source_reference": True,
        },
    }

    def parse_failure(output, *, expected_node_id):
        assert output == "bounded-supervisor-envelope"
        assert expected_node_id == "tests/test_breach.py::test_alias_style"
        return failure

    monkeypatch.setattr(
        observation_protocol,
        "pytest_observer_failure",
        parse_failure,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            observation_protocol.PYTEST_OBSERVER_FAILED,
            stdout="bounded-supervisor-envelope",
            stderr="",
        ),
    )

    observed = runner._run_exact_visible_observations(
        {
            "test_command": ["pytest"],
            "breach_attack_ids": ["alias_style"],
            "breach_test_file": "tests/test_breach.py",
        },
        {"container_id": "authoring"},
    )

    assert observed == [{
        "language": "python",
        "id": "tests/test_breach.py::test_alias_style",
        "status": "failed",
        "line": "tests/test_breach.py::test_alias_style",
        "failure": failure,
    }]


def test_exact_visible_observation_rejects_malformed_failure_envelope(
    monkeypatch,
):
    from crucible.common.runners import observations as observation_protocol

    def reject_failure(_output, *, expected_node_id):
        assert expected_node_id == "tests/test_breach.py::test_alias_style"
        raise ValueError("malformed pytest observer failure envelope")

    monkeypatch.setattr(
        observation_protocol,
        "pytest_observer_failure",
        reject_failure,
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            observation_protocol.PYTEST_OBSERVER_FAILED,
            stdout="untrusted-noise",
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="malformed pytest observer"):
        runner._run_exact_visible_observations(
            {
                "test_command": ["pytest"],
                "breach_attack_ids": ["alias_style"],
                "breach_test_file": "tests/test_breach.py",
            },
            {"container_id": "authoring"},
        )


def test_exact_visible_observation_rejects_failure_on_passing_status(
    monkeypatch,
):
    from crucible.common.runners import observations as observation_protocol

    monkeypatch.setattr(
        observation_protocol,
        "pytest_observer_failure",
        lambda _output, *, expected_node_id: {
            "schema": "crucible.pytest.failure.v1",
        },
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0],
            observation_protocol.PYTEST_OBSERVER_PASSED,
            stdout="unexpected-envelope",
            stderr="",
        ),
    )

    with pytest.raises(ValueError, match="incompatible status"):
        runner._run_exact_visible_observations(
            {
                "test_command": ["pytest"],
                "breach_attack_ids": ["alias_style"],
                "breach_test_file": "tests/test_breach.py",
            },
            {"container_id": "authoring"},
        )


def _authoring_inspect_payload(
    *,
    instance_id: str,
    image_digest: str,
    workspace_source: Path,
    phase: str = "authoring",
    candidate_patch_sha256: str | None = None,
) -> str:
    workspace_source = workspace_source.resolve()
    labels = {
        "io.crucible.managed": "true",
        "io.crucible.phase": phase,
        "io.crucible.image-digest": image_digest,
        "io.crucible.task": runner.canonical_fingerprint(
            instance_id
        ).split(":", 1)[1],
        "io.crucible.owner-host": runner.socket.gethostname(),
        "io.crucible.owner-pid": str(os.getpid()),
        "io.crucible.owner-start": runner._process_start_token(
            os.getpid()
        ),
    }
    if candidate_patch_sha256 is not None:
        labels["io.crucible.candidate-patch-sha256"] = candidate_patch_sha256
    return json.dumps({
        "State": {"Running": True},
        "Config": {
            "Image": image_digest,
            "Labels": labels,
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": str(workspace_source),
                "Destination": "/testbed",
                "RW": True,
                "Propagation": "rprivate",
            },
            {
                "Type": "bind",
                "Source": str(workspace_source / ".git"),
                "Destination": "/testbed/.git",
                "RW": False,
                "Propagation": "rprivate",
            },
        ],
    })


def test_clean_baseline_accepts_patch_bound_visible_verifier(
    tmp_path,
    monkeypatch,
):
    instance_id = "org__repo-1"
    image_digest = "image@sha256:" + "f" * 64
    candidate_patch_sha256 = "a" * 64
    verifier_workspace = tmp_path / "visible-verifier"
    (verifier_workspace / ".git").mkdir(parents=True)

    def fake_run(command, **kwargs):
        assert command[1:3] == ["inspect", "--format"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_authoring_inspect_payload(
                instance_id=instance_id,
                image_digest=image_digest,
                workspace_source=verifier_workspace,
                phase="visible-verifier",
                candidate_patch_sha256=candidate_patch_sha256,
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    sources = runner._validated_authoring_bind_sources({
        "container_id": "visible-verifier-123",
        "instance_id": instance_id,
        "image_digest": image_digest,
        "candidate_patch_sha256": candidate_patch_sha256,
    })

    assert sources == runner._AuthoringBindSources(
        str(verifier_workspace),
        str(verifier_workspace / ".git"),
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_metadata_hash",
        "invalid_metadata_hash",
        "missing_label_hash",
        "mismatched_label_hash",
        "unexpected_phase",
        "unexpected_mount",
    ],
)
def test_clean_baseline_rejects_unbound_visible_verifier(
    tmp_path,
    monkeypatch,
    tamper,
):
    instance_id = "org__repo-1"
    image_digest = "image@sha256:" + "f" * 64
    candidate_patch_sha256 = "a" * 64
    verifier_workspace = tmp_path / "visible-verifier"
    (verifier_workspace / ".git").mkdir(parents=True)
    inspection = json.loads(_authoring_inspect_payload(
        instance_id=instance_id,
        image_digest=image_digest,
        workspace_source=verifier_workspace,
        phase="visible-verifier",
        candidate_patch_sha256=candidate_patch_sha256,
    ))
    metadata = {
        "container_id": "visible-verifier-123",
        "instance_id": instance_id,
        "image_digest": image_digest,
        "candidate_patch_sha256": candidate_patch_sha256,
    }
    if tamper == "missing_metadata_hash":
        metadata.pop("candidate_patch_sha256")
    elif tamper == "invalid_metadata_hash":
        metadata["candidate_patch_sha256"] = "A" * 64
        inspection["Config"]["Labels"][
            "io.crucible.candidate-patch-sha256"
        ] = "A" * 64
    elif tamper == "missing_label_hash":
        inspection["Config"]["Labels"].pop(
            "io.crucible.candidate-patch-sha256"
        )
    elif tamper == "mismatched_label_hash":
        inspection["Config"]["Labels"][
            "io.crucible.candidate-patch-sha256"
        ] = "b" * 64
    elif tamper == "unexpected_phase":
        inspection["Config"]["Labels"]["io.crucible.phase"] = "verification"
    else:
        inspection["Mounts"].append({
            "Type": "bind",
            "Source": str(tmp_path),
            "Destination": "/unexpected-host-path",
            "RW": False,
            "Propagation": "rprivate",
        })

    def fake_run(command, **kwargs):
        assert command[1:3] == ["inspect", "--format"]
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(inspection), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert runner._validated_authoring_bind_sources(metadata) is None


def test_parse_summary_extracts_final_counts():
    assert _parse_pytest_summary(
        "intermediate 1 passed\n"
        "3 passed, 2 failed, 1 error, 4 skipped in 0.5s"
    ) == {
        "tests_passed": 3.0,
        "tests_failed": 2.0,
        "tests_errored": 1.0,
        "tests_skipped": 4.0,
    }


def test_pytest_parser_tracks_terminal_discovery_outcomes():
    outcome = _pytest_outcome(subprocess.CompletedProcess(
        [],
        0,
        "1 passed, 2 skipped, 3 xfailed, 1 xpassed, 4 deselected, "
        "2 warnings in 0.5s\n",
        "diagnostic written to stderr\n",
    ))

    assert outcome.returncode == 0
    assert outcome.tests_run == 7
    assert dict(outcome.summary)["tests_deselected"] == 4.0
    assert dict(outcome.summary)["tests_warnings"] == 2.0


def test_pytest_parser_rejects_substantive_output_after_summary():
    outcome = _pytest_outcome(subprocess.CompletedProcess(
        [],
        0,
        "2 passed in 0.1s\nTraceback: outer runner crashed\n",
        "",
    ))

    assert outcome.returncode == 2
    assert outcome.tests_run is None
    assert outcome.summary == ()


def test_pytest_parser_ignores_nested_inner_error_for_passing_outer_suite():
    output = (
        "----------------------------- Captured stdout call "
        "-----------------------------\n"
        "=========================== short test summary info "
        "============================\n"
        "ERROR inner_test.py::TestIt::test_it - assert False\n"
        "===================== 1 passed, 1 error in 0.01s "
        "=====================\n"
        "=========================== short test summary info "
        "============================\n"
        "PASSED testing/test_unittest.py::test_traceback_pruning\n"
        "======================= 62 passed, 9 skipped in 2.42s "
        "=======================\n"
    )
    candidate = _pytest_outcome(
        subprocess.CompletedProcess([], 0, output, "")
    )
    baseline = _pytest_outcome(
        subprocess.CompletedProcess([], 0, output, "")
    )

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="f" * 64
    )

    assert candidate.returncode == 0
    assert candidate.tests_run == 71
    assert candidate.failures == ()
    assert comparison.mode == "baseline_match"
    assert comparison.effective_returncode == 0


def test_pytest_parser_scopes_failures_to_final_outer_short_summary():
    output = (
        "=========================== short test summary info "
        "============================\n"
        "ERROR inner_test.py::TestIt::test_it - assert False\n"
        "===================== 1 passed, 1 error in 0.01s "
        "=====================\n"
        "=========================== short test summary info "
        "============================\n"
        "FAILED testing/test_widget.py::test_value - AssertionError\n"
        "========================= 1 failed in 0.10s "
        "=========================\n"
    )

    outcome = _pytest_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 1
    assert outcome.tests_run == 1
    assert outcome.failures == (
        ("testing/test_widget.py::test_value", "FAILED", "AssertionError"),
    )


def test_pytest_parser_rejects_unpaired_nested_failure_summary():
    output = (
        "=========================== short test summary info "
        "============================\n"
        "ERROR inner_test.py::TestIt::test_it - assert False\n"
        "===================== 1 passed, 1 error in 0.01s "
        "=====================\n"
        "outer test continued after the nested run\n"
        "========================= 1 failed in 0.10s "
        "=========================\n"
    )

    outcome = _pytest_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.tests_run is None
    assert outcome.summary == ()
    assert outcome.failures == ()


def test_pytest_passing_candidate_must_preserve_executed_test_count():
    candidate = _pytest_outcome(subprocess.CompletedProcess(
        [], 0, "2 passed in 0.1s\n", ""
    ))
    baseline = _pytest_outcome(subprocess.CompletedProcess(
        [], 0, "3 passed in 0.1s\n", ""
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="f" * 64
    )

    assert comparison.mode == "incomparable"
    assert comparison.effective_returncode == 1


def test_pytest_equal_passing_outcomes_are_comparable():
    candidate = _pytest_outcome(subprocess.CompletedProcess(
        [], 0, "3 passed in 0.1s\n", ""
    ))
    baseline = _pytest_outcome(subprocess.CompletedProcess(
        [], 0, "3 passed in 0.2s\n", ""
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=True, cache_key="a" * 64
    )

    assert comparison.mode == "baseline_match"
    assert comparison.effective_returncode == 0


@pytest.mark.parametrize(
    ("instance_id", "returncode", "tests_run", "summary_key", "count"),
    [
        ("astropy__astropy-7606", 0, 242, "tests_warnings", 1.0),
        ("astropy__astropy-8872", 1, 2, "tests_errored", 2.0),
    ],
)
def test_pytest_parser_accepts_real_visible_harness_shapes(
    instance_id,
    returncode,
    tests_run,
    summary_key,
    count,
):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "products/swebench/tests/fixtures/harness_outputs"
        / instance_id
        / "test_output.txt"
    )
    test_output = fixture.read_text(encoding="utf-8").partition(
        "\n+ : '>>>>> End Test Output'"
    )[0] + "\n"

    outcome = _pytest_outcome(
        subprocess.CompletedProcess([], returncode, test_output, "")
    )

    assert outcome.returncode == returncode
    assert outcome.tests_run == tests_run
    assert dict(outcome.summary)[summary_key] == count


def test_resolve_test_command_unknown_repo_returns_none():
    assert _resolve_test_command("unknown/repo", "0") is None


@pytest.mark.parametrize(
    "path",
    (
        "pkg/__tests__/helper.py",
        "pkg/spec/helper.py",
        "pkg/conftest.py",
        "pkg/widget_test.py",
    ),
)
def test_visible_selector_uses_shared_structural_test_policy(path: str):
    assert not _is_python_source_path(path)


def test_visible_selector_does_not_drop_test_substring_source():
    assert _is_python_source_path("pkg/contest.py")


@pytest.mark.parametrize(
    "field", [
        "test_patch", "fail_to_pass", "pass_to_pass",
        "FAIL_TO_PASS", "PASS_TO_PASS", "patch",
    ],
)
def test_hidden_evaluation_fields_fail_before_subprocess(
    field, monkeypatch,
):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    rc, payload = _invoke({
        "role": "dev_verify",
        "swebench_metadata": {
            "container_id": "sandbox",
            field: "secret",
        },
    })
    assert rc == 2
    assert not called
    assert payload["values"]["hidden_evaluation_data_rejected"] == 1.0


def test_missing_container_fails_before_subprocess(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("host fallback attempted"),
    )
    rc, payload = _invoke({
        "role": "dev_verify",
        "swebench_metadata": {"repo": "org/repo", "version": "1"},
    })
    assert rc == 2
    assert payload["values"]["container_required"] == 1.0


def test_visible_tests_run_only_through_container(monkeypatch):
    captured: dict = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            command, 0, stdout="2 passed in 0.1s\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "tests/unit"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "repo": "org/repo",
            "version": "1",
        },
    })
    assert rc == 0
    assert captured["command"][:5] == [
        "docker", "exec", "--workdir", "/testbed", "sandbox-123",
    ]
    assert payload["values"]["visible_tests_only"] == 1.0
    assert payload["values"]["tests_passed"] == 2.0


def test_astropy_changed_source_selects_exact_visible_sibling_test(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"astropy/modeling/separable.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=b".crucible_build.lock\0",
                    stderr=b"",
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"astropy/modeling/tests/test_separable.py\0"
                    b"astropy/modeling/tests/test_other.py\0"
                ),
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="8 passed in 0.1s\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "repo": "astropy/astropy",
            "version": "4.3",
        },
    })

    assert rc == 0
    assert len(calls) == 4
    assert calls[-1][5:7] == ["bash", "-c"]
    assert calls[-1][-1].endswith(
        "pytest -rA astropy/modeling/tests/test_separable.py"
    )
    assert payload["values"]["visible_test_selection_applied"] == 1.0
    assert payload["values"]["visible_test_selection_target_count"] == 1.0
    assert payload["values"]["visible_test_selection_changed_source_count"] == 1.0
    assert "git diff --name-only HEAD" in payload["raw_output"]
    assert (
        'selected_targets=["astropy/modeling/tests/test_separable.py"]'
        in payload["raw_output"]
    )


def test_astropy_reviewed_timeseries_source_map_selects_shared_suites():
    source_path = "astropy/timeseries/core.py"
    expected_targets = (
        "astropy/timeseries/tests/test_common.py",
        "astropy/timeseries/tests/test_sampled.py",
        "astropy/timeseries/tests/test_binned.py",
    )

    assert runner._map_changed_astropy_sources(
        (source_path,), (*expected_targets, "astropy/timeseries/tests/test_io.py")
    ) == ((source_path,), expected_targets)


def test_astropy_reviewed_source_map_requires_every_target():
    source_path = "astropy/timeseries/core.py"
    incomplete_targets = tuple(
        target
        for target in runner._ASTROPY_VISIBLE_TEST_PATH_MAP[source_path]
        if not target.endswith("test_binned.py")
    )

    assert runner._map_changed_astropy_sources(
        (source_path,), incomplete_targets
    ) is None


def test_astropy_reviewed_source_map_requires_full_source_coverage():
    source_path = "astropy/timeseries/core.py"

    assert runner._map_changed_astropy_sources(
        (source_path, "astropy/timeseries/downsample.py"),
        tuple(runner._ASTROPY_VISIBLE_TEST_PATH_MAP[source_path]),
    ) is None


def test_astropy_timeseries_runner_uses_reviewed_targets(monkeypatch):
    source_path = "astropy/timeseries/core.py"
    expected_targets = tuple(
        runner._ASTROPY_VISIBLE_TEST_PATH_MAP[source_path]
    )
    git_results = iter(((source_path,), (), expected_targets))
    monkeypatch.setattr(
        runner,
        "_docker_git_paths",
        lambda *args, **kwargs: next(git_results),
    )

    selection = runner._select_visible_pytest_targets(
        {},
        {"container_id": "sandbox-123", "repo": "astropy/astropy"},
        ["pytest", "-rA"],
    )

    assert selection.mode == "targeted"
    assert selection.targets == expected_targets


@pytest.mark.parametrize(
    ("source_path", "test_paths"),
    [
        (
            "src/_pytest/logging.py",
            (
                "testing/logging/test_fixture.py",
                "testing/logging/test_formatter.py",
                "testing/logging/test_reporting.py",
            ),
        ),
        ("src/_pytest/unittest.py", ("testing/test_unittest.py",)),
    ],
)
def test_pytest_project_reviewed_source_map(source_path, test_paths):
    assert runner._map_changed_pytest_project_sources(
        (source_path,), (*test_paths, "testing/test_unrelated.py")
    ) == ((source_path,), test_paths)


def test_pytest_project_reviewed_source_map_requires_full_coverage():
    assert runner._map_changed_pytest_project_sources(
        ("src/_pytest/logging.py", "src/_pytest/config/__init__.py"),
        tuple(runner._PYTEST_VISIBLE_TEST_PATH_MAP["src/_pytest/logging.py"]),
    ) is None


@pytest.mark.parametrize(
    ("source_path", "expected_targets"),
    [
        (
            "sympy/geometry/point.py",
            ("sympy/geometry/tests/test_point.py",),
        ),
        (
            "sympy/core/function.py",
            (
                "sympy/core/tests/test_function.py",
                "sympy/utilities/tests/test_lambdify.py",
            ),
        ),
    ],
)
def test_sympy_native_runner_uses_reviewed_targets(
    monkeypatch, source_path, expected_targets
):
    git_results = iter(((source_path,), (), expected_targets))
    monkeypatch.setattr(
        runner,
        "_docker_git_paths",
        lambda *args, **kwargs: next(git_results),
    )
    command = [
        "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning",
        "bin/test",
        "-C",
        "--verbose",
    ]

    selection = runner._select_visible_pytest_targets(
        {}, {"container_id": "sandbox-123", "repo": "sympy/sympy"}, command
    )

    assert selection.mode == "targeted"
    assert selection.targets == expected_targets
    shell_command = runner._shell_test_command(
        [*command, *selection.targets],
        metadata={"repo": "sympy/sympy"},
    )
    assert "&& env PYTHONWARNINGS=" in shell_command
    assert " PYTHONHASHSEED=1 bin/test " in shell_command
    assert " bin/test -C --verbose " in shell_command
    assert " --verbose --seed=1 " in shell_command


@pytest.mark.parametrize(
    ("metadata", "suffix"),
    [
        ({"repo": "other/sympy"}, ["sympy/core/tests/test_function.py"]),
        ({"repo": "sympy/sympy"}, ["--seed=99"]),
    ],
)
def test_sympy_determinism_is_not_injected_into_rejected_commands(
    metadata, suffix
):
    shell_command = runner._shell_test_command(
        [*runner._SYMPY_NATIVE_COMMAND_PREFIX, *suffix],
        metadata=metadata,
    )

    assert "PYTHONHASHSEED=" not in shell_command
    assert "--seed=1" not in shell_command


@pytest.mark.parametrize(
    "metadata,command",
    [
        (
            {"repo": "other/sympy"},
            list(runner._SYMPY_NATIVE_COMMAND_PREFIX),
        ),
        (
            {"repo": "sympy/sympy"},
            ["bin/test", "-C", "--verbose"],
        ),
        (
            {"repo": "sympy/sympy"},
            [*runner._SYMPY_NATIVE_COMMAND_PREFIX, "--pdb"],
        ),
        (
            {"repo": "sympy/sympy"},
            [*runner._SYMPY_NATIVE_COMMAND_PREFIX, "sympy/core/function.py"],
        ),
    ],
)
def test_sympy_native_runner_rejects_unreviewed_command_shapes(
    metadata, command
):
    assert runner._sympy_runtests_index(command, metadata) is None


def _sympy_native_failure_output(
    *,
    bodies=("RecursionError: maximum recursion depth exceeded\n",),
    passed=5,
    skipped=0,
    expected_failed=0,
    failed=0,
    seconds="0.35",
    preamble="random seed: 123\n",
    framed=True,
):
    blocks = []
    for index, body in enumerate(bodies):
        node = (
            "sympy/geometry/tests/test_point.py:"
            f"test_preexisting_{index}"
        )
        blocks.append(
            "_" * 80
            + "\n"
            + "_" * 8
            + f" {node} "
            + "_" * 8
            + "\n"
            + body
        )
    fields = [f"{passed} passed"]
    if failed:
        fields.append(f"{failed} failed")
    if skipped:
        fields.append(f"{skipped} skipped")
    if expected_failed:
        fields.append(f"{expected_failed} expected to fail")
    if bodies:
        fields.append(f"{len(bodies)} exceptions")
    trailer = ", ".join(fields) + ", \n" + f"in {seconds} seconds"
    summary = (
        "\n========== tests finished: " + trailer + " ==========\n"
        if framed
        else "\n tests finished: " + trailer + " \n"
    )
    return preamble + "".join(blocks) + summary + (
        "DO *NOT* COMMIT!\n" if failed or bodies else ""
    )


def test_sympy_native_parser_accepts_multiline_trailer_and_exact_failures():
    output = _sympy_native_failure_output(
        bodies=tuple(f"RuntimeError: base {index}\n" for index in range(7)),
        passed=91,
        skipped=26,
        expected_failed=2,
        seconds="8.98",
        framed=False,
    )

    outcome = runner._sympy_outcome(
        subprocess.CompletedProcess([], 1, output, "DeprecationWarning\n")
    )

    assert outcome.returncode == 1
    assert outcome.parser_kind == "sympy-native"
    assert outcome.tests_run == 126
    assert len(outcome.failures) == 7
    assert all(status == "ERROR" for _, status, _ in outcome.failures)
    assert dict(outcome.summary) == {
        "tests_errored": 7.0,
        "tests_expected_failed": 2.0,
        "tests_failed": 0.0,
        "tests_passed": 91.0,
        "tests_skipped": 26.0,
        "tests_unexpected_successes": 0.0,
    }


def test_sympy_native_clean_base_forgives_only_exact_failure_bodies(
    monkeypatch,
):
    candidate = runner._sympy_outcome(subprocess.CompletedProcess(
        [],
        1,
        _sympy_native_failure_output(
            seconds="0.31", preamble="random seed: 123\n"
        ),
        "candidate warning\n",
    ))
    baseline = runner._sympy_outcome(subprocess.CompletedProcess(
        [],
        1,
        _sympy_native_failure_output(
            seconds="0.99", preamble="random seed: 987\n"
        ),
        "baseline warning\n",
    ))
    monkeypatch.setattr(
        runner,
        "_cached_baseline_outcome",
        lambda identity, callback, **kwargs: (baseline, True, "a" * 64, ""),
    )
    command = [
        *runner._SYMPY_NATIVE_COMMAND_PREFIX,
        "sympy/geometry/tests/test_point.py",
    ]

    comparison = runner._baseline_comparison(
        {
            "instance_id": "sympy__sympy-11618",
            "repo": "sympy/sympy",
            "version": "1.0",
            "image_digest": "image@sha256:" + "b" * 64,
        },
        "c" * 40,
        command,
        _VisibleTestSelection(
            mode="targeted",
            reason="test",
            targets=("sympy/geometry/tests/test_point.py",),
        ),
        candidate,
    )

    assert comparison.mode == "preexisting_only"
    assert comparison.effective_returncode == 0
    assert comparison.preexisting_failures == 1


def test_sympy_native_failure_fingerprint_remains_exact():
    plain = runner._sympy_failure_fingerprint(
        "traceback\nRecursionError: maximum recursion depth exceeded\n"
    )
    while_calling = runner._sympy_failure_fingerprint(
        "traceback\nRecursionError: maximum recursion depth exceeded "
        "while calling a Python object\n"
    )
    in_comparison = runner._sympy_failure_fingerprint(
        "traceback\nRecursionError: maximum recursion depth exceeded "
        "in comparison\n"
    )

    assert plain != while_calling
    assert plain != in_comparison
    assert while_calling != in_comparison
    assert plain != runner._sympy_failure_fingerprint(
        "different traceback\nRecursionError: maximum recursion depth exceeded\n"
    )
    assert plain != runner._sympy_failure_fingerprint(
        "traceback\nRuntimeError: maximum recursion depth exceeded\n"
    )


def test_sympy_native_changed_failure_body_remains_blocking():
    candidate = runner._sympy_outcome(subprocess.CompletedProcess(
        [],
        1,
        _sympy_native_failure_output(bodies=("RuntimeError: candidate\n",)),
        "",
    ))
    baseline = runner._sympy_outcome(subprocess.CompletedProcess(
        [],
        1,
        _sympy_native_failure_output(bodies=("RuntimeError: baseline\n",)),
        "",
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="d" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1
    assert comparison.worsened_failures == 1


@pytest.mark.parametrize(
    "output,returncode",
    [
        ("tests finished: 0 passed, in 0.10 seconds\n", 0),
        ("tests finished: 1 passed, in 0.10 seconds\ncrashed\n", 0),
        ("tests finished: 1 passed, 1 exceptions, in 0.10 seconds\n", 1),
        ("tests finished: 1 passed, in 0.10 seconds\nDO *NOT* COMMIT!\n", 0),
    ],
)
def test_sympy_native_malformed_or_inconsistent_output_fails_closed(
    output, returncode
):
    outcome = runner._sympy_outcome(
        subprocess.CompletedProcess([], returncode, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.failures == ()
    assert outcome.summary == ()


def test_changed_test_path_is_excluded_and_falls_back_to_full_suite(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"astropy/modeling/tests/test_separable.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            raise AssertionError("test-only changes must not query candidate tests")
        return subprocess.CompletedProcess(
            command, 0, stdout="8 passed in 0.1s\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "swebench_metadata": {"container_id": "sandbox-123"},
    })

    assert rc == 0
    assert len(calls) == 2
    assert calls[-1][-1].endswith("pytest -rA")
    assert payload["values"]["visible_test_selection_applied"] == 0.0
    assert payload["values"]["visible_test_selection_full_suite_fallback"] == 1.0
    assert "no changed non-test Python source files" in payload["raw_output"]
    assert "selected_targets=[]" in payload["raw_output"]


def test_no_safe_candidate_retains_full_pytest_suite(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"astropy/modeling/separable.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"astropy/modeling/tests/test_unrelated.py\0",
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            command, 0, stdout="8 passed in 0.1s\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "swebench_metadata": {"container_id": "sandbox-123"},
    })

    assert rc == 0
    assert len(calls) == 4
    assert calls[-1][-1].endswith("pytest -rA")
    assert payload["values"]["visible_test_selection_applied"] == 0.0
    assert payload["values"]["visible_test_selection_full_suite_fallback"] == 1.0
    assert "no safe unambiguous test target" in payload["raw_output"]


def test_explicit_test_args_are_preserved_without_git_selection(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5] == "git":
            raise AssertionError("explicit test_args must bypass Git selection")
        return subprocess.CompletedProcess(
            command, 0, stdout="8 passed in 0.1s\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "test_args": ["astropy/modeling/tests/test_separable.py"],
        "swebench_metadata": {"container_id": "sandbox-123"},
    })

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][-1].endswith(
        "pytest -rA astropy/modeling/tests/test_separable.py"
    )
    assert payload["values"]["visible_test_selection_applied"] == 0.0
    assert payload["values"]["visible_test_selection_explicit_args"] == 1.0
    assert "mode=preserved_explicit_test_args" in payload["raw_output"]


def test_unknown_non_pytest_command_is_preserved_without_git_selection(
    monkeypatch,
):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5] == "git":
            raise AssertionError("non-pytest commands must bypass Git selection")
        return subprocess.CompletedProcess(
            command, 0, stdout="suite passed\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["./scripts/runtests.py", "--verbosity", "2"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "repo": "org/repo",
        },
    })

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][-1].endswith("./scripts/runtests.py --verbosity 2")
    assert payload["values"]["visible_test_selection_non_pytest"] == 1.0


def test_django_changed_source_selects_tracked_native_test_app(monkeypatch):
    calls: list[list[str]] = []
    pass_output = (
        "----------------------------------------------------------------------\n"
        "Ran 14 tests in 0.123s\n\nOK\n"
    )
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 0, pass_output, "")
    )
    monkeypatch.setattr(
        runner,
        "_cached_baseline_outcome",
        lambda identity, callback, **kwargs: (baseline, True, "a" * 64, ""),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"django/core/validators.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"tests/auth_tests/__init__.py\0"
                    b"tests/auth_tests/test_validators.py\0"
                    b"tests/validators/__init__.py\0"
                    b"tests/validators/tests.py\0"
                ),
                stderr=b"",
            )
        if command[5:7] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="e" * 40 + "\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=pass_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["./tests/runtests.py", "--verbosity", "2"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "instance_id": "django__django-10097",
            "repo": "django/django",
            "version": "2.2",
            "image_digest": "image@sha256:" + "a" * 64,
        },
    })

    assert rc == 0
    assert len(calls) == 5
    assert calls[-1][-1].endswith(
        "./tests/runtests.py --verbosity 2 validators"
    )
    assert payload["values"]["visible_test_selection_applied"] == 1.0
    assert payload["values"]["visible_test_selection_non_pytest"] == 0.0
    assert payload["values"]["tests_passed"] == 14.0
    assert payload["values"]["tests_failed"] == 0.0
    assert payload["values"]["tests_errored"] == 0.0
    assert "mode=baseline_match" in payload["raw_output"]
    assert 'selected_targets=["validators"]' in payload["raw_output"]


def _django_failure_output(reason: str) -> str:
    return (
        "======================================================================\n"
        "FAIL: test_value (queries.tests.QueryTests)\n"
        "----------------------------------------------------------------------\n"
        "Traceback (most recent call last):\n"
        '  File "/testbed/tests/queries/tests.py", line 10, in test_value\n'
        f"AssertionError: {reason}\n\n"
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.100s\n\n"
        "FAILED (failures=1)\n"
    )


def test_django_reviewed_multi_source_map_selects_one_native_test_app(
    monkeypatch,
):
    calls: list[list[str]] = []
    pass_output = (
        "----------------------------------------------------------------------\n"
        "Ran 87 tests in 1.000s\n\nOK\n"
    )
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 0, pass_output, "")
    )
    monkeypatch.setattr(
        runner,
        "_cached_baseline_outcome",
        lambda identity, callback, **kwargs: (baseline, True, "b" * 64, ""),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"django/db/models/sql/query.py\0"
                    b"django/db/models/sql/compiler.py\0"
                ),
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"tests/queries/__init__.py\0"
                    b"tests/queries/test_query.py\0"
                ),
                stderr=b"",
            )
        if command[5:7] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="e" * 40 + "\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=pass_output,
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["./tests/runtests.py", "--verbosity", "2"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "instance_id": "django__django-10554",
            "repo": "django/django",
            "version": "3.0",
            "image_digest": "image@sha256:" + "a" * 64,
        },
    })

    assert rc == 0
    assert calls[-1][-1].endswith(
        "./tests/runtests.py --verbosity 2 queries"
    )
    assert payload["values"]["visible_test_selection_applied"] == 1.0
    assert payload["values"][
        "visible_test_selection_full_suite_fallback"
    ] == 0.0
    assert payload["values"]["tests_passed"] == 87.0


def test_django_unmapped_source_falls_back_to_exact_clean_base(monkeypatch):
    calls: list[list[str]] = []
    output = _django_failure_output("known base failure")
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )
    monkeypatch.setattr(
        runner,
        "_cached_baseline_outcome",
        lambda identity, callback, **kwargs: (baseline, True, "d" * 64, ""),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"django/db/models/options.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"tests/queries/__init__.py\0"
                    b"tests/queries/test_query.py\0"
                ),
                stderr=b"",
            )
        if command[5:7] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="e" * 40 + "\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 1, stdout=output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["./tests/runtests.py", "--verbosity", "2"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "instance_id": "django__django-unmapped",
            "repo": "django/django",
            "version": "3.0",
            "image_digest": "image@sha256:" + "a" * 64,
        },
    })

    assert rc == 0
    assert calls[-1][-1].endswith("./tests/runtests.py --verbosity 2")
    assert payload["values"][
        "visible_test_selection_full_suite_fallback"
    ] == 1.0
    assert payload["values"]["visible_test_candidate_tests_failed"] == 1.0
    assert payload["values"]["tests_failed"] == 0.0
    assert payload["values"]["visible_test_preexisting_failures"] == 1.0
    assert "mode=preexisting_only" in payload["raw_output"]


def test_django_same_test_with_changed_failure_body_remains_blocking():
    candidate = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, _django_failure_output("candidate regression"), ""
    ))
    baseline = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, _django_failure_output("known base failure"), ""
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=True, cache_key="e" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1
    assert comparison.worsened_failures == 1


@pytest.mark.parametrize(
    ("instance_id", "tests_run", "failures", "errors"),
    [
        ("django__django-10097", 12_317, 47, 550),
        ("django__django-11532", 158, 0, 18),
        ("django__django-14792", 25, 2, 0),
    ],
)
def test_django_parser_accepts_real_visible_harness_shapes(
    instance_id,
    tests_run,
    failures,
    errors,
):
    fixture = (
        Path(__file__).resolve().parents[1]
        / "products/swebench/tests/fixtures/harness_outputs"
        / instance_id
        / "test_output.txt"
    )
    test_output = fixture.read_text(encoding="utf-8").partition(
        "\n+ : '>>>>> End Test Output'"
    )[0] + "\n"

    outcome = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, test_output, ""
    ))

    assert outcome.returncode == 1
    assert outcome.parser_kind == "django-unittest"
    assert outcome.tests_run == tests_run
    assert len(outcome.failures) == failures + errors
    assert sum(
        status == "FAILED" for _, status, _ in outcome.failures
    ) == failures
    assert sum(
        status == "ERROR" for _, status, _ in outcome.failures
    ) == errors


@pytest.mark.parametrize(
    "output",
    [
        "----------------------------------------------------------------------\n"
        "Ran 0 tests in 0.000s\n\nOK\n",
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nFAILED (failures=1)\n",
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nOK (unknown=1)\n",
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nOK (skipped=1, skipped=1)\n",
    ],
)
def test_django_malformed_success_output_fails_closed(output):
    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 0, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.summary == ()
    assert outcome.failures == ()


def test_django_parser_uses_final_paired_suite_trailer():
    earlier = (
        "----------------------------------------------------------------------\n"
        "Ran 999 tests in 9.999s\n\nFAILED (errors=3)\n"
    )
    final = (
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK (skipped=1)\n"
    )

    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 0, earlier + final, "")
    )

    assert outcome.returncode == 0
    assert outcome.tests_run == 2
    assert dict(outcome.summary)["tests_skipped"] == 1.0


def test_django_parser_accepts_bounded_database_teardown_postamble():
    output = (
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK (skipped=1)\n"
        "Destroying test database for alias 'default' "
        "('file:memorydb_default?mode=memory&cache=shared')...\n"
    )

    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 0, output, "")
    )

    assert outcome.returncode == 0
    assert outcome.tests_run == 2
    assert dict(outcome.summary)["tests_skipped"] == 1.0


def test_django_parser_rejects_outer_crash_after_suite_trailer():
    baseline = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, _django_failure_output("known base failure"), ""
    ))
    crashed = runner._django_outcome(subprocess.CompletedProcess(
        [],
        1,
        _django_failure_output("known base failure")
        + "Traceback (most recent call last):\n"
        + "RuntimeError: outer runner crashed\n",
        "",
    ))

    comparison = _compare_pytest_outcomes(
        crashed, baseline, cache_hit=False, cache_key="d" * 64
    )

    assert crashed.summary == ()
    assert crashed.failures == ()
    assert comparison.mode == "incomparable"
    assert comparison.effective_returncode == 2


def test_django_error_to_failure_transition_remains_blocking():
    failed_output = _django_failure_output("same body")
    error_output = failed_output.replace(
        "FAIL: test_value", "ERROR: test_value"
    ).replace("FAILED (failures=1)", "FAILED (errors=1)")
    candidate = runner._django_outcome(
        subprocess.CompletedProcess([], 1, failed_output, "")
    )
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 1, error_output, "")
    )

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="f" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1
    assert comparison.worsened_failures == 1


def test_django_increased_skip_count_blocks_passing_candidate():
    baseline_output = (
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK\n"
    )
    candidate_output = (
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK (skipped=1)\n"
    )
    candidate = runner._django_outcome(
        subprocess.CompletedProcess([], 0, candidate_output, "")
    )
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 0, baseline_output, "")
    )

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=True, cache_key="a" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1


def test_django_changed_tests_run_blocks_passing_candidate():
    candidate = runner._django_outcome(subprocess.CompletedProcess(
        [],
        0,
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nOK\n",
        "",
    ))
    baseline = runner._django_outcome(subprocess.CompletedProcess(
        [],
        0,
        "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nOK\n",
        "",
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=True, cache_key="b" * 64
    )

    assert comparison.mode == "incomparable"
    assert comparison.effective_returncode == 1


def test_django_unexpected_success_cannot_be_forgiven():
    output = (
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\n"
        "FAILED (unexpected successes=1)\n"
    )
    candidate = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )
    baseline = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="c" * 64
    )

    assert dict(candidate.summary)["tests_unexpected_successes"] == 1.0
    assert comparison.mode == "incomparable"
    assert comparison.effective_returncode == 1


def test_django_duplicate_failure_identity_cannot_be_forgiven():
    section = (
        "======================================================================\n"
        "FAIL: test_value (queries.tests.QueryTests)\n"
        "----------------------------------------------------------------------\n"
        "AssertionError: same\n"
    )
    output = (
        section
        + section
        + "----------------------------------------------------------------------\n"
        "Ran 2 tests in 0.001s\n\nFAILED (failures=2)\n"
    )
    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.failures == ()
    assert outcome.summary == ()


def test_django_failure_header_cannot_consume_final_suite_separator():
    output = (
        "======================================================================\n"
        "FAIL: test_value (queries.tests.QueryTests)\n"
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nFAILED (failures=1)\n"
    )

    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.failures == ()
    assert outcome.summary == ()


def test_django_prior_suite_failures_cannot_satisfy_final_suite_counts():
    prior = _django_failure_output("prior suite failure")
    final = (
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nFAILED (failures=1)\n"
    )

    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 1, prior + final, "")
    )

    assert outcome.returncode == 2
    assert outcome.failures == ()
    assert outcome.summary == ()


@pytest.mark.parametrize(
    "output",
    [
        "----------------------------------------------------------------------\n"
        + "Ran " + "9" * 5_000 + " tests in 0.001s\n\nOK\n",
        "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nFAILED (failures="
        + "9" * 5_000
        + ")\n",
    ],
)
def test_django_oversized_numeric_fields_fail_closed(output):
    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 2
    assert outcome.tests_run is None


def test_django_multiple_subtest_failures_may_exceed_tests_run():
    sections = "".join(
        "======================================================================\n"
        f"FAIL: test_value (queries.tests.QueryTests) (value={value})\n"
        "----------------------------------------------------------------------\n"
        f"AssertionError: {value}\n"
        for value in (1, 2)
    )
    output = (
        sections
        + "----------------------------------------------------------------------\n"
        "Ran 1 test in 0.001s\n\nFAILED (failures=2)\n"
    )

    outcome = runner._django_outcome(
        subprocess.CompletedProcess([], 1, output, "")
    )

    assert outcome.returncode == 1
    assert outcome.tests_run == 1
    assert len(outcome.failures) == 2


def test_django_separator_only_assertion_lines_are_fingerprinted():
    candidate = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, _django_failure_output("line one\n----\nline two"), ""
    ))
    baseline = runner._django_outcome(subprocess.CompletedProcess(
        [], 1, _django_failure_output("line one\n====\nline two"), ""
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="e" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.worsened_failures == 1


def test_django_changed_non_python_behavior_path_forces_full_suite(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b"django/core/validators.py\0"
                    b"django/contrib/admin/static/admin/js/core.js\0"
                ),
                stderr=b"",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=_django_failure_output("candidate failure"),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    selection = runner._select_visible_pytest_targets(
        {},
        {"container_id": "sandbox-123", "repo": "django/django"},
        ["./tests/runtests.py", "--verbosity", "2"],
    )

    assert selection.mode == "full_suite_fallback"
    assert selection.changed_source_count == 1
    assert len(calls) == 1
    assert "not a mappable" in selection.reason


def test_django_untracked_helper_forces_full_suite(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"django/core/validators.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "ls-files"] and "--others" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    b".crucible_build.lock\0"
                    b"django/core/generated_helper.py\0"
                ),
                stderr=b"",
            )
        raise AssertionError("untracked coverage failure must stop selection")

    monkeypatch.setattr(subprocess, "run", fake_run)
    selection = runner._select_visible_pytest_targets(
        {},
        {"container_id": "sandbox-123", "repo": "django/django"},
        ["./tests/runtests.py", "--verbosity", "2"],
    )

    assert selection.mode == "full_suite_fallback"
    assert selection.changed_source_count == 1
    assert "untracked path" in selection.reason
    assert len(calls) == 2


def _failure_outcome(node_id: str, reason: str) -> _PytestOutcome:
    output = (
        "=========================== short test summary info "
        "============================\n"
        f"FAILED {node_id} - {reason}\n"
        "===================== 1 failed, 458 passed in 2.0s "
        "=====================\n"
    )
    return _pytest_outcome(subprocess.CompletedProcess([], 1, output, ""))


def test_clean_base_unchanged_astropy_13236_failure_is_nonblocking():
    node_id = (
        "astropy/table/tests/test_table.py::TestTable::"
        "test_values_equal_part1"
    )
    candidate = _failure_outcome(
        node_id, "Failed: DID NOT RAISE <class 'TypeError'>"
    )
    baseline = _failure_outcome(
        node_id, "Failed: DID NOT RAISE <class 'TypeError'>"
    )

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="a" * 64
    )

    assert comparison.mode == "preexisting_only"
    assert comparison.effective_returncode == 0
    assert comparison.preexisting_failures == 1
    assert comparison.new_failures == 0
    assert comparison.worsened_failures == 0


def test_clean_base_same_node_with_different_failure_remains_blocking():
    node_id = "tests/test_widget.py::test_value"
    candidate = _failure_outcome(node_id, "AssertionError: candidate regression")
    baseline = _failure_outcome(node_id, "AssertionError: known base failure")

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=True, cache_key="b" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1
    assert comparison.worsened_failures == 1


def test_clean_base_pass_makes_candidate_failure_new_and_blocking():
    candidate = _failure_outcome(
        "tests/test_widget.py::test_value", "AssertionError"
    )
    baseline = _pytest_outcome(subprocess.CompletedProcess(
        [], 0, "459 passed in 2.0s\n", ""
    ))

    comparison = _compare_pytest_outcomes(
        candidate, baseline, cache_hit=False, cache_key="c" * 64
    )

    assert comparison.mode == "regressed"
    assert comparison.effective_returncode == 1
    assert comparison.new_failures == 1


@pytest.mark.parametrize(
    ("mode", "command"),
    [
        (
            "preserved_explicit_test_args",
            ["pytest", "-rA", "tests/test_widget.py"],
        ),
        ("full_suite_fallback", ["pytest", "-rA"]),
    ],
)
def test_clean_base_comparison_covers_every_direct_pytest_selection_mode(
    monkeypatch,
    mode,
    command,
):
    candidate = _failure_outcome(
        "tests/test_widget.py::test_value", "AssertionError: base failure"
    )
    baseline = _failure_outcome(
        "tests/test_widget.py::test_value", "AssertionError: base failure"
    )
    monkeypatch.setattr(
        runner,
        "_cached_baseline_outcome",
        lambda identity, callback, **kwargs: (baseline, True, "d" * 64, ""),
    )

    comparison = runner._baseline_comparison(
        {
            "instance_id": "org__repo-1",
            "repo": "org/repo",
            "version": "1",
            "image_digest": "image@sha256:" + "a" * 64,
        },
        "b" * 40,
        command,
        _VisibleTestSelection(mode=mode, reason="test"),
        candidate,
    )

    assert comparison.mode == "preexisting_only"
    assert comparison.effective_returncode == 0


def test_targeted_failure_without_immutable_identity_fails_closed(monkeypatch):
    def fake_run(command, **kwargs):
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=b"pkg/widget.py\0", stderr=b""
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command, 0, stdout=b"pkg/tests/test_widget.py\0", stderr=b""
            )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=(
                "=========================== short test summary info "
                "============================\n"
                "FAILED pkg/tests/test_widget.py::test_value - AssertionError\n"
                "========================= 1 failed in 0.1s "
                "=========================\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc, payload = _invoke({
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "swebench_metadata": {"container_id": "sandbox-123"},
    })

    assert rc == 1
    assert payload["values"]["returncode"] == 1.0
    assert payload["values"]["visible_test_baseline_unavailable"] == 1.0
    assert "mode=unavailable" in payload["raw_output"]


def test_clean_base_cache_reuses_only_exact_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CRUCIBLE_SWEBENCH_BASELINE_CACHE", str(tmp_path / "baselines")
    )
    baseline = _failure_outcome(
        "tests/test_widget.py::test_value", "AssertionError"
    )
    identity = {
        "schema": "test",
        "image_digest": "image@sha256:" + "a" * 64,
        "base_commit": "b" * 40,
        "runner_sha256": "c" * 64,
        "command": ["pytest", "tests/test_widget.py"],
    }
    calls = 0

    def run_baseline():
        nonlocal calls
        calls += 1
        return baseline

    first, first_hit, first_key, first_error = _cached_baseline_outcome(
        identity, run_baseline
    )
    second, second_hit, second_key, second_error = _cached_baseline_outcome(
        identity, run_baseline
    )
    changed_identity = {
        **identity,
        "image_digest": "image@sha256:" + "d" * 64,
    }
    third, third_hit, third_key, third_error = _cached_baseline_outcome(
        changed_identity, run_baseline
    )

    assert first == second == third == baseline
    assert not first_hit
    assert second_hit
    assert not third_hit
    assert first_key == second_key
    assert third_key != first_key
    assert first_error == second_error == third_error == ""
    assert calls == 2


def test_clean_base_identity_binds_execution_recipe_and_selection():
    metadata = {
        "instance_id": "org__repo-1",
        "repo": "org/repo",
        "version": "1",
        "image_digest": "image@sha256:" + "a" * 64,
    }
    selection = _VisibleTestSelection(
        mode="targeted",
        reason="test",
        targets=("pkg/tests/test_widget.py",),
    )
    command = ["pytest", "-rA", *selection.targets]

    identity = _baseline_identity(metadata, "b" * 40, command, selection)
    changed_command = _baseline_identity(
        metadata, "b" * 40, ["pytest", "-q", *selection.targets], selection
    )
    changed_selection = _baseline_identity(
        metadata,
        "b" * 40,
        ["pytest", "-rA", "pkg/tests/test_other.py"],
        _VisibleTestSelection(
            mode="targeted",
            reason="test",
            targets=("pkg/tests/test_other.py",),
        ),
    )

    assert identity is not None
    assert changed_command is not None
    assert changed_selection is not None
    assert identity["image_digest"] == metadata["image_digest"]
    assert identity["base_commit"] == "b" * 40
    assert identity["instance_id"] == metadata["instance_id"]
    assert identity["outcome_parser"] == {
        "schema": runner._OUTCOME_PARSER_SCHEMA,
        "kind": "pytest",
    }
    assert identity["environment"]["source_recipe"] == (
        "immutable-image-provisioned-plus-git-object-base"
    )
    assert identity["command"] == command
    assert identity["selected_targets"] == list(selection.targets)
    assert len(identity["runner_sha256"]) == 64
    assert _identity_key(identity) != _identity_key(changed_command)
    assert _identity_key(identity) != _identity_key(changed_selection)

    django_identity = _baseline_identity(
        {
            **metadata,
            "instance_id": "django__django-10097",
            "repo": "django/django",
            "version": "2.2",
        },
        "b" * 40,
        ["./tests/runtests.py", "--verbosity", "2", "validators"],
        _VisibleTestSelection(
            mode="targeted", reason="test", targets=("validators",)
        ),
    )
    assert django_identity is not None
    assert django_identity["outcome_parser"]["kind"] == "django-unittest"
    assert django_identity["environment"]["source_recipe"] == (
        "git-object-base-only"
    )


def test_astropy_13236_runner_reuses_clean_base_and_reports_delta(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "_prepare_immutable_pytest_seed",
        lambda **kwargs: ("seed-volume", "seed-container"),
    )
    monkeypatch.setenv(
        "CRUCIBLE_SWEBENCH_BASELINE_CACHE", str(tmp_path / "baselines")
    )
    instance_id = "astropy__astropy-13236"
    image_digest = "image@sha256:" + "f" * 64
    authoring_workspace = tmp_path / "authoring"
    (authoring_workspace / ".git").mkdir(parents=True)
    calls: list[list[str]] = []
    failure_output = (
        "=========================== short test summary info "
        "============================\n"
        "FAILED astropy/table/tests/test_table.py::TestTable::"
        "test_values_equal_part1 - Failed: DID NOT RAISE <class 'TypeError'>\n"
        "===================== 1 failed, 458 passed in 2.0s "
        "=====================\n"
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["inspect", "--format"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_authoring_inspect_payload(
                    instance_id=instance_id,
                    image_digest=image_digest,
                    workspace_source=authoring_workspace,
                ),
                stderr="",
            )
        if command[1:3] == ["ps", "-aq"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        if command[5:7] == ["git", "diff"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=b"astropy/table/table.py\0", stderr=b""
            )
        if command[5:7] == ["git", "ls-files"]:
            if "--others" in command:
                return subprocess.CompletedProcess(
                    command, 0, stdout=b"", stderr=b""
                )
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"astropy/table/tests/test_table.py\0",
                stderr=b"",
            )
        if command[5:7] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="e" * 40 + "\n", stderr=""
            )
        return subprocess.CompletedProcess(
            command, 1, stdout=failure_output, stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    parameters = {
        "role": "dev_verify",
        "test_command": ["pytest", "-rA"],
        "swebench_metadata": {
            "container_id": "sandbox-123",
            "instance_id": instance_id,
            "repo": "astropy/astropy",
            "version": "4.3",
            "image_digest": image_digest,
        },
    }

    first_rc, first_payload = _invoke(parameters)
    baseline_command = next(command for command in calls if command[1] == "run")
    first_baseline_runs = sum(command[1] == "run" for command in calls)
    second_rc, second_payload = _invoke(parameters)
    all_baseline_runs = sum(command[1] == "run" for command in calls)

    assert first_rc == second_rc == 0
    assert first_payload["values"]["visible_test_candidate_returncode"] == 1.0
    assert first_payload["values"]["visible_test_baseline_cache_hit"] == 0.0
    assert second_payload["values"]["visible_test_baseline_cache_hit"] == 1.0
    assert first_payload["values"]["visible_test_preexisting_failures"] == 1.0
    assert first_payload["values"]["returncode"] == 0.0
    assert first_payload["values"]["tests_failed"] == 0.0
    assert first_payload["values"]["tests_errored"] == 0.0
    assert first_payload["values"]["visible_test_candidate_tests_failed"] == 1.0
    assert first_payload["values"]["visible_test_candidate_tests_errored"] == 0.0
    assert "mode=preexisting_only" in first_payload["raw_output"]
    assert first_baseline_runs == all_baseline_runs == 1
    assert "--network" in baseline_command
    assert baseline_command[baseline_command.index("--network") + 1] == "none"
    assert "--cap-drop" in baseline_command
    assert baseline_command[baseline_command.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in baseline_command
    assert baseline_command[baseline_command.index("--user") + 1] == (
        f"{os.getuid()}:{os.getgid()}"
    )
    assert "--cap-add" not in baseline_command
    assert "--volumes-from" not in baseline_command
    source_mounts = [
        baseline_command[index + 1]
        for index, value in enumerate(baseline_command)
        if value == "--mount"
    ]
    assert source_mounts == [
        "type=bind,source="
        + str(authoring_workspace / ".git")
        + ",target="
        + runner._BASELINE_SOURCE_GIT_DIR
        + ",readonly,bind-propagation=rprivate",
        "type=volume,source=seed-volume,target="
        + runner._BASELINE_TESTBED
        + ",volume-nocopy",
    ]
    assert not any(
        baseline_command[index + 1].startswith(runner._BASELINE_TESTBED + ":")
        for index, value in enumerate(baseline_command)
        if value == "--tmpfs"
    )
    assert "git reset --hard --quiet " + "e" * 40 in baseline_command[-1]
    assert "git clean -ffd --quiet" in baseline_command[-1]
    assert "git clean -ffdx --quiet" not in baseline_command[-1]
    assert "pytest -rA astropy/table/tests/test_table.py" in baseline_command[-1]

    # Exercise the consumer contract, not only the runner's process exit: the
    # generic build gate must not reinterpret the preserved raw candidate
    # failure count as a partial failure.
    from crucible.workflows import _build_verify

    gate_result = _build_verify.RunnerResult(
        role="dev_verify",
        runner="crucible.products.swebench.runners.swebench_pytest",
        returncode=first_rc,
        values=first_payload["values"],
        raw_output=json.dumps(first_payload).encode(),
        stderr=b"",
        duration_s=first_payload["values"]["duration_s"],
    )
    monkeypatch.setattr(
        _build_verify, "dev_verify", lambda *args, **kwargs: gate_result
    )
    gate = _build_verify.build_and_verify(
        object(), cwd=str(tmp_path), gate="dev_verify"
    )
    assert gate.classification == "success"


def test_clean_baseline_uses_unique_managed_bounded_sandbox(
    tmp_path, monkeypatch,
):
    calls: list[list[str]] = []
    reconciled_scopes: list[str] = []
    nonces = iter(("1" * 32, "2" * 32))
    instance_id = "org__repo-1"
    image_digest = "image@sha256:" + "f" * 64
    authoring_workspace = tmp_path / "authoring"
    (authoring_workspace / ".git").mkdir(parents=True)

    monkeypatch.setattr(runner.secrets, "token_hex", lambda size: next(nonces))
    monkeypatch.setattr(
        runner,
        "reconcile_orphaned_containers",
        lambda *, scope, run_docker: reconciled_scopes.append(scope) or (),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_immutable_pytest_seed",
        lambda **kwargs: ("seed-volume", "seed-container"),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["inspect", "--format"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_authoring_inspect_payload(
                    instance_id=instance_id,
                    image_digest=image_digest,
                    workspace_source=authoring_workspace,
                ),
                stderr="",
            )
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        if command[1:3] == ["volume", "rm"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        assert command[1] == "run"
        return subprocess.CompletedProcess(
            command, 0, stdout="3 passed in 0.1s\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    metadata = {
        "container_id": "authoring-123",
        "instance_id": instance_id,
        "image_digest": image_digest,
    }
    outcomes = [
        _run_clean_baseline(
            metadata,
            "e" * 40,
            ["pytest", "-rA", "tests/test_widget.py"],
            "a" * 64,
        )
        for _ in range(2)
    ]

    assert all(outcome is not None and outcome.returncode == 0 for outcome in outcomes)
    run_commands = [command for command in calls if command[1] == "run"]
    assert len(run_commands) == 2
    names = [command[command.index("--name") + 1] for command in run_commands]
    assert names[0] != names[1]
    inspect_commands = [command for command in calls if command[1] == "inspect"]
    assert len(inspect_commands) == 2
    assert all(command[-1] == "authoring-123" for command in inspect_commands)
    assert reconciled_scopes == [
        runner._BASELINE_CONTAINER_SCOPE,
        runner._BASELINE_CONTAINER_SCOPE,
    ]

    for command in run_commands:
        assert "--read-only" in command
        assert command[command.index("--entrypoint") + 1] == "bash"
        assert command[command.index("--user") + 1] == (
            f"{os.getuid()}:{os.getgid()}"
        )
        assert "--cap-add" not in command
        assert "--volumes-from" not in command
        source_mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--mount"
        ]
        assert source_mounts == [
            "type=bind,source="
            + str(authoring_workspace / ".git")
            + ",target="
            + runner._BASELINE_SOURCE_GIT_DIR
            + ",readonly,bind-propagation=rprivate",
            "type=volume,source=seed-volume,target="
            + runner._BASELINE_TESTBED
            + ",volume-nocopy",
        ]
        assert "type=bind,source=" not in source_mounts[-1]
        labels = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--label"
        ]
        assert "io.crucible.managed=true" in labels
        assert any(label.startswith("io.crucible.owner-pid=") for label in labels)
        assert any(
            label == "io.crucible.visible-baseline-key=" + "a" * 64
            for label in labels
        )
        tmpfs_mounts = [
            command[index + 1]
            for index, value in enumerate(command)
            if value == "--tmpfs"
        ]
        assert len(tmpfs_mounts) == 1
        assert any(
            mount.startswith("/tmp:") and "size=4096m" in mount
            for mount in tmpfs_mounts
        )
        shell_command = command[-1]
        assert str(authoring_workspace) not in shell_command
        assert (
            "cd "
            + runner._BASELINE_SOURCE_GIT_DIR
            + " && tar -cf - . | tar --no-same-owner -xf - -C "
            + runner._BASELINE_TESTBED
            + "/.git"
            in shell_command
        )
        assert "tar --no-same-owner" in shell_command
        assert "git reset --hard --quiet " + "e" * 40 in command[-1]
        assert "git clean -ffd --quiet" in command[-1]
        assert "git clean -ffdx --quiet" not in command[-1]
        assert (
            'test -z "$(git status --porcelain=v1 --untracked-files=all)"'
            in shell_command
        )
        assert shell_command.index("git clean -ffd --quiet") < (
            shell_command.rindex("cd " + runner._BASELINE_TESTBED)
        )
        assert "pytest -rA tests/test_widget.py" in command[-1]

    removed_names = [
        command[-1] for command in calls if command[1:3] == ["rm", "-f"]
    ]
    assert removed_names == [
        names[0], "seed-container", names[1], "seed-container"
    ]


def test_django_clean_baseline_seeds_only_from_read_only_git_objects(
    tmp_path,
    monkeypatch,
):
    calls: list[list[str]] = []
    instance_id = "django__django-unmapped"
    image_digest = "image@sha256:" + "f" * 64
    authoring_workspace = tmp_path / "authoring"
    (authoring_workspace / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        runner,
        "reconcile_orphaned_containers",
        lambda *, scope, run_docker: (),
    )

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["inspect", "--format"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=_authoring_inspect_payload(
                    instance_id=instance_id,
                    image_digest=image_digest,
                    workspace_source=authoring_workspace,
                ),
                stderr="",
            )
        if command[1:3] == ["rm", "-f"]:
            return subprocess.CompletedProcess(
                command, 0, stdout="", stderr=""
            )
        assert command[1] == "run"
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "----------------------------------------------------------------------\n"
                "Ran 2 tests in 0.001s\n\nOK\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    outcome = _run_clean_baseline(
        {
            "container_id": "authoring-123",
            "instance_id": instance_id,
            "repo": "django/django",
            "version": "3.0",
            "image_digest": image_digest,
        },
        "e" * 40,
        ["./tests/runtests.py", "--verbosity", "2"],
        "a" * 64,
    )

    assert outcome is not None
    assert outcome.returncode == 0
    run_command = next(command for command in calls if command[1] == "run")
    source_mounts = [
        run_command[index + 1]
        for index, value in enumerate(run_command)
        if value == "--mount"
    ]
    assert source_mounts == [
        "type=bind,source="
        + str(authoring_workspace / ".git")
        + ",target="
        + runner._BASELINE_SOURCE_GIT_DIR
        + ",readonly,bind-propagation=rprivate"
    ]
    assert runner._BASELINE_SOURCE_WORKTREE not in run_command[-1]
    assert "git clean -ffdx --quiet" in run_command[-1]
    assert "git clean -ffd --quiet" not in run_command[-1]


@pytest.mark.parametrize(
    "tamper",
    [
        "stale_owner",
        "unexpected_mount",
        "wrong_image",
        "shared_propagation",
        "mismatched_git_source",
        "symlink_git_source",
    ],
)
def test_clean_baseline_rejects_untrusted_authoring_bind_source(
    tmp_path, monkeypatch,
    tamper,
):
    instance_id = "org__repo-1"
    image_digest = "image@sha256:" + "f" * 64
    authoring_workspace = tmp_path / "authoring"
    (authoring_workspace / ".git").mkdir(parents=True)
    inspection = json.loads(_authoring_inspect_payload(
        instance_id=instance_id,
        image_digest=image_digest,
        workspace_source=authoring_workspace,
    ))
    if tamper == "stale_owner":
        inspection["Config"]["Labels"]["io.crucible.owner-start"] = "0"
    elif tamper == "unexpected_mount":
        inspection["Mounts"].append({
            "Type": "bind",
            "Destination": "/unexpected-host-path",
            "RW": False,
        })
    elif tamper == "wrong_image":
        inspection["Config"]["Image"] = "other@sha256:" + "0" * 64
    elif tamper == "shared_propagation":
        inspection["Mounts"][1]["Propagation"] = "rshared"
    elif tamper == "mismatched_git_source":
        other_git = tmp_path / "other.git"
        other_git.mkdir()
        inspection["Mounts"][1]["Source"] = str(other_git)
    else:
        real_git = authoring_workspace / "real.git"
        real_git.mkdir()
        (authoring_workspace / ".git").rmdir()
        (authoring_workspace / ".git").symlink_to(real_git, target_is_directory=True)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["inspect", "--format"]
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(inspection), stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    outcome = _run_clean_baseline(
        {
            "container_id": "authoring-123",
            "instance_id": instance_id,
            "image_digest": image_digest,
        },
        "e" * 40,
        ["pytest", "tests/test_widget.py"],
        "a" * 64,
    )

    assert outcome is None
    assert len(calls) == 1


def test_clean_baseline_requires_active_authoring_container(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("Docker must not run"),
    )

    assert _run_clean_baseline(
        {"image_digest": "image@sha256:" + "f" * 64},
        "e" * 40,
        ["pytest", "tests/test_widget.py"],
        "a" * 64,
    ) is None


def test_build_is_provisioned_without_subprocess(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("build subprocess attempted"),
    )
    rc, payload = _invoke({
        "role": "build",
        "swebench_metadata": {"container_id": "sandbox"},
    })
    assert rc == 0
    assert payload["values"]["swebench_build_skipped"] == 1.0


def test_baseline_cache_lock_obeys_runner_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "CRUCIBLE_SWEBENCH_BASELINE_CACHE", str(tmp_path / "baselines")
    )

    def contended_lock(_descriptor, operation):
        if operation & fcntl.LOCK_UN:
            return None
        raise BlockingIOError

    monkeypatch.setattr(fcntl, "flock", contended_lock)
    started = time.monotonic()
    outcome, cache_hit, _key, error = _cached_baseline_outcome(
        {"schema": "deadline-test"},
        lambda: pytest.fail("a contended cache must not start a baseline"),
        deadline=started + 0.02,
    )

    assert outcome is None
    assert not cache_hit
    assert error == "baseline cache lock deadline expired"
    assert time.monotonic() - started < 0.5


def test_docker_cleanup_actions_share_one_absolute_deadline(monkeypatch):
    observed_timeouts: list[float] = []
    clock = iter((90.0, 96.0))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))

    def fake_run(command, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner._best_effort_docker_cleanup(
        ["rm", "-f", "baseline"], deadline=100.0
    )
    runner._best_effort_docker_cleanup(
        ["volume", "rm", "-f", "seed"], deadline=100.0
    )

    assert observed_timeouts == [10.0, 4.0]


def test_visible_test_pipeline_threads_one_runner_deadline(monkeypatch):
    deadline = time.monotonic() + 60.0
    observed: list[tuple[str, float | None]] = []
    selection = _VisibleTestSelection(mode="targeted", reason="test")

    monkeypatch.setattr(
        runner,
        "_select_visible_pytest_targets",
        lambda parameters, metadata, command, *, deadline=None: (
            observed.append(("selection", deadline)) or selection
        ),
    )
    monkeypatch.setattr(
        runner,
        "_container_base_commit",
        lambda metadata, *, deadline=None: (
            observed.append(("commit", deadline)) or "b" * 40
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_candidate_tests",
        lambda metadata, command, *, deadline: (
            observed.append(("candidate", deadline))
            or subprocess.CompletedProcess(
                command, 0, "1 passed in 0.1s\n", ""
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "_baseline_comparison",
        lambda metadata, base_commit, command, selection, candidate,
        *, deadline=None: (
            observed.append(("baseline", deadline))
            or _BaselineComparison(mode="not_needed", effective_returncode=0)
        ),
    )

    runner._run_visible_tests(
        {"test_command": ["pytest"]},
        {
            "container_id": "sandbox",
            "image_digest": "image@sha256:" + "a" * 64,
        },
        deadline=deadline,
    )

    assert observed == [
        ("selection", deadline),
        ("commit", deadline),
        ("candidate", deadline),
        ("baseline", deadline),
    ]
