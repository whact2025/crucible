# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.deepswe.runners.deepswe_dev_verify``.

Hermetic — fakes ``subprocess.run`` (docker exec) and feeds a
BenchConfig on a StringIO stdin. No real docker / container / tests.

Coverage:
- per-language command shape (python / rust / go / ts / js)
- pytest count parsing → tests_passed / tests_failed
- cargo count parsing (summed across crate result lines)
- success requires rc==0 AND tests_failed==0
- failing tests (rc!=0) → non-zero exit
- pytest rc==0 but failed>0 → still non-zero (defensive)
- unknown language → exit 2
- missing container_id metadata → exit 2
- role refusal (build)
- timeout surfaces as returncode 124
- malformed stdin → exit 2
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shlex
import subprocess
from typing import Any

import pytest

from crucible.common.runners.pytest_policy import PYTEST_DEV_VERIFY_MAXFAIL_ARG
from crucible.products.deepswe.runners import deepswe_dev_verify as dv


@pytest.fixture(autouse=True)
def _logical_contained_exec_adapter(monkeypatch):
    """Keep runner-outcome tests hermetic at the containment seam.

    Process-tree staging and teardown have dedicated tests.  This adapter
    retains the historical one-call argv shape so this module can continue to
    assert the language command rendered by the runner without emulating a
    Docker daemon and its inspect/staging lifecycle.
    """

    def _run(container_id, command, *, timeout, cleanup_timeout):
        bounded_command = (
            "exec timeout --signal=TERM "
            f"--kill-after={cleanup_timeout:.6f}s "
            f"{timeout:.6f}s bash -c {shlex.quote(command)}"
        )
        full = ["docker", "exec", container_id, "bash", "-c", bounded_command]
        try:
            result = subprocess.run(
                full,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout + cleanup_timeout,
            )
        except subprocess.TimeoutExpired:
            return 124, "", f"dev_verify timed out after {timeout:.6f}s"
        except FileNotFoundError as exc:
            return 127, "", f"docker not on PATH: {exc}"
        return result.returncode, result.stdout, result.stderr

    monkeypatch.setattr(dv, "run_contained_docker_exec", _run)


def _make_stdin(parameters: dict) -> io.StringIO:
    return io.StringIO(json.dumps({"parameters": parameters}))


def _fake_exec(rc: int = 0, stdout: str = "", stderr: str = ""):
    calls: list[list[str]] = []

    def _fake(args, **kwargs):
        calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args, returncode=rc, stdout=stdout, stderr=stderr,
        )

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


def _params(language: str = "python", container_id: str = "auth123") -> dict:
    return {
        "language": language,
        "role": "dev_verify",
        "deepswe_metadata": {"container_id": container_id},
    }


def _run(parameters: dict) -> tuple[int, dict]:
    out = io.StringIO()
    rc = dv.main(stdin=_make_stdin(parameters), stdout=out)
    payload = json.loads(out.getvalue()) if out.getvalue().strip() else {}
    return rc, payload


def _passing_output(language: str) -> str:
    if language == "python":
        return "1 passed in 0.10s"
    if language == "rust":
        return "test result: ok. 1 passed; 0 failed; 0 ignored"
    if language == "go":
        return json.dumps({
            "Action": "pass",
            "Package": "example.test/pkg",
            "Test": "TestVisible",
        })
    if language in {"typescript", "javascript"}:
        return "Tests:       1 passed, 1 total"
    raise AssertionError(f"no passing fixture for {language}")


@pytest.mark.parametrize(
    ("language", "returncode", "output", "expected_key"),
    [
        pytest.param(
            "typescript",
            125,
            (
                "Error: write EPIPE\n"
                "crucible supervisor: command left background descendants; "
                "all descendants were terminated"
            ),
            "dev_verify_supervisor_unavailable",
            id="typescript-supervisor-epipe",
        ),
        pytest.param(
            "go",
            1,
            (
                "github.com/prometheus/compliance/remotewrite/@v/v1.mod: "
                "Get https://proxy.golang.org/module: dial tcp: lookup "
                "proxy.golang.org: network is unreachable"
            ),
            "dev_verify_dependency_unavailable",
            id="go-sealed-dependency",
        ),
    ],
)
def test_candidate_infrastructure_values_detect_runner_owned_failures(
    language,
    returncode,
    output,
    expected_key,
):
    values = dv._candidate_infrastructure_values(
        language=language,
        process_returncode=returncode,
        output=output,
        parser_kind="go-json" if language == "go" else "unsupported",
        clean_probe=True,
    )

    assert values[expected_key] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0


@pytest.mark.parametrize(
    ("language", "returncode", "output", "signature_key"),
    [
        pytest.param(
            "typescript",
            125,
            (
                "Error: write EPIPE\n"
                "crucible supervisor: command left background descendants; "
                "all descendants were terminated"
            ),
            "dev_verify_supervisor_signature",
            id="candidate-typescript-epipe",
        ),
        pytest.param(
            "go",
            1,
            (
                "go: downloading github.com/acme/module v1.0.0\n"
                "Get https://proxy.golang.org/acme: dial tcp: lookup "
                "proxy.golang.org: network is unreachable"
            ),
            "dev_verify_dependency_signature",
            id="candidate-go-dependency",
        ),
    ],
)
def test_candidate_only_infrastructure_signatures_remain_actionable(
    language,
    returncode,
    output,
    signature_key,
):
    values = dv._candidate_infrastructure_values(
        language=language,
        process_returncode=returncode,
        output=output,
        parser_kind="go-json" if language == "go" else "unsupported",
    )

    assert values[signature_key] == 1.0
    assert values["dev_verify_supervisor_unavailable"] == 0.0
    assert values["dev_verify_dependency_unavailable"] == 0.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0


def test_exact_clean_provenance_admits_matching_candidate_signature():
    command = dv._LANG_TO_COMMAND["go"]
    clean_values = {
        "dev_verify_supervisor_unavailable": 0.0,
        "dev_verify_dependency_unavailable": 1.0,
    }
    provenance = dv._clean_infrastructure_provenance(
        language="go",
        command=command,
        values=clean_values,
    )
    metadata = {dv.CLEAN_INFRASTRUCTURE_PROVENANCE_KEY: provenance}

    flags = dv._validated_clean_infrastructure_flags(
        metadata,
        language="go",
        command=command,
    )
    values = dv._candidate_infrastructure_values(
        language="go",
        process_returncode=1,
        output=(
            "go: downloading github.com/acme/module v1.0.0\n"
            "proxy.golang.org: network is unreachable"
        ),
        parser_kind="go-json",
        proven_clean_flags=flags,
    )

    assert flags == frozenset({"dev_verify_dependency_unavailable"})
    assert values["dev_verify_dependency_unavailable"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert not dv._validated_clean_infrastructure_flags(
        metadata,
        language="go",
        command=command + " --candidate-modified",
    )


def test_candidate_infrastructure_values_leave_model_compile_error_actionable():
    values = dv._candidate_infrastructure_values(
        language="typescript",
        process_returncode=1,
        output="src/index.ts(4,1): error TS1005: '}' expected.",
        parser_kind="jest",
    )

    assert values["dev_verify_infrastructure_unavailable"] == 0.0


def test_candidate_supervisor_descendants_from_model_remain_actionable():
    values = dv._candidate_infrastructure_values(
        language="typescript",
        process_returncode=125,
        output=(
            "model test server stayed alive\n"
            "crucible supervisor: command left background descendants; "
            "all descendants were terminated"
        ),
        parser_kind="jest",
    )

    assert values["dev_verify_supervisor_unavailable"] == 0.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0


def test_candidate_command_not_found_remains_actionable():
    values = dv._candidate_infrastructure_values(
        language="typescript",
        process_returncode=127,
        output="/bin/bash: npm: command not found",
        parser_kind="jest",
    )

    assert values["dev_verify_executable_unavailable"] == 0.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0


def test_unsupported_runner_parser_is_infrastructure_unavailable():
    values = dv._candidate_infrastructure_values(
        language="typescript",
        process_returncode=1,
        output="suite exited before emitting a supported structured summary",
        parser_kind="unsupported",
        clean_probe=True,
    )

    assert values["dev_verify_parser_unavailable"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0


def test_candidate_only_unsupported_parser_remains_actionable():
    values = dv._candidate_infrastructure_values(
        language="typescript",
        process_returncode=1,
        output="candidate changed its reporter and emitted no supported frame",
        parser_kind="unsupported",
    )

    assert values["dev_verify_parser_signature"] == 1.0
    assert values["dev_verify_parser_unavailable"] == 0.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0


# ---------- clean-baseline containment ----------


def test_clean_baseline_exec_probes_use_contained_path(monkeypatch):
    image = "example.invalid/repo@sha256:" + "a" * 64
    deadline = dv._RunnerDeadline.from_parameters({"runner_timeout_s": 10})
    host_calls: list[list[str]] = []
    contained_calls: list[tuple[str, str, float, object]] = []

    def fake_host(args, **_kwargs):
        host_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout=image, stderr="")

    def fake_contained(container_id, command, *, timeout, deadline):
        contained_calls.append((container_id, command, timeout, deadline))
        output = "b" * 40 if "rev-parse" in command else ""
        return 0, output, ""

    monkeypatch.setattr(dv.subprocess, "run", fake_host)
    monkeypatch.setattr(dv, "_run_docker_exec", fake_contained)

    assert dv._probe_clean_baseline_container(
        {
            "baseline_container_id": "baseline-id",
            "image_digest": image,
            "base_commit": "b" * 40,
        },
        deadline=deadline,
    ) == ""
    assert len(host_calls) == 1
    assert host_calls[0][:2] == ["docker", "inspect"]
    assert all("exec" not in call for call in host_calls)
    assert [call[1] for call in contained_calls] == [
        "cd /app && git rev-parse --verify 'HEAD^{commit}'",
        "cd /app && git status --porcelain=v1 -z --untracked-files=all",
    ]
    assert all(call[2] == 30.0 and call[3] is deadline for call in contained_calls)


# ---------- command shape per language ----------


@pytest.mark.parametrize("language,fragment", [
    ("python", f"pytest {PYTEST_DEV_VERIFY_MAXFAIL_ARG}"),
    ("rust", "cargo test"),
    ("go", "go test -json ./..."),
    ("typescript", "npm test"),
    ("javascript", "npm test"),
])
def test_command_shape_per_language(language, fragment, monkeypatch):
    fake = _fake_exec(rc=0, stdout=_passing_output(language))
    monkeypatch.setattr(subprocess, "run", fake)
    _run(_params(language=language))
    # docker exec <id> bash -c "<command>"
    cmd = fake.calls[0]
    assert cmd[:3] == ["docker", "exec", "auth123"]
    assert cmd[3:5] == ["bash", "-c"]
    assert "cd /app &&" in cmd[5]
    assert fragment in cmd[5]


def test_direct_deepswe_rust_runner_honors_nested_spec_parameters(monkeypatch):
    fake = _fake_exec(rc=0, stdout="test result: ok. 1 passed; 0 failed; 0 ignored")
    monkeypatch.setattr(subprocess, "run", fake)
    parameters = _params(language="rust")
    parameters["per_language"] = {
        "rust": {
            "crate": "fabric-render-ovrtx",
            "cargo_features": ["ovrtx"],
            "cargo_test_args": ["--", "--test-threads=1"],
        },
    }

    rc, _payload = _run(parameters)

    assert rc == 0
    command = fake.calls[0][5]
    assert "cargo build --quiet -p fabric-render-ovrtx --features ovrtx" in command
    assert "cargo test -p fabric-render-ovrtx --features ovrtx" in command
    assert "-- --test-threads=1" in command


@pytest.mark.parametrize(
    "selection",
    [
        {"crate": 7},
        {"crate": ""},
        {"cargo_features": "ovrtx"},
        {"cargo_features": ["ovrtx", 7]},
        {"cargo_test_args": "-- --test-threads=1"},
    ],
)
def test_deepswe_rust_runner_rejects_malformed_selection(selection):
    parameters = _params(language="rust")
    parameters.update(selection)

    rc, payload = _run(parameters)

    assert rc == 2
    assert payload["values"]["returncode"] == 2.0
    assert "Rust runner parameter" in payload["raw_output"]


# ---------- compile-then-test (build is part of every language flow) ----------


@pytest.mark.parametrize("language,build_fragment", [
    ("rust", "cargo build"),
    ("go", "go build ./..."),
    ("typescript", "npm run build --if-present"),
    ("javascript", "npm run build --if-present"),
])
def test_build_gate_precedes_tests_per_language(language, build_fragment, monkeypatch):
    """The in-container gate compiles BEFORE testing (single `&&` command),
    so go/ts/js get the build gate they previously lacked. The build
    fragment must precede the test fragment in the one docker-exec command."""
    fake = _fake_exec(rc=0, stdout=_passing_output(language))
    monkeypatch.setattr(subprocess, "run", fake)
    _run(_params(language=language))
    cmd = fake.calls[0][5]
    assert build_fragment in cmd, cmd
    # compile runs before the test phase
    test_tok = {"rust": "cargo test", "go": "go test",
                "typescript": "npm test", "javascript": "npm test"}[language]
    assert cmd.index(build_fragment) < cmd.index(test_tok), cmd
    assert "&&" in cmd


def test_python_has_no_separate_build_step(monkeypatch):
    """python is interpreted — pytest collection is its import/compile gate,
    so no build prefix is injected (the command stays pytest-only)."""
    fake = _fake_exec(rc=0, stdout="==== 1 passed in 0.1s ====")
    monkeypatch.setattr(subprocess, "run", fake)
    _run(_params(language="python"))
    cmd = fake.calls[0][5]
    assert "pytest" in cmd
    assert "build" not in cmd


def test_build_failure_fails_the_gate(monkeypatch):
    """A compile failure (the `&&` short-circuits before tests) returns the
    build's non-zero rc — failing the dev_verify gate, so a signature/symbol
    deviation surfaces as a build error (the whole point of the gate)."""
    # tsc-style compile error: combined command returns non-zero, no test
    # summary in the output.
    fake = _fake_exec(
        rc=2, stdout="", stderr="src/index.ts(42,5): error TS2554: Expected 0 arguments.",
    )
    monkeypatch.setattr(subprocess, "run", fake)
    rc, payload = _run(_params(language="typescript"))
    assert rc != 0
    assert payload["values"]["returncode"] == 2.0
    # the build error rides in raw_output so the Defender sees it as feedback
    assert "TS2554" in payload["raw_output"]


# ---------- count parsing ----------


def test_pytest_counts_parsed(monkeypatch):
    out = "==== 5 passed, 2 skipped in 0.4s ===="
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0, stdout=out))
    rc, payload = _run(_params(language="python"))
    assert rc == 0
    assert payload["values"]["tests_passed"] == 5.0
    assert payload["values"]["tests_skipped"] == 2.0


def test_pytest_failed_nonzero_exit(monkeypatch):
    out = (
        "================ short test summary info ================\n"
        "FAILED tests/test_x.py::test_x - AssertionError\n"
        "==== 3 passed, 1 failed in 0.5s ===="
    )
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=1, stdout=out))
    rc, payload = _run(_params(language="python"))
    assert rc != 0
    assert payload["values"]["tests_failed"] == 1.0


def test_pytest_rc_zero_but_failed_counts_still_fails(monkeypatch):
    """Defensive: if rc==0 but the summary shows a failure, treat as
    failure rather than reporting success."""
    out = (
        "================ short test summary info ================\n"
        "FAILED tests/test_x.py::test_x - AssertionError\n"
        "==== 3 passed, 1 failed in 0.5s ===="
    )
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0, stdout=out))
    rc, payload = _run(_params(language="python"))
    assert rc != 0
    assert payload["values"]["tests_failed"] == 1.0


def test_python_exact_observation_propagates_bounded_failure_hint(monkeypatch):
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
        dv,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (
            observation_protocol.PYTEST_OBSERVER_FAILED,
            "bounded-supervisor-envelope",
            "",
        ),
    )

    observed = dv._python_exact_observations("authoring", {
        "breach_attack_ids": ["alias_style"],
        "breach_test_file": "tests/test_breach.py",
    })

    assert observed == [{
        "language": "python",
        "id": "tests/test_breach.py::test_alias_style",
        "status": "failed",
        "line": "tests/test_breach.py::test_alias_style",
        "failure": failure,
    }]


def test_python_exact_observation_oserror_is_typed_infrastructure(
    monkeypatch,
) -> None:
    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )

    def fake_exec(_container_id, command, **_kwargs):
        if "_crucible_exact_pytest_observer" in command:
            raise OSError("docker exec channel disappeared")
        return 0, "1 passed in 0.10s", ""

    monkeypatch.setattr(dv, "_run_docker_exec", fake_exec)
    parameters = _params(language="python")
    parameters.update({
        "breach_observations": True,
        "breach_attack_ids": ["transport"],
        "breach_test_file": "tests/test_breach.py",
    })

    rc, payload = _run(parameters)

    assert rc == 2
    assert payload[BREACH_OBSERVATION_FAILURE_KEY]["kind"] == (
        "infrastructure"
    )
    assert payload["test_observations"] == []


def test_cargo_counts_summed_across_crates(monkeypatch):
    out = (
        "test result: ok. 4 passed; 0 failed; 1 ignored; 0 measured\n"
        "test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured\n"
    )
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0, stdout=out))
    rc, payload = _run(_params(language="rust"))
    assert rc == 0
    assert payload["values"]["tests_passed"] == 10.0
    assert payload["values"]["tests_failed"] == 0.0
    assert payload["values"]["tests_skipped"] == 1.0


def test_language_read_from_metadata(monkeypatch):
    """Production threads language through deepswe_metadata (the starter
    spec carries no LanguagesConstraint, so _inject_languages never sets
    a top-level ``language``). Metadata must be honored even with no
    top-level key."""
    fake = _fake_exec(rc=0, stdout=_passing_output("rust"))
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "role": "dev_verify",
        "deepswe_metadata": {"container_id": "auth123", "language": "rust"},
    }
    rc, _ = _run(params)
    assert rc == 0
    assert "cargo test" in fake.calls[0][5]


def test_metadata_language_takes_precedence_over_top_level(monkeypatch):
    """When both are set, metadata wins (it's the production-authoritative
    source); the top-level read is only a direct/unit-invocation fallback."""
    fake = _fake_exec(rc=0, stdout=_passing_output("go"))
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "language": "python",
        "role": "dev_verify",
        "deepswe_metadata": {"container_id": "auth123", "language": "go"},
    }
    rc, _ = _run(params)
    assert rc == 0
    assert "go test -json ./..." in fake.calls[0][5]


def test_raw_output_emitted(monkeypatch):
    """The KPI payload carries a top-level ``raw_output`` (combined
    stdout+stderr, capped) — breach.py::_extract_pytest_output reads it
    to give the Defender round-over-round test feedback."""
    out = "==== 1 failed in 0.2s ====\nE   assert 1 == 2\n"
    monkeypatch.setattr(
        subprocess, "run", _fake_exec(rc=1, stdout=out, stderr="trace\n"),
    )
    _, payload = _run(_params(language="python"))
    assert "raw_output" in payload
    assert "1 failed" in payload["raw_output"]
    assert "trace" in payload["raw_output"]


def test_raw_output_capped_to_tier_repair_budget(monkeypatch):
    big = "x" * 20_000
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0, stdout=big))
    _, payload = _run(_params(language="go"))
    assert len(payload["raw_output"]) == 8_000


def test_python_failure_batch_retains_three_short_failure_details(monkeypatch):
    output = (
        "." * 4_200
        + "\n=================================== FAILURES "
        "===================================\n"
        "______________________________ test_first "
        "______________________________\n"
        "tests/test_generated.py:10: in test_first\n"
        "    assert actual == 'foo_bar'\n"
        "E   AssertionError: expected snake_case alias, got foobar\n"
        "______________________________ test_second "
        "_____________________________\n"
        "tests/test_generated.py:20: in test_second\n"
        "    make_shape(required_after_optional=True)\n"
        "E   ValueError: required field ordering fixture is invalid\n"
        "______________________________ test_third "
        "______________________________\n"
        "tests/test_generated.py:30: in test_third\n"
        "    assert actual == 'FOO_BAR'\n"
        "E   AssertionError: expected upper snake alias, got FOOBAR\n"
        "=========================== short test summary info "
        "============================\n"
        "FAILED tests/test_generated.py::test_first - AssertionError: "
        "expected snake_case alias\n"
        "FAILED tests/test_generated.py::test_second - ValueError: required "
        "field ordering fixture\n"
        "FAILED tests/test_generated.py::test_third - AssertionError: "
        "expected upper snake alias\n"
        "========================= 3 failed, 7 passed in 0.42s "
        "=========================\n"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_exec(rc=1, stdout=output),
    )

    rc, payload = _run(_params(language="python"))

    assert rc == 1
    assert payload["values"]["tests_failed"] == 3.0
    assert payload["values"]["tests_passed"] == 7.0
    feedback = payload["raw_output"]
    assert "expected snake_case alias, got foobar" in feedback
    assert "required field ordering fixture is invalid" in feedback
    assert "expected upper snake alias, got FOOBAR" in feedback
    assert "tests/test_generated.py::test_first" in feedback
    assert "tests/test_generated.py::test_second" in feedback
    assert "tests/test_generated.py::test_third" in feedback


def test_go_json_requires_observed_tests(monkeypatch):
    """Go uses machine-readable terminal events and rejects vacuous success."""
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_exec(rc=0, stdout=_passing_output("go")),
    )
    rc, payload = _run(_params(language="go"))
    assert rc == 0
    assert payload["values"]["returncode"] == 0.0
    assert payload["values"]["tests_passed"] == 1.0


def test_go_failure_returncode(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=2, stderr="FAIL\n"))
    rc, _ = _run(_params(language="go"))
    assert rc == 2


# ---------- error surfaces ----------


def test_unknown_language_exit_2(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0))
    rc, payload = _run(_params(language="cobol"))
    assert rc == 2
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 1.0
    assert payload["values"]["dev_verify_language_unavailable"] == 1.0


def test_missing_container_id_exit_2(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0))
    params = {"language": "python", "deepswe_metadata": {}}
    rc, payload = _run(params)
    assert rc == 2
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 1.0
    assert payload["values"]["dev_verify_container_unavailable"] == 1.0


def test_metadata_not_dict_exit_2(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0))
    params = {"language": "python", "deepswe_metadata": "nope"}
    rc, payload = _run(params)
    assert rc == 2
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 1.0
    assert payload["values"]["dev_verify_metadata_unavailable"] == 1.0


def test_role_build_refused(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_exec(rc=0))
    params = _params()
    params["role"] = "build"
    rc, _ = _run(params)
    assert rc == 2


def test_role_verify_tolerated(monkeypatch):
    """_build_verify injects the role it dispatched under; dev_verify
    is dispatched under the dev_verify role, but verify is tolerated
    so a role-injection mismatch doesn't hard-fail."""
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_exec(rc=0, stdout=_passing_output("python")),
    )
    params = _params()
    params["role"] = "verify"
    rc, _ = _run(params)
    assert rc == 0


def test_timeout_returncode_124(monkeypatch):
    def _timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout"))
    monkeypatch.setattr(subprocess, "run", _timeout)
    rc, payload = _run(_params())
    assert rc == 124
    assert payload["values"]["returncode"] == 124.0


def test_unusable_containment_source_is_terminal_before_parsing(monkeypatch):
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (
            dv.CONTAINMENT_UNUSABLE_RC,
            "",
            "durably unusable; quarantine_FAILED",
        ),
    )
    monkeypatch.setattr(
        dv,
        "parse_test_outcome",
        lambda *_args, **_kwargs: pytest.fail(
            "unusable source must not enter ordinary outcome parsing"
        ),
    )

    rc, payload = _run(_params())

    assert rc == dv.CONTAINMENT_UNUSABLE_RC
    assert payload["values"] == {
        "returncode": float(dv.CONTAINMENT_UNUSABLE_RC),
        "dev_verify_candidate_process_returncode": float(
            dv.CONTAINMENT_UNUSABLE_RC
        ),
        "dev_verify_source_unusable": 1.0,
        "dev_verify_infrastructure_unavailable": 1.0,
    }
    assert "quarantine_FAILED" in payload["raw_output"]


_PYTEST_FAILING_OUTPUT = (
    "================ short test summary info ================\n"
    "FAILED tests/test_api.py::test_contract - AssertionError: expected\n"
    "================ 1 failed in 0.10s ================\n"
)
_NOT_STARTED_VALUES = {
    "returncode": float(dv.CONTAINMENT_NOT_STARTED_RC),
    "dev_verify_candidate_process_returncode": float(
        dv.CONTAINMENT_NOT_STARTED_RC
    ),
    "dev_verify_command_not_started": 1.0,
    "dev_verify_infrastructure_unavailable": 1.0,
}


def test_command_that_never_started_is_terminal_before_parsing(monkeypatch):
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (
            dv.CONTAINMENT_NOT_STARTED_RC,
            "",
            "docker inspect -- failed rc=1: Error: No such object: auth123",
        ),
    )
    monkeypatch.setattr(
        dv,
        "parse_test_outcome",
        lambda *_args, **_kwargs: pytest.fail(
            "a command that never started has no outcome to parse"
        ),
    )

    rc, payload = _run(_params())

    assert rc == dv.CONTAINMENT_NOT_STARTED_RC
    assert payload["values"] == _NOT_STARTED_VALUES
    assert "No such object: auth123" in payload["raw_output"]
    assert payload["test_observations"] == []


def test_host_error_at_the_gate_command_is_infrastructure_unavailable(
    monkeypatch,
):
    def _host_error(*_args, **_kwargs):
        raise PermissionError(13, "helper cache is not writable")

    monkeypatch.setattr(dv, "run_contained_docker_exec", _host_error)
    monkeypatch.setattr(
        dv,
        "parse_test_outcome",
        lambda *_args, **_kwargs: pytest.fail(
            "a host error leaves no outcome to parse"
        ),
    )

    rc, payload = _run(_params())

    # The error may arise before or after the command started (the helper
    # cache, or spawning the Docker client for the attestation), so it is
    # reported as unavailable containment, not as a command that never ran.
    assert rc == 2
    assert payload["values"] == {
        "returncode": 2.0,
        "dev_verify_infrastructure_unavailable": 1.0,
        "dev_verify_containment_unavailable": 1.0,
    }
    assert "PermissionError" in payload["raw_output"]
    assert "helper cache is not writable" in payload["raw_output"]


@pytest.mark.parametrize(
    "breach_round",
    [
        pytest.param(False, id="tier-gate"),
        pytest.param(True, id="breach-gate"),
    ],
)
def test_command_that_never_started_is_not_fed_back_to_the_model(
    monkeypatch,
    tmp_path,
    breach_round,
):
    """Across the real runner subprocess: Docker fails before the command."""
    from crucible.workflows import _build_verify as bv
    from crucible.workflows import breach, tier
    from tests.test_build_verify import _DEEPSWE_DEV_VERIFY_RUNNER, _spec

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker-calls.log"
    fake_docker = bin_dir / "docker"
    fake_docker.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(docker_log))}\n"
        "printf 'Error: No such object: gone-container\\n' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSWE_HELPER_DIR", str(tmp_path / "helper-cache"),
    )

    extra_parameters: dict = {
        "deepswe_metadata": {
            "container_id": "gone-container",
            "language": "python",
        },
    }
    if breach_round:
        # The per-round request of the breach workflow, passed as it does.
        extra_parameters = bv.trusted_runner_overrides(
            {**extra_parameters, **_BREACH_GATE_PARAMETERS}
        )

    outcome = bv.build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        slice_id="s1",
        cwd=str(tmp_path),
        log_dir=tmp_path / "logs",
        extra_parameters=extra_parameters,
        gate="dev_verify",
    )

    assert outcome.classification == "dev_verify_unavailable"
    assert outcome.dev_verify is not None
    assert outcome.dev_verify.returncode == dv.CONTAINMENT_NOT_STARTED_RC
    assert outcome.dev_verify.values == _NOT_STARTED_VALUES
    assert not tier._is_model_retriable_build_failure(outcome)
    assert docker_log.read_text(encoding="utf-8").splitlines() == [
        "inspect -- gone-container",
    ]
    # A tier reads the classification.  A breach round reads the typed
    # failure, which ends the round before any model role sees the gate.
    failure = breach._gate_observation_failure(outcome)
    if breach_round:
        assert failure is not None
        assert failure["kind"] == "infrastructure"
        assert "contained command never started" in failure["detail"]
        assert "No such object: gone-container" in failure["detail"]
    else:
        assert failure is None


@pytest.mark.parametrize(
    ("process_rc", "stdout", "stderr", "expected_rc"),
    [
        pytest.param(
            125,
            "1 passed in 0.10s",
            (
                "crucible supervisor: command left background descendants; "
                "all descendants were terminated"
            ),
            2,
            id="descendants",
        ),
        pytest.param(124, "", "dev_verify timed out", 124, id="timeout"),
        pytest.param(1, _PYTEST_FAILING_OUTPUT, "", 1, id="failing"),
        pytest.param(
            127, "", "/bin/bash: pytest: command not found", 127,
            id="command-not-found",
        ),
    ],
)
def test_started_command_statuses_are_not_reported_as_not_started(
    monkeypatch,
    process_rc,
    stdout,
    stderr,
    expected_rc,
):
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_exec(rc=process_rc, stdout=stdout, stderr=stderr),
    )

    rc, payload = _run(_params())

    values = payload["values"]
    assert "dev_verify_command_not_started" not in values
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert values["dev_verify_candidate_process_returncode"] == float(process_rc)
    assert rc == expected_rc
    assert values["returncode"] == float(expected_rc)


# A gate run sends more contained commands after the gate's own: per declared
# test file a path-state probe and a focused command in the candidate (and in
# the clean-base container), and per attack an exact observation.

_DECLARED_TESTS = ("tests/test_declared.py", "tests/test_second.py")
_DOCKER_FAILED_FIRST = (
    dv.CONTAINMENT_NOT_STARTED_RC,
    "",
    "docker inspect -- failed rc=1: Cannot connect to the Docker daemon",
)


def _evidence_unavailable(rc: int, values: dict) -> bool:
    """Ask the build/verify layer whether a gate result lacks model evidence."""
    from crucible.workflows import _build_verify as bv

    return bv._dev_verify_evidence_unavailable(
        bv.RunnerResult(
            role="dev_verify",
            runner="crucible.products.deepswe.runners.deepswe_dev_verify",
            returncode=rc,
            values=values,
            raw_output=b"",
            stderr=b"",
            duration_s=0.0,
        )
    )


def _declared_parameters(*, baseline: bool) -> dict:
    metadata = {"container_id": "candidate", "language": "python"}
    if baseline:
        metadata.update({
            "baseline_container_id": "baseline",
            "image_digest": "registry.test/image@sha256:" + "a" * 64,
            "base_commit": "b" * 40,
        })
    return {
        "role": "dev_verify",
        "deepswe_metadata": metadata,
        dv.DECLARED_VERIFICATION_SURFACE_KEY: {
            "schema_version": "crucible.declared-verification-surface.v1",
            "slices": [{
                "slice_id": "declared-tests",
                "files_in_scope": [],
                "creates_paths": list(_DECLARED_TESTS),
            }],
        },
    }


def _declared_gate_exec(
    calls: list[tuple[str, str]],
    *,
    never_started: tuple[str, str],
    gate: tuple[int, str, str] = (0, "1 passed in 0.10s", ""),
):
    """Answer one gate run with declared files; one later command never starts."""

    def _exec(container_id, command, **_kwargs):
        if "readlink -f" in command:
            command_kind = "path-state"
        elif any(test_file in command for test_file in _DECLARED_TESTS):
            command_kind = "focused"
        else:
            command_kind = "suite"
        calls.append((container_id, command_kind))
        if (container_id, command_kind) == never_started:
            return _DOCKER_FAILED_FIRST
        if command_kind == "path-state":
            return 0, "safe\n", ""
        if (container_id, command_kind) == ("candidate", "suite"):
            return gate
        return 0, "1 passed in 0.10s", ""

    return _exec


_LATER_COMMANDS = [
    pytest.param("candidate", "path-state", id="candidate-path-state"),
    pytest.param("candidate", "focused", id="candidate-focused"),
    pytest.param("baseline", "path-state", id="baseline-path-state"),
    pytest.param("baseline", "focused", id="baseline-focused"),
]


@pytest.mark.parametrize(("container", "kind"), _LATER_COMMANDS)
def test_later_command_that_never_started_ends_the_declared_probes_unavailable(
    monkeypatch,
    tmp_path,
    container,
    kind,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    monkeypatch.setattr(
        dv,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        _declared_gate_exec(calls, never_started=(container, kind)),
    )

    rc, payload = _run(_declared_parameters(baseline=container == "baseline"))

    # The gate's own command ran and passed.  Docker failed before a later
    # command existed, which is no evidence about the candidate: the gate is
    # unavailable, and the Docker text is not a test failure to repair.
    assert rc == 2
    values = payload["values"]
    assert values["returncode"] == 2.0
    assert values["tests_passed"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert values["dev_verify_declared_probe_unavailable"] == 1.0
    assert _evidence_unavailable(rc, values)
    assert "status=runner_error" in payload["raw_output"]
    assert "contained command never started" in payload["raw_output"]
    assert "Cannot connect to the Docker daemon" in payload["raw_output"]
    # Nothing follows the command that never started: not the first file's
    # remaining commands and not the second declared file.
    assert calls[-1] == (container, kind)
    assert calls.count((container, kind)) == 1
    # Only the suite's clean-base outcome is cached, never one for a focused
    # command that did not run.
    assert len(list(tmp_path.glob("*.json"))) == int(container == "baseline")


@pytest.mark.parametrize(("container", "kind"), _LATER_COMMANDS)
def test_gate_command_timeout_stays_a_timeout_when_a_later_command_never_started(
    monkeypatch,
    tmp_path,
    container,
    kind,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    monkeypatch.setattr(
        dv,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        _declared_gate_exec(
            calls,
            never_started=(container, kind),
            gate=(124, "", "dev_verify timed out"),
        ),
    )

    rc, payload = _run(_declared_parameters(baseline=container == "baseline"))

    # The gate's own command started and timed out, which the candidate is
    # charged with.  A later command that never started is reported and does
    # not turn that timeout into an unavailable gate.
    assert rc == 124
    values = payload["values"]
    assert values["returncode"] == 124.0
    assert values["dev_verify_candidate_process_returncode"] == 124.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert not _evidence_unavailable(rc, values)
    assert values["dev_verify_declared_probe_unavailable"] == 1.0
    assert "status=runner_error" in payload["raw_output"]
    assert "contained command never started" in payload["raw_output"]
    assert "Cannot connect to the Docker daemon" in payload["raw_output"]
    # The probes still end at the command that never started; a suite that
    # timed out has no clean-base run, so nothing is cached.
    assert calls[0] == ("candidate", "suite")
    assert ("baseline", "suite") not in calls
    assert calls[-1] == (container, kind)
    assert calls.count((container, kind)) == 1
    assert not list(tmp_path.glob("*.json"))


def test_observation_command_that_never_started_is_typed_infrastructure(
    monkeypatch,
) -> None:
    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
    )

    def fake_exec(_container_id, command, **_kwargs):
        if "_crucible_exact_pytest_observer" in command:
            return _DOCKER_FAILED_FIRST
        return 0, "1 passed in 0.10s", ""

    monkeypatch.setattr(dv, "_run_docker_exec", fake_exec)
    parameters = _params(language="python")
    parameters.update({
        "breach_observations": True,
        "breach_attack_ids": ["transport"],
        "breach_test_file": "tests/test_breach.py",
    })

    rc, payload = _run(parameters)

    # Not a failure of the attack channel: the observer never ran.
    assert rc == 2
    failure = payload[BREACH_OBSERVATION_FAILURE_KEY]
    assert failure["kind"] == "infrastructure"
    assert "contained command never started" in failure["detail"]
    assert "Cannot connect to the Docker daemon" in failure["detail"]
    assert payload["test_observations"] == []


@pytest.mark.parametrize(
    ("observe", "never_started", "answers"),
    [
        pytest.param(
            lambda: dv._python_exact_observations("candidate", {
                "breach_attack_ids": ["transport"],
                "breach_test_file": "tests/test_breach.py",
            }),
            "_crucible_exact_pytest_observer",
            {},
            id="python-exact",
        ),
        pytest.param(
            lambda: dv._rust_exact_observations("candidate", {
                "breach_attack_ids": ["attack_race_contract"],
            }),
            "-- --list",
            {},
            id="rust-discovery",
        ),
        pytest.param(
            lambda: dv._rust_exact_observations("candidate", {
                "breach_attack_ids": ["attack_race_contract"],
            }),
            "--exact",
            {"-- --list": "tests::test_attack_race_contract: test\n"},
            id="rust-exact",
        ),
        pytest.param(
            lambda: dv._go_exact_observations("candidate", {
                "breach_test_file": "pkg/breach_test.go",
                "breach_attack_ids": ["race_a"],
            }),
            "go list",
            {},
            id="go-package",
        ),
        pytest.param(
            lambda: dv._go_exact_observations("candidate", {
                "breach_test_file": "pkg/breach_test.go",
                "breach_attack_ids": ["race_a"],
            }),
            "go test -json",
            {"go list": "example/pkg\n"},
            id="go-exact",
        ),
        pytest.param(
            lambda: dv._javascript_exact_observations(
                "candidate",
                {
                    "deepswe_metadata": {"language": "javascript"},
                    "breach_test_file": "tests/breach_tests.js",
                    "breach_attack_ids": ["race_a"],
                },
                parser_kind="jest",
            ),
            "package.json",
            {},
            id="javascript-package",
        ),
        pytest.param(
            lambda: dv._javascript_exact_observations(
                "candidate",
                {
                    "deepswe_metadata": {"language": "javascript"},
                    "breach_test_file": "tests/breach_tests.js",
                    "breach_attack_ids": ["race_a"],
                },
                parser_kind="jest",
            ),
            "--testNamePattern",
            {"package.json": "/app\n"},
            id="javascript-exact",
        ),
        pytest.param(
            lambda: dv._declared_test_probe_command(
                "candidate",
                "crates/core/tests/contract.rs",
                language="rust",
                parser_kind="libtest",
                metadata={},
                parameters={},
            ),
            "Cargo.toml",
            {},
            id="rust-declared-package",
        ),
    ],
)
def test_every_later_candidate_command_reports_that_it_never_started(
    monkeypatch,
    observe,
    never_started,
    answers,
):
    calls: list[str] = []

    def fake_exec(_container_id, command, **_kwargs):
        calls.append(command)
        if never_started in command:
            return _DOCKER_FAILED_FIRST
        for marker, stdout in answers.items():
            if marker in command:
                return 0, stdout, ""
        raise AssertionError(f"unexpected contained command: {command}")

    monkeypatch.setattr(dv, "_run_docker_exec", fake_exec)

    # Neither a runner_error observation nor a channel failure: each of
    # these would file a Docker failure under the candidate's test code.
    with pytest.raises(dv._ContainedCommandNotStarted) as caught:
        observe()

    assert never_started in calls[-1]
    assert "Cannot connect to the Docker daemon" in str(caught.value)


def test_started_later_command_keeps_its_status(monkeypatch):
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (124, "partial output", "timed out"),
    )

    # Only the not-started code is taken out of the statuses a caller reads.
    assert dv._run_started_docker_exec("candidate", "true") == (
        124,
        "partial output",
        "timed out",
    )


def test_malformed_stdin_exit_2():
    out = io.StringIO()
    rc = dv.main(stdin=io.StringIO("not json"), stdout=out)
    assert rc == 2
    payload = json.loads(out.getvalue())
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 1.0
    assert payload["values"]["dev_verify_runner_protocol_unavailable"] == 1.0


def test_declared_probe_runner_error_is_explicitly_unavailable(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_exec(rc=0, stdout=_passing_output("python")),
    )
    monkeypatch.setattr(
        dv,
        "_run_declared_test_probes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("declared probe setup is invalid")
        ),
    )
    params = _params()
    params[dv.DECLARED_VERIFICATION_SURFACE_KEY] = ["tests/test_added.py"]

    rc, payload = _run(params)

    assert rc == 2
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 1.0
    assert payload["values"]["dev_verify_declared_probe_unavailable"] == 1.0


@pytest.mark.parametrize(
    "stdin",
    [
        pytest.param(
            lambda: io.StringIO("[" * 50_000 + "0" + "]" * 50_000),
            id="recursive-json",
        ),
        pytest.param(lambda: io.BytesIO(b"\xff"), id="invalid-utf8"),
    ],
)
def test_bench_config_decode_failures_exit_2(stdin):
    rc = dv.main(stdin=stdin(), stdout=io.StringIO())

    assert rc == 2


# ---------- seam: build_and_verify(gate="dev_verify") end-to-end ----------


def _make_instance(language: str):
    """Construct a DeepSWEInstance directly (no fixture tree). Only the
    fields synthesize_starter_spec + the worker read need to be real."""
    from crucible.products.deepswe.dataset import DeepSWEInstance

    return DeepSWEInstance(
        task_id="seam-task-1",
        ext_id="seam-ext-1",
        repo="ex/seam",
        repository_url="https://github.com/ex/seam",
        language=language,
        category="feature_request",
        display_title="T",
        display_description="D",
        base_commit_hash="0" * 40,
        docker_image="public.ecr.aws/x/seam:e",
        cpus=2,
        memory_mb=8192,
        storage_mb=20480,
        build_timeout_sec=1800.0,
        allow_internet=False,
        verifier_timeout_sec=1800.0,
        agent_timeout_sec=5400.0,
        instruction="Do the thing.",
    )


def test_dev_verify_seam_threads_language_without_top_level(monkeypatch):
    """The PRODUCTION seam: synthesize_starter_spec → build_and_verify(
    gate="dev_verify") → deepswe_dev_verify, driven by the SAME payload
    shape the worker emits (language only in deepswe_metadata, no
    top-level ``language``). This is the seam the P0 bug lived in:
    earlier unit tests hand-fed {"language": "python"} at the top level
    and so masked the broken language threading. Here the dev_verify
    runner must dispatch on the metadata language alone.

    ``build_and_verify`` invokes the runner as a real subprocess
    (``python -m <runner>``); we fake its bounded subprocess helper at that
    layer to invoke the runner's ``main`` in-process against the piped stdin,
    so the whole payload contract is exercised without docker."""
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify as bv

    inst = _make_instance("rust")
    spec = synthesize_starter_spec(inst)
    assert spec.project_context.runners["dev_verify"] == (
        "crucible.products.deepswe.runners.deepswe_dev_verify"
    )

    # The worker's deepswe_metadata: container_id + language, NO
    # top-level language key (mirrors _build_run_architect_kwargs).
    extra_parameters = {
        "deepswe_metadata": {
            "container_id": "authcid",
            "language": inst.language,
        }
    }
    process_settlement_digest = "sha256:" + "a" * 64

    seen: dict = {}

    def _fake_runner_subprocess(
        argv,
        *,
        config_json,
        cwd=None,
        timeout=None,
        env=None,
        settlement_callback=None,
        writer_epoch_members=None,
    ):
        assert settlement_callback is not None
        assert writer_epoch_members is None
        # argv == [python, "-m", "<runner module>"]; run the runner's
        # main() in-process against the piped BenchConfig stdin.
        module = importlib.import_module(argv[-1])
        cfg = json.loads(config_json.decode("utf-8"))
        seen["parameters"] = cfg["parameters"]

        # The runner's docker exec is itself a subprocess.run — but
        # we've patched the symbol it imported, so stub by language.
        def _docker_exec(args, **kw):
            seen["docker_argv"] = list(args)
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout="test result: ok. 3 passed; 0 failed; 0 ignored\n",
                stderr="",
            )

        monkeypatch.setattr(dv.subprocess, "run", _docker_exec)
        out = io.StringIO()
        rc = module.main(stdin=io.StringIO(json.dumps(cfg)), stdout=out)
        settlement_callback(process_settlement_digest)
        return subprocess.CompletedProcess(
            args=argv, returncode=rc,
            stdout=out.getvalue().encode("utf-8"), stderr=b"",
        )

    monkeypatch.setattr(
        bv,
        "_run_runner_subprocess",
        _fake_runner_subprocess,
    )

    outcome = bv.build_and_verify(
        spec, slice_id="s1", cwd="/tmp",
        extra_parameters=extra_parameters, gate="dev_verify",
    )

    # The runner dispatched on the rust command, proving language
    # threaded through metadata (no top-level key was set).
    assert "cargo test" in seen["docker_argv"][5]
    assert seen["docker_argv"][:3] == ["docker", "exec", "authcid"]
    # No top-level language was injected by the seam.
    assert "language" not in seen["parameters"]
    # Gate passed → success classification, dev_verify field populated.
    assert outcome.classification == "success"
    assert outcome.dev_verify is not None
    assert outcome.dev_verify.values["tests_passed"] == 3.0
    assert (
        outcome.dev_verify.process_settlement_digest
        == process_settlement_digest
    )


# ---------- per-project dev_verify_command override ----------


def test_override_honored_over_language_default(monkeypatch):
    """A ``dev_verify_command`` in deepswe_metadata wins over the
    per-language default — the pilot-calibrated escape hatch for
    projects whose native orchestrator the default bypasses."""
    fake = _fake_exec(rc=0, stdout=_passing_output("python"))
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "auth123",
            "language": "python",
            "dev_verify_command": "cd /app && nox -s test",
        },
    }
    rc, _ = _run(params)
    assert rc == 0
    # The override ran verbatim; the python default (pytest) did not.
    assert "cd /app && nox -s test" in fake.calls[0][5]
    assert "timeout --signal=TERM" in fake.calls[0][5]
    assert "pytest" not in fake.calls[0][5]


def test_rust_override_honored_for_ordinary_dev_verify(monkeypatch):
    """Rust's project-native suite remains active outside sealed breach."""
    fake = _fake_exec(rc=0, stdout=_passing_output("rust"))
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "auth123",
            "language": "rust",
            "dev_verify_command": "cd /app && cargo test --test project_suite",
        },
    }

    rc, _ = _run(params)

    assert rc == 0
    assert "cd /app && cargo test --test project_suite" in fake.calls[0][5]
    assert "cargo build --quiet" not in fake.calls[0][5]


def test_override_rescues_unknown_language(monkeypatch):
    """With an override present, an unsupported / unset language no
    longer hard-fails — the override is a complete command on its own."""
    fake = _fake_exec(rc=0, stdout="  1 passing (1ms)\n")
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "auth123",
            "language": "cobol",
            "dev_verify_command": "cd /app && make check",
            "dev_verify_parser": "mocha",
        },
    }
    rc, _ = _run(params)
    assert rc == 0
    assert "cd /app && make check" in fake.calls[0][5]
    assert "timeout --signal=TERM" in fake.calls[0][5]


def test_empty_override_falls_back_to_language_default(monkeypatch):
    """A blank / whitespace override is ignored — the per-language
    default still applies (no accidental empty-command exec)."""
    fake = _fake_exec(rc=0, stdout=_passing_output("python"))
    monkeypatch.setattr(subprocess, "run", fake)
    params = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "auth123",
            "language": "python",
            "dev_verify_command": "   ",
        },
    }
    rc, _ = _run(params)
    assert rc == 0
    assert f"pytest {PYTEST_DEV_VERIFY_MAXFAIL_ARG}" in fake.calls[0][5]


# ---------- override resolution table ----------


def test_resolve_override_task_wins_over_repo(monkeypatch):
    """task_id entry takes precedence over a repo-level entry."""
    from crucible.products.deepswe import dev_verify_overrides as ov

    monkeypatch.setattr(ov, "_TASK_OVERRIDES", {"t1": "cd /app && nox"})
    monkeypatch.setattr(ov, "_REPO_OVERRIDES", {"org/r": "cd /app && tox"})
    assert ov.resolve_dev_verify_command(task_id="t1", repo="org/r") == (
        "cd /app && nox"
    )


def test_resolve_override_repo_fallback(monkeypatch):
    """No task entry → repo-level entry applies."""
    from crucible.products.deepswe import dev_verify_overrides as ov

    monkeypatch.setattr(ov, "_TASK_OVERRIDES", {})
    monkeypatch.setattr(ov, "_REPO_OVERRIDES", {"org/r": "cd /app && tox"})
    assert ov.resolve_dev_verify_command(task_id="t1", repo="org/r") == (
        "cd /app && tox"
    )


def test_resolve_override_none_when_unregistered(monkeypatch):
    """No entry at either level → None (caller uses language default)."""
    from crucible.products.deepswe import dev_verify_overrides as ov

    monkeypatch.setattr(ov, "_TASK_OVERRIDES", {})
    monkeypatch.setattr(ov, "_REPO_OVERRIDES", {})
    assert ov.resolve_dev_verify_command(task_id="t1", repo="org/r") is None


def test_calibrated_go_overrides_present():
    """Regression guard for the 2026-06-16 Go calibration: `go test ./...`
    fails at baseline on these repos, so the production table must carry a
    scoped, baseline-green override (not fall back to the broken default)."""
    from crucible.products.deepswe.dev_verify_overrides import (
        resolve_dev_verify_command,
    )
    abs_cmd = resolve_dev_verify_command(task_id="x", repo="abs-lang/abs")
    assert abs_cmd is not None
    assert "CONTEXT=abs" in abs_cmd            # env fix for TestEnv/TestCommand
    assert "go test -json" in abs_cmd and "/js" in abs_cmd

    al_cmd = resolve_dev_verify_command(task_id="x", repo="rhysd/actionlint")
    assert al_cmd is not None
    assert "go test -json" in al_cmd and "/scripts/" in al_cmd


def test_calibrated_typescript_overrides_present():
    """Regression guard for the 2026-06-16 TS calibration: the `npm test`
    default mis-fires per-project (typecheck/bundle tests fail at
    baseline), so each repo carries its real unit-test runner."""
    from crucible.products.deepswe.dev_verify_overrides import (
        resolve_dev_verify_command,
    )
    aw = resolve_dev_verify_command(task_id="x", repo="jeffijoe/awilix")
    assert aw is not None and "jest" in aw and "rollup" in aw  # skip bundle suite
    ark = resolve_dev_verify_command(task_id="x", repo="arktypeio/arktype")
    assert ark is not None and "pnpm test" in ark
    cl = resolve_dev_verify_command(task_id="x", repo="bombshell-dev/clack")
    assert cl is not None and "pnpm test" in cl
    optique = resolve_dev_verify_command(task_id="x", repo="dahlia/optique")
    assert optique is not None and "/packages/core" in optique
    ink = resolve_dev_verify_command(task_id="x", repo="vadimdemedes/ink")
    assert ink is not None and "ava --tap" in ink


def test_calibrated_javascript_exact_probe_profiles_are_sealed():
    from crucible.products.deepswe.dev_verify_overrides import (
        resolve_javascript_probe_profile,
    )

    optique = resolve_javascript_probe_profile(
        task_id="optique-conditional-option-dependencies"
    )
    assert optique is not None
    assert optique.parser_kind == "node-tap"
    assert optique.workdir == "/app/packages/core"
    assert optique.setup_command == "pnpm exec tsdown"

    clack = resolve_javascript_probe_profile(
        task_id="clack-async-autocomplete-options"
    )
    assert clack is not None
    assert clack.parser_kind == "vitest"
    assert clack.workdir == "/app/packages/prompts"

    ink = resolve_javascript_probe_profile(task_id="ink-grid-box-layout")
    assert ink is not None
    assert ink.parser_kind == "ava-tap"
    assert ink.executable[-1] == "ava"


def test_calibrated_per_project_default_misfire_overrides():
    """Regression guard for the 2026-06-16 6-default verification: even
    python/rust/js defaults mis-fire/no-op per-project (dev_verify is
    per-project, not per-language). These carry baseline-green overrides."""
    from crucible.products.deepswe.dev_verify_overrides import (
        resolve_dev_verify_command,
    )
    # bandit: real suite is tests/ (root pytest collects lint-sample data).
    b = resolve_dev_verify_command(task_id="x", repo="PyCQA/bandit")
    assert b is not None and "pytest tests/" in b
    assert PYTEST_DEV_VERIFY_MAXFAIL_ARG in b
    assert " -x" not in b
    # oxvg: workspace — root cargo test collects 0 tests.
    o = resolve_dev_verify_command(task_id="x", repo="noahbald/oxvg")
    assert o is not None and "--workspace" in o
    # katex: jest only (skip yarn lint/ts + yarn-version pin).
    k = resolve_dev_verify_command(task_id="x", repo="KaTeX/KaTeX")
    assert k is not None and "jest" in k
    # testem: task-level scoped mocha (excludes headless-failing browser tests).
    t = resolve_dev_verify_command(
        task_id="testem-bail-on-test-failure", repo="testem/testem",
    )
    assert t is not None and "mocha" in t and "bail_on_test_failure" in t
    assert "browser_launcher" not in t  # the headless-failing files are excluded
    # Explicit-file list must include the mirrored breach test (mocha
    # globs *_tests.js → breach.py writes tests/breach_tests.js) so the
    # per-round breach gate actually runs the adversarial tests.
    assert "breach_tests.js" in t


# ---------- paths the gate command itself creates in the checkout ----------


_GATE_CREATED = ("Cargo.lock", "coverage")


class _ScriptedContainer:
    """Answer the runner's contained commands and keep them in order.

    The presence probe and the removal are recognized by the exact script
    the runner renders for the recorded paths; anything else is the gate.
    An exception given as an answer is raised, as a host error would be.
    """

    def __init__(
        self,
        *,
        gate=(0, "1 passed in 0.10s", ""),
        present: tuple[str, ...] = (),
        probe=None,
        removal=None,
    ) -> None:
        self.gate = gate
        self.present = present
        self.probe = probe
        self.removal = removal
        self.calls: list[tuple[str, str, str]] = []
        self._removals = {
            dv._gate_created_removal_script(paths): len(paths)
            for paths in (_GATE_CREATED, _GATE_CREATED[:1], _GATE_CREATED[1:])
        }

    def kinds(self) -> list[str]:
        return [kind for kind, _container, _command in self.calls]

    def __call__(self, container_id, command, *, timeout=None, deadline=None):
        del timeout, deadline
        if command == dv._gate_created_probe_script(_GATE_CREATED):
            self.calls.append(("probe", container_id, command))
            if isinstance(self.probe, BaseException):
                raise self.probe
            if self.probe is not None:
                return self.probe
            return 0, "".join(
                "present\n" if path in self.present else "absent\n"
                for path in _GATE_CREATED
            ), ""
        if command in self._removals:
            self.calls.append(("removal", container_id, command))
            if isinstance(self.removal, BaseException):
                raise self.removal
            if self.removal is not None:
                return self.removal
            return 0, "removed\n" * self._removals[command], ""
        self.calls.append(("gate", container_id, command))
        if isinstance(self.gate, BaseException):
            raise self.gate
        return self.gate


def _gate_created_params(paths: tuple[str, ...] = _GATE_CREATED) -> dict:
    params = _params()
    params["deepswe_metadata"][dv.GATE_CREATED_PATHS_KEY] = (
        dv.gate_created_paths_evidence(paths)
    )
    return params


def test_gate_created_paths_evidence_is_exact_and_bounded():
    assert dv.gate_created_paths_evidence(
        ("coverage", "Cargo.lock", "coverage")
    ) == {
        "schema_version": dv.GATE_CREATED_PATHS_SCHEMA,
        "paths": ["Cargo.lock", "coverage"],
    }
    assert dv._validated_gate_created_paths(
        _gate_created_params()["deepswe_metadata"]
    ) == _GATE_CREATED
    assert dv._validated_gate_created_paths({}) == ()

    at_most = tuple(
        f"p{index:03d}" for index in range(dv.MAX_GATE_CREATED_PATHS)
    )
    assert dv.gate_created_paths_evidence(at_most) is not None
    largest = tuple(f"{index:02d}" + "x" * 254 for index in range(64))
    assert sum(map(len, largest)) == dv.MAX_GATE_CREATED_PATH_BYTES
    assert dv.gate_created_paths_evidence(largest) is not None

    for unbindable in (
        (),
        (*at_most, "one-more"),
        (*largest, "y"),
        ("/app/absolute",),
        (".",),
        ("..",),
        ("a/../b",),
        ("a//b",),
        ("a/",),
        ("./a",),
        (".git/hooks/pre-commit",),
        ("pkg/.GIT",),
        ("line\nbreak",),
        ("tab\tname",),
        (os.fsdecode(b"caf\xe9"),),
        ("coverage", "coverage/html"),
        ("",),
        (7,),
        ("coverage", 7),
        "coverage",
        b"coverage",
        None,
        7,
    ):
        # Never an exception: the caller continues with an unbound record.
        assert dv.gate_created_paths_evidence(unbindable) is None, unbindable


def test_recorded_gate_created_paths_are_removed_after_the_gate_run(
    monkeypatch,
):
    container = _ScriptedContainer()
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    assert rc == 0
    assert container.kinds() == ["probe", "gate", "removal"]
    assert {candidate for _kind, candidate, _command in container.calls} == {
        "auth123"
    }
    assert container.calls[1][2] == dv._LANG_TO_COMMAND["python"]
    assert container.calls[2][2] == dv._gate_created_removal_script(
        _GATE_CREATED
    )
    values = payload["values"]
    assert values["dev_verify_gate_created_paths_removed"] == 2.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert "dev_verify_gate_residue_unavailable" not in values
    assert values["returncode"] == 0.0


@pytest.mark.parametrize(
    ("present", "removed"),
    [
        pytest.param(("Cargo.lock",), ("coverage",), id="one-present"),
        pytest.param(_GATE_CREATED, (), id="all-present"),
    ],
)
def test_recorded_path_present_before_the_gate_run_is_left_in_place(
    monkeypatch,
    present,
    removed,
):
    container = _ScriptedContainer(present=present)
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    assert rc == 0
    if removed:
        # Only what was absent before this run can be this run's residue.
        assert container.kinds() == ["probe", "gate", "removal"]
        assert container.calls[2][2] == dv._gate_created_removal_script(
            removed
        )
        assert payload["values"]["dev_verify_gate_created_paths_removed"] == 1.0
    else:
        assert container.kinds() == ["probe", "gate"]
        assert "dev_verify_gate_created_paths_removed" not in payload["values"]


def test_gate_created_paths_are_removed_after_a_failing_gate_run(monkeypatch):
    container = _ScriptedContainer(gate=(1, _PYTEST_FAILING_OUTPUT, ""))
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    assert rc == 1
    assert container.kinds() == ["probe", "gate", "removal"]
    assert payload["values"]["dev_verify_gate_created_paths_removed"] == 2.0
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 0.0


@pytest.mark.parametrize(
    ("failure", "kinds"),
    [
        pytest.param(
            {"probe": (1, "", "no such container")}, ["probe"], id="probe",
        ),
        pytest.param(
            {"probe": (0, "absent\n", "")}, ["probe"], id="probe-short-output",
        ),
        pytest.param(
            {"removal": (44, "", "")},
            ["probe", "gate", "removal"],
            id="removal-refused",
        ),
        pytest.param(
            {"removal": (0, "removed\nkept\n", "")},
            ["probe", "gate", "removal"],
            id="removal-unknown-line",
        ),
        pytest.param(
            {"removal": PermissionError(13, "helper cache is not writable")},
            ["probe", "gate", "removal"],
            id="removal-host-error",
        ),
    ],
)
def test_gate_created_path_handling_failure_is_infrastructure_unavailable(
    monkeypatch,
    failure,
    kinds,
):
    container = _ScriptedContainer(**failure)
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    # A gate whose own residue could not be handled is neither a pass nor
    # the slice's fault; without the probe the gate command does not run.
    assert rc == 2
    assert container.kinds() == kinds
    values = payload["values"]
    assert values["returncode"] == 2.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert values["dev_verify_gate_residue_unavailable"] == 1.0
    assert "dev_verify_gate_created_paths_removed" not in values
    assert "gate-created path" in payload["raw_output"]


def test_gate_created_paths_are_not_removed_after_the_runner_deadline(
    monkeypatch,
):
    container = _ScriptedContainer(
        removal=dv._RunnerDeadlineExceeded(
            "workflow-owned development-verifier deadline expired"
        ),
    )
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    # No mutation after the workflow-owned window, and a timeout stays one.
    assert rc == 124
    assert container.kinds() == ["probe", "gate", "removal"]
    values = payload["values"]
    assert values["returncode"] == 124.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert "dev_verify_gate_residue_unavailable" not in values
    assert "dev_verify_gate_created_paths_removed" not in values


@pytest.mark.parametrize(
    ("removal", "removed"),
    [
        pytest.param(None, 2.0, id="removal-succeeds"),
        pytest.param((44, "", ""), None, id="removal-refused"),
        pytest.param(
            (124, "", "dev_verify timed out"), None, id="removal-times-out",
        ),
        pytest.param(
            PermissionError(13, "helper cache is not writable"),
            None,
            id="removal-host-error",
        ),
    ],
)
def test_gate_command_timeout_stays_a_timeout_with_recorded_paths(
    monkeypatch,
    removal,
    removed,
):
    container = _ScriptedContainer(
        gate=(124, "", "dev_verify timed out"),
        removal=removal,
    )
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    # The removal still runs after a gate command that timed out.  Whatever
    # becomes of it, the timeout stays attributed to the candidate: a failed
    # removal is reported and does not make the gate "unavailable".
    assert rc == 124
    assert container.kinds() == ["probe", "gate", "removal"]
    values = payload["values"]
    assert values["returncode"] == 124.0
    assert values["dev_verify_candidate_process_returncode"] == 124.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert not _evidence_unavailable(rc, values)
    if removed is None:
        assert values["dev_verify_gate_residue_unavailable"] == 1.0
        assert "dev_verify_gate_created_paths_removed" not in values
        assert "[crucible gate-created paths]" in payload["raw_output"]
        assert "status=unavailable" in payload["raw_output"]
    else:
        assert values["dev_verify_gate_created_paths_removed"] == removed
        assert "dev_verify_gate_residue_unavailable" not in values


@pytest.mark.parametrize(
    "returncode",
    [dv.CONTAINMENT_NOT_STARTED_RC, dv.CONTAINMENT_UNUSABLE_RC],
)
def test_gate_created_paths_are_not_removed_without_a_usable_gate_run(
    monkeypatch,
    returncode,
):
    container = _ScriptedContainer(gate=(returncode, "", "docker failed"))
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_gate_created_params())

    # Nothing ran, or the container is gone: there is nothing to remove.
    assert rc == returncode
    assert container.kinds() == ["probe", "gate"]
    assert "dev_verify_gate_created_paths_removed" not in payload["values"]


# A breach round runs this runner for its pre-fix and its post-fix gate.  It
# reads the typed observation failure of the result and never the
# infrastructure flag, so a gate that ends before its observations has to
# carry one.

_BREACH_GATE_PARAMETERS = {
    "breach_observations": True,
    "breach_attack_ids": ["transport"],
    "breach_test_file": "tests/test_breach.py",
}
_HOST_ERROR = PermissionError(13, "helper cache is not writable")
_RESIDUE_UNAVAILABLE_VALUES = {
    "returncode": 2.0,
    "dev_verify_infrastructure_unavailable": 1.0,
    "dev_verify_gate_residue_unavailable": 1.0,
}


def _breach_reads(rc: int, payload: dict) -> dict[str, str] | None:
    """Ask the breach workflow what it reads from one gate result."""
    from crucible.workflows import _build_verify as bv
    from crucible.workflows import breach

    return breach._gate_observation_failure(
        bv.BuildVerifyOutcome(
            classification="dev_verify_unavailable",
            build=None,
            verify=None,
            dev_verify=bv.RunnerResult(
                role="dev_verify",
                runner="crucible.products.deepswe.runners.deepswe_dev_verify",
                returncode=rc,
                values=payload["values"],
                raw_output=json.dumps(payload).encode("utf-8"),
                stderr=b"",
                duration_s=0.0,
            ),
        )
    )


@pytest.mark.parametrize(
    "breach_round",
    [
        pytest.param(True, id="breach-gate"),
        pytest.param(False, id="tier-gate"),
    ],
)
@pytest.mark.parametrize(
    ("record", "answers", "kinds", "expected_rc", "expected_values", "detail"),
    [
        pytest.param(
            False,
            {"gate": _DOCKER_FAILED_FIRST},
            ["gate"],
            dv.CONTAINMENT_NOT_STARTED_RC,
            _NOT_STARTED_VALUES,
            (
                "contained command never started: ",
                "Cannot connect to the Docker daemon",
            ),
            id="gate-never-started",
        ),
        pytest.param(
            False,
            {"gate": (dv.CONTAINMENT_NOT_STARTED_RC, "", "")},
            ["gate"],
            dv.CONTAINMENT_NOT_STARTED_RC,
            _NOT_STARTED_VALUES,
            ("contained command never started: no diagnostic",),
            id="gate-never-started-without-diagnostic",
        ),
        pytest.param(
            True,
            {"gate": _DOCKER_FAILED_FIRST},
            ["probe", "gate"],
            dv.CONTAINMENT_NOT_STARTED_RC,
            _NOT_STARTED_VALUES,
            (
                "contained command never started: ",
                "Cannot connect to the Docker daemon",
            ),
            id="gate-never-started-after-the-probe",
        ),
        pytest.param(
            False,
            {"gate": _HOST_ERROR},
            ["gate"],
            2,
            {
                "returncode": 2.0,
                "dev_verify_infrastructure_unavailable": 1.0,
                "dev_verify_containment_unavailable": 1.0,
            },
            (
                "contained execution failed on the host",
                "PermissionError",
                "helper cache is not writable",
            ),
            id="gate-host-error",
        ),
        pytest.param(
            True,
            {"probe": _DOCKER_FAILED_FIRST},
            ["probe"],
            2,
            _RESIDUE_UNAVAILABLE_VALUES,
            (
                "gate-created path probe failed",
                f"rc={dv.CONTAINMENT_NOT_STARTED_RC}",
                "Cannot connect to the Docker daemon",
            ),
            id="probe-never-started",
        ),
        pytest.param(
            True,
            {"probe": (0, "absent\n", "")},
            ["probe"],
            2,
            _RESIDUE_UNAVAILABLE_VALUES,
            ("gate-created path probe failed", "lines=1/2"),
            id="probe-short-output",
        ),
        pytest.param(
            True,
            {"probe": _HOST_ERROR},
            ["probe"],
            2,
            _RESIDUE_UNAVAILABLE_VALUES,
            ("helper cache is not writable",),
            id="probe-host-error",
        ),
    ],
)
def test_gate_that_ends_before_its_observations_is_typed_for_a_breach_round(
    monkeypatch,
    breach_round,
    record,
    answers,
    kinds,
    expected_rc,
    expected_values,
    detail,
):
    from crucible.common.runners.breach_observation import (
        BREACH_OBSERVATION_FAILURE_KEY,
        BREACH_OBSERVATION_FAILURE_SCHEMA,
    )

    container = _ScriptedContainer(**answers)
    monkeypatch.setattr(dv, "_run_docker_exec", container)
    parameters = _gate_created_params() if record else _params()
    if breach_round:
        parameters.update(_BREACH_GATE_PARAMETERS)

    rc, payload = _run(parameters)

    # The exit is the same for a tier gate and a breach gate, and no
    # observer command follows it.
    assert rc == expected_rc
    assert container.kinds() == kinds
    assert payload["values"] == expected_values
    assert payload["test_observations"] == []
    read = _breach_reads(rc, payload)
    if not breach_round:
        assert BREACH_OBSERVATION_FAILURE_KEY not in payload
        assert read is None
        return
    # Without the envelope the round would not end here: the missing
    # observations would become runner_error blockers, and the Docker text
    # the repair diagnostics of a paid Test Author turn.
    failure = payload[BREACH_OBSERVATION_FAILURE_KEY]
    assert set(failure) == {"schema_version", "kind", "detail"}
    assert failure["schema_version"] == BREACH_OBSERVATION_FAILURE_SCHEMA
    assert failure["kind"] == "infrastructure"
    for fragment in detail:
        assert fragment in failure["detail"]
    # The breach workflow accepts the envelope as it is; a malformed one
    # would come back as a ``channel`` failure with another detail.
    assert read == failure


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param(
            {"schema_version": "other.v1", "paths": ["coverage"]},
            id="wrong-schema",
        ),
        pytest.param(
            {
                "schema_version": dv.GATE_CREATED_PATHS_SCHEMA,
                "paths": ["coverage"],
                "extra": True,
            },
            id="extra-key",
        ),
        pytest.param(
            {
                "schema_version": dv.GATE_CREATED_PATHS_SCHEMA,
                "paths": ["coverage", "Cargo.lock"],
            },
            id="unsorted",
        ),
        pytest.param(
            {
                "schema_version": dv.GATE_CREATED_PATHS_SCHEMA,
                "paths": ["/app/coverage"],
            },
            id="absolute",
        ),
        pytest.param(
            {"schema_version": dv.GATE_CREATED_PATHS_SCHEMA, "paths": ["."]},
            id="workspace-root",
        ),
        pytest.param(
            {"schema_version": dv.GATE_CREATED_PATHS_SCHEMA, "paths": []},
            id="empty",
        ),
        pytest.param(["coverage"], id="bare-list"),
        pytest.param(None, id="null"),
    ],
)
def test_malformed_gate_created_paths_evidence_is_refused_before_any_command(
    monkeypatch,
    evidence,
):
    monkeypatch.setattr(
        dv,
        "_run_docker_exec",
        lambda *_args, **_kwargs: pytest.fail(
            "a malformed record must not reach the container"
        ),
    )
    params = _params()
    params["deepswe_metadata"][dv.GATE_CREATED_PATHS_KEY] = evidence

    rc, payload = _run(params)

    assert rc == 2
    values = payload["values"]
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert values["dev_verify_runner_configuration_unavailable"] == 1.0
    assert dv.GATE_CREATED_PATHS_KEY in payload["raw_output"]


def test_gate_created_paths_evidence_survives_the_build_and_verify_seam(
    monkeypatch,
):
    """The record travels as runtime metadata through the real gate seam."""
    from crucible.products.deepswe.spec_synthesis import synthesize_starter_spec
    from crucible.workflows import _build_verify as bv

    container = _ScriptedContainer()

    def _fake_runner_subprocess(
        argv,
        *,
        config_json,
        cwd=None,
        timeout=None,
        env=None,
        settlement_callback=None,
        writer_epoch_members=None,
    ):
        # Run the runner's main() in-process on the exact piped BenchConfig.
        monkeypatch.setattr(dv, "_run_docker_exec", container)
        out = io.StringIO()
        rc = importlib.import_module(argv[-1]).main(
            stdin=io.StringIO(config_json.decode("utf-8")),
            stdout=out,
        )
        settlement_callback("sha256:" + "a" * 64)
        return subprocess.CompletedProcess(
            args=argv,
            returncode=rc,
            stdout=out.getvalue().encode("utf-8"),
            stderr=b"",
        )

    monkeypatch.setattr(bv, "_run_runner_subprocess", _fake_runner_subprocess)

    outcome = bv.build_and_verify(
        synthesize_starter_spec(_make_instance("python")),
        slice_id="s1",
        cwd="/tmp",
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "auth123",
                "language": "python",
                dv.GATE_CREATED_PATHS_KEY: dv.gate_created_paths_evidence(
                    _GATE_CREATED
                ),
            },
        },
        gate="dev_verify",
    )

    assert outcome.classification == "success"
    assert container.kinds() == ["probe", "gate", "removal"]
    assert outcome.dev_verify is not None
    assert outcome.dev_verify.values[
        "dev_verify_gate_created_paths_removed"
    ] == 2.0


def test_gate_run_without_gate_created_paths_evidence_runs_one_command(
    monkeypatch,
):
    container = _ScriptedContainer()
    monkeypatch.setattr(dv, "_run_docker_exec", container)

    rc, payload = _run(_params())

    assert rc == 0
    assert container.kinds() == ["gate"]
    assert "dev_verify_gate_created_paths_removed" not in payload["values"]
