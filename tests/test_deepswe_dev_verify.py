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
