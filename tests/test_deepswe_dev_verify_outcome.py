# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict, hermetic DeepSWE development-verifier outcome contracts."""

from __future__ import annotations

import io
import hashlib
import json
import subprocess

import pytest

from crucible.products.deepswe.runners import deepswe_dev_verify as runner
from crucible.products.deepswe.runners import dev_verify_outcome as outcome
from crucible.common.runners.declared_verification import (
    accepted_verification_reference_payload,
)
from crucible.common.runners.repair_progress import (
    REPAIR_FAILURE_SET_OUTPUT_KEY,
    parse_repair_failure_set,
)


_PYTEST_PASS = "1 passed in 0.10s"
_PYTEST_FAILURE = (
    "================ short test summary info ================\n"
    "FAILED tests/test_api.py::test_contract - AssertionError: expected value\n"
    "================ 1 failed in 0.10s ================\n"
)
_PYTEST_WORSENED = (
    "================ short test summary info ================\n"
    "FAILED tests/test_api.py::test_contract - TypeError: wrong shape\n"
    "================ 1 failed in 0.10s ================\n"
)


def _pytest_failure_batch(count: int, *, truncated: bool) -> str:
    lines = ["================ short test summary info ================"]
    lines.extend(
        "FAILED tests/test_batch.py::test_"
        f"{index:02d} - AssertionError: stable failure {index:02d}"
        for index in range(count)
    )
    if truncated:
        lines.append(
            "!!!!!!!!!!!!!!!! stopping after "
            f"{count} failures !!!!!!!!!!!!!!!!"
        )
    lines.append(f"================ {count} failed in 0.10s ================")
    return "\n".join(lines) + "\n"


def _parse(
    language: str,
    text: str,
    *,
    rc: int = 0,
    command: str = "",
    parser_hint: str | None = None,
) -> outcome.TestOutcome:
    return outcome.parse_test_outcome(
        language=language,
        command=command,
        actual_returncode=rc,
        stdout=text,
        stderr="",
        parser_hint=parser_hint,
    )


def test_pytest_rejects_zero_test_and_unanchored_success():
    empty = _parse("python", "", rc=0)
    unanchored = _parse("python", "helper says 20 passed", rc=0)
    all_skipped = _parse("python", "3 skipped in 0.10s", rc=0)

    assert empty.returncode == 2
    assert not empty.structurally_valid
    assert unanchored.returncode == 2
    assert "terminal summary" in unanchored.structural_error
    assert all_skipped.returncode == 2


def test_pytest_uses_outer_terminal_summary_and_exact_failure_identity():
    parsed = _parse(
        "python",
        "inner plugin: 99 passed in 0.01s\n" + _PYTEST_FAILURE,
        rc=1,
    )

    assert parsed.structurally_valid
    assert parsed.returncode == 1
    assert parsed.tests_run == 1
    assert parsed.failures[0][0] == "tests/test_api.py::test_contract"
    assert parsed.failures[0][2].startswith("sha256:")


def test_pytest_outcome_schema_tracks_failure_set_completeness():
    complete = _parse("python", _PYTEST_FAILURE, rc=1)
    truncated = _parse(
        "python",
        _pytest_failure_batch(16, truncated=True),
        rc=1,
    )

    assert outcome.PARSER_SCHEMA == "crucible.deepswe.dev-verify-outcome.v4"
    assert complete.failure_set_complete
    assert not truncated.failure_set_complete
    assert truncated.structurally_valid
    assert truncated.to_payload()["failure_set_complete"] is False

    legacy = complete.to_payload()
    legacy["schema"] = "crucible.deepswe.dev-verify-outcome.v3"
    legacy.pop("failure_set_complete")
    with pytest.raises(ValueError, match="unsupported.*schema"):
        outcome.TestOutcome.from_payload(legacy)


@pytest.mark.parametrize(
    ("baseline_truncated", "candidate_truncated"),
    [(True, False), (False, True), (True, True)],
)
def test_clean_base_never_forgives_incomplete_pytest_failure_set(
    baseline_truncated,
    candidate_truncated,
):
    baseline = _parse(
        "python",
        _pytest_failure_batch(16, truncated=baseline_truncated),
        rc=1,
    )
    candidate = _parse(
        "python",
        _pytest_failure_batch(16, truncated=candidate_truncated),
        rc=1,
    )

    comparison = outcome.compare_outcomes(candidate, baseline)

    assert comparison.mode == "incomparable"
    assert comparison.effective_returncode == 1
    assert "truncated or incomplete" in comparison.reason


def test_clean_base_forgives_only_identical_pytest_failure():
    baseline = _parse("python", _PYTEST_FAILURE, rc=1)
    unchanged = _parse("python", _PYTEST_FAILURE, rc=1)
    worsened = _parse("python", _PYTEST_WORSENED, rc=1)

    accepted = outcome.compare_outcomes(unchanged, baseline)
    rejected = outcome.compare_outcomes(worsened, baseline)

    assert accepted.mode == "preexisting_only"
    assert accepted.effective_returncode == 0
    assert accepted.preexisting_failures == 1
    assert rejected.mode == "regressed"
    assert rejected.effective_returncode == 1
    assert rejected.worsened_failures == 1


def test_clean_base_rejects_test_count_loss_but_allows_proved_breach_delta():
    baseline = _parse("python", "2 passed in 0.10s")
    fewer = _parse("python", _PYTEST_PASS)
    added_project_test = _parse("python", "3 passed in 0.10s")
    with_attack = _parse("python", "3 passed in 0.10s")

    assert outcome.compare_outcomes(fewer, baseline).mode == "incomparable"
    assert outcome.compare_outcomes(added_project_test, baseline).mode == (
        "baseline_match"
    )
    comparison = outcome.compare_outcomes(
        with_attack,
        baseline,
        expected_extra_tests=1,
    )
    assert comparison.mode == "baseline_match"
    assert comparison.effective_returncode == 0


def test_libtest_requires_nonzero_observed_tests_and_consistent_result():
    zero = _parse(
        "rust",
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured",
    )
    passed = _parse(
        "rust",
        "test test_visible ... ok\n"
        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured",
    )

    assert zero.returncode == 2
    assert not zero.structurally_valid
    assert passed.structurally_valid
    assert passed.tests_run == 1


def test_libtest_qualifies_duplicate_failure_names_by_cargo_harness():
    stdout = (
        "running 1 test\n"
        "test shared::contract ... FAILED\n\n"
        "---- shared::contract stdout ----\nalpha mismatch\n\n"
        "failures:\n    shared::contract\n\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured\n"
        "running 1 test\n"
        "test shared::contract ... FAILED\n\n"
        "---- shared::contract stdout ----\nbeta mismatch\n\n"
        "failures:\n    shared::contract\n\n"
        "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured\n"
    )
    stderr = (
        "     Running unittests src/lib.rs "
        "(target/debug/deps/alpha-12345678)\n"
        "     Running unittests src/lib.rs "
        "(target/debug/deps/beta-abcdef12)\n"
    )

    parsed = outcome.parse_test_outcome(
        language="rust",
        command="cargo test --workspace",
        actual_returncode=1,
        stdout=stdout,
        stderr=stderr,
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 2
    assert [failure[0] for failure in parsed.failures] == [
        "unittests src/lib.rs@alpha::shared::contract",
        "unittests src/lib.rs@beta::shared::contract",
    ]
    assert parsed.failures[0][2] != parsed.failures[1][2]


def test_go_json_requires_unique_terminal_test_records():
    record = json.dumps(
        {
            "Action": "pass",
            "Package": "example.test/pkg",
            "Test": "TestVisible",
        }
    )
    passed = _parse("go", record, command="go test -json ./...")
    duplicate = _parse(
        "go",
        record + "\n" + record,
        command="go test -json ./...",
    )

    assert passed.structurally_valid
    assert passed.tests_run == 1
    assert duplicate.returncode == 2
    assert "duplicate terminal" in duplicate.structural_error


def test_jest_vitest_and_mocha_have_strict_nonvacuous_adapters():
    jest = _parse(
        "typescript",
        "Tests:       2 passed, 2 total\n",
        command="npx jest",
    )
    mocha = _parse(
        "javascript",
        "  3 passing (20ms)\n",
        command="npx mocha",
    )
    vitest = _parse(
        "typescript",
        " Test Files  1 passed (1)\n      Tests  2 passed (2)\n",
        command="pnpm test",
    )
    unknown = _parse(
        "typescript",
        "npm completed successfully\n",
        command="npm test",
    )

    assert jest.structurally_valid and jest.tests_run == 2
    assert mocha.structurally_valid and mocha.tests_run == 3
    assert vitest.structurally_valid and vitest.tests_run == 2
    assert vitest.parser_kind == "vitest"
    assert unknown.returncode == 2
    assert "no strict parser" in unknown.structural_error


def test_mocha_failure_identity_includes_indented_test_title():
    parsed = _parse(
        "javascript",
        "  0 passing (20ms)\n"
        "  2 failing\n\n"
        "  1) shared suite\n"
        "       first contract:\n"
        "     AssertionError: first\n\n"
        "  2) shared suite\n"
        "       second contract:\n"
        "     AssertionError: second\n",
        rc=1,
        command="npx mocha",
    )

    assert parsed.structurally_valid
    assert [failure[0] for failure in parsed.failures] == [
        "shared suite > first contract",
        "shared suite > second contract",
    ]


def test_mocha_uses_only_detailed_failures_after_terminal_summary():
    parsed = _parse(
        "javascript",
        "  shared suite\n"
        "    ✔ passes\n"
        "    1) failing contract\n\n"
        "  1 passing (20ms)\n"
        "  1 failing\n\n"
        "  1) shared suite\n"
        "       failing contract:\n"
        "     AssertionError: expected true\n",
        rc=1,
        command="npx mocha",
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 2
    assert dict(parsed.summary) == {
        "tests_failed": 1,
        "tests_passed": 1,
        "tests_skipped": 0,
    }
    assert [failure[0] for failure in parsed.failures] == [
        "shared suite > failing contract"
    ]


def test_mocha_prior_failing_report_does_not_taint_clean_final_report():
    parsed = _parse(
        "javascript",
        "  old suite\n"
        "    1) old contract\n\n"
        "  0 passing (10ms)\n"
        "  1 failing\n\n"
        "  1) old suite\n"
        "       old contract:\n"
        "     AssertionError: old failure\n\n"
        "  final suite\n"
        "    ✔ clean contract\n\n"
        "  1 passing (5ms)\n",
        command="npx mocha",
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 1
    assert dict(parsed.summary) == {
        "tests_failed": 0,
        "tests_passed": 1,
        "tests_skipped": 0,
    }
    assert parsed.failures == ()


def test_mocha_final_failing_report_owns_summary_and_failure_details():
    parsed = _parse(
        "javascript",
        "  prior suite\n"
        "    ✔ prior pass\n\n"
        "  1 passing (5ms)\n\n"
        "  final suite\n"
        "    ✔ final pass\n"
        "    1) first contract\n"
        "    2) second contract\n\n"
        "  1 passing (20ms)\n"
        "  2 failing\n\n"
        "  1) final suite\n"
        "       first contract:\n"
        "     AssertionError: first\n\n"
        "  2) final suite\n"
        "       second contract:\n"
        "     AssertionError: second\n",
        rc=1,
        command="npx mocha",
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 3
    assert dict(parsed.summary) == {
        "tests_failed": 2,
        "tests_passed": 1,
        "tests_skipped": 0,
    }
    assert [failure[0] for failure in parsed.failures] == [
        "final suite > first contract",
        "final suite > second contract",
    ]


def test_mocha_rejects_nonsequential_detailed_failure_enumerators():
    parsed = _parse(
        "javascript",
        "  1 passing (20ms)\n"
        "  2 failing\n\n"
        "  1) final suite\n"
        "       first contract:\n"
        "     AssertionError: first\n\n"
        "  3) final suite\n"
        "       third contract:\n"
        "     AssertionError: third\n",
        rc=1,
        command="npx mocha",
    )

    assert not parsed.structurally_valid
    assert parsed.returncode == 2
    assert "enumerators did not exactly match" in parsed.structural_error


@pytest.mark.parametrize(
    "candidate_output",
    [
        "application output only\n  115 passing counterfeit\n",
        "  1 passing (1ms)\napplication output after reporter\n",
    ],
)
def test_mocha_rejects_counterfeit_or_nonterminal_passing_summary(
    candidate_output,
):
    baseline = _parse(
        "javascript",
        "  115 passing (2s)\n",
        command="npx mocha",
    )
    candidate = _parse(
        "javascript",
        candidate_output,
        command="npx mocha",
    )

    assert baseline.structurally_valid
    assert not candidate.structurally_valid
    assert candidate.returncode == 2


def test_mocha_rejects_noncanonical_summary_kind_order():
    parsed = _parse(
        "javascript",
        "  1 pending\n"
        "  1 passing (1ms)\n",
        command="npx mocha",
    )

    assert not parsed.structurally_valid
    assert "result-kind order" in parsed.structural_error


def test_mocha_numbered_failure_diagnostic_is_not_a_reporter_header():
    parsed = _parse(
        "javascript",
        "  1 passing (20ms)\n"
        "  1 failing\n\n"
        "  1) shared suite\n"
        "       failing contract:\n"
        "     AssertionError: numbered diagnostic follows\n"
        "  2) diagnostic list item\n"
        "      at Context.<anonymous> (tests/example.js:10:3)\n",
        rc=1,
        command="npx mocha",
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 2
    assert [failure[0] for failure in parsed.failures] == [
        "shared suite > failing contract"
    ]


def test_mocha_replays_observed_testem_119_pass_1_failure_frame():
    stdout = (
        "  mochaAdapter\n"
        "    when the Runner instance is used\n"
        "      ✔ should emit an all-test-results event\n"
        "    1) edge_case_qunit_queque_clear_only_on_done\n\n"
        "  119 passing (2s)\n"
        "  1 failing\n\n"
        "  1) edge_case_qunit_queque_clear_only_on_done:\n"
        "     TypeError: qunitAdapter is not a function\n"
        "      at Context.<anonymous> (tests/breach_tests.js:296:5)\n"
        "      at process.processImmediate (node:internal/timers:504:21)\n"
    )
    stderr = (
        "Warning: Cannot find any files matching pattern "
        '"tests/reporter_bail_output_tests.js"\n'
        "info Seeking for config file...\n"
    )

    parsed = outcome.parse_test_outcome(
        language="javascript",
        command="npx mocha",
        actual_returncode=1,
        stdout=stdout,
        stderr=stderr,
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 120
    assert dict(parsed.summary) == {
        "tests_failed": 1,
        "tests_passed": 119,
        "tests_skipped": 0,
    }
    assert [failure[0] for failure in parsed.failures] == [
        "edge_case_qunit_queque_clear_only_on_done"
    ]


_NODE_TAP_FAILURE = """TAP version 13
# Subtest: suite
    # Subtest: passes
    ok 1 - passes
      ---
      type: 'test'
      ...
    # Subtest: fails
    not ok 2 - fails
      ---
      type: 'test'
      location: '/app/test/breach.test.js:5:3'
      error: 'boom'
      ...
    # Subtest: skips
    ok 3 - skips # SKIP SAFETY: unavailable
      ---
      type: 'test'
      ...
    1..3
not ok 1 - suite
  ---
  type: 'suite'
  error: '1 subtest failed'
  ...
1..1
# tests 3
# suites 1
# pass 1
# fail 1
# cancelled 0
# skipped 1
# todo 0
# duration_ms 20.1
"""


def test_node_tap_uses_leaf_results_and_discards_suite_wrapper():
    parsed = _parse(
        "javascript",
        _NODE_TAP_FAILURE,
        rc=1,
        command="node --test --test-reporter=tap",
    )

    assert parsed.structurally_valid
    assert parsed.parser_kind == "node-tap"
    assert parsed.tests_run == 3
    assert parsed.failures[0][0] == "test/breach.test.js::suite > fails"


def test_ava_requires_runner_owned_tap_reporter():
    unsupported = _parse(
        "typescript",
        "1 test passed",
        command="npx ava",
    )
    tap = _parse(
        "typescript",
        "TAP version 13\n"
        "ok 1 - visible contract\n\n"
        "1..1\n# tests 1\n# pass 1\n# fail 0\n",
        command="npx ava --tap",
    )

    assert unsupported.returncode == 2
    assert "deterministic --tap reporter" in unsupported.structural_error
    assert tap.structurally_valid
    assert tap.parser_kind == "ava-tap"


def test_vitest_failure_identity_is_stable_and_nonvacuous():
    parsed = _parse(
        "typescript",
        " FAIL  tests/api.test.ts > suite > attack title\n"
        "AssertionError: expected value\n"
        " Test Files  1 failed (1)\n"
        "      Tests  1 failed | 2 passed (3)\n",
        rc=1,
        parser_hint="vitest",
    )

    assert parsed.structurally_valid
    assert parsed.tests_run == 3
    assert parsed.failures[0][0] == ("tests/api.test.ts > suite > attack title")


def test_jest_vitest_and_mocha_exact_attack_probes_are_runner_owned(
    monkeypatch,
):
    jest_payload = json.dumps(
        {
            "testResults": [
                {
                    "assertionResults": [
                        {
                            "title": "attack title",
                            "fullName": "suite attack title",
                            "status": "passed",
                        }
                    ],
                }
            ],
        }
    )
    mocha_payload = json.dumps(
        {
            "passes": [{"title": "attack title", "fullTitle": "suite attack title"}],
            "failures": [],
            "pending": [],
        }
    )
    outputs = iter(
        (
            (0, jest_payload, ""),
            (0, jest_payload, ""),
            (0, mocha_payload, ""),
        )
    )
    commands: list[str] = []

    def fake_exec(_container_id, command, **_kwargs):
        commands.append(command)
        return next(outputs)

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_nearest_javascript_package_dir",
        lambda *_args, **_kwargs: "/app",
    )
    common = {
        "breach_test_file": "tests/breach.test.ts",
        "breach_attack_ids": ["attack title"],
    }
    jest = runner._javascript_exact_observations(
        "candidate",
        {**common, "deepswe_metadata": {"language": "typescript"}},
        parser_kind="jest",
    )
    vitest = runner._javascript_exact_observations(
        "candidate",
        {
            **common,
            "deepswe_metadata": {
                "language": "typescript",
                "dev_verify_command": "cd /app && pnpm test",
            },
        },
        parser_kind="vitest",
    )
    mocha = runner._javascript_exact_observations(
        "candidate",
        {**common, "deepswe_metadata": {"language": "javascript"}},
        parser_kind="mocha",
    )

    assert jest[0]["status"] == "passed"
    assert vitest[0]["status"] == "passed"
    assert mocha[0]["status"] == "passed"
    assert "--runTestsByPath" in commands[0]
    assert "pnpm exec vitest run" in commands[1]
    assert "--reporter=json" in commands[1]
    assert "--reporter json" in commands[2]


@pytest.mark.parametrize(
    ("parser_kind", "language"),
    [
        ("jest", "javascript"),
        ("vitest", "typescript"),
        ("mocha", "javascript"),
        ("node-tap", "javascript"),
        ("ava-tap", "javascript"),
    ],
)
@pytest.mark.parametrize("named_skip", [False, True])
def test_javascript_exact_probe_distinguishes_missing_from_named_skip(
    monkeypatch,
    parser_kind,
    language,
    named_skip,
):
    title = "attack title" if named_skip else "unrelated title"
    if parser_kind in {"jest", "vitest"}:
        status = "pending" if parser_kind == "jest" else "skipped"
        output = json.dumps({
            "testResults": [{
                "assertionResults": [{"title": title, "status": status}],
            }],
        })
    elif parser_kind == "mocha":
        output = json.dumps({
            "passes": [],
            "failures": [],
            "pending": [{"title": title}],
        })
    else:
        output = (
            "TAP version 13\n"
            f"# Subtest: {title}\n"
            f"ok 1 - {title} # SKIP SAFETY: unavailable\n"
            "  ---\n"
            "  type: 'test'\n"
            "  ...\n"
            "1..1\n# tests 1\n# pass 0\n# fail 0\n# skipped 1\n"
        )

    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (0, output, ""),
    )
    monkeypatch.setattr(
        runner,
        "_nearest_javascript_package_dir",
        lambda *_args, **_kwargs: "/app",
    )
    extension = "ts" if language == "typescript" else "js"
    observations = runner._javascript_exact_observations(
        "candidate",
        {
            "breach_test_file": f"tests/breach.test.{extension}",
            "breach_attack_ids": ["attack title"],
            "deepswe_metadata": {"language": language},
        },
        parser_kind=parser_kind,
    )

    assert observations[0]["status"] == (
        "not_observed" if named_skip else "runner_error"
    )


def test_node_and_ava_exact_probes_use_sealed_task_package_profiles(monkeypatch):
    node_tap = (
        "TAP version 13\n"
        "# Subtest: attack title\n"
        "ok 1 - attack title\n"
        "  ---\n"
        "  type: 'test'\n"
        "  ...\n"
        "1..1\n# tests 1\n# suites 0\n# pass 1\n# fail 0\n"
        "# cancelled 0\n# skipped 0\n# todo 0\n"
    )
    ava_tap = (
        "TAP version 13\n"
        "ok 1 - attack title\n\n"
        "1..1\n# tests 1\n# pass 1\n# fail 0\n"
    )
    outputs = iter(((0, node_tap, ""), (0, ava_tap, "")))
    commands: list[str] = []

    def fake_exec(_container_id, command, **_kwargs):
        commands.append(command)
        return next(outputs)

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    common = {
        "breach_attack_ids": ["attack title"],
    }
    node = runner._javascript_exact_observations(
        "candidate",
        {
            **common,
            "breach_test_file": "packages/core/src/breach.test.ts",
            "deepswe_metadata": {
                "language": "typescript",
                "task_id": "optique-conditional-option-dependencies",
            },
        },
        parser_kind="node-tap",
    )
    ava = runner._javascript_exact_observations(
        "candidate",
        {
            **common,
            "breach_test_file": "test/breach.test.tsx",
            "deepswe_metadata": {
                "language": "typescript",
                "task_id": "ink-grid-box-layout",
            },
        },
        parser_kind="ava-tap",
    )

    assert node[0]["status"] == "passed"
    assert ava[0]["status"] == "passed"
    assert commands[0].startswith("cd /app/packages/core && pnpm exec tsdown && ")
    assert "node --experimental-transform-types --test" in commands[0]
    assert "src/breach.test.ts" in commands[0]
    assert commands[1].startswith("cd /app && npx --no-install ava --tap")
    assert "test/breach.test.tsx" in commands[1]


def test_baseline_cache_identity_binds_every_executable_input(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    first_identity = outcome.baseline_identity(
        image_digest="registry.test/image@sha256:" + "1" * 64,
        base_commit="2" * 40,
        command="cd /app && pytest -q",
        language="python",
        parser_hint="pytest",
    )
    changed_command = outcome.baseline_identity(
        image_digest="registry.test/image@sha256:" + "1" * 64,
        base_commit="2" * 40,
        command="cd /app && pytest tests -q",
        language="python",
        parser_hint="pytest",
    )
    calls = 0

    def produce():
        nonlocal calls
        calls += 1
        return _parse("python", _PYTEST_PASS)

    first, hit_1, key_1, error_1 = outcome.cached_baseline_outcome(
        first_identity, produce
    )
    second, hit_2, key_2, error_2 = outcome.cached_baseline_outcome(
        first_identity, produce
    )

    assert first == second
    assert not hit_1 and hit_2
    assert key_1 == key_2
    assert not error_1 and not error_2
    assert calls == 1
    assert outcome.identity_key(changed_command) != key_1


def test_non_object_baseline_cache_entry_is_regenerated(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    identity = outcome.baseline_identity(
        image_digest="registry.test/image@sha256:" + "1" * 64,
        base_commit="2" * 40,
        command="cd /app && pytest -q",
        language="python",
        parser_hint="pytest",
    )
    key = outcome.identity_key(identity)
    (tmp_path / f"{key}.json").write_text("[]", encoding="utf-8")
    calls = 0

    def produce():
        nonlocal calls
        calls += 1
        return _parse("python", _PYTEST_PASS)

    baseline, cache_hit, observed_key, error = (
        outcome.cached_baseline_outcome(identity, produce)
    )

    assert baseline == _parse("python", _PYTEST_PASS)
    assert cache_hit is False
    assert observed_key == key
    assert error == ""
    assert calls == 1


def _run_runner(parameters: dict) -> tuple[int, dict]:
    stdout = io.StringIO()
    rc = runner.main(
        stdin=io.StringIO(json.dumps({"parameters": parameters})),
        stdout=stdout,
    )
    return rc, json.loads(stdout.getvalue())


def _baseline_parameters() -> dict:
    return {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "baseline_container_id": "baseline",
            "language": "python",
            "image_digest": "registry.test/image@sha256:" + "a" * 64,
            "base_commit": "b" * 40,
        },
    }


def _declared_surface(*paths: str) -> dict:
    return {
        "schema_version": "crucible.declared-verification-surface.v1",
        "slices": [{
            "slice_id": "declared-tests",
            "files_in_scope": [],
            "creates_paths": list(paths),
        }],
    }


def _accepted_python_reference(
    test_file: str,
    output: str,
    *,
    returncode: int = 0,
    surface: dict | None = None,
) -> dict:
    command = runner._declared_test_probe_command(
        "candidate",
        test_file,
        language="python",
        parser_kind="pytest",
        metadata={},
        parameters={},
    )
    parsed = outcome.parse_test_outcome(
        language="python",
        command=command,
        actual_returncode=returncode,
        stdout=output,
        stderr="",
        parser_hint="pytest",
    )
    assert parsed.structurally_valid
    return accepted_verification_reference_payload(
        predecessor_surface=surface or _declared_surface(test_file),
        language="python",
        parser_kind="pytest",
        outcomes=((
            test_file,
            hashlib.sha256(command.encode("utf-8")).hexdigest(),
            parsed.to_payload(),
        ),),
    )


@pytest.mark.parametrize(
    ("qunit_output", "expected_rc", "expected_passed", "expected_failed"),
    [
        ("  0 passing (1ms)\n", 2, 1.0, 1.0),
        ("  4 passing (5ms)\n", 0, 2.0, 0.0),
    ],
)
def test_declared_testem_files_are_individually_nonvacuous_and_candidate_only(
    tmp_path,
    monkeypatch,
    qunit_output,
    expected_rc,
    expected_passed,
    expected_failed,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    primary_command = (
        "cd /app && npx mocha tests/config_tests.js "
        "tests/mocha_adapter_tests.js"
    )
    jasmine = "tests/jasmine2_adapter_tests.js"
    qunit = "tests/qunit_adapter_tests.js"
    calls: list[tuple[str, str]] = []

    def fake_exec(container_id, command, **_kwargs):
        calls.append((container_id, command))
        if container_id == "baseline":
            if "readlink -f" in command:
                return 0, "absent\n", ""
            assert command == primary_command
            return 0, "  115 passing (2s)\n", ""
        if "readlink -f" in command:
            return 0, "safe\n", ""
        if "package.json" in command:
            return 0, "/app\n", ""
        if jasmine in command:
            assert qunit not in command
            return 0, "  12 passing (20ms)\n", ""
        if qunit in command:
            assert jasmine not in command
            return 0, qunit_output, ""
        assert command == primary_command
        return 0, "  115 passing (2s)\n", ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    parameters = _baseline_parameters()
    parameters["deepswe_metadata"].update({
        "language": "javascript",
        "task_id": "generic-unprofiled-task",
        "dev_verify_command": primary_command,
    })
    parameters["declared_verification_surface"] = _declared_surface(
        jasmine,
        qunit,
    )

    rc, payload = _run_runner(parameters)

    assert rc == expected_rc
    values = payload["values"]
    assert values["dev_verify_declared_test_files_required"] == 2.0
    assert values["dev_verify_declared_test_files_executed"] == 2.0
    assert values["dev_verify_declared_test_files_passed"] == expected_passed
    assert values["dev_verify_declared_test_files_failed"] == expected_failed
    baseline_test_commands = [
        command
        for cid, command in calls
        if cid == "baseline" and "readlink -f" not in command
    ]
    assert baseline_test_commands == [primary_command]
    assert not any(
        cid == "baseline" and "--reporter spec" in command
        for cid, command in calls
    )
    probe_commands = [
        command for cid, command in calls
        if cid == "candidate" and "--reporter spec" in command
    ]
    assert len(probe_commands) == 2
    assert any(jasmine in command and qunit not in command for command in probe_commands)
    assert any(qunit in command and jasmine not in command for command in probe_commands)
    if expected_rc:
        assert qunit in payload["raw_output"]
        assert "tests_run=None" in payload["raw_output"]


@pytest.mark.parametrize(
    (
        "baseline_state",
        "candidate_focused",
        "baseline_focused",
        "expected_rc",
        "expected_mode",
    ),
    [
        ("absent", _PYTEST_PASS, None, 0, "baseline=absent"),
        ("safe", _PYTEST_FAILURE, _PYTEST_FAILURE, 0, "preexisting_only"),
        ("safe", _PYTEST_FAILURE, _PYTEST_PASS, 1, "regressed"),
    ],
)
def test_declared_focused_probe_uses_baseline_only_when_exact_path_exists(
    tmp_path,
    monkeypatch,
    baseline_state,
    candidate_focused,
    baseline_focused,
    expected_rc,
    expected_mode,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    test_file = "tests/test_declared.py"
    focused_calls = []

    def fake_exec(container_id, command, **_kwargs):
        if "readlink -f" in command:
            return 0, (
                f"{baseline_state}\n" if container_id == "baseline"
                else "safe\n"
            ), ""
        if test_file in command:
            focused_calls.append((container_id, command))
            output = (
                baseline_focused
                if container_id == "baseline" else candidate_focused
            )
            assert output is not None
            return (1 if "failed" in output else 0), output, ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    parameters = _baseline_parameters()
    parameters["declared_verification_surface"] = _declared_surface(test_file)

    rc, payload = _run_runner(parameters)

    assert rc == expected_rc
    assert expected_mode in payload["raw_output"]
    values = payload["values"]
    if baseline_state == "safe":
        assert values[
            "dev_verify_declared_test_baseline_files_compared"
        ] == 1.0
        assert [container for container, _ in focused_calls] == [
            "candidate",
            "baseline",
        ]
    else:
        assert values[
            "dev_verify_declared_test_baseline_files_absent"
        ] == 1.0
        assert [container for container, _ in focused_calls] == ["candidate"]
    if expected_mode == "preexisting_only":
        assert values["dev_verify_declared_test_preexisting_failures"] == 1.0
    if expected_rc == 1:
        failure_set = parse_repair_failure_set(
            payload[REPAIR_FAILURE_SET_OUTPUT_KEY],
        )
        assert len(failure_set.failures) == 1
        assert failure_set.failures[0].failure_id.endswith(
            '"tests/test_api.py::test_contract"]'
        )
    else:
        assert REPAIR_FAILURE_SET_OUTPUT_KEY not in payload


@pytest.mark.parametrize(
    (
        "accepted_output",
        "candidate_output",
        "candidate_rc",
        "baseline_state",
        "baseline_output",
        "expected_rc",
        "reason",
    ),
    [
        # A later tier may not undo a predecessor's accepted fix merely
        # because the exact old failure also existed on the pristine base.
        (
            _PYTEST_PASS,
            _PYTEST_FAILURE,
            1,
            "safe",
            _PYTEST_FAILURE,
            1,
            "fixed failure reverted",
        ),
        # Green-vs-green is still a regression when focused coverage drops.
        (
            "2 passed in 0.10s",
            _PYTEST_PASS,
            0,
            "safe",
            _PYTEST_PASS,
            2,
            "two tests collapsed to one",
        ),
        # A predecessor-created file has no pristine baseline; its accepted
        # count remains authoritative for every later tier.
        (
            "2 passed in 0.10s",
            _PYTEST_PASS,
            0,
            "absent",
            None,
            2,
            "created-file coverage collapsed",
        ),
    ],
)
def test_accepted_predecessor_reference_blocks_exact_later_tier_regressions(
    tmp_path,
    monkeypatch,
    accepted_output,
    candidate_output,
    candidate_rc,
    baseline_state,
    baseline_output,
    expected_rc,
    reason,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    test_file = "tests/test_declared.py"
    baseline_focused_calls = 0

    def fake_exec(container_id, command, **_kwargs):
        nonlocal baseline_focused_calls
        if "readlink -f" in command:
            return 0, (
                f"{baseline_state}\n" if container_id == "baseline"
                else "safe\n"
            ), ""
        if test_file in command:
            if container_id == "baseline":
                baseline_focused_calls += 1
                assert baseline_output is not None
                return (
                    1 if "failed" in baseline_output else 0,
                    baseline_output,
                    "",
                )
            return candidate_rc, candidate_output, ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    parameters = _baseline_parameters()
    parameters["declared_verification_surface"] = _declared_surface(test_file)
    parameters["accepted_declared_verification_reference"] = (
        _accepted_python_reference(test_file, accepted_output)
    )

    rc, payload = _run_runner(parameters)

    assert rc == expected_rc, reason
    assert payload["values"][
        "dev_verify_declared_test_accepted_files_compared"
    ] == 1.0
    assert "baseline=accepted_predecessor" in payload["raw_output"]
    # Once an accepted checkpoint exists, pristine focused evidence can no
    # longer override it, even when that exact base path exists.
    assert baseline_focused_calls == 0


def test_accepted_predecessor_reference_fails_closed_on_missing_outcome(
    monkeypatch,
):
    test_file = "tests/test_declared.py"
    calls = []

    def fake_exec(_container_id, command, **_kwargs):
        calls.append(command)
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    surface = _declared_surface(test_file)
    parameters = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": surface,
        "accepted_declared_verification_reference": (
            accepted_verification_reference_payload(
                predecessor_surface=surface,
                language="python",
                parser_kind="pytest",
                outcomes=(),
            )
        ),
    }

    rc, payload = _run_runner(parameters)

    assert rc == 2
    assert len(calls) == 1  # primary suite only; no unbound focused probe
    assert "does not bind every predecessor" in payload["raw_output"]


def test_accepted_predecessor_reference_rejects_production_path_extra(
    monkeypatch,
):
    test_file = "tests/test_declared.py"
    production_file = "src/service.py"
    surface = _declared_surface(test_file, production_file)
    reference = _accepted_python_reference(
        test_file,
        _PYTEST_PASS,
        surface=surface,
    )
    reference["outcomes"].append({
        "path": production_file,
        "command_sha256": "c" * 64,
        "outcome": _parse("python", _PYTEST_PASS).to_payload(),
    })
    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (0, _PYTEST_PASS, ""),
    )

    rc, payload = _run_runner({
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": surface,
        "accepted_declared_verification_reference": reference,
    })

    assert rc == 2
    assert "does not bind every predecessor" in payload["raw_output"]


def test_accepted_verification_reference_cache_is_strictly_bounded():
    paths = tuple(f"tests/test_contract_{index}.py" for index in range(65))
    parsed = _parse("python", _PYTEST_PASS)

    with pytest.raises(ValueError, match="outcomes must be bounded"):
        accepted_verification_reference_payload(
            predecessor_surface=_declared_surface(*paths),
            language="python",
            parser_kind="pytest",
            outcomes=(
                (path, hashlib.sha256(path.encode()).hexdigest(), parsed.to_payload())
                for path in paths
            ),
        )


def test_declared_python_file_uses_strict_exact_candidate_probe(monkeypatch):
    test_file = "tests/test_declared.py"
    commands = []

    def fake_exec(_container_id, command, **_kwargs):
        commands.append(command)
        if "readlink -f" in command:
            return 0, "safe\n", ""
        if test_file in command:
            return 0, "2 passed in 0.10s", ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    parameters = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": _declared_surface(test_file),
    }

    rc, payload = _run_runner(parameters)

    assert rc == 0
    assert payload["values"]["dev_verify_declared_test_files_passed"] == 1.0
    exact = [command for command in commands if test_file in command]
    assert len(exact) == 2  # sandbox path proof plus the one-file pytest probe
    assert "pytest tests/test_declared.py" in exact[-1]


def test_repair_progress_envelope_rejection_preserves_verifier_failure(
    monkeypatch,
):
    test_file = "tests/test_declared.py"

    def fake_exec(_container_id, command, **_kwargs):
        if "readlink -f" in command:
            return 0, "safe\n", ""
        if test_file in command:
            return 1, _PYTEST_FAILURE, ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "build_repair_failure_set",
        lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("envelope exceeds common transport bounds")
        ),
    )
    parameters = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": _declared_surface(test_file),
    }

    rc, payload = _run_runner(parameters)

    assert rc == 1
    assert payload["values"]["dev_verify_declared_test_files_failed"] == 1.0
    assert REPAIR_FAILURE_SET_OUTPUT_KEY not in payload


def test_truncated_declared_failure_cannot_authorize_repair_progress(
    monkeypatch,
):
    test_file = "tests/test_declared.py"
    truncated = _pytest_failure_batch(16, truncated=True)

    def fake_exec(_container_id, command, **_kwargs):
        if "readlink -f" in command:
            return 0, "safe\n", ""
        if test_file in command:
            return 1, truncated, ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    parameters = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": _declared_surface(test_file),
    }

    rc, payload = _run_runner(parameters)

    assert rc == 1
    assert payload["values"]["dev_verify_declared_test_files_failed"] == 1.0
    assert REPAIR_FAILURE_SET_OUTPUT_KEY not in payload


def test_declared_rust_file_maps_only_to_direct_integration_target(monkeypatch):
    monkeypatch.setattr(
        runner,
        "_nearest_rust_package_dir",
        lambda *_args, **_kwargs: "/app/crates/core",
    )
    command = runner._declared_test_probe_command(
        "candidate",
        "crates/core/tests/contract.rs",
        language="rust",
        parser_kind="rust-libtest",
        metadata={},
        parameters={"cargo_features": ["serde"]},
    )

    assert command.startswith("cd /app/crates/core && cargo test")
    assert "--test contract" in command
    assert "--features serde" in command
    assert command.endswith("-- --test-threads=1")

    with pytest.raises(ValueError, match="direct Cargo integration target"):
        runner._declared_test_probe_command(
            "candidate",
            "crates/core/tests/support/mod.rs",
            language="rust",
            parser_kind="rust-libtest",
            metadata={},
            parameters={},
        )


def test_declared_surface_rejects_unsafe_path_after_primary_suite(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (
            calls.append((_args, _kwargs)) or (0, _PYTEST_PASS, "")
        ),
    )
    parameters = {
        "role": "dev_verify",
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "python",
        },
        "declared_verification_surface": _declared_surface(
            "../tests/test_escape.py"
        ),
    }

    rc, payload = _run_runner(parameters)

    assert rc == 2
    assert len(calls) == 1
    assert "repository-relative" in payload["raw_output"]
    assert payload["values"][
        "dev_verify_declared_test_files_structural_failures"
    ] == 1.0


def test_absolute_runner_deadline_stops_later_probes_and_breach_execs(
    monkeypatch,
):
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    calls = []

    def fake_contained_exec(
        _container_id, command, *, timeout, cleanup_timeout,
    ):
        assert timeout > 0
        assert cleanup_timeout > 0
        calls.append(command)
        clock.now += 0.6
        return 0, "  1 passing (1ms)\n", ""

    monkeypatch.setattr(runner.time, "monotonic", clock)
    monkeypatch.setattr(
        runner, "run_contained_docker_exec", fake_contained_exec,
    )
    monkeypatch.setattr(
        runner,
        "_validate_declared_container_path",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        runner,
        "_javascript_probe_context",
        lambda _container, path, **_kwargs: (
            "/app",
            ("npx", "--no-install", "mocha"),
            "",
            path,
        ),
    )
    first = "tests/first_tests.js"
    second = "tests/second_tests.js"
    parameters = {
        "role": "dev_verify",
        "runner_timeout_s": 1.2,
        "deepswe_metadata": {
            "container_id": "candidate",
            "language": "javascript",
            "dev_verify_command": "cd /app && npx mocha tests/core_tests.js",
        },
        "declared_verification_surface": _declared_surface(first, second),
        "breach_observations": True,
        "breach_test_file": "tests/breach_tests.js",
        "breach_attack_ids": ["deadline_attack"],
    }

    rc, payload = _run_runner(parameters)

    assert rc == 124
    assert payload["values"]["returncode"] == 124.0
    assert len(calls) == 2  # primary suite + first declared file only
    assert first in calls[1]
    assert all(second not in command for command in calls)
    assert all("breach_tests.js" not in command for command in calls)
    assert "deadline expired" in payload["raw_output"]


@pytest.mark.parametrize(
    ("path", "language", "expected"),
    [
        ("tests/test_api.py", "python", True),
        ("tests/api_test.py", "python", True),
        ("src/base_test.py", "python", False),
        ("src/test_set.py", "python", False),
        ("tests/qunit_adapter_tests.js", "javascript", True),
        ("tests/adapter_contract.js", "javascript", True),
        ("test/browser_abort.js", "javascript", True),
        ("testing/client_abort.ts", "typescript", True),
        ("tests/fixtures/processes/echo.js", "javascript", False),
        ("tests/fixtures/processes/ignore_sigterm.js", "javascript", False),
        ("tests/helpers/browser_abort.js", "javascript", False),
        ("tests/fixtures/browser_abort_tests.js", "javascript", True),
        ("src/adapter_contract.js", "javascript", False),
        ("lib/runners/browser_test_runner.js", "javascript", False),
        ("packages/core/api.test.ts", "typescript", True),
        ("crates/core/tests/contract.rs", "rust", True),
        ("crates/core/src/lib.rs", "rust", False),
        ("pkg/api_test.go", "go", False),
    ],
)
def test_declared_native_test_path_classification_is_conservative(
    path,
    language,
    expected,
):
    assert runner._is_declared_native_test_path(path, language) is expected


def test_runner_clean_base_turns_identical_red_suite_into_effective_pass(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    calls = []

    def fake_exec(container_id, command, **_kwargs):
        calls.append((container_id, command))
        return 1, _PYTEST_FAILURE, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )

    rc, payload = _run_runner(_baseline_parameters())

    assert rc == 0
    assert [call[0] for call in calls] == ["candidate", "baseline"]
    assert payload["values"]["tests_failed"] == 0.0
    assert payload["values"]["dev_verify_candidate_tests_failed"] == 1.0
    assert payload["values"]["dev_verify_preexisting_failures"] == 1.0
    assert "mode=preexisting_only" in payload["raw_output"]


def test_clean_probe_infrastructure_cannot_be_forgiven_as_preexisting(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    go_failure = "\n".join((
        json.dumps({
            "Action": "output",
            "Package": "example.invalid/project",
            "Test": "TestDependency",
            "Output": "dependency unavailable",
        }),
        json.dumps({
            "Action": "fail",
            "Package": "example.invalid/project",
            "Test": "TestDependency",
        }),
    ))
    network_failure = (
        "github.com/acme/module/@v/v1.0.0.mod: "
        "Get https://proxy.golang.org/acme: dial tcp: lookup "
        "proxy.golang.org: network is unreachable"
    )

    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (1, go_failure, network_failure),
    )
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )
    parameters = _baseline_parameters()
    parameters["deepswe_metadata"]["language"] = "go"
    parameters["deepswe_metadata"]["dev_verify_clean_probe"] = True

    rc, payload = _run_runner(parameters)

    assert "mode=preexisting_only" in payload["raw_output"]
    assert payload["values"]["dev_verify_preexisting_failures"] == 1.0
    assert payload["values"]["dev_verify_dependency_unavailable"] == 1.0
    assert payload["values"]["dev_verify_clean_probe_unavailable"] == 1.0
    assert payload["values"]["returncode"] == 2.0
    assert rc == 2


def test_runner_blocks_later_seventeenth_failure_hidden_by_maxfail(
    tmp_path,
    monkeypatch,
):
    """The common first 16 cannot prove a later candidate failure absent."""

    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    truncated = _pytest_failure_batch(16, truncated=True)

    # Both processes stop after the same 16 stable failures. The candidate's
    # newly introduced seventeenth failure is later in collection order and
    # therefore absent from its bounded output; incompleteness must block the
    # otherwise-identical baseline comparison.
    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (1, truncated, ""),
    )
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )

    rc, payload = _run_runner(_baseline_parameters())

    assert rc == 1
    assert payload["values"]["dev_verify_failure_set_complete"] == 0.0
    assert payload["values"]["dev_verify_baseline_passed"] == 0.0
    assert payload["values"]["dev_verify_baseline_attempted"] == 1.0
    assert payload["values"]["dev_verify_baseline_unavailable"] == 0.0
    assert payload["values"]["dev_verify_candidate_incomparable"] == 1.0
    assert "mode=incomparable" in payload["raw_output"]
    assert "truncated or incomplete" in payload["raw_output"]


def test_structurally_invalid_candidate_does_not_claim_baseline_attempt(
    monkeypatch,
):
    calls = []

    def fake_exec(container_id, _command, **_kwargs):
        calls.append(container_id)
        return 1, "src/module.py: SyntaxError: invalid syntax", ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)

    rc, payload = _run_runner(_baseline_parameters())

    assert rc == 2
    assert calls == ["candidate"]
    assert payload["values"]["dev_verify_candidate_incomparable"] == 1.0
    assert payload["values"]["dev_verify_baseline_attempted"] == 0.0
    assert payload["values"]["dev_verify_baseline_unavailable"] == 0.0
    assert payload["values"]["dev_verify_infrastructure_unavailable"] == 0.0


def test_runner_clean_base_blocks_new_candidate_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))

    def fake_exec(container_id, _command, **_kwargs):
        if container_id == "candidate":
            return 1, _PYTEST_FAILURE, ""
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )

    rc, payload = _run_runner(_baseline_parameters())

    assert rc == 1
    assert payload["values"]["dev_verify_new_failures"] == 1.0
    assert "mode=regressed" in payload["raw_output"]


def test_runner_rejects_partial_baseline_identity(monkeypatch):
    parameters = _baseline_parameters()
    del parameters["deepswe_metadata"]["base_commit"]
    monkeypatch.setattr(
        runner,
        "_run_docker_exec",
        lambda *_args, **_kwargs: (0, _PYTEST_PASS, ""),
    )

    rc, payload = _run_runner(parameters)

    assert rc == 2
    assert payload["values"]["dev_verify_baseline_unavailable"] == 1.0
    assert "metadata is incomplete" in payload["raw_output"]


def test_clean_base_suite_that_never_started_is_unavailable_and_not_cached(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    calls = []

    def fake_exec(container_id, _command, **_kwargs):
        calls.append(container_id)
        if container_id == "baseline":
            return (
                runner.CONTAINMENT_NOT_STARTED_RC,
                "",
                "docker inspect -- failed rc=1: "
                "Cannot connect to the Docker daemon",
            )
        return 0, _PYTEST_PASS, ""

    monkeypatch.setattr(runner, "_run_docker_exec", fake_exec)
    monkeypatch.setattr(
        runner,
        "_probe_clean_baseline_container",
        lambda _baseline, **_kwargs: "",
    )

    rc, payload = _run_runner(_baseline_parameters())

    # The status of this one later command goes to the outcome parser, which
    # finds no test summary in it: no clean-base outcome, no comparison.
    assert rc == 2
    assert calls == ["candidate", "baseline"]
    values = payload["values"]
    assert values["tests_passed"] == 1.0
    assert values["dev_verify_baseline_attempted"] == 1.0
    assert values["dev_verify_baseline_unavailable"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert not list(tmp_path.glob("*.json"))

    # Nothing was cached: the next gate run sends the command again.
    assert _run_runner(_baseline_parameters())[0] == 2
    assert calls == ["candidate", "baseline", "candidate", "baseline"]


# ---------- paths a baseline command creates in the clean-base checkout ----------


_GATE_CREATED = ("coverage",)
_DECLARED_TEST = "tests/test_declared.py"


class _Checkouts:
    """Candidate and baseline containers that keep what a test command creates.

    Every test command leaves ``coverage`` behind until a removal runs in the
    same container; ``git status`` reports it for as long as it is there.
    """

    def __init__(self, *, baseline_removal=None) -> None:
        self.created = {"candidate": False, "baseline": False}
        self.baseline_removal = baseline_removal
        self.calls: list[tuple[str, str]] = []

    def kinds(self, container_id: str) -> list[str]:
        return [kind for container, kind in self.calls if container == container_id]

    def inspect(self, _args, **_kwargs):
        # The host-side image-digest probe of the clean-base container.
        return subprocess.CompletedProcess(
            _args, 0, stdout="registry.test/image@sha256:" + "a" * 64, stderr="",
        )

    def __call__(self, container_id, command, **_kwargs):
        if "git rev-parse" in command:
            self.calls.append((container_id, "head"))
            return 0, "b" * 40 + "\n", ""
        if "git status" in command:
            self.calls.append((container_id, "status"))
            dirty = "?? coverage\0" if self.created[container_id] else ""
            return 0, dirty, ""
        if command == runner._gate_created_probe_script(_GATE_CREATED):
            self.calls.append((container_id, "probe"))
            return 0, "absent\n", ""
        if command == runner._gate_created_removal_script(_GATE_CREATED):
            self.calls.append((container_id, "removal"))
            if container_id == "baseline" and self.baseline_removal is not None:
                return self.baseline_removal
            was_created = self.created[container_id]
            self.created[container_id] = False
            return 0, "removed\n" if was_created else "absent\n", ""
        if "readlink -f" in command:
            return 0, "safe\n", ""
        self.calls.append(
            (container_id, "focused" if _DECLARED_TEST in command else "suite")
        )
        self.created[container_id] = True
        return 0, _PYTEST_PASS, ""


def _gate_created_baseline_parameters(*, recorded: bool, declared: bool) -> dict:
    parameters = _baseline_parameters()
    if recorded:
        parameters["deepswe_metadata"][runner.GATE_CREATED_PATHS_KEY] = (
            runner.gate_created_paths_evidence(_GATE_CREATED)
        )
    if declared:
        parameters["declared_verification_surface"] = _declared_surface(
            _DECLARED_TEST
        )
    return parameters


def test_unrecorded_baseline_residue_makes_the_next_clean_base_probe_fail(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    checkouts = _Checkouts()
    monkeypatch.setattr(runner, "_run_docker_exec", checkouts)
    monkeypatch.setattr(runner.subprocess, "run", checkouts.inspect)

    rc, payload = _run_runner(
        _gate_created_baseline_parameters(recorded=False, declared=True)
    )

    # The suite's baseline run left ``coverage`` in the clean-base checkout,
    # so the declared-file probes of the same gate run find it dirty.
    assert checkouts.kinds("baseline") == ["head", "status", "suite", "head", "status"]
    assert rc == 2
    values = payload["values"]
    assert values["dev_verify_baseline_passed"] == 1.0
    assert values["dev_verify_declared_probe_unavailable"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    assert "baseline worktree cleanliness" in payload["raw_output"]


def test_baseline_runs_remove_recorded_gate_created_paths_from_the_clean_checkout(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    checkouts = _Checkouts()
    monkeypatch.setattr(runner, "_run_docker_exec", checkouts)
    monkeypatch.setattr(runner.subprocess, "run", checkouts.inspect)
    parameters = _gate_created_baseline_parameters(recorded=True, declared=True)

    rc, payload = _run_runner(parameters)

    # Each baseline command, the suite and the focused file, is followed by
    # its removal, so every clean-base probe finds the checkout clean.
    assert checkouts.kinds("baseline") == [
        "head", "status", "suite", "removal",
        "head", "status", "focused", "removal",
    ]
    assert checkouts.kinds("candidate") == [
        "probe", "suite", "focused", "removal",
    ]
    assert checkouts.created == {"candidate": False, "baseline": False}
    assert rc == 0
    values = payload["values"]
    assert values["dev_verify_baseline_passed"] == 1.0
    assert values["dev_verify_declared_test_baseline_files_compared"] == 1.0
    assert values["dev_verify_declared_test_files_passed"] == 1.0
    assert values["dev_verify_infrastructure_unavailable"] == 0.0
    assert values["dev_verify_gate_created_paths_removed"] == 1.0

    # Both baseline outcomes are cached now: the next gate run sends no test
    # command to the clean-base container and removes nothing there.
    checkouts.calls.clear()
    rc, payload = _run_runner(parameters)

    assert rc == 0
    assert checkouts.kinds("baseline") == ["head", "status"]
    assert payload["values"]["dev_verify_baseline_cache_hit"] == 1.0
    assert payload["values"]["dev_verify_declared_test_baseline_cache_hits"] == 1.0


@pytest.mark.parametrize("declared", [False, True])
def test_baseline_checkout_that_cannot_be_restored_is_unavailable(
    tmp_path,
    monkeypatch,
    declared,
):
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_BASELINE_CACHE", str(tmp_path))
    # The suite's baseline outcome is cached first when the focused removal
    # is the one under test, so only that removal can fail.
    checkouts = _Checkouts()
    monkeypatch.setattr(runner, "_run_docker_exec", checkouts)
    monkeypatch.setattr(runner.subprocess, "run", checkouts.inspect)
    if declared:
        assert _run_runner(
            _gate_created_baseline_parameters(recorded=True, declared=False)
        )[0] == 0
    cached = sorted(path.name for path in tmp_path.glob("*.json"))
    checkouts.calls.clear()
    checkouts.baseline_removal = (44, "", "")

    rc, payload = _run_runner(
        _gate_created_baseline_parameters(recorded=True, declared=declared)
    )

    assert rc == 2
    assert checkouts.kinds("baseline")[-2:] == [
        "focused" if declared else "suite",
        "removal",
    ]
    values = payload["values"]
    assert values["dev_verify_infrastructure_unavailable"] == 1.0
    if declared:
        assert values["dev_verify_declared_probe_unavailable"] == 1.0
    else:
        assert values["dev_verify_baseline_unavailable"] == 1.0
    assert "clean-base checkout was not restored" in payload["raw_output"]
    assert "rc=44" in payload["raw_output"]
    # An outcome from a checkout that was not restored is not cached.
    assert sorted(path.name for path in tmp_path.glob("*.json")) == cached
