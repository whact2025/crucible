# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Strict public-suite outcomes and clean-base comparison for DeepSWE.

This module deliberately knows nothing about the hidden DeepSWE verifier.  It
parses only project-native development-test output and can cache one outcome
from a separately managed, pristine authoring container.  The cache identity
binds the immutable image, base commit, exact suite command, language, and
parser schema so stale evidence cannot be reused after any executable input
changes.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable


PARSER_SCHEMA = "crucible.deepswe.dev-verify-outcome.v4"
BASELINE_CACHE_SCHEMA = "crucible.deepswe.dev-verify-baseline.v1"
_CACHE_ENV = "CRUCIBLE_DEEPSWE_BASELINE_CACHE"
_MAX_FAILURES = 2_048
_MAX_FAILURE_FIELD_CHARS = 4_096
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_VOLATILE_RE = re.compile(r"(?:0x[0-9a-fA-F]+|\b\d+(?:\.\d+)?(?:ms|s)\b)")
_COUNT_RE = re.compile(
    r"(?P<count>\d{1,9})\s+"
    r"(?P<kind>passed|failed|errors?|skipped|xfailed|xpassed|deselected|"
    r"warnings?|reruns?|pending|todo)",
    re.IGNORECASE,
)
_PYTEST_TERMINAL_RE = re.compile(
    r"^(?:=+\s*)?(?P<counts>.+?)\s+in\s+"
    r"\d+(?:\.\d+)?(?:s|\s+seconds?)(?:\s+\([^\n]*\))?\s*(?:=+)?$",
    re.IGNORECASE,
)
_PYTEST_SUMMARY_HEADER_RE = re.compile(
    r"^=+\s+short test summary info\s+=+$", re.IGNORECASE
)
_PYTEST_FAILURE_RE = re.compile(r"^(FAILED|ERROR)\s+(.+)$")
_PYTEST_INCOMPLETE_FAILURE_SET_RE = re.compile(
    r"^!+\s*(?:stopping after \d+ failures?|"
    r"Interrupted:\s+\d+ errors? during collection)\s*!+$",
    re.IGNORECASE,
)
_LIBTEST_RESULT_RE = re.compile(
    r"^test result:\s+(?P<status>ok|FAILED)\.\s+"
    r"(?P<passed>\d{1,9}) passed;\s+"
    r"(?P<failed>\d{1,9}) failed;\s+"
    r"(?P<ignored>\d{1,9}) ignored(?:;|$)",
    re.IGNORECASE,
)
_LIBTEST_FAILURE_RE = re.compile(r"^test\s+(.+?)\s+\.\.\.\s+FAILED$")
_LIBTEST_FAILURE_BODY_RE = re.compile(r"^----\s+(.+?)\s+stdout\s+----$")
_CARGO_RUNNING_RE = re.compile(r"^\s*Running\s+(.+?)\s*$")
_CARGO_DOC_TEST_RE = re.compile(r"^\s*Doc-tests\s+(.+?)\s*$")
_JEST_TESTS_RE = re.compile(r"^Tests:\s+(?P<body>.+)$", re.IGNORECASE)
_JEST_FAILURE_RE = re.compile(r"^\s*●\s+(.+?)\s*$")
_VITEST_TESTS_RE = re.compile(
    r"^\s*Tests\s+(?P<body>.+?)\s+\((?P<total>\d{1,9})\)\s*$",
    re.IGNORECASE,
)
_VITEST_FAILURE_RE = re.compile(r"^\s*FAIL\s{1,}(.+?)\s*$")
_MOCHA_DURATION = r"\d+(?:\.\d+)?(?:ms|s|m|h|d|w|y)"
_MOCHA_PASSING_RE = re.compile(
    rf"^ {{2}}(?P<count>\d{{1,9}}) passing \({_MOCHA_DURATION}\)$"
)
_MOCHA_FAILING_RE = re.compile(r"^ {2}(?P<count>\d{1,9}) failing$")
_MOCHA_PENDING_RE = re.compile(r"^ {2}(?P<count>\d{1,9}) pending$")
_MOCHA_SUMMARY_PREFIX_RE = re.compile(
    r"^ {0,2}\d{1,9} (?:passing|failing|pending)\b"
)
_MOCHA_FAILURE_RE = re.compile(
    r"^ {2}(?P<number>\d{1,9})\) (?P<headline>\S.*?)\s*$"
)
_TAP_VERSION_RE = re.compile(r"^\s*TAP version 13\s*$", re.IGNORECASE)
_TAP_SUBTEST_RE = re.compile(r"^(?P<indent> *)# Subtest:\s*(?P<title>.+?)\s*$")
_TAP_RESULT_RE = re.compile(
    r"^(?P<indent> *)(?P<result>ok|not ok)\s+\d+"
    r"(?:\s*-\s*(?P<title>.*?))?\s*$",
    re.IGNORECASE,
)
_TAP_DIRECTIVE_RE = re.compile(
    r"\s+#\s*(?P<directive>SKIP|TODO)\b.*$", re.IGNORECASE
)
_TAP_SUMMARY_RE = re.compile(
    r"^\s*#\s*(?P<kind>tests|suites|pass|fail|cancelled|skip|skipped|todo)"
    r"\s+(?P<count>\d{1,9})\s*$",
    re.IGNORECASE,
)
_TAP_TYPE_RE = re.compile(r"^\s*type:\s*['\"]?(?P<type>test|suite)['\"]?\s*$")
_TAP_LOCATION_RE = re.compile(
    r"^\s*location:\s*['\"]?(?P<location>.+?)['\"]?\s*$"
)


@dataclasses.dataclass(frozen=True)
class TestOutcome:
    """One structurally validated project-suite result."""

    parser_kind: str
    returncode: int
    tests_run: int | None
    summary: tuple[tuple[str, int], ...]
    failures: tuple[tuple[str, str, str], ...]
    failure_set_complete: bool
    output_sha256: str
    structural_error: str = ""

    @property
    def structurally_valid(self) -> bool:
        return not self.structural_error and self.returncode in {0, 1}

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": PARSER_SCHEMA,
            "parser_kind": self.parser_kind,
            "returncode": self.returncode,
            "tests_run": self.tests_run,
            "summary": [list(item) for item in self.summary],
            "failures": [list(item) for item in self.failures],
            "failure_set_complete": self.failure_set_complete,
            "output_sha256": self.output_sha256,
            "structural_error": self.structural_error,
        }

    @classmethod
    def from_payload(cls, raw: object) -> "TestOutcome":
        if not isinstance(raw, dict) or raw.get("schema") != PARSER_SCHEMA:
            raise ValueError("unsupported DeepSWE dev-verify outcome schema")
        expected = {
            "schema",
            "parser_kind",
            "returncode",
            "tests_run",
            "summary",
            "failures",
            "failure_set_complete",
            "output_sha256",
            "structural_error",
        }
        if set(raw) != expected:
            raise ValueError("invalid DeepSWE dev-verify outcome fields")
        parser_kind = raw["parser_kind"]
        returncode = raw["returncode"]
        tests_run = raw["tests_run"]
        failure_set_complete = raw["failure_set_complete"]
        output_sha256 = raw["output_sha256"]
        structural_error = raw["structural_error"]
        if (
            not isinstance(parser_kind, str)
            or not parser_kind
            or isinstance(returncode, bool)
            or not isinstance(returncode, int)
            or returncode not in {0, 1, 2}
            or (
                tests_run is not None
                and (
                    isinstance(tests_run, bool)
                    or not isinstance(tests_run, int)
                    or tests_run < 0
                )
            )
            or not isinstance(failure_set_complete, bool)
            or not isinstance(output_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", output_sha256) is None
            or not isinstance(structural_error, str)
        ):
            raise ValueError("invalid DeepSWE dev-verify outcome scalar")
        summary = _tuple_pairs(raw["summary"])
        failures = _tuple_failures(raw["failures"])
        outcome = cls(
            parser_kind=parser_kind,
            returncode=returncode,
            tests_run=tests_run,
            summary=summary,
            failures=failures,
            failure_set_complete=failure_set_complete,
            output_sha256=output_sha256,
            structural_error=structural_error,
        )
        if outcome.structurally_valid and (tests_run is None or tests_run <= 0):
            raise ValueError("valid DeepSWE dev-verify outcome is vacuous")
        if outcome.returncode == 0 and not outcome.failure_set_complete:
            raise ValueError("passing DeepSWE outcome has an incomplete failure set")
        if not outcome.structurally_valid and outcome.failure_set_complete:
            raise ValueError("invalid DeepSWE outcome claims a complete failure set")
        return outcome


@dataclasses.dataclass(frozen=True)
class BaselineComparison:
    """Candidate effect relative to the same suite on the clean base."""

    mode: str
    effective_returncode: int
    reason: str
    cache_hit: bool = False
    cache_key: str = ""
    baseline_returncode: int | None = None
    preexisting_failures: int = 0
    new_failures: int = 0
    worsened_failures: int = 0


@dataclasses.dataclass(frozen=True)
class TapTestRecord:
    """One leaf result from a strict TAP13 stream."""

    title: str
    identity: str
    status: str
    diagnostic: str
    line: str


def _tuple_pairs(raw: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(raw, list) or len(raw) > 128:
        raise ValueError("invalid DeepSWE outcome summary")
    pairs: list[tuple[str, int]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            or item[1] < 0
        ):
            raise ValueError("invalid DeepSWE outcome summary item")
        pairs.append((item[0], item[1]))
    return tuple(pairs)


def _tuple_failures(raw: object) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(raw, list) or len(raw) > _MAX_FAILURES:
        raise ValueError("invalid DeepSWE outcome failures")
    failures: list[tuple[str, str, str]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(field, str) for field in item)
            or any(len(field) > _MAX_FAILURE_FIELD_CHARS for field in item)
            or item[1] not in {"FAILED", "ERROR"}
        ):
            raise ValueError("invalid DeepSWE outcome failure item")
        failures.append((item[0], item[1], item[2]))
    if len({item[0] for item in failures}) != len(failures):
        raise ValueError("duplicate DeepSWE outcome failure identity")
    return tuple(failures)


def _normalize(text: str) -> str:
    return " ".join(_VOLATILE_RE.sub("<volatile>", text).split())


def _fingerprint(text: str) -> str:
    normalized = _normalize(text)
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _invalid(
    parser_kind: str,
    actual_returncode: int,
    combined: str,
    reason: str,
) -> TestOutcome:
    return TestOutcome(
        parser_kind=parser_kind,
        returncode=2,
        tests_run=None,
        summary=(),
        failures=(),
        failure_set_complete=False,
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
        structural_error=(f"{reason}; process_returncode={actual_returncode}")[:2_000],
    )


def _finish_outcome(
    *,
    parser_kind: str,
    actual_returncode: int,
    combined: str,
    summary: dict[str, int],
    failures: tuple[tuple[str, str, str], ...],
    failure_set_complete: bool = True,
) -> TestOutcome:
    tests_run = sum(
        summary.get(key, 0)
        for key in (
            "tests_passed",
            "tests_failed",
            "tests_errored",
            "tests_skipped",
            "tests_expected_failed",
            "tests_unexpected_successes",
        )
    )
    failed = summary.get("tests_failed", 0) + summary.get("tests_errored", 0)
    executed = summary.get("tests_passed", 0) + failed
    structurally_valid = bool(
        tests_run > 0
        and executed > 0
        and failed == len(failures)
        and len({failure[0] for failure in failures}) == len(failures)
        and (
            actual_returncode == 0
            and failed == 0
            or actual_returncode != 0
            and failed > 0
        )
    )
    if not structurally_valid:
        return TestOutcome(
            parser_kind=parser_kind,
            returncode=2,
            tests_run=tests_run if tests_run > 0 else None,
            summary=tuple(sorted(summary.items())),
            failures=tuple(sorted(failures)),
            failure_set_complete=False,
            output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
            structural_error=(
                "test output was vacuous or inconsistent with process status; "
                f"process_returncode={actual_returncode}"
            ),
        )
    return TestOutcome(
        parser_kind=parser_kind,
        returncode=0 if actual_returncode == 0 else 1,
        tests_run=tests_run,
        summary=tuple(sorted(summary.items())),
        failures=tuple(sorted(failures)),
        failure_set_complete=failure_set_complete,
        output_sha256=hashlib.sha256(combined.encode("utf-8")).hexdigest(),
    )


def _count_summary(body: str) -> dict[str, int]:
    aliases = {
        "passed": "tests_passed",
        "failed": "tests_failed",
        "error": "tests_errored",
        "errors": "tests_errored",
        "skipped": "tests_skipped",
        "xfailed": "tests_expected_failed",
        "xpassed": "tests_unexpected_successes",
        "deselected": "tests_deselected",
        "rerun": "tests_rerun",
        "reruns": "tests_rerun",
        "pending": "tests_skipped",
        "todo": "tests_skipped",
    }
    summary: dict[str, int] = {}
    for match in _COUNT_RE.finditer(body):
        key = aliases.get(match.group("kind").lower())
        if key is not None:
            summary[key] = summary.get(key, 0) + int(match.group("count"))
    return summary


def _pytest_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = stdout + stderr
    parsed: list[tuple[str, list[str], int]] = []
    for stream in (stdout, stderr):
        lines = _ANSI_RE.sub("", stream).splitlines()
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        if not nonempty:
            continue
        terminal_index = nonempty[-1]
        match = _PYTEST_TERMINAL_RE.fullmatch(lines[terminal_index].strip())
        if match is not None:
            parsed.append((match.group("counts"), lines, terminal_index))
    if len(parsed) != 1:
        return _invalid(
            "pytest",
            actual_returncode,
            combined,
            "pytest emitted no unique terminal summary",
        )
    body, lines, terminal_index = parsed[0]
    summary = _count_summary(body)
    failure_set_complete = not any(
        _PYTEST_INCOMPLETE_FAILURE_SET_RE.fullmatch(line.strip())
        for stream in (stdout, stderr)
        for line in _ANSI_RE.sub("", stream).splitlines()
    )
    failed = summary.get("tests_failed", 0) + summary.get("tests_errored", 0)
    failures: list[tuple[str, str, str]] = []
    if failed:
        headers = [
            index
            for index, line in enumerate(lines[:terminal_index])
            if _PYTEST_SUMMARY_HEADER_RE.fullmatch(line.strip())
        ]
        if not headers:
            return _invalid(
                "pytest",
                actual_returncode,
                combined,
                "pytest failures lacked an authoritative short summary",
            )
        for raw_line in lines[headers[-1] + 1 : terminal_index]:
            match = _PYTEST_FAILURE_RE.match(raw_line.strip())
            if match is None:
                continue
            status, remainder = match.groups()
            node_id, separator, reason = remainder.partition(" - ")
            node_id = node_id.strip()
            if not node_id or len(node_id) > _MAX_FAILURE_FIELD_CHARS:
                continue
            failures.append(
                (
                    node_id,
                    "ERROR" if status == "ERROR" else "FAILED",
                    _fingerprint(reason if separator else raw_line),
                )
            )
    return _finish_outcome(
        parser_kind="pytest",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=tuple(failures),
        failure_set_complete=failure_set_complete,
    )


def _cargo_harness_label(raw: str) -> str:
    """Return a stable Cargo harness label without the artifact hash."""

    target = " ".join(raw.split())
    match = re.fullmatch(r"(?P<kind>.+?)\s+\((?P<binary>[^()]*)\)", target)
    if match is None:
        return target
    binary = Path(match.group("binary")).name
    binary = re.sub(r"-[0-9a-fA-F]{8,}$", "", binary)
    return f"{match.group('kind')}@{binary}"


def _libtest_harness_labels(stdout: str, stderr: str) -> list[str]:
    labels: list[str] = []
    # Cargo progress normally lives on stderr while libtest records live on
    # stdout. Preserve the Cargo stream's order and only fall back to stdout
    # for direct/custom invocations that route everything through one stream.
    for stream in (stderr, stdout):
        found: list[str] = []
        for line in _ANSI_RE.sub("", stream).splitlines():
            if match := _CARGO_RUNNING_RE.match(line):
                found.append(_cargo_harness_label(match.group(1)))
            elif match := _CARGO_DOC_TEST_RE.match(line):
                found.append(f"doc-tests {match.group(1).strip()}")
        if found:
            labels.extend(found)
            if stream is stderr:
                break
    return labels


def _libtest_result_sections(stdout: str, stderr: str) -> list[list[str]]:
    """Split libtest output into harness-local sections.

    Keeping each result line with the preceding records prevents identical
    test names in two workspace members from sharing a failure body.
    """

    sections: list[list[str]] = []
    for stream in (stdout, stderr):
        lines = _ANSI_RE.sub("", stream).splitlines()
        start = 0
        for index, line in enumerate(lines):
            if _LIBTEST_RESULT_RE.match(line.strip()) is None:
                continue
            sections.append(lines[start : index + 1])
            start = index + 1
    return sections


def _libtest_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = _ANSI_RE.sub("", stdout + stderr)
    summary = {
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
    }
    sections = _libtest_result_sections(stdout, stderr)
    labels = _libtest_harness_labels(stdout, stderr)
    failures: list[tuple[str, str, str]] = []
    for ordinal, lines in enumerate(sections):
        match = next(
            (
                found
                for line in reversed(lines)
                if (found := _LIBTEST_RESULT_RE.match(line.strip())) is not None
            ),
            None,
        )
        assert match is not None
        summary["tests_passed"] += int(match.group("passed"))
        summary["tests_failed"] += int(match.group("failed"))
        summary["tests_skipped"] += int(match.group("ignored"))
        harness = labels[ordinal] if ordinal < len(labels) else f"harness[{ordinal}]"
        failure_bodies: dict[str, str] = {}
        body_headers = [
            (index, body_match.group(1).strip())
            for index, line in enumerate(lines)
            if (
                body_match := _LIBTEST_FAILURE_BODY_RE.match(line.strip())
            ) is not None
        ]
        for position, (index, raw_id) in enumerate(body_headers):
            end = (
                body_headers[position + 1][0]
                if position + 1 < len(body_headers)
                else next(
                    (
                        candidate
                        for candidate in range(index + 1, len(lines))
                        if lines[candidate].strip() == "failures:"
                    ),
                    len(lines),
                )
            )
            failure_bodies[raw_id] = "\n".join(lines[index + 1 : end])
        for line in lines:
            failure_match = _LIBTEST_FAILURE_RE.match(line.strip())
            if failure_match is None:
                continue
            raw_id = failure_match.group(1).strip()
            failures.append(
                (
                    f"{harness}::{raw_id}",
                    "FAILED",
                    _fingerprint(failure_bodies.get(raw_id, raw_id)),
                )
            )
    if not sections:
        return _invalid(
            "rust-libtest",
            actual_returncode,
            combined,
            "libtest emitted no result records",
        )
    return _finish_outcome(
        parser_kind="rust-libtest",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=tuple(failures),
    )


def _go_json_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = stdout + stderr
    outputs: dict[tuple[str, str], list[str]] = {}
    terminals: dict[tuple[str, str], str] = {}
    malformed = 0
    for raw_line in stdout.splitlines():
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if not isinstance(record, dict):
            malformed += 1
            continue
        package = record.get("Package")
        test_id = record.get("Test")
        action = record.get("Action")
        if not isinstance(package, str) or not isinstance(test_id, str):
            continue
        key = (package, test_id)
        output = record.get("Output")
        if isinstance(output, str):
            outputs.setdefault(key, []).append(output)
        if action in {"pass", "fail", "skip"}:
            if key in terminals:
                return _invalid(
                    "go-json",
                    actual_returncode,
                    combined,
                    "go test emitted duplicate terminal test records",
                )
            terminals[key] = str(action)
    if malformed:
        return _invalid(
            "go-json",
            actual_returncode,
            combined,
            "go test -json emitted malformed records",
        )
    summary = {
        "tests_passed": sum(action == "pass" for action in terminals.values()),
        "tests_failed": sum(action == "fail" for action in terminals.values()),
        "tests_skipped": sum(action == "skip" for action in terminals.values()),
    }
    failures = tuple(
        (
            f"{package}::{test_id}",
            "FAILED",
            _fingerprint("".join(outputs.get((package, test_id), ()))),
        )
        for (package, test_id), action in terminals.items()
        if action == "fail"
    )
    return _finish_outcome(
        parser_kind="go-json",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=failures,
    )


def _jest_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = _ANSI_RE.sub("", stdout + stderr)
    matches = [
        match
        for line in combined.splitlines()
        if (match := _JEST_TESTS_RE.match(line.strip())) is not None
    ]
    if not matches:
        return _invalid(
            "jest",
            actual_returncode,
            combined,
            "Jest emitted no Tests summary",
        )
    summary = _count_summary(matches[-1].group("body"))
    lines = combined.splitlines()
    failure_indexes = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := _JEST_FAILURE_RE.match(line)) is not None
    ]
    failures: list[tuple[str, str, str]] = []
    for position, (index, node_id) in enumerate(failure_indexes):
        end = (
            failure_indexes[position + 1][0]
            if position + 1 < len(failure_indexes)
            else len(lines)
        )
        failures.append(
            (
                node_id,
                "FAILED",
                _fingerprint("\n".join(lines[index + 1 : end])),
            )
        )
    return _finish_outcome(
        parser_kind="jest",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=tuple(failures),
    )


def _vitest_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = _ANSI_RE.sub("", stdout + stderr)
    lines = combined.splitlines()
    matches = [
        match for line in lines if (match := _VITEST_TESTS_RE.match(line)) is not None
    ]
    if not matches:
        return _invalid(
            "vitest",
            actual_returncode,
            combined,
            "Vitest emitted no terminal Tests summary",
        )
    terminal = matches[-1]
    summary = _count_summary(terminal.group("body"))
    if sum(summary.values()) != int(terminal.group("total")):
        return _invalid(
            "vitest",
            actual_returncode,
            combined,
            "Vitest Tests summary did not account for its total",
        )
    failure_indexes = [
        (index, match.group(1).strip())
        for index, line in enumerate(lines)
        if (match := _VITEST_FAILURE_RE.match(line)) is not None
    ]
    failures: list[tuple[str, str, str]] = []
    for position, (index, node_id) in enumerate(failure_indexes):
        end = (
            failure_indexes[position + 1][0]
            if position + 1 < len(failure_indexes)
            else next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if _VITEST_TESTS_RE.match(lines[candidate]) is not None
                ),
                len(lines),
            )
        )
        failures.append(
            (
                node_id,
                "FAILED",
                _fingerprint("\n".join(lines[index + 1 : end])),
            )
        )
    return _finish_outcome(
        parser_kind="vitest",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=tuple(failures),
    )


def strict_tap_test_records(
    stdout: str,
    stderr: str,
    *,
    node_style: bool,
) -> tuple[TapTestRecord, ...]:
    """Extract TAP13 leaf results with stable suite/source identities.

    Node emits both a leaf ``not ok`` record and a parent-suite ``not ok``
    record for one failing nested test. The diagnostic ``type`` field is the
    authority used to discard the suite wrapper. AVA is flat TAP and does not
    emit that field, so its records remain leaves by construction.
    """

    combined = _ANSI_RE.sub("", stdout + stderr)
    lines = combined.splitlines()
    stack: dict[int, str] = {}
    raw_records: list[tuple[int, int, str, str, str, str]] = []
    for index, line in enumerate(lines):
        if subtest := _TAP_SUBTEST_RE.match(line):
            indent = len(subtest.group("indent"))
            stack = {depth: title for depth, title in stack.items() if depth < indent}
            stack[indent] = subtest.group("title").strip()
            continue
        result = _TAP_RESULT_RE.match(line)
        if result is None:
            continue
        indent = len(result.group("indent"))
        raw_title = (result.group("title") or "").strip()
        directive = ""
        if directive_match := _TAP_DIRECTIVE_RE.search(raw_title):
            directive = directive_match.group("directive").lower()
            raw_title = raw_title[: directive_match.start()].rstrip()
        path = [stack[depth] for depth in sorted(stack) if depth <= indent]
        if not path or path[-1] != raw_title:
            path.append(raw_title or f"tap-result-{index}")
        raw_records.append(
            (
                index,
                indent,
                result.group("result").lower(),
                raw_title,
                " > ".join(path),
                directive,
            )
        )

    records: list[TapTestRecord] = []
    for position, (index, _indent, result, title, identity, directive) in enumerate(
        raw_records
    ):
        end = (
            raw_records[position + 1][0]
            if position + 1 < len(raw_records)
            else len(lines)
        )
        diagnostic_lines = lines[index + 1 : end]
        diagnostic_type = next(
            (
                match.group("type").lower()
                for line in diagnostic_lines
                if (match := _TAP_TYPE_RE.match(line)) is not None
            ),
            "",
        )
        if node_style and diagnostic_type == "suite":
            continue
        location = next(
            (
                match.group("location")
                for line in diagnostic_lines
                if (match := _TAP_LOCATION_RE.match(line)) is not None
            ),
            "",
        )
        if location:
            source = re.sub(r":\d+:\d+$", "", location).removeprefix("/app/")
            identity = f"{source}::{identity}"
        if directive in {"skip", "todo"}:
            status = "not_observed"
        elif result == "ok":
            status = "passed"
        else:
            status = "failed"
        records.append(
            TapTestRecord(
                title=title,
                identity=identity,
                status=status,
                diagnostic="\n".join(diagnostic_lines),
                line=lines[index],
            )
        )
    return tuple(records)


def _tap_outcome(
    actual_returncode: int,
    stdout: str,
    stderr: str,
    *,
    parser_kind: str,
    node_style: bool,
) -> TestOutcome:
    combined = _ANSI_RE.sub("", stdout + stderr)
    lines = combined.splitlines()
    if sum(_TAP_VERSION_RE.match(line) is not None for line in lines) != 1:
        return _invalid(
            parser_kind,
            actual_returncode,
            combined,
            "test runner emitted no unique TAP version 13 stream",
        )
    terminal: dict[str, int] = {}
    aliases = {"skip": "skipped", "skipped": "skipped"}
    for line in lines:
        if (match := _TAP_SUMMARY_RE.match(line)) is None:
            continue
        key = aliases.get(match.group("kind").lower(), match.group("kind").lower())
        terminal[key] = int(match.group("count"))
    if not {"tests", "pass", "fail"}.issubset(terminal):
        return _invalid(
            parser_kind,
            actual_returncode,
            combined,
            "TAP stream lacked terminal tests/pass/fail accounting",
        )
    accounted = sum(
        terminal.get(key, 0)
        for key in ("pass", "fail", "cancelled", "skipped", "todo")
    )
    if accounted != terminal["tests"]:
        return _invalid(
            parser_kind,
            actual_returncode,
            combined,
            "TAP terminal accounting did not equal its test total",
        )
    summary = {
        "tests_passed": terminal.get("pass", 0),
        "tests_failed": terminal.get("fail", 0),
        "tests_errored": terminal.get("cancelled", 0),
        "tests_skipped": terminal.get("skipped", 0),
        "tests_expected_failed": terminal.get("todo", 0),
    }
    records = strict_tap_test_records(stdout, stderr, node_style=node_style)
    failures = tuple(
        (
            record.identity,
            "FAILED",
            _fingerprint(record.diagnostic or record.line),
        )
        for record in records
        if record.status == "failed"
    )
    return _finish_outcome(
        parser_kind=parser_kind,
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=failures,
    )


def _node_tap_outcome(
    actual_returncode: int, stdout: str, stderr: str
) -> TestOutcome:
    return _tap_outcome(
        actual_returncode,
        stdout,
        stderr,
        parser_kind="node-tap",
        node_style=True,
    )


def _ava_tap_outcome(
    actual_returncode: int, stdout: str, stderr: str
) -> TestOutcome:
    return _tap_outcome(
        actual_returncode,
        stdout,
        stderr,
        parser_kind="ava-tap",
        node_style=False,
    )


def _mocha_failure_identity(
    lines: list[str],
    *,
    index: int,
    end: int,
    headline: str,
) -> str:
    """Include Mocha's indented test-title continuation in the identity."""

    parts = [headline.rstrip(":")]
    for line in lines[index + 1 : end]:
        stripped = line.strip()
        if not stripped:
            if len(parts) > 1:
                break
            continue
        indent = len(line) - len(line.lstrip(" "))
        # Spec reporter failure titles are indented seven or more spaces;
        # the exception body begins at five. This is stable across nested
        # suite depths because each additional title component adds indent.
        if indent < 7:
            break
        parts.append(stripped.rstrip(":"))
    return " > ".join(part for part in parts if part)


def _mocha_failure_header(
    lines: list[str],
    *,
    index: int,
) -> tuple[int, str] | None:
    """Recognize one Base.list header, excluding numbered body diagnostics."""

    match = _MOCHA_FAILURE_RE.fullmatch(lines[index])
    if match is None or index == 0 or lines[index - 1].strip():
        return None
    headline = match.group("headline").strip()
    if not headline.endswith(":"):
        continuations: list[str] = []
        for line in lines[index + 1 :]:
            stripped = line.strip()
            if not stripped:
                break
            indent = len(line) - len(line.lstrip(" "))
            if indent < 7:
                break
            continuations.append(stripped)
        if not continuations or not continuations[-1].endswith(":"):
            return None
    return int(match.group("number")), headline


def _mocha_outcome(actual_returncode: int, stdout: str, stderr: str) -> TestOutcome:
    combined = _ANSI_RE.sub("", stdout + stderr)
    # Mocha's bundled reporters write their Base.epilogue frame to stdout.
    # Parse that stream alone so warnings and application diagnostics written
    # to stderr cannot become a second, attacker-controlled result frame.
    lines = _ANSI_RE.sub("", stdout).splitlines()

    summary_patterns = (
        ("tests_passed", _MOCHA_PASSING_RE),
        ("tests_failed", _MOCHA_FAILING_RE),
        ("tests_skipped", _MOCHA_PENDING_RE),
    )
    summary_blocks: list[list[tuple[int, str, int]]] = []
    current_block: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        parsed = next(
            (
                (kind, int(match.group("count")))
                for kind, pattern in summary_patterns
                if (match := pattern.fullmatch(line)) is not None
            ),
            None,
        )
        if parsed is None:
            if current_block:
                summary_blocks.append(current_block)
                current_block = []
            continue
        kind, count = parsed
        current_block.append((index, kind, count))
    if current_block:
        summary_blocks.append(current_block)
    if not summary_blocks:
        return _invalid(
            "mocha",
            actual_returncode,
            combined,
            "Mocha emitted no canonical terminal summary block on stdout",
        )

    terminal_block = summary_blocks[-1]
    terminal_kinds = [kind for _index, kind, _count in terminal_block]
    valid_kind_orders = {
        ("tests_passed",),
        ("tests_passed", "tests_skipped"),
        ("tests_passed", "tests_failed"),
        ("tests_passed", "tests_skipped", "tests_failed"),
    }
    if tuple(terminal_kinds) not in valid_kind_orders:
        return _invalid(
            "mocha",
            actual_returncode,
            combined,
            "Mocha terminal summary block used a noncanonical result-kind order",
        )
    terminal_counts = {kind: count for _index, kind, count in terminal_block}
    passed = terminal_counts.get("tests_passed", 0)
    failed = terminal_counts.get("tests_failed", 0)
    pending = terminal_counts.get("tests_skipped", 0)
    if (
        "tests_failed" in terminal_counts and failed == 0
        or "tests_skipped" in terminal_counts and pending == 0
    ):
        return _invalid(
            "mocha",
            actual_returncode,
            combined,
            "Mocha terminal summary emitted a zero optional result kind",
        )
    summary = {
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_skipped": pending,
    }
    summary_end = terminal_block[-1][0]
    trailing_lines = lines[summary_end + 1 :]
    if any(_MOCHA_SUMMARY_PREFIX_RE.match(line) for line in trailing_lines):
        return _invalid(
            "mocha",
            actual_returncode,
            combined,
            "Mocha terminal frame contained trailing summary-like output",
        )
    failure_indexes: list[tuple[int, int, str]] = []
    for index in range(summary_end + 1, len(lines)):
        header = _mocha_failure_header(lines, index=index)
        if header is not None:
            number, headline = header
            failure_indexes.append((index, number, headline))
    failure_numbers = [number for _index, number, _headline in failure_indexes]
    if failure_numbers != list(range(1, failed + 1)):
        return _invalid(
            "mocha",
            actual_returncode,
            combined,
            "Mocha detailed failure enumerators did not exactly match the "
            "terminal failing count",
        )
    if not failed:
        if any(line.strip() for line in trailing_lines):
            return _invalid(
                "mocha",
                actual_returncode,
                combined,
                "Mocha passing summary was not the terminal stdout frame",
            )
    else:
        first_failure = failure_indexes[0][0]
        if any(line.strip() for line in lines[summary_end + 1 : first_failure]):
            return _invalid(
                "mocha",
                actual_returncode,
                combined,
                "Mocha detailed failures did not immediately follow the summary",
            )
        if any(
            index <= 0 or lines[index - 1].strip()
            for index, _number, _headline in failure_indexes
        ):
            return _invalid(
                "mocha",
                actual_returncode,
                combined,
                "Mocha detailed failure headers lacked reporter boundaries",
            )
    failures: list[tuple[str, str, str]] = []
    for position, (index, _number, headline) in enumerate(failure_indexes):
        end = (
            failure_indexes[position + 1][0]
            if position + 1 < len(failure_indexes)
            else len(lines)
        )
        node_id = _mocha_failure_identity(
            lines,
            index=index,
            end=end,
            headline=headline,
        )
        failures.append(
            (
                node_id,
                "FAILED",
                _fingerprint("\n".join(lines[index + 1 : end])),
            )
        )
    return _finish_outcome(
        parser_kind="mocha",
        actual_returncode=actual_returncode,
        combined=combined,
        summary=summary,
        failures=tuple(failures),
    )


def parse_test_outcome(
    *,
    language: str,
    command: str,
    actual_returncode: int,
    stdout: str,
    stderr: str,
    parser_hint: str | None = None,
) -> TestOutcome:
    """Parse one suite result, rejecting zero-test and ambiguous success."""

    hint = (parser_hint or "").strip().lower()
    hint = {
        "node": "node-tap",
        "node-test": "node-tap",
        "node:test": "node-tap",
        "ava": "ava-tap",
    }.get(hint, hint)
    command_lower = command.lower()
    if not hint:
        if language == "python":
            hint = "pytest"
        elif language == "rust":
            hint = "rust-libtest"
        elif language == "go":
            hint = "go-json"
        elif re.search(r"\bava\b", command_lower):
            hint = "ava-tap" if "--tap" in command_lower else "ava-unsupported"
        elif re.search(r"\bnode\b[^\n;&|]*\s--test\b", command_lower):
            hint = "node-tap"
        elif "jest" in command_lower:
            hint = "jest"
        elif "mocha" in command_lower:
            hint = "mocha"
        elif "vitest" in command_lower:
            hint = "vitest"
        elif re.search(r"(?m)^Tests:\s+", stdout + stderr):
            hint = "jest"
        elif any(
            _VITEST_TESTS_RE.match(line) is not None
            for line in _ANSI_RE.sub("", stdout + stderr).splitlines()
        ):
            hint = "vitest"
        elif re.search(r"(?m)^\s*\d+\s+passing\b", stdout + stderr):
            hint = "mocha"
        elif (
            re.search(r"(?m)^\s*TAP version 13\s*$", stdout + stderr)
            and re.search(r"(?m)^\s*#\s*suites\s+\d+\s*$", stdout + stderr)
        ):
            hint = "node-tap"
    parsers = {
        "pytest": _pytest_outcome,
        "rust-libtest": _libtest_outcome,
        "go-json": _go_json_outcome,
        "jest": _jest_outcome,
        "vitest": _vitest_outcome,
        "mocha": _mocha_outcome,
        "node-tap": _node_tap_outcome,
        "ava-tap": _ava_tap_outcome,
    }
    parser = parsers.get(hint)
    if parser is None:
        reason = (
            "AVA output is unsupported unless the Crucible-owned command "
            "selects AVA's deterministic --tap reporter"
            if hint == "ava-unsupported"
            else "no strict parser is configured for the development suite"
        )
        return _invalid(
            hint or "unsupported",
            actual_returncode,
            stdout + stderr,
            reason,
        )
    return parser(actual_returncode, stdout, stderr)


def _failure_total(outcome: TestOutcome) -> int:
    summary = dict(outcome.summary)
    return summary.get("tests_failed", 0) + summary.get("tests_errored", 0)


def compare_outcomes(
    candidate: TestOutcome,
    baseline: TestOutcome,
    *,
    expected_extra_tests: int = 0,
    cache_hit: bool = False,
    cache_key: str = "",
) -> BaselineComparison:
    """Forgive only byte-stable base failures; reject vacuous deltas."""

    common = {
        "cache_hit": cache_hit,
        "cache_key": cache_key,
        "baseline_returncode": baseline.returncode,
    }
    blocking_rc = max(1, candidate.returncode)
    if not candidate.structurally_valid or not baseline.structurally_valid:
        return BaselineComparison(
            "incomparable",
            blocking_rc,
            "candidate or baseline outcome was structurally invalid",
            **common,
        )
    if candidate.parser_kind != baseline.parser_kind:
        return BaselineComparison(
            "incomparable",
            blocking_rc,
            "candidate and baseline parser kinds differ",
            **common,
        )
    if (
        candidate.returncode == 1
        and baseline.returncode == 1
        and (
            not candidate.failure_set_complete
            or not baseline.failure_set_complete
        )
    ):
        return BaselineComparison(
            "incomparable",
            blocking_rc,
            "candidate or baseline failure set was truncated or incomplete",
            **common,
        )
    assert candidate.tests_run is not None
    assert baseline.tests_run is not None
    minimum_candidate_tests = baseline.tests_run + max(0, expected_extra_tests)
    if candidate.tests_run < minimum_candidate_tests:
        return BaselineComparison(
            "incomparable",
            blocking_rc,
            "candidate did not execute every clean-base and expected attack test",
            **common,
        )
    candidate_summary = dict(candidate.summary)
    baseline_summary = dict(baseline.summary)
    for key in (
        "tests_skipped",
        "tests_expected_failed",
        "tests_unexpected_successes",
        "tests_deselected",
        "tests_rerun",
    ):
        if candidate_summary.get(key, 0) > baseline_summary.get(key, 0):
            return BaselineComparison(
                "regressed",
                blocking_rc,
                "candidate increased skipped or non-passing outcomes",
                worsened_failures=1,
                **common,
            )
    candidate_total = _failure_total(candidate)
    baseline_total = _failure_total(baseline)
    if candidate.returncode == 0:
        return BaselineComparison(
            "baseline_match",
            0,
            "candidate suite passed without regressing the clean base",
            preexisting_failures=baseline_total,
            **common,
        )
    if baseline.returncode == 0:
        return BaselineComparison(
            "regressed",
            1,
            "candidate introduced failures on a passing clean base",
            new_failures=candidate_total,
            **common,
        )
    if candidate_total != len(candidate.failures) or baseline_total != len(
        baseline.failures
    ):
        return BaselineComparison(
            "incomparable",
            blocking_rc,
            "candidate or baseline failure identities were incomplete",
            **common,
        )
    baseline_failures = {
        node_id: (status, reason) for node_id, status, reason in baseline.failures
    }
    new_failures = 0
    worsened_failures = 0
    preexisting_failures = 0
    for node_id, status, reason in candidate.failures:
        prior = baseline_failures.get(node_id)
        if prior is None:
            new_failures += 1
        elif prior != (status, reason):
            worsened_failures += 1
        else:
            preexisting_failures += 1
    if new_failures or worsened_failures or candidate_total > baseline_total:
        return BaselineComparison(
            "regressed",
            1,
            "candidate introduced or worsened clean-base failures",
            preexisting_failures=preexisting_failures,
            new_failures=new_failures,
            worsened_failures=worsened_failures,
            **common,
        )
    return BaselineComparison(
        "preexisting_only",
        0,
        "all candidate failures are unchanged clean-base failures",
        preexisting_failures=preexisting_failures,
        **common,
    )


def baseline_identity(
    *,
    image_digest: str,
    base_commit: str,
    command: str,
    language: str,
    parser_hint: str | None,
) -> dict[str, str]:
    return {
        "schema": BASELINE_CACHE_SCHEMA,
        "parser_schema": PARSER_SCHEMA,
        "image_digest": image_digest,
        "base_commit": base_commit,
        "command": command,
        "language": language,
        "parser_hint": (parser_hint or ""),
    }


def identity_key(identity: dict[str, str]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_root() -> Path:
    configured = os.environ.get(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path(f"/data/.crucible-deepswe-visible-baselines-{os.geteuid()}")


def _secure_cache_root(root: Path) -> bool:
    try:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        stat_result = root.lstat()
        if root.is_symlink() or not root.is_dir():
            return False
        if stat_result.st_uid != os.geteuid():
            return False
        os.chmod(root, 0o700)
    except OSError:
        return False
    return True


def cached_baseline_outcome(
    identity: dict[str, str],
    producer: Callable[[], TestOutcome | None],
) -> tuple[TestOutcome | None, bool, str, str]:
    """Atomically load or populate one immutable clean-base outcome."""

    key = identity_key(identity)
    root = _cache_root()
    if not _secure_cache_root(root):
        return None, False, key, "baseline cache root is unavailable or unsafe"
    cache_path = root / f"{key}.json"
    lock_path = root / f"{key}.lock"
    try:
        with lock_path.open("a+b") as lock_file:
            os.chmod(lock_path, 0o600)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            if cache_path.is_file() and not cache_path.is_symlink():
                try:
                    raw = json.loads(cache_path.read_text(encoding="utf-8"))
                    if (
                        isinstance(raw, dict)
                        and raw.get("identity") == identity
                    ):
                        return (
                            TestOutcome.from_payload(raw.get("outcome")),
                            True,
                            key,
                            "",
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            outcome = producer()
            if outcome is None or not outcome.structurally_valid:
                return (
                    outcome,
                    False,
                    key,
                    (
                        outcome.structural_error
                        if outcome is not None
                        else "clean-base runner did not produce an outcome"
                    ),
                )
            payload = json.dumps(
                {"identity": identity, "outcome": outcome.to_payload()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fd, temporary = tempfile.mkstemp(prefix=f".{key}.", dir=root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, cache_path)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            return outcome, False, key, ""
    except OSError as exc:
        return (
            None,
            False,
            key,
            f"baseline cache operation failed: {type(exc).__name__}",
        )


__all__ = [
    "BASELINE_CACHE_SCHEMA",
    "PARSER_SCHEMA",
    "BaselineComparison",
    "TapTestRecord",
    "TestOutcome",
    "baseline_identity",
    "cached_baseline_outcome",
    "compare_outcomes",
    "identity_key",
    "parse_test_outcome",
    "strict_tap_test_records",
]
