# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Boundary validation for the arguments that workflows share."""

from __future__ import annotations

import argparse

import pytest

from crucible.workflows._args import (
    SPEC_READ_AGAINST_HELP,
    SPEC_READ_AGAINST_OPTION,
    SPEC_READ_AGAINST_PARENT,
    SPEC_READ_AGAINST_VALUES,
    SPEC_READ_AGAINST_WRITE_ROOT,
    add_budget_arg,
    add_max_turns_arg,
    add_spec_arg,
    add_spec_read_against_arg,
    nonnegative_int_arg,
    validate_nonnegative_int,
)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_max_turns_requires_a_positive_integer(value: str):
    parser = argparse.ArgumentParser()
    add_max_turns_arg(parser, default=5, scope="per call")

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-turns", value])


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf"])
def test_cumulative_budget_rejects_invalid_values(value: str):
    parser = argparse.ArgumentParser()
    add_budget_arg(parser, kind="cumulative")

    with pytest.raises(SystemExit):
        parser.parse_args(["--budget-usd", value])


def test_cumulative_budget_allows_zero_as_an_explicit_no_call_cap():
    parser = argparse.ArgumentParser()
    add_budget_arg(parser, kind="cumulative")

    assert parser.parse_args(["--budget-usd", "0"]).budget_usd == 0.0


@pytest.mark.parametrize("value", ["0", "3"])
def test_nonnegative_integer_allows_zero_and_positive_values(value: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=nonnegative_int_arg)

    assert parser.parse_args(["--count", value]).count == int(value)


def test_nonnegative_integer_rejects_negative_values():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=nonnegative_int_arg)

    with pytest.raises(SystemExit):
        parser.parse_args(["--count", "-1"])


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_programmatic_nonnegative_integer_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="count must be a non-negative integer"):
        validate_nonnegative_int(value, name="count")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_per_call_budget_requires_a_positive_finite_value(value: str):
    parser = argparse.ArgumentParser()
    add_budget_arg(parser, kind="per_call")

    with pytest.raises(SystemExit):
        parser.parse_args(["--budget-usd", value])


def _spec_read_against_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workflow")
    add_spec_arg(parser)
    add_spec_read_against_arg(parser)
    return parser


def test_spec_read_against_has_one_spelling_and_two_values():
    assert SPEC_READ_AGAINST_OPTION == "--spec-read-against"
    assert SPEC_READ_AGAINST_WRITE_ROOT == "write-root"
    assert SPEC_READ_AGAINST_PARENT == "parent"
    assert SPEC_READ_AGAINST_VALUES == ("write-root", "parent")


@pytest.mark.parametrize("value", SPEC_READ_AGAINST_VALUES)
def test_spec_read_against_takes_each_value_beside_the_spec(value: str):
    parser = _spec_read_against_parser()

    before = parser.parse_args(["--spec-read-against", value, "--spec", "corpus"])
    after = parser.parse_args(["--spec", "corpus", f"--spec-read-against={value}"])

    assert before.spec_read_against == after.spec_read_against == value
    assert str(before.spec) == "corpus"


def test_spec_read_against_is_no_statement_when_absent():
    parsed = _spec_read_against_parser().parse_args(["--spec", "corpus"])

    assert parsed.spec_read_against is None


@pytest.mark.parametrize("value", [
    "..", ".", "/srv/checkout", "checkout/proj", "write_root", "writeroot",
    "write-root/", "WRITE-ROOT", "PARENT", "Parent", "root", "above", "",
])
def test_spec_read_against_is_one_of_two_words_and_never_a_directory(
    value: str, capsys: pytest.CaptureFixture[str],
):
    with pytest.raises(SystemExit) as exit_info:
        _spec_read_against_parser().parse_args(
            ["--spec", "corpus", "--spec-read-against", value],
        )

    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_spec_read_against_needs_a_value(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        _spec_read_against_parser().parse_args(["--spec", "corpus", "--spec-read-against"])

    assert exit_info.value.code == 2
    assert "expected one argument" in capsys.readouterr().err


def test_spec_read_against_help_is_the_shared_text(capsys: pytest.CaptureFixture[str]):
    # Both words are explained, the value is said to be no directory and to
    # be stored nowhere, and the text says in which checkout it is needed.
    assert SPEC_READ_AGAINST_HELP == (
        "What the paths of the Spec are relative to: the directory this run "
        "writes in (write-root) or the directory above it (parent). The value is "
        "one of these two words, not a directory. Needed in a checkout whose "
        "root holds a directory of its own name, when a path of the Spec begins "
        "with that name or a scope entry of the Spec is that name itself; and, "
        "where a refusal asks for it, in a checkout that holds no entry of its "
        "own name, for a Spec with a scope entry that is that name itself. In a "
        "checkout that holds a directory of its own name the value is given on "
        "every run of such a Spec, the same each time, by where the Spec was "
        "planned; it is not stored."
    )
    # A formatter that does not wrap shows the text as the parser holds it.
    parser = argparse.ArgumentParser(
        prog="workflow",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, width=100_000),
    )
    add_spec_arg(parser)
    add_spec_read_against_arg(parser)

    shown = parser.format_help()

    # Optional, and named twice: in the usage and on its own line. That line
    # holds the one spelling and the two values and nothing else, and the
    # line below it is the shared text and nothing else.
    assert "[--spec-read-against {write-root,parent}]" in shown
    assert shown.count(SPEC_READ_AGAINST_OPTION) == 2
    lines = shown.splitlines()
    at = lines.index("  --spec-read-against {write-root,parent}")
    assert lines[at + 1].strip() == SPEC_READ_AGAINST_HELP
    with pytest.raises(SystemExit):
        _spec_read_against_parser().parse_args(["--help"])
    assert "--spec-read-against {write-root,parent}" in capsys.readouterr().out
