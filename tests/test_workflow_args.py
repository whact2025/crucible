# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Boundary validation for shared workflow resource arguments."""

from __future__ import annotations

import argparse

import pytest

from crucible.workflows._args import (
    add_budget_arg,
    add_max_turns_arg,
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
