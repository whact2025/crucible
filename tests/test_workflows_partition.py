# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Generic Handoff obligations retain exact gate and output authority."""

from types import SimpleNamespace

from crucible.constraints import AuthoringFinalizationConstraint
from crucible.workflows._partition import _handoff_verification_obligation


def test_output_producing_obligation_uses_generic_constraint() -> None:
    selected = AuthoringFinalizationConstraint(
        slice_id="native-build",
        materializes_paths=("reports/native-build.json",),
        gate="dev_verify",
    )
    spec = SimpleNamespace(constraints=(selected,))
    assert _handoff_verification_obligation(spec, "native-build") == (
        "dev_verify", ("reports/native-build.json",),
    )
    assert _handoff_verification_obligation(spec, "ordinary") == ("verify", None)


def test_duplicate_output_authority_is_rejected() -> None:
    selected = AuthoringFinalizationConstraint(
        slice_id="native-build",
        materializes_paths=("reports/native-build.json",),
    )
    spec = SimpleNamespace(constraints=(selected, selected))
    assert _handoff_verification_obligation(spec, "native-build") is None
