# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the read-only state script under .claude/skills/_shared/bin/.

Two regressions captured here:

1. ``_RUN_DIR_RE`` must match the actual run-dir stamping in
   ``assay.Corpus.open_run`` — ``YYYYMMDDTHHMMSSZ_<recipe>_<tag>``,
   with a trailing ``Z`` on the timestamp.
2. ``_derive_target`` must extract ``<target>`` from the canonical
   path ``.../.crucible/specs/<target>/<spec>/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / ".claude/skills/_shared/bin/spec_state.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("spec_state", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("spec_state", mod)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Run-dir regex
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_workflow",
    [
        # Canonical assay.Corpus.open_run stamp.
        ("20260516T053200Z_implement_a7b2", "implement"),
        ("20260514T130502Z_architect_deadbeef", "architect"),
        ("20260101T000000Z_breach_0001", "breach"),
        # Suite and underscore-bearing workflow names.
        ("20260514T130502Z_swebench_a1b2", "swebench"),
        ("20260514T130502Z_deepswe_a1b2", "deepswe"),
        ("20260514T130502Z_journal_recover_ff00", "journal_recover"),
    ],
)
def test_run_dir_regex_matches_assay_stamping(name, expected_workflow):
    mod = _load_module()
    m = mod._RUN_DIR_RE.match(name)
    assert m is not None, f"regex did not match {name!r}"
    assert m.group(1) == expected_workflow


def test_run_dir_regex_rejects_garbage():
    mod = _load_module()
    for name in ("not_a_run", "abc_def_ghi", "20260516_implement_a7b2"):
        # The middle one is the tricky case: it lacks the timestamp shape.
        # The last lacks the `T` separator.
        assert mod._RUN_DIR_RE.match(name) is None, f"should have rejected {name!r}"


# ----------------------------------------------------------------------
# target derivation from canonical path
# ----------------------------------------------------------------------


def test_derive_target_canonical_layout(tmp_path):
    mod = _load_module()
    spec_dir = tmp_path / ".crucible" / "specs" / "kit" / "demo_spec"
    spec_dir.mkdir(parents=True)
    assert mod._derive_target(spec_dir) == "kit"


def test_derive_target_with_hyphenated_target(tmp_path):
    mod = _load_module()
    spec_dir = tmp_path / ".crucible" / "specs" / "fabric-usd" / "fr_pipeline"
    spec_dir.mkdir(parents=True)
    assert mod._derive_target(spec_dir) == "fabric-usd"


def test_derive_target_outside_canonical_returns_empty(tmp_path):
    mod = _load_module()
    spec_dir = tmp_path / "not_a_corpus" / "demo"
    spec_dir.mkdir(parents=True)
    assert mod._derive_target(spec_dir) == ""


# ----------------------------------------------------------------------
# End-to-end smoke: full output against a synthetic Corpus
# ----------------------------------------------------------------------


def test_full_state_report_counts_stages(tmp_path):
    mod = _load_module()
    spec_dir = tmp_path / ".crucible" / "specs" / "kit" / "demo_spec"
    (spec_dir / "versions").mkdir(parents=True)
    (spec_dir / "versions" / "v0_deadbeef.json").write_text("{}")
    (spec_dir / "spec.yaml").write_text(
        "project_context:\n"
        "  project: kit\n"
        "  submodule_path: null\n",
    )
    runs = spec_dir / "runs"
    runs.mkdir()
    for name in (
        "20260514T130502Z_architect_a1b2",
        "20260514T140000Z_swebench_c3d4",
        "20260515T100000Z_reconcile_e5f6",
        "20260515T110000Z_implement_0987",
    ):
        (runs / name).mkdir()

    paths = mod._resolve_paths(spec_dir)
    state = mod._spec_state(spec_dir)

    assert paths["target"] == "kit"
    assert paths["project"] == "kit"
    assert paths["submodule_path"] == ""  # `null` collapses to empty
    assert state["spec_version"] == "0"
    assert state["runs_total"] == "4"
    assert state["runs_build"] == "2"  # architect + implement
    assert state["runs_benchmark"] == "1"  # swebench
    assert state["runs_recovery"] == "1"  # reconcile
    assert state["runs_design"] == "0"


def test_state_report_does_not_create_subdirs(tmp_path):
    """Regression guard: the script must not call Corpus(spec_dir)."""
    mod = _load_module()
    spec_dir = tmp_path / "fresh_spec"
    spec_dir.mkdir()
    mod._resolve_paths(spec_dir)
    mod._spec_state(spec_dir)
    # Side-effecting subdirs from Corpus.__init__ must NOT be present.
    for sub in ("versions", "artifacts", "transcripts", "diffs", "traces", "runs"):
        assert not (spec_dir / sub).exists(), f"unexpected side-effect dir: {sub}"
