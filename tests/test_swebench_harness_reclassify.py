# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.swebench.harness_reclassify``.

Each rule has at least one positive fixture (rule must fire) and one
negative fixture (rule must NOT fire). Negative coverage prevents
overly-broad rules from passing the suite.

The rule functions take only the fields they need (FAIL_TO_PASS,
PASS_TO_PASS) — duck-typed access keeps these tests hermetic without
importing the full swebench TestSpec construction pipeline.
"""
from __future__ import annotations

import dataclasses

import json
from pathlib import Path

import pytest

from crucible.products.swebench.harness_reclassify import (
    ReclassifyVerdict,
    detect_m4_collection_error,
    format_reclassification_reason,
    has_existing_reclassification,
    reclassify,
    rule_m1_tox_wrapped_pytest,
    rule_m2_parametrize_template,
    rule_m3_django_docstring_concat,
)

_FIXTURES = Path(__file__).parent.parent / "products" / "swebench" / "tests" / "fixtures" / "harness_outputs"


@dataclasses.dataclass
class _MinimalTestSpec:
    """Hermetic stand-in for ``swebench.harness.test_spec.test_spec.TestSpec``.

    Carries only the fields the rules access. ``dataclasses.replace``
    works on this just as on the real TestSpec, so the rule's
    ``dataclasses.replace(test_spec, FAIL_TO_PASS=..., PASS_TO_PASS=...)``
    pattern is faithfully exercised here.
    """
    instance_id: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    repo: str = ""
    version: str = ""


# ---------- M2: parametrize-template F2P/P2P entry ----------


def test_m2_fires_on_astropy_7606_real_shape():
    """Pinned to astropy-7606's ACTUAL dataset shape: FAIL_TO_PASS is
    the concrete test ``test_unknown_unit3`` (no template), and
    PASS_TO_PASS contains 1 parametrize template
    ``test_compose_roundtrip[]`` alongside 240 concrete parametrize
    cases. The rule must filter the P2P template only, leaving F2P
    untouched.

    Verified against the live dataset entry at
    ``.crucible/datasets/swebench/verified.json`` and the committed
    fixture
    ``crucible/products/swebench/tests/fixtures/harness_outputs/astropy__astropy-7606/dataset_entry.json``.
    """
    spec = _MinimalTestSpec(
        instance_id="astropy__astropy-7606",
        FAIL_TO_PASS=["astropy/units/tests/test_units.py::test_unknown_unit3"],
        PASS_TO_PASS=[
            "astropy/units/tests/test_units.py::test_compose_roundtrip[]",
            "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]",
            "astropy/units/tests/test_units.py::test_compose_roundtrip[%]",
        ],
    )
    result = rule_m2_parametrize_template("", spec)
    assert result is not None, "rule must fire on empty-bracket P2P"
    _log, normalized, evidence = result
    assert normalized.FAIL_TO_PASS == [
        "astropy/units/tests/test_units.py::test_unknown_unit3"
    ], "F2P has no template; must be unchanged"
    assert normalized.PASS_TO_PASS == [
        "astropy/units/tests/test_units.py::test_compose_roundtrip[unit0]",
        "astropy/units/tests/test_units.py::test_compose_roundtrip[%]",
    ], "empty-bracket P2P entry must be filtered out; concrete cases must remain"
    assert "0 F2P templates" in evidence
    assert "1 P2P templates" in evidence


def test_m2_does_not_fire_on_well_formed_pytest_ids():
    """Pinned to django-11734's dataset shape: F2P is a Django
    unittest-style test ID, no parametrize templates anywhere. Rule
    must NOT fire — otherwise upstream's verdict would be silently
    altered.
    """
    spec = _MinimalTestSpec(
        instance_id="django__django-11734",
        FAIL_TO_PASS=["test_subquery_exclude_outerref (queries.tests.ExcludeTests)"],
        PASS_TO_PASS=["test_exclude_extra (queries.tests.ExcludeTests)"],
    )
    result = rule_m2_parametrize_template("", spec)
    assert result is None, (
        "rule must not fire on well-formed test IDs; django-11734 should "
        "fall through to upstream's verdict unchanged"
    )


def test_m2_does_not_fire_on_concrete_parametrize_ids():
    """Concrete parametrize cases like ``test_X[unit0]`` are not
    templates and pytest emits them normally — the rule must not
    treat them as misparse-prone.
    """
    spec = _MinimalTestSpec(
        instance_id="example__example-1",
        FAIL_TO_PASS=["mod/tests/test_x.py::test_y[case_a]"],
        PASS_TO_PASS=["mod/tests/test_x.py::test_y[case_b]"],
    )
    assert rule_m2_parametrize_template("", spec) is None


def test_m2_handles_empty_template_in_p2p_only():
    """Some datasets carry the template in P2P but not F2P. The rule
    still fires — but only filters P2P.
    """
    spec = _MinimalTestSpec(
        instance_id="example__example-2",
        FAIL_TO_PASS=["mod/tests/test_x.py::test_real"],
        PASS_TO_PASS=[
            "mod/tests/test_x.py::test_template[]",
            "mod/tests/test_x.py::test_template[a]",
        ],
    )
    result = rule_m2_parametrize_template("", spec)
    assert result is not None
    _log, normalized, _evidence = result
    assert normalized.FAIL_TO_PASS == ["mod/tests/test_x.py::test_real"]
    assert normalized.PASS_TO_PASS == ["mod/tests/test_x.py::test_template[a]"]


# ---------- M1: tox-wrapped non-verbose pytest log normalization ----------


def test_m1_fires_on_sphinx_8595_real_fixture():
    """Pinned to sphinx-8595's real test_output: tox-wrapped pytest
    with ``py39: exit 0`` and ``1 passed, 7 warnings`` summary, no
    upstream-parseable PASSED <id> lines, one F2P test in the
    duration block. The rule must fire and synthesize a
    ``PASSED tests/test_ext_autodoc_automodule.py::test_empty_all``
    line inserted before the END marker.
    """
    log_path = _FIXTURES / "sphinx-doc__sphinx-8595" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    text = log_path.read_text(errors="replace")
    spec = _MinimalTestSpec(
        instance_id="sphinx-doc__sphinx-8595",
        FAIL_TO_PASS=["tests/test_ext_autodoc_automodule.py::test_empty_all"],
        PASS_TO_PASS=[],
    )
    result = rule_m1_tox_wrapped_pytest(text, spec)
    assert result is not None, "M1 must fire on tox-wrapped exit-0 log"
    normalized_log, normalized_spec, evidence = result
    # The F2P test appears in the synthesized PASSED line.
    assert (
        "PASSED tests/test_ext_autodoc_automodule.py::test_empty_all"
        in normalized_log
    ), "synthesized PASSED line must include the duration-block test ID"
    # Synthesis is inserted before the END marker so upstream's
    # split-between-markers parsing picks it up.
    end_pos = normalized_log.index(">>>>> End Test Output")
    synth_pos = normalized_log.index(
        "PASSED tests/test_ext_autodoc_automodule.py::test_empty_all"
    )
    assert synth_pos < end_pos, (
        "synthesized PASSED line must be inserted BEFORE the "
        "END_TEST_OUTPUT marker so upstream parser sees it"
    )
    # Spec is untouched (M1 is log-only).
    assert normalized_spec is spec
    assert "tox exit_0" in evidence
    assert "1 passed" in evidence


def test_m1_does_not_fire_on_explicit_pytest_passed_lines():
    """Synthetic: log already has ``PASSED tests/foo.py::test_x``
    lines (the normal verbose-pytest case). Upstream parser handles
    these natively; M1 must NOT fire — otherwise we'd be
    double-synthesizing.
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "PASSED tests/foo.py::test_x\n"
        "py39: exit 0 (0.5 seconds)\n"
        "1 passed in 0.5s\n"
        ">>>>> End Test Output\n"
    )
    spec = _MinimalTestSpec("x", ["tests/foo.py::test_x"], [])
    assert rule_m1_tox_wrapped_pytest(synthetic, spec) is None


def test_m1_does_not_fire_on_tox_failure():
    """Synthetic: tox emits ``py39: exit 1`` — the wrapped pytest
    failed. M1 must NOT fire (we'd silently mark a failing run
    as resolved).
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "0.01s call    tests/foo.py::test_x\n"
        "1 failed in 0.5s\n"
        "py39: exit 1 (0.5 seconds)\n"
        ">>>>> End Test Output\n"
    )
    spec = _MinimalTestSpec("x", ["tests/foo.py::test_x"], [])
    assert rule_m1_tox_wrapped_pytest(synthetic, spec) is None


def test_m1_does_not_fire_on_django_unittest_ok():
    """Synthetic: Django unittest output with ``... ok``. No tox
    markers, no pytest summary. M1 must NOT fire (this is
    parse_log_django territory; M5 will handle it once Phase 0
    diagnoses the upstream gap).
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "test_X (mod.tests.Class) ... ok\n"
        "OK\n"
        ">>>>> End Test Output\n"
    )
    spec = _MinimalTestSpec("x", ["test_X (mod.tests.Class)"], [])
    assert rule_m1_tox_wrapped_pytest(synthetic, spec) is None


def test_m1_does_not_fire_when_mixed_pass_fail_summary():
    """Synthetic: pytest summary ``2 passed, 1 failed`` — there's at
    least one real failure. M1 must NOT fire — synthesizing PASSED
    lines for all duration-listed tests would silently flip a true
    failure to resolved.
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "0.01s call    tests/foo.py::test_x\n"
        "0.01s call    tests/foo.py::test_y\n"
        "0.01s call    tests/foo.py::test_z\n"
        "2 passed, 1 failed in 0.5s\n"
        "py39: exit 1\n"
        ">>>>> End Test Output\n"
    )
    spec = _MinimalTestSpec("x", ["tests/foo.py::test_x"], [])
    assert rule_m1_tox_wrapped_pytest(synthetic, spec) is None


def test_m1_composer_resolves_sphinx_8595_via_synthesis(tmp_path):
    """End-to-end via composer: real sphinx-8595 fixture + real
    ``swebench.harness.test_spec.TestSpec`` + real
    ``swebench.harness.grading.get_eval_report``. Before M1: upstream
    sees no PASSED lines, marks F2P as fail-by-absence,
    ``resolved=False``. After M1 synthesizes PASSED line: upstream
    finds the test in its status_map and marks ``resolved=True``.
    """
    log_path = _FIXTURES / "sphinx-doc__sphinx-8595" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("sphinx-doc__sphinx-8595")
    prediction = {
        "instance_id": "sphinx-doc__sphinx-8595",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "resolved", (
        f"after M1 synthesis, upstream get_eval_report should return "
        f"resolved=True. Got status={verdict.status!r}; upstream entry: "
        f"{verdict.upstream_report.get('sphinx-doc__sphinx-8595', {})}"
    )
    assert verdict.reclass_reason == "M1_tox_wrapped_pytest"
    assert "tox exit_0" in verdict.evidence_excerpt
    # The normalized log was written to tmp_path; verify the file
    # exists and contains the synthesized PASSED line.
    normalized_files = list(tmp_path.glob("*.reclass.txt"))
    assert len(normalized_files) == 1, (
        f"composer should have written exactly one normalized log; "
        f"found {normalized_files}"
    )
    normalized_text = normalized_files[0].read_text()
    assert "PASSED tests/test_ext_autodoc_automodule.py::test_empty_all" in normalized_text


# ---------- M3: Django docstring concatenated with next test ----------


def test_m3_fires_on_django_14792_real_fixture():
    """Pinned to django-14792's real test_output: a single line
    concatenates the docstring of one test with the name + status of
    the next:
        ``The _get_timezone_name() helper must return the offset for
        fixed offset ... test_is_aware (...) ... ok``
    The rule must split this into two lines, preserving the trailing
    ``ok`` on each.
    """
    log_path = _FIXTURES / "django__django-14792" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    text = log_path.read_text(errors="replace")
    spec = _MinimalTestSpec(
        instance_id="django__django-14792",
        FAIL_TO_PASS=[
            "The _get_timezone_name() helper must return the offset for fixed offset",
            "test_is_aware (utils_tests.test_timezone.TimezoneTests)",
        ],
        PASS_TO_PASS=[],
    )
    result = rule_m3_django_docstring_concat(text, spec)
    assert result is not None
    normalized_log, _normalized_spec, evidence = result
    assert (
        "The _get_timezone_name() helper must return the offset for "
        "fixed offset ... ok" in normalized_log
    )
    assert (
        "test_is_aware (utils_tests.test_timezone.TimezoneTests) ... ok"
        in normalized_log
    )
    assert "split 1 Django docstring-concat" in evidence


def test_m3_does_not_fire_on_normal_django_test_line():
    """A clean ``test_X (mod.Cls) ... ok`` line (no docstring prefix)
    must not trip M3 — the parser handles it natively.
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "test_one (mod.tests.MyTestCase) ... ok\n"
        "test_two (mod.tests.MyTestCase) ... FAIL\n"
        ">>>>> End Test Output\n"
    )
    spec = _MinimalTestSpec("x", [], [])
    assert rule_m3_django_docstring_concat(synthetic, spec) is None


def test_m3_does_not_fire_on_pytest_output():
    """Pytest-shaped lines (with `::`) don't match the Django
    docstring-concat shape; M3 must NOT fire.
    """
    synthetic = (
        "PASSED tests/foo.py::test_x\n"
        "FAILED tests/foo.py::test_y\n"
    )
    spec = _MinimalTestSpec("x", [], [])
    assert rule_m3_django_docstring_concat(synthetic, spec) is None


def test_m3_handles_fail_status():
    """If the trailing status is FAIL or ERROR, the split must
    preserve it for both lines so parse_log_django marks the test
    as failed (not promoted to PASSED).
    """
    synthetic = (
        "Some docstring ... test_x (mod.tests.MyClass) ... FAIL\n"
    )
    spec = _MinimalTestSpec("x", [], [])
    result = rule_m3_django_docstring_concat(synthetic, spec)
    assert result is not None
    normalized, _spec, _evidence = result
    assert "Some docstring ... FAIL" in normalized
    assert "test_x (mod.tests.MyClass) ... FAIL" in normalized


def test_phase0_m5_does_not_exist__django_10097_stays_unresolved(tmp_path):
    """**Phase 0 finding (slice 5):** django-10097's "unresolved" verdict
    is NOT a parser bug. The upstream parse_log_django correctly maps
    all 438 F2P entries (including the 13 docstring-style entries) to
    PASSED. The "unresolved" comes from 5 genuine P2P regressions the
    Defender's patch introduced
    (``test_add (generic_inline_admin.tests.GenericInlineAdminWithUniqueTogetherTest)``
    + 4 others).

    The anchor originally listed this as "M5 — unittest ok lines
    present but unpicked." That categorization was wrong; this test
    locks the correction in. If a future M5-style rule false-promotes
    this iid to resolved, this test fails.
    """
    log_path = _FIXTURES / "django__django-10097" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("django__django-10097")
    prediction = {
        "instance_id": "django__django-10097",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "unresolved", (
        f"django-10097 has P2P regressions and must stay unresolved; "
        f"got {verdict.status!r}"
    )
    assert verdict.reclass_reason == "", (
        f"no rule should fire on django-10097; got {verdict.reclass_reason!r}"
    )
    inst = verdict.upstream_report["django__django-10097"]
    ts = inst["tests_status"]
    assert len(ts["FAIL_TO_PASS"]["success"]) == 438
    assert len(ts["FAIL_TO_PASS"]["failure"]) == 0
    assert len(ts["PASS_TO_PASS"]["failure"]) >= 1, (
        "django-10097's unresolved verdict comes from P2P regressions"
    )


def test_phase0_m5_does_not_exist__django_11532_stays_unresolved(tmp_path):
    """Companion to the django-10097 case (slice 5). django-11532 has
    F2P=1/0 pass/fail but 18 P2P regressions. Real cascade-quality
    miss, not a parser artifact. No rule should fire.
    """
    log_path = _FIXTURES / "django__django-11532" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("django__django-11532")
    prediction = {
        "instance_id": "django__django-11532",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "unresolved"
    assert verdict.reclass_reason == ""
    inst = verdict.upstream_report["django__django-11532"]
    assert len(inst["tests_status"]["PASS_TO_PASS"]["failure"]) >= 1


def test_phase0_correction__sphinx_9229_stays_unresolved(tmp_path):
    """**Anchor correction (slice 6, 2026-05-27):** sphinx-9229 was
    originally categorized as M1 (tox-wrapped pytest artifact). The
    slice-6 post-hoc-script dry-run exposed that the test_output
    actually has ``1 failed, 13 passed`` and ``py39: exit 1`` — the
    F2P test (``test_class_alias_having_doccomment``) genuinely
    failed. M1's negative guards (no failures in summary, exit=0)
    correctly reject this case.

    This test locks the correction in. If a future slice loosens
    M1's guards and false-promotes sphinx-9229, this test fails.
    """
    log_path = _FIXTURES / "sphinx-doc__sphinx-9229" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("sphinx-doc__sphinx-9229")
    prediction = {
        "instance_id": "sphinx-doc__sphinx-9229",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "unresolved", (
        f"sphinx-9229 has a real F2P failure; M1 must not fire. "
        f"Got {verdict.status!r}"
    )
    assert verdict.reclass_reason == ""


def test_m3_composer_resolves_django_14792(tmp_path):
    """End-to-end via composer: real django-14792 fixture + real
    TestSpec + real ``get_eval_report``. Before M3: the concatenated
    line maps to a composite key matching neither dataset F2P entry;
    both fail by absence; ``resolved=False``. After M3 splits the
    line: both keys land in the status_map; ``resolved=True``.
    """
    log_path = _FIXTURES / "django__django-14792" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("django__django-14792")
    prediction = {
        "instance_id": "django__django-14792",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "resolved", (
        f"after M3 splits the concat line, get_eval_report should mark "
        f"both F2P entries passed and return resolved=True. "
        f"Got status={verdict.status!r}; upstream entry: "
        f"{verdict.upstream_report.get('django__django-14792', {})}"
    )
    assert verdict.reclass_reason == "M3_django_docstring_concat"


# ---------- M4: pytest collection-error pre-classifier ----------


def test_m4_fires_on_astropy_8872_real_fixture():
    """Pinned to astropy-8872's real test_output: pytest hit
    ``DeprecationWarning: distutils Version classes`` at collection
    time on ``test_quantity.py`` and emitted
    ``Interrupted: 2 errors during collection``. No tests ran.
    M4 must fire with status ``infra_fail``.
    """
    log_path = (
        _FIXTURES / "astropy__astropy-8872" / "test_output.txt"
    )
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    text = log_path.read_text(errors="replace")
    result = detect_m4_collection_error(text)
    assert result is not None, (
        "M4 must fire on astropy-8872's collection-error test_output"
    )
    status, evidence = result
    assert status == "infra_fail"
    assert "2 errors during collection" in evidence


def test_m4_does_not_fire_on_astropy_7606_successful_run():
    """astropy-7606's test_output has 240+ PASSED lines and no
    collection errors. M4 must NOT fire — otherwise a normal
    successful run would get spuriously promoted to ``infra_fail``.
    """
    log_path = (
        _FIXTURES / "astropy__astropy-7606" / "test_output.txt"
    )
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    text = log_path.read_text(errors="replace")
    result = detect_m4_collection_error(text)
    assert result is None, (
        "M4 must not fire on a log that has successful test execution"
    )


def test_m4_does_not_fire_on_simple_test_failure():
    """A normal test_output with one FAIL (not a collection error)
    must not trip M4. Synthetic shape matching parse_log_django's
    expected input.
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "test_some_thing (mod.tests.Class) ... ok\n"
        "test_other (mod.tests.Class) ... FAIL\n"
        ">>>>> End Test Output\n"
    )
    assert detect_m4_collection_error(synthetic) is None


def test_m4_does_not_fire_when_interrupted_but_tests_ran():
    """Edge: pytest can emit 'errors during collection' for one
    file while still running tests in other files. The guard
    requires NO successful test execution evidence — so this
    multi-file case must NOT trip M4 (the partial run's verdict
    should flow through upstream get_eval_report normally).
    """
    synthetic = (
        ">>>>> Start Test Output\n"
        "Interrupted: 1 errors during collection\n"
        "PASSED tests/test_other.py::test_one\n"
        "PASSED tests/test_other.py::test_two\n"
        ">>>>> End Test Output\n"
    )
    assert detect_m4_collection_error(synthetic) is None


def test_m4_composer_returns_infra_fail_for_astropy_8872(tmp_path):
    """End-to-end via the composer: astropy-8872's real test_output
    must produce ``status="infra_fail"`` without consulting upstream
    get_eval_report (the upstream call would mislead since no tests
    ran). ``upstream_report`` is empty in this path.
    """
    log_path = (
        _FIXTURES / "astropy__astropy-8872" / "test_output.txt"
    )
    if not log_path.exists():
        pytest.skip(f"fixture not present: {log_path}")
    spec = _load_real_spec("astropy__astropy-8872")
    prediction = {
        "instance_id": "astropy__astropy-8872",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "infra_fail"
    assert verdict.reclass_reason == "M4_collection_error"
    assert verdict.upstream_report == {}, (
        "M4 short-circuits the upstream call; upstream_report must be empty"
    )
    assert "2 errors during collection" in verdict.evidence_excerpt


# ---------- composer end-to-end (F2 from slice-1 critique) ----------


def _load_real_spec(iid: str):
    """Build the parser-facing spec entirely from a committed fixture.

    Upstream ``make_test_spec`` downloads requirement files for some
    repositories. The reclassifier needs only these five fields, so a local
    duck-typed spec exercises the same log parser without network access.
    """
    entry_path = _FIXTURES / iid / "dataset_entry.json"
    if not entry_path.exists():
        pytest.skip(f"dataset fixture not present: {entry_path}")
    inst = json.loads(entry_path.read_text())

    def _cases(name: str) -> list[str]:
        value = inst.get(name, [])
        return list(json.loads(value) if isinstance(value, str) else value)

    return _MinimalTestSpec(
        instance_id=inst["instance_id"],
        FAIL_TO_PASS=_cases("FAIL_TO_PASS"),
        PASS_TO_PASS=_cases("PASS_TO_PASS"),
        repo=inst["repo"],
        version=inst["version"],
    )


def test_composer_resolves_astropy_7606_via_m2(tmp_path):
    """End-to-end: real test_output + real TestSpec + real
    ``get_eval_report``. Before reclassification, upstream marks
    astropy-7606 as F2P-fail because the P2P template
    ``test_compose_roundtrip[]`` is misclassified by absence. After
    M2 fires and filters P2P, upstream's verdict flips to resolved.

    Validates the composer's full path: rule firing, normalized-log
    handling (no-op for M2 since spec changes, not log), upstream
    delegation, and status mapping.
    """
    log_path = _FIXTURES / "astropy__astropy-7606" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"test_output fixture not present: {log_path}")

    spec = _load_real_spec("astropy__astropy-7606")

    # Sanity: confirm the fixture spec has the template entry the rule
    # is supposed to fix. If this assertion ever fails the fixture
    # drifted from the upstream dataset and the test is no longer
    # exercising what its name claims.
    assert any(t.endswith("[]") for t in spec.PASS_TO_PASS), (
        "fixture spec must contain at least one parametrize template "
        "in PASS_TO_PASS for M2 to be testable here"
    )

    prediction = {
        "instance_id": "astropy__astropy-7606",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture — patch content irrelevant; the harness "
                       "log already reflects post-patch state)",
    }
    verdict = reclassify(
        log_path, spec, prediction, tmp_log_dir=tmp_path,
    )
    assert isinstance(verdict, ReclassifyVerdict)
    assert verdict.status == "resolved", (
        f"after M2 filters the P2P template, get_eval_report should "
        f"return resolved=True. Got status={verdict.status!r}; "
        f"upstream report instance entry: "
        f"{verdict.upstream_report.get('astropy__astropy-7606', {})}"
    )
    assert verdict.reclass_reason == "M2_parametrize_template"
    assert "1 P2P templates" in verdict.evidence_excerpt
    # The upstream report carries the verdict source — must be present
    # and must contain the iid key (the no-iid-found path is tested
    # separately).
    assert "astropy__astropy-7606" in verdict.upstream_report


def test_composer_returns_verdict_unknown_on_iid_mismatch(
    tmp_path, monkeypatch,
):
    """If the upstream report doesn't carry an entry for the
    prediction's instance_id (schema drift, mid-flight upstream
    change, or third-party get_eval_report shim), the composer must
    return ``verdict_unknown`` rather than silently mapping the
    missing entry to ``unresolved``. The guard is defensive: today's
    upstream always populates the prediction.instance_id key, so this
    failure mode is reachable only via monkeypatch — but it's exactly
    the layer-confusion class of bug the goal doc's risk section
    warns against, so the guard exists by design.
    """
    log_path = _FIXTURES / "astropy__astropy-7606" / "test_output.txt"
    if not log_path.exists():
        pytest.skip(f"test_output fixture not present: {log_path}")
    spec = _load_real_spec("astropy__astropy-7606")
    prediction = {
        "instance_id": "astropy__astropy-7606",
        "model_name_or_path": "fixture",
        "model_patch": "(fixture)",
    }

    # Stub get_eval_report to simulate the API-drift scenario: returns
    # a report shaped like upstream's but keyed by a different iid
    # than the prediction asked for.
    def _stub_get_eval_report(test_spec, prediction, test_log_path, include_tests_status):
        return {"some__other_iid": {"resolved": True, "patch_successfully_applied": True}}

    monkeypatch.setattr(
        "swebench.harness.grading.get_eval_report",
        _stub_get_eval_report,
    )
    verdict = reclassify(log_path, spec, prediction, tmp_log_dir=tmp_path)
    assert verdict.status == "verdict_unknown", (
        f"composer must surface iid mismatch, not silently map to unresolved; "
        f"got status={verdict.status!r}"
    )
    assert verdict.reclass_reason == "iid_not_in_upstream_report"
    # Upstream report is still carried whole for audit, even when we
    # can't extract a verdict from it.
    assert verdict.upstream_report == {
        "some__other_iid": {"resolved": True, "patch_successfully_applied": True}
    }


# ---------- Shared helpers (C2 symmetry) ----------


def test_format_reclassification_reason_encoding():
    """The reason string's leading whitespace-separated token must
    be the rule_name — this is the dedup key. Format must stay
    stable across crucible's inline integration and post-hoc script.
    """
    verdict = ReclassifyVerdict(
        status="resolved",
        reclass_reason="M2_parametrize_template",
        evidence_excerpt="dropped 1 P2P templates",
        upstream_report={},
    )
    reason = format_reclassification_reason(verdict)
    assert reason.split(None, 1)[0] == "M2_parametrize_template"
    assert "upstream_status=resolved" in reason
    assert "evidence=dropped 1 P2P templates" in reason


def test_has_existing_reclassification_dedup_key(tmp_path):
    """has_existing_reclassification matches on the first reason
    token, only fires for kind='reclassified', and rejects other
    lifecycle kinds.
    """
    pytest.importorskip("assay.corpus.store")
    pytest.importorskip("assay.types.artifacts")
    from assay.corpus.store import Corpus
    from assay.types.artifacts import ExecutionContext, LifecycleEvent

    corpus = Corpus(tmp_path / "test-corpus")

    def _exec() -> ExecutionContext:
        return ExecutionContext(
            timestamp="2026-05-27T00:00:00+00:00",
            machine="test", cwd="/", toolchain={"python": "3.12"},
            backend=None, model=None, recipe="test",
            recipe_commit=None, gpu_model=None, run_id="test",
        )

    assert not has_existing_reclassification(corpus, "M2_parametrize_template")

    corpus.put_record(
        LifecycleEvent(
            kind="landed", reason="M2_parametrize_template wrong-kind",
            affected_versions=(), execution=_exec(),
        ),
        kind="lifecycle_events",
    )
    assert not has_existing_reclassification(corpus, "M2_parametrize_template")

    corpus.put_record(
        LifecycleEvent(
            kind="reclassified",
            reason="M2_parametrize_template upstream_status=resolved evidence=...",
            affected_versions=(), execution=_exec(),
        ),
        kind="lifecycle_events",
    )
    assert has_existing_reclassification(corpus, "M2_parametrize_template")
    assert not has_existing_reclassification(corpus, "M1_tox_wrapped_pytest")


# ---------- Inline integration (C3: cover _stamp_inline_reclassification) ----------


def _build_swebench_instance(entry: dict):
    """Construct a real SwebenchInstance from a fixture dataset entry."""
    from crucible.products.swebench.dataset import SwebenchInstance
    return SwebenchInstance(
        instance_id=entry["instance_id"],
        repo=entry["repo"],
        base_commit=entry["base_commit"],
        patch=entry.get("patch", ""),
        test_patch=entry.get("test_patch", ""),
        problem_statement=entry["problem_statement"],
        fail_to_pass=tuple(json.loads(entry["FAIL_TO_PASS"])),
        pass_to_pass=tuple(json.loads(entry["PASS_TO_PASS"])),
        version=entry["version"],
    )


def _corpus_with_spec(path: Path):
    """Create the per-instance lineage required by eval persistence."""
    from assay import ProjectContext, Spec
    from assay.corpus.store import Corpus

    corpus = Corpus(path)
    corpus.put_spec(Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="swebench-test",
            repo_url="https://example.invalid/repo.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Test SWE-bench eval persistence.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    ))
    return corpus


def test_stamp_inline_reclassification_writes_bins_for_artifact(tmp_path):
    """Happy path: an unresolved astropy-7606 EvalReport triggers M2;
    `_stamp_inline_reclassification` writes a reclassified Measurement
    bin + reclassified LifecycleEvent.
    """
    pytest.importorskip("swebench.harness.test_spec.test_spec")
    from crucible.workflows.swebench import _stamp_inline_reclassification

    log_path = _FIXTURES / "astropy__astropy-7606" / "test_output.txt"
    entry_path = _FIXTURES / "astropy__astropy-7606" / "dataset_entry.json"
    if not (log_path.exists() and entry_path.exists()):
        pytest.skip("astropy-7606 fixtures not present")
    inst = _build_swebench_instance(json.loads(entry_path.read_text()))

    import types
    mock_report = types.SimpleNamespace(
        per_instance=[types.SimpleNamespace(
            instance_id=inst.instance_id, resolved=False,
        )],
        model="fixture-model",
    )

    harness_output = tmp_path / "harness_output"
    nested = harness_output / "logs" / "run_evaluation" / "x" / "y" / inst.instance_id
    nested.mkdir(parents=True)
    (nested / "test_output.txt").write_text(log_path.read_text())

    corpus = _corpus_with_spec(tmp_path / "corpus")
    _stamp_inline_reclassification(
        mock_report, inst, corpus=corpus,
        harness_output_dir=harness_output, run_id="inline_test",
    )

    reclass_records = [
        m for m in corpus.iter_measurements()
        if m.bench_id == "swebench/aggregate_reclassified"
    ]
    assert len(reclass_records) == 1
    rec = reclass_records[0]
    assert rec.values["resolved"] == 1.0
    assert rec.config_hash == "inline_test"

    events = list((corpus.spec_dir / "artifacts" / "lifecycle_events").glob("*.bin"))
    reclass_events = [
        e for e in events
        if json.loads(e.read_text()).get("kind") == "reclassified"
    ]
    assert len(reclass_events) == 1
    event_data = json.loads(reclass_events[0].read_text())
    assert event_data["reason"].startswith("M2_parametrize_template")
    latest = corpus.latest_spec()
    assert len(latest.measurement_refs) == 1
    assert len(latest.lifecycle_event_refs) == 1
    assert event_data["affected_versions"] == [latest.version]


def test_stamp_inline_reclassification_skips_when_resolved(tmp_path):
    """harness raw verdict already resolved=True → no reclass write."""
    from crucible.workflows.swebench import _stamp_inline_reclassification
    from crucible.products.swebench.dataset import SwebenchInstance

    inst = SwebenchInstance(
        instance_id="x__x-1", repo="x/x", base_commit="abc",
        patch="", test_patch="", problem_statement="",
        fail_to_pass=(), pass_to_pass=(), version="1",
    )
    import types
    mock_report = types.SimpleNamespace(
        per_instance=[types.SimpleNamespace(
            instance_id="x__x-1", resolved=True,
        )],
        model="x",
    )
    corpus = _corpus_with_spec(tmp_path / "corpus")
    _stamp_inline_reclassification(
        mock_report, inst, corpus=corpus,
        harness_output_dir=tmp_path / "harness_output_doesnt_exist",
        run_id="inline_test",
    )
    assert not list(corpus.iter_measurements())


def test_stamp_inline_reclassification_skips_when_no_test_output(tmp_path):
    """No test_output.txt under harness_output_dir → silent no-op."""
    from assay.corpus.store import Corpus
    from crucible.workflows.swebench import _stamp_inline_reclassification
    from crucible.products.swebench.dataset import SwebenchInstance

    inst = SwebenchInstance(
        instance_id="x__x-1", repo="x/x", base_commit="abc",
        patch="", test_patch="", problem_statement="",
        fail_to_pass=(), pass_to_pass=(), version="1",
    )
    import types
    mock_report = types.SimpleNamespace(
        per_instance=[types.SimpleNamespace(
            instance_id="x__x-1", resolved=False,
        )],
        model="x",
    )
    corpus = Corpus(tmp_path / "corpus")
    empty_harness = tmp_path / "harness_output_empty"
    empty_harness.mkdir()
    _stamp_inline_reclassification(
        mock_report, inst, corpus=corpus,
        harness_output_dir=empty_harness, run_id="inline_test",
    )
    assert not list(corpus.iter_measurements())


def test_stamp_inline_reclassification_dedups_on_re_invocation(tmp_path):
    """Calling twice for the same (iid, rule_name) must write exactly
    one reclassified event. Symmetric with the post-hoc script's
    idempotency. Closes Surface 1 critique C2 at the inline path.
    """
    pytest.importorskip("swebench.harness.test_spec.test_spec")
    from crucible.workflows.swebench import _stamp_inline_reclassification

    log_path = _FIXTURES / "astropy__astropy-7606" / "test_output.txt"
    entry_path = _FIXTURES / "astropy__astropy-7606" / "dataset_entry.json"
    if not (log_path.exists() and entry_path.exists()):
        pytest.skip("astropy-7606 fixtures not present")
    inst = _build_swebench_instance(json.loads(entry_path.read_text()))
    import types
    mock_report = types.SimpleNamespace(
        per_instance=[types.SimpleNamespace(
            instance_id=inst.instance_id, resolved=False,
        )],
        model="fixture",
    )
    harness_output = tmp_path / "harness_output"
    nested = harness_output / "logs" / "x"
    nested.mkdir(parents=True)
    (nested / "test_output.txt").write_text(log_path.read_text())
    corpus = _corpus_with_spec(tmp_path / "corpus")

    _stamp_inline_reclassification(
        mock_report, inst, corpus=corpus,
        harness_output_dir=harness_output, run_id="run_1",
    )
    _stamp_inline_reclassification(
        mock_report, inst, corpus=corpus,
        harness_output_dir=harness_output, run_id="run_2",
    )
    events = list((corpus.spec_dir / "artifacts" / "lifecycle_events").glob("*.bin"))
    reclass_events = [
        e for e in events
        if json.loads(e.read_text()).get("kind") == "reclassified"
    ]
    assert len(reclass_events) == 1, (
        f"second invocation should dedup; got {len(reclass_events)} events"
    )


def test_detect_m4_distutils_subcause_positive_negative():
    """The narrow detector must fire only when BOTH the M4 collection
    pattern AND the distutils Version classes warning are present.
    """
    from crucible.products.swebench.harness_reclassify import (
        detect_m4_distutils_subcause,
    )
    # Real astropy-8872 fixture: both conditions hold.
    fixture = _FIXTURES / "astropy__astropy-8872" / "test_output.txt"
    if fixture.exists():
        assert detect_m4_distutils_subcause(fixture.read_text()) is not None
    # Synthetic: M4 without distutils — must not fire.
    assert detect_m4_distutils_subcause(
        ">>>>> Start Test Output\n"
        "ERROR collecting tests/test_x.py\n"
        "Interrupted: 1 errors during collection\n"
        ">>>>> End Test Output\n"
    ) is None
    # Synthetic: distutils warning without M4 (tests ran) — must not fire.
    assert detect_m4_distutils_subcause(
        ">>>>> Start Test Output\n"
        "PASSED tests/test_x.py::test_one\n"
        "DeprecationWarning: distutils Version classes are deprecated\n"
        "1 passed in 0.1s\n"
        ">>>>> End Test Output\n"
    ) is None


def test_m2_does_not_mutate_input_spec():
    """The rule returns a new spec via ``dataclasses.replace``; the
    original must be untouched (immutability of dataset records is a
    Corpus contract).
    """
    original = _MinimalTestSpec(
        instance_id="example__example-3",
        FAIL_TO_PASS=["foo[]"],
        PASS_TO_PASS=["bar"],
    )
    result = rule_m2_parametrize_template("", original)
    assert result is not None
    assert original.FAIL_TO_PASS == ["foo[]"], "rule mutated input F2P"
    assert original.PASS_TO_PASS == ["bar"], "rule mutated input P2P"
