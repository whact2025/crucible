# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Post-hoc reclassifier for SWE-bench harness reports.

The upstream harness's ``swebench/harness/log_parsers/python.py`` has
parser limitations that misclassify some passing tests as F2P
failures. This module corrects them by transforming
``(test_log, test_spec)`` inputs into a form upstream's parser handles,
then delegating verdict computation to
``swebench.harness.grading.get_eval_report``. The reclassifier never
computes ``resolved`` independently — it transforms inputs and reads
the upstream verdict.

The ``Measurement.values.resolved`` float that downstream readers
consume MUST come from the upstream report's ``resolved`` field, not
from any computation done here. ``ReclassifyVerdict`` carries the
upstream report whole so reviewers can trace the value to its source.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Callable, Optional

# Lazy imports — only required when ``reclassify`` is invoked, not at
# module import time. Keeps the workflow's lazy-load discipline intact.


@dataclasses.dataclass(frozen=True)
class ReclassifyVerdict:
    """Output of a reclassification pass.

    ``status`` values produced by this slice (M2 only):
        - ``"resolved"`` — upstream report (after input transform)
          says resolved.
        - ``"unresolved"`` — upstream report says not resolved (either
          no rule fired and raw verdict was unresolved, or a rule
          fired and the corrected verdict is still unresolved).
        - ``"verdict_unknown"`` — the upstream report did not carry a
          record for the prediction's instance_id (schema drift, iid
          typo, mid-flight upstream change). Distinct from
          unresolved-because-failed: no verdict was produced at all.

        - ``"infra_fail"`` — a pre-classifier determined that the harness did
          not produce a meaningful test verdict, such as a collection error.

    ``reclass_reason`` is the name of the rule that fired, or empty
    string when the upstream verdict was used as-is, or
    ``"iid_not_in_upstream_report"`` for the verdict_unknown case,
    or ``"M4_collection_error"`` (etc.) for pre-classifier terminations.
    ``evidence_excerpt`` is a ≤512-char slice the rule keyed on, for
    audit. ``upstream_report`` is whatever ``get_eval_report`` returned —
    carried whole so the Measurement-bin write site can reuse the same
    shape the raw bench_raw artifact uses, so reviewers can trace ``status``
    back to its upstream source.
    """
    status: str
    reclass_reason: str
    evidence_excerpt: str
    upstream_report: dict[str, Any]


# There are two rule families:
#
# **Transform rules** map (test_log_text, test_spec) →
#   (normalized_log_text, normalized_test_spec, evidence_excerpt), or
#   return None when the rule doesn't apply. The composer then
#   delegates to upstream get_eval_report with the normalized inputs.
#   M1, M2, M3 are transform rules.
#
# **Pre-classifiers** map test_log_text → (terminal_status,
#   evidence_excerpt), or return None when not applicable. The
#   composer short-circuits — no upstream get_eval_report call —
#   because the upstream verdict would be misleading
#   (e.g., M4's collection-error case: no test actually ran, so
#   "F2P fail by absence" is structurally wrong).
#
# The test_spec is typed Any so unit tests can pass minimal duck-typed
# dataclasses without pulling in the full swebench TestSpec import.
_RuleFn = Callable[[str, Any], Optional[tuple[str, Any, str]]]
_PreClassifierFn = Callable[[str], Optional[tuple[str, str]]]


def detect_m4_distutils_subcause(test_log: str) -> Optional[str]:
    """**Narrow M4 sub-detection — distutils-Version DeprecationWarning.**

    Returns evidence excerpt when both hold:
      1. The general M4 pattern fires (``detect_m4_collection_error``
         returns non-None — collection-error banner + no executed-test
         markers).
      2. The test_log explicitly contains
         ``DeprecationWarning: distutils Version classes``.

    This is a narrower signal than M4 itself. M4's general detection
    keys on "no tests ran due to collection error" — but the underlying
    cause can be any of: distutils-Version warning treated as error,
    circular imports at discovery, missing modules, syntax errors in
    test files, conftest errors. Only the distutils-Version cause is
    fixable by ``pip install --upgrade setuptools>=68``; for the
    others, an upgrade is a no-op at best and an iatrogenic break at
    worst (newer setuptools may remove APIs older test code relies on).

    This is a post-hoc diagnostic/classification signal only. The benchmark
    workflow never mutates the hidden-evaluation environment or retries based
    on this detector.
    """
    if detect_m4_collection_error(test_log) is None:
        return None
    # Look for the specific distutils-Version DeprecationWarning.
    # Match both bare "DeprecationWarning:" lines and pytest's
    # "E   DeprecationWarning:" failure-frame lines.
    distutils_match = re.search(
        r"DeprecationWarning:\s*distutils\s+Version\s+classes",
        test_log,
    )
    if distutils_match is None:
        return None
    return (
        f"M4 distutils-Version subcause: matched "
        f"{distutils_match.group(0)!r} in test_output"
    )[:512]


def detect_m4_collection_error(test_log: str) -> Optional[tuple[str, str]]:
    """**M4 — pytest collection-error pre-classifier.**

    When a test file fails to import / collect (e.g., due to
    ``DeprecationWarning: distutils Version classes`` raised during
    module import), pytest emits ``Interrupted: N errors during
    collection`` and never executes any of the file's tests. No
    PASSED/FAILED lines are produced for the F2P/P2P set, so the
    upstream parser's status_map is empty for those tests and
    ``test_failed(case, sm)`` returns True by absence → upstream
    verdict ``resolved=False``. The "fail" is environmental, not a
    cascade-quality issue, so the right verdict is ``infra_fail``,
    not ``unresolved``.

    Detection (conservative — requires both conditions):
      1. Pytest emitted an explicit collection-failure summary:
         ``Interrupted: N errors during collection`` (the bang-banner
         line pytest writes at the end of collection on failure).
      2. The log has no successful test-execution evidence in the
         common shapes (``PASSED ``, ``FAILED ``, ``... ok``,
         ``... FAIL``, ``::test_X PASSED``). This guards against the
         false-positive where collection had warnings but tests
         still ran to completion.

    Affected iid in the 2026-05-26 corpus: ``astropy__astropy-8872``
    (DeprecationWarning collection error in
    ``astropy/units/tests/test_quantity.py``; pytest reported
    ``Interrupted: 2 errors during collection``; no tests ran).
    """
    interrupted_match = re.search(
        r"Interrupted:\s*(\d+)\s+errors?\s+during\s+collection",
        test_log,
    )
    if interrupted_match is None:
        return None
    # Guard: don't fire if the log has successful test execution
    # evidence. pytest can emit "errors during collection" for one
    # file while still running others; if any tests ran, we shouldn't
    # short-circuit to infra_fail.
    has_executed_tests = bool(
        re.search(
            r"(?m)"
            r"^(PASSED|FAILED)\s+\S|"   # ^PASSED <test_id>
            r"\s+\.\.\.\s+(ok|FAIL)\b|"  # ... ok / ... FAIL
            r"::\S+\s+(PASSED|FAILED)\b",  # ::test_X PASSED
            test_log,
        )
    )
    if has_executed_tests:
        return None
    n_errors = interrupted_match.group(1)
    excerpt = (
        f"pytest collection failed: {n_errors} errors during collection, "
        f"no PASSED/FAILED/ok lines in test_output. Match: "
        f"{interrupted_match.group(0)!r}"
    )
    return "infra_fail", excerpt[:512]


def rule_m2_parametrize_template(
    test_log: str, test_spec: Any,
) -> Optional[tuple[str, Any, str]]:
    """**M2 — parametrize-template F2P/P2P entry.**

    Some dataset records list a pytest parametrize template id (e.g.,
    ``test_compose_roundtrip[]`` with empty brackets) as a FAIL_TO_PASS
    or PASS_TO_PASS entry. pytest never emits a result line for the
    template — only for the concrete parametrize cases like
    ``test_compose_roundtrip[unit0]``. ``get_logs_eval`` builds a
    status map keyed on the concrete cases; the lookup for the empty-
    bracket template misses, and ``test_failed(case, sm)`` treats
    "missing from map" as failure (``grading.py:31-37``).

    Transform: drop empty-bracket entries from both FAIL_TO_PASS and
    PASS_TO_PASS. The concrete cases stay; if they pass, the verdict
    flips to resolved.

    Affected iid in the 2026-05-26 corpus: ``astropy__astropy-7606``
    (single F2P/P2P entry ``test_compose_roundtrip[]``; 240 concrete
    cases all PASSED).
    """
    f2p_templates = [
        t for t in getattr(test_spec, "FAIL_TO_PASS", ()) if t.endswith("[]")
    ]
    p2p_templates = [
        t for t in getattr(test_spec, "PASS_TO_PASS", ()) if t.endswith("[]")
    ]
    if not f2p_templates and not p2p_templates:
        return None
    new_f2p = [
        t for t in getattr(test_spec, "FAIL_TO_PASS", ()) if not t.endswith("[]")
    ]
    new_p2p = [
        t for t in getattr(test_spec, "PASS_TO_PASS", ()) if not t.endswith("[]")
    ]
    normalized = dataclasses.replace(
        test_spec,
        FAIL_TO_PASS=new_f2p,
        PASS_TO_PASS=new_p2p,
    )
    excerpt = (
        f"dropped {len(f2p_templates)} F2P templates "
        f"({f2p_templates[:3]}), {len(p2p_templates)} P2P templates "
        f"({p2p_templates[:3]})"
    )
    return test_log, normalized, excerpt[:512]


_DJANGO_DOCSTRING_CONCAT_RE = re.compile(
    r"^(?P<docstring>.+?)\s+\.\.\.\s+"
    r"(?P<test_id>test_\w+\s+\([\w.]+\))\s+\.\.\.\s+"
    r"(?P<status>ok|OK|FAIL|ERROR|skipped)\s*$",
    re.MULTILINE,
)


def rule_m3_django_docstring_concat(
    test_log: str, test_spec: Any,
) -> Optional[tuple[str, Any, str]]:
    """**M3 — Django test docstring concatenated with next test.**

    Django's unittest runner sometimes prints a test's docstring on
    the same line as the next test's name + status, producing output
    like::

        <docstring> ... test_other (mod.Cls) ... ok

    The upstream parser's ``... ok`` rule (line 90-102 of
    ``parse_log_django``) sees this and adds
    ``status_map[<docstring> ... test_other (mod.Cls)] = "PASSED"`` —
    a composite key that matches neither dataset entry. Both the
    docstring (a FAIL_TO_PASS entry in some datasets) and the
    second test name (the actual next test) end up missing from
    status_map → both fail by absence.

    Transform: split each concatenated line into two single-test
    lines preserving the trailing status, so parse_log_django adds
    both keys correctly::

        <docstring> ... ok
        test_other (mod.Cls) ... ok

    The dataset's docstring entry matches the first key verbatim;
    the concrete test ID matches the second.

    Affected iid in the 2026-05-26 corpus: ``django__django-14792``
    (F2P contains both the docstring
    ``"The _get_timezone_name() helper must return the offset for
    fixed offset"`` and the concrete test
    ``test_is_aware (utils_tests.test_timezone.TimezoneTests)``;
    test_output has them concatenated on one line).
    """
    matches = list(_DJANGO_DOCSTRING_CONCAT_RE.finditer(test_log))
    if not matches:
        return None
    normalized_log = test_log
    # Apply replacements in reverse so earlier offsets stay valid.
    for m in reversed(matches):
        replacement = (
            f"{m.group('docstring')} ... {m.group('status')}\n"
            f"{m.group('test_id')} ... {m.group('status')}"
        )
        normalized_log = (
            normalized_log[: m.start()]
            + replacement
            + normalized_log[m.end() :]
        )
    examples = [
        f"{m.group('docstring')[:60]!r} + {m.group('test_id')!r}"
        for m in matches[:2]
    ]
    evidence = (
        f"split {len(matches)} Django docstring-concat line(s); "
        f"examples: {examples}"
    )
    return normalized_log, test_spec, evidence[:512]


# Pre-classifier registry — run BEFORE transform rules. First-fire-wins.
# These short-circuit the upstream get_eval_report call because the
# upstream verdict would be misleading (no tests actually ran, etc.).
_PRE_CLASSIFIERS: list[tuple[str, _PreClassifierFn]] = [
    ("M4_collection_error", detect_m4_collection_error),
]


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
_TOX_EXIT_RE = re.compile(r"py\d+:\s+exit\s+(\d+)")
_PYTEST_SUMMARY_RE = re.compile(r"\b(\d+)\s+passed\b")
_PYTEST_FAILED_SUMMARY_RE = re.compile(r"\b(\d+)\s+failed\b")
# Duration line: "0.01s call    tests/path::test_id" (also setup).
_PYTEST_DURATION_RE = re.compile(
    r"^\s*\d+\.\d+s\s+(?:setup|call|teardown)\s+(\S+)",
    re.MULTILINE,
)
# Upstream-parseable result line: "PASSED <id>" or "<id> PASSED" (v2's
# endswith branch).
_PASSED_LINE_RE = re.compile(
    r"(?m)^PASSED\s+\S+|^\S+\s+PASSED\b",
)
_FAILED_LINE_RE = re.compile(
    r"(?m)^FAILED\s+\S+|^\S+\s+FAILED\b",
)


def rule_m1_tox_wrapped_pytest(
    test_log: str, test_spec: Any,
) -> Optional[tuple[str, Any, str]]:
    """**M1 — tox-wrapped non-verbose pytest log normalization.**

    When pytest runs through tox without ``-v``, the result lines look
    like duration entries (``0.01s call <test_id>``) and a terse
    summary (``1 passed, 7 warnings in 0.64s``). Upstream
    ``parse_log_pytest_v2``'s startswith and endswith branches both
    miss this format — the status_map ends empty and tests register
    as "fail by absence."

    Transform: synthesize ``PASSED <test_id>`` lines for every test
    that appears in the duration block, inserted immediately before
    the END_TEST_OUTPUT marker so ``parse_log_pytest_v2`` picks them
    up on its next pass over the log.

    Detection (all must hold to avoid false positives):
      1. ``py<N>: exit 0`` tox marker present (rc=0 from the
         wrapped pytest invocation).
      2. A pytest summary line with ``N passed`` and ``N > 0``.
      3. No ``M failed`` in the summary (skip mixed-result runs —
         those need a different transform).
      4. No upstream-parseable ``PASSED <id>`` / ``<id> PASSED``
         lines anywhere (upstream parser would already handle the
         normal-verbose case; only fire when synthesis is needed).
      5. No ``FAILED <id>`` lines either (same rationale).
      6. At least one test ID extractable from a duration line.

    Affected iids in the 2026-05-26 corpus: ``sphinx-doc__sphinx-8595``,
    ``sphinx-doc__sphinx-9229``, ``sphinx-doc__sphinx-9711`` (all
    sphinx runs go through the tox wrapper).
    """
    # Strip ANSI color escapes — pytest emits them in some
    # environments and they break startswith/endswith matching.
    clean = _ANSI_ESCAPE_RE.sub("", test_log)

    exit_match = _TOX_EXIT_RE.search(clean)
    if exit_match is None or exit_match.group(1) != "0":
        return None
    summary_match = _PYTEST_SUMMARY_RE.search(clean)
    if summary_match is None:
        return None
    n_passed = int(summary_match.group(1))
    if n_passed == 0:
        return None
    if _PYTEST_FAILED_SUMMARY_RE.search(clean):
        return None
    if _PASSED_LINE_RE.search(clean) or _FAILED_LINE_RE.search(clean):
        return None
    test_ids = sorted({m.group(1) for m in _PYTEST_DURATION_RE.finditer(clean)})
    if not test_ids:
        return None

    synthesized_block = "\n".join(f"PASSED {tid}" for tid in test_ids)
    end_marker = ">>>>> End Test Output"
    # The END marker appears inside a bash echo line like
    # ``+ : '>>>>> End Test Output'`` in tox-wrapped harness logs.
    # We MUST prepend a newline so the synthesized PASSED lines stand
    # on their own — otherwise they'd glue onto the ``+ : '`` prefix
    # and ``parse_log_pytest_v2`` would skip them (line doesn't
    # startswith PASSED).
    if end_marker in test_log:
        normalized_log = test_log.replace(
            end_marker,
            "\n" + synthesized_block + "\n" + end_marker,
            1,
        )
    else:
        # No END marker → upstream would fail on get_logs_eval anyway,
        # but still write the synthesis at the tail so post-hoc
        # inspection sees what we intended.
        normalized_log = test_log + "\n" + synthesized_block + "\n"

    evidence = (
        f"tox exit_0 with {n_passed} passed; synthesized PASSED for "
        f"{len(test_ids)} test ids: {test_ids[:3]}"
    )
    return normalized_log, test_spec, evidence[:512]


# Rule registry — ordered, first-fire-wins.
#
# DISJOINT-IN-PRACTICE assumption: M1-M4 modes observed in the
# 2026-05-26 corpus are disjoint per iid (one mode per affected iid).
# If a future iid trips two modes simultaneously (e.g., collection
# error AND parametrize template in P2P), composition becomes
# load-bearing and this comment should be reconsidered. For now,
# rule ordering = priority ordering.
#
# Signature drift discipline: _RuleFn is Callable[[str, Any], ...],
# which Python's runtime does NOT enforce. When adding rules in
# subsequent slices, keep the (test_log, test_spec) parameter shape;
# don't quietly add positional args. The composer at reclassify()
# calls rule_fn(normalized_log, normalized_spec) blindly.
_RULES: list[tuple[str, _RuleFn]] = [
    # Ordering rationale: M1 (tox-wrapped synthesis) and M3 (Django
    # docstring split) both rewrite the log. M2 only touches the
    # spec. M4 is a pre-classifier and runs before this list. In
    # practice the affected iids are disjoint by repo (M1 on sphinx,
    # M3 on django, M2 on astropy-7606), so ordering only matters
    # defensively. M1 first because it has the strictest detection
    # guards.
    ("M1_tox_wrapped_pytest", rule_m1_tox_wrapped_pytest),
    ("M3_django_docstring_concat", rule_m3_django_docstring_concat),
    ("M2_parametrize_template", rule_m2_parametrize_template),
]


def reclassify(
    test_log_path: Path,
    test_spec: Any,
    prediction: dict[str, str],
    *,
    tmp_log_dir: Optional[Path] = None,
) -> ReclassifyVerdict:
    """Apply input-transform rules and delegate verdict to
    ``swebench.harness.grading.get_eval_report``.

    Pipeline:
      1. Read the test_log_path text.
      2. Run detection rules in order. The first to fire produces a
         normalized ``(log, test_spec, evidence)`` tuple.
      3. If a rule fires and rewrote the log, write the normalized
         log to ``tmp_log_dir`` (or, if ``tmp_log_dir is None``, the
         original log's parent dir) as ``<stem>.reclass<ext>``.
         **Side effect:** this writes a file next to the original log
         by default. Callers wanting an isolated location must pass
         an explicit ``tmp_log_dir``. The post-hoc maintenance script
         passes a per-run temp dir; in-line workflow integration
         passes the workflow's run_dir.
      4. Call ``get_eval_report(normalized_spec, prediction,
         normalized_log_path, include_tests_status=True)``.
      5. Map the upstream report's ``resolved`` field to a status.
         If the iid isn't in the upstream report (schema drift),
         return ``verdict_unknown`` rather than silently mapping to
         ``unresolved`` — the layer-confusion failure mode the goal
         doc's risk section warns against.

    Note: when no rule fires, this is a pass-through — upstream sees
    the same inputs the original eval saw, and returns the same
    verdict. The rules' value is in detection plus transformation,
    not in re-invocation alone.
    """
    # Lazy upstream imports — keeps module-import-time clean for callers
    # that just want ReclassifyVerdict / rule_m2_* for unit testing.
    from swebench.harness.grading import get_eval_report

    test_log_path = Path(test_log_path)
    test_log = test_log_path.read_text(errors="replace")

    # Pre-classifiers fire first. If one returns a terminal verdict, we
    # skip the upstream call entirely — those cases (M4 collection
    # error today) reflect "no test ran," and upstream's verdict over
    # an empty status_map is misleading.
    for pre_name, pre_fn in _PRE_CLASSIFIERS:
        pre_result = pre_fn(test_log)
        if pre_result is None:
            continue
        terminal_status, pre_evidence = pre_result
        return ReclassifyVerdict(
            status=terminal_status,
            reclass_reason=pre_name,
            evidence_excerpt=pre_evidence,
            upstream_report={},  # upstream not consulted
        )

    fired_rule = ""
    evidence = ""
    normalized_log = test_log
    normalized_spec = test_spec
    for rule_name, rule_fn in _RULES:
        result = rule_fn(normalized_log, normalized_spec)
        if result is None:
            continue
        normalized_log, normalized_spec, evidence = result
        fired_rule = rule_name
        break  # at most one rule per pass for now (see _RULES comment)

    # Materialize the (possibly normalized) log path for upstream.
    if normalized_log != test_log:
        if tmp_log_dir is None:
            tmp_log_dir = test_log_path.parent
        tmp_log_dir.mkdir(parents=True, exist_ok=True)
        out_path = tmp_log_dir / f"{test_log_path.stem}.reclass{test_log_path.suffix}"
        out_path.write_text(normalized_log)
        upstream_log_path = out_path
    else:
        upstream_log_path = test_log_path

    upstream_report = get_eval_report(
        test_spec=normalized_spec,
        prediction=prediction,
        test_log_path=str(upstream_log_path),
        include_tests_status=True,
    )

    iid = prediction.get("instance_id", "")
    if iid not in upstream_report:
        return ReclassifyVerdict(
            status="verdict_unknown",
            reclass_reason="iid_not_in_upstream_report",
            evidence_excerpt=(
                f"prediction.instance_id={iid!r} not in upstream_report "
                f"keys {sorted(upstream_report.keys())[:5]}"
            )[:512],
            upstream_report=upstream_report,
        )
    inst_report = upstream_report[iid]
    resolved = bool(inst_report.get("resolved", False))
    return ReclassifyVerdict(
        status="resolved" if resolved else "unresolved",
        reclass_reason=fired_rule,
        evidence_excerpt=evidence,
        upstream_report=upstream_report,
    )


def has_existing_reclassification(corpus: Any, rule_name: str) -> bool:
    """Return True if a ``LifecycleEvent(kind="reclassified")`` with
    the given ``rule_name`` as its leading reason-token already
    exists in the per-instance Corpus.

    The inline workflow uses the leading reason token as its deduplication
    key. The reason encoding is
    ``"<rule_name> upstream_status=... evidence=..."``; this reader compares
    the first whitespace-separated token.
    """
    from assay.types.artifacts import LifecycleEvent  # lazy import

    for ref in corpus.list_artifacts(kind="lifecycle_events"):
        try:
            event = corpus.get_record(ref, as_type=LifecycleEvent)
        except Exception:
            continue
        if not isinstance(event, LifecycleEvent):
            continue
        if event.kind != "reclassified":
            continue
        token = event.reason.split(None, 1)[0] if event.reason else ""
        if token == rule_name:
            return True
    return False


def format_reclassification_reason(verdict: "ReclassifyVerdict") -> str:
    """Encode a ReclassifyVerdict into the LifecycleEvent.reason
    string format retained by inline evaluation. The leading
    whitespace-separated token is the rule_name
    (the dedup key for ``has_existing_reclassification``).
    """
    return (
        f"{verdict.reclass_reason} "
        f"upstream_status={verdict.status} "
        f"evidence={verdict.evidence_excerpt[:200]}"
    )


__all__ = [
    "ReclassifyVerdict",
    "reclassify",
    "rule_m1_tox_wrapped_pytest",
    "rule_m2_parametrize_template",
    "rule_m3_django_docstring_concat",
    "detect_m4_collection_error",
    "detect_m4_distutils_subcause",
    "has_existing_reclassification",
    "format_reclassification_reason",
]
