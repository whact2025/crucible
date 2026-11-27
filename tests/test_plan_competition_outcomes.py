# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""A competition that ends without a plan tells a ruling from a lost call.

``plan_competition_inconclusive`` is kept when a ruling ended the run: the
comparison answered, or every candidate that existed was refused by the size
limit, the decoder, a deterministic gate or its reviewers. It is also kept
when the records do not prove that a lost call ended the run. A run that lost
a call while a candidate had no ruling against it, or in which no planner
returned a draft, reports the status single-track planning gives that loss.
Each test shows both sides of one such line: the kept status first, the named
loss last.
"""

from __future__ import annotations

import itertools
import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ContentRejected,
    Evidence,
    ExternalCancellation,
    Finding,
    IncompleteEvidence,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.primitives.plan import _serialize_tier_dag

from crucible.backends.openai_compat import OpenAIBackend
from crucible.common.cost import CostAccumulator
from crucible.rubrics import load
from crucible.workflows import _outcome, _plan_selection, architect, plan
from crucible.workflows import _plan_competition as competition
from crucible.workflows._plan_validation import build_pre_review_candidate_validator
from tests import test_plan_competition as harness
from tests.test_backends_openai_compat import _StubToolSet, _stub_response, _tool_call


KEPT = "plan_competition_inconclusive"
NO_CANDIDATE = "no independently reviewed candidate was selected"
ABSTAINED = "comparison selected no acceptable plan"
SELECTOR_LOST = "selector review failed or its accounting is incomplete"
BUNDLE_REFUSED = "candidate evidence is invalid: "

DRAFTS = ("plan_1", "plan_2", "plan_3")
CRITICS = tuple(f"{name}_critic" for name in DRAFTS)
JUDGES = tuple(f"{name}_judge" for name in DRAFTS)
# The fixture finishes its three drafts in reverse order.
DRAFT_CALLS = ["plan_3", "plan_2", "plan_1"]
REVIEW_CALLS = [label for name in DRAFTS for label in (f"{name}_critic", f"{name}_judge")]

TRANSPORT = TransportError(message="connection reset")
TRANSPORT_FINAL = TransportError(message="refused for good", retriable=False)
TURNS = MaxTurnsExceeded(turns_used=6, turns_limit=5, message="turn limit")
CONTENT = ContentRejected(message="refused")
BUDGET = BudgetExhausted(spent_usd=0.01, limit_usd=0.01, message="ceiling")
EVIDENCE = IncompleteEvidence(missing=("promotion_evidence",), message="evidence")

# One byte above the size limit of a draft reply.
OVERSIZE = "x" * (1024 * 1024 + 1)

ABSTAIN = {
    "mode": "none",
    "selected_id": None,
    "selected_digest": None,
    "synthesized_dag": None,
}


# --- scripted faults: each takes the fixture's result and returns another ----


def _lost(error=TRANSPORT):
    """A typed backend error with complete evidence and no reply."""
    return lambda result: replace(
        result, final_message="", transcript=(), stop_reason="error", error=error
    )


def _says(text):
    return lambda result: replace(result, final_message=text)


def _citing_a_missing_file(plan_json):
    """The same plan with a citation the deterministic gate refuses."""
    payload = json.loads(plan_json)
    payload["tiers"][0]["slices"][0]["directive"]["read_only_references"] = [
        {
            "reference_id": "phantom",
            "path": "missing_source.py",
            "sha256": "sha256:" + "0" * 64,
            "start_line": 1,
            "end_line": 1,
        }
    ]
    return payload


def _cites_a_missing_file(result):
    """A parseable draft that is refused before any review."""
    return replace(
        result, final_message=json.dumps(_citing_a_missing_file(result.final_message))
    )


def _raises(_result):
    raise ConnectionError("socket closed")


def _each(labels, fault):
    return dict.fromkeys(labels, fault)


def _script(patch, faults, rulings, comparison, rule_set_refuses=None):
    """Script the fixture for one run.

    ``faults`` maps a backend label to a fault. ``rulings`` maps a reviewer
    label to the decision its parsed verdict is given, with a model-sourced
    finding, so the verdict is a ruling and not a control-plane failure.
    ``comparison`` is what the comparison reviewer answers. ``rule_set_refuses``
    names the draft the competition's own rule-set recheck refuses.
    """
    reply = harness.Backend.invoke

    def invoke(backend, invocation):
        result = reply(backend, invocation)
        fault = faults.get(backend.label)
        return result if fault is None else fault(result)

    patch.setattr(harness.Backend, "invoke", invoke)

    review = plan.strict_review

    def ruled(artifact, *args, **kwargs):
        verdict = review(artifact, *args, **kwargs)
        label = getattr(kwargs.get("backend"), "label", None)
        decision = rulings.get(label)
        if decision is None:
            return verdict
        return replace(
            verdict,
            decision=decision,
            reasoning=f"{label} ruled {decision}",
            findings=(
                Finding(
                    id=f"{label}_{decision}",
                    severity="High",
                    category="other",
                    evidence=(
                        Evidence(
                            kind="source_quote",
                            location="json:$.tiers[0]",
                            excerpt="implement the thing",
                        ),
                    ),
                    suggested_action="fixture ruling",
                    source="llm",
                    criterion_id="intent_coverage",
                ),
            ),
        )

    patch.setattr(plan, "strict_review", ruled)

    if comparison is not None:
        selector = harness.ComparisonBackend
        patch.setattr(
            harness, "ComparisonBackend", lambda _changes: selector(comparison)
        )

    if rule_set_refuses is not None:
        validate = competition.validate
        refused = harness._dag(rule_set_refuses)

        def recheck(spec, **kwargs):
            verdict = validate(spec, **kwargs)
            if (
                kwargs["ruleset"].name != "competition_candidate"
                or spec.tier_dag != refused
            ):
                return verdict
            return replace(
                verdict,
                decision="reject",
                findings=(
                    Finding(
                        id="fixture_rule_set_refusal",
                        severity="High",
                        category="other",
                        evidence=(
                            Evidence(
                                kind="source_reference_validation",
                                location="tiers[0]",
                                excerpt="the complete rule set refuses this draft",
                            ),
                        ),
                        suggested_action="fixture refusal",
                        source="deterministic",
                    ),
                ),
            )

        patch.setattr(competition, "validate", recheck)


def _terminal_class(status):
    return _outcome.classify_terminal_status(status).value


def _compete(
    tmp_path,
    monkeypatch,
    *,
    faults=None,
    rulings=None,
    comparison=None,
    rule_set_refuses=None,
    pool=None,
    **pool_kwargs,
):
    """Run one three-planner competition on a fresh fixture project."""
    _workspace, root, corpus, spec = harness._repo(tmp_path)
    pool = pool or harness.Pool(**pool_kwargs)
    run_dir = corpus.spec_dir / "runs/competition"
    with monkeypatch.context() as patch:
        _script(patch, faults or {}, rulings or {}, comparison, rule_set_refuses)
        outcome = competition.run_plan_competition(
            spec,
            pool=pool,
            corpus=corpus,
            cwd=str(root),
            run_dir=run_dir,
            planner_role="plan_planner",
            critic_role="plan_critic",
            judge_role="plan_judge",
            critic_rubric=load("plan_critic"),
            judge_rubric=load("plan_judge"),
            budget_usd=10,
            cancel_event=pool.cancel_event,
        )
    report = json.loads((run_dir / "plan_competition.json").read_text())
    return SimpleNamespace(
        outcome=outcome,
        status=outcome.status,
        reason=outcome.reason,
        terminal_class=_terminal_class(outcome.status),
        report=report,
        rows={row["id"]: row["status"] for row in report["candidates"]},
        calls=[label for label, _ in pool.order],
        corpus=corpus,
        spec=spec,
    )


def _nothing_landed(run):
    """No plan is staged or published, and the report tells the same end."""
    assert run.outcome.accepted is False and run.outcome.decision == "inconclusive"
    assert run.outcome.spec.tier_dag is None
    assert run.corpus.latest_spec() == run.spec
    assert (run.report["status"], run.report["reason"]) == (run.status, run.reason)


def _is_kept(run, reason, calls):
    """The run keeps the competition's own status, with this reason and class."""
    assert (run.status, run.terminal_class) == (KEPT, "contract_error"), run.reason
    assert run.reason == reason or (
        reason.endswith(": ") and run.reason.startswith(reason)
    ), run.reason
    assert run.calls == calls
    _nothing_landed(run)


def _is_named(run, status, terminal_class, reason, calls):
    """The run reports the lost call, and made no call beyond the scenario's."""
    assert (run.status, run.terminal_class) == (status, terminal_class), run.reason
    assert run.reason.startswith(reason), run.reason
    assert run.calls == calls
    _nothing_landed(run)


# --- the same project planned by one planner ----------------------------------


class _Planner:
    """A single-track Planner backend; ``script`` gives each call its fault."""

    name = "competition-fixture"
    model = "competition-model"
    capability = "text"

    def __init__(self, script):
        self.script, self.invocations = script, []

    def invoke(self, invocation):
        self.invocations.append(invocation)
        text = _serialize_tier_dag(harness._dag("plan_1")).decode()
        result = InvocationResult(
            final_message=text,
            transcript=(Message(role="assistant", content=text),),
            usage=Usage(
                input_tokens=11, output_tokens=1, cost_usd=0.01, cache_read_tokens=1
            ),
            stop_reason="end_turn",
            error=None,
            completeness=harness.COMPLETE,
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(sequence=0, kind="request", name="invoke"),
                    BackendExecutionEvent(sequence=1, kind="terminal", name="complete"),
                ),
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=(
                    UsagePricingSegment(
                        input_tokens=11,
                        output_tokens=1,
                        cache_creation_tokens=0,
                        cache_read_tokens=1,
                        cost_usd=0.01,
                    ),
                ),
                complete=True,
            ),
        )
        fault = self.script(len(self.invocations))
        return result if fault is None else fault(result)


def _plan_once(tmp_path, monkeypatch, *, planner=lambda call: None, faults=None, rulings=None):
    """Run single-track planning; its reviewers are ``single_critic``/``single_judge``."""
    _workspace, root, corpus, spec = harness._repo(tmp_path)
    pool = harness.Pool()
    backend = _Planner(planner)
    with monkeypatch.context() as patch:
        _script(patch, faults or {}, rulings or {}, None)
        outcome = plan.run_plan(
            spec,
            planner_backend=backend,
            critic_backend=harness.Backend(
                pool, "plan_critic", "single_critic", Path("unused"), None
            ),
            judge_backend=harness.Backend(
                pool, "plan_judge", "single_judge", Path("unused"), None
            ),
            corpus=corpus,
            critic_rubric=load("plan_critic"),
            judge_rubric=load("plan_judge"),
            cwd=str(root),
            budget_usd=10,
            allowed_tools=(),
            candidate_validator=build_pre_review_candidate_validator(
                spec=spec, project_root=root
            ),
            max_candidate_validation_repairs=1,
        )
    return SimpleNamespace(
        outcome=outcome,
        status=outcome.status,
        terminal_class=_terminal_class(outcome.status),
        planner_calls=len(backend.invocations),
    )


# --- no planner returned a draft ----------------------------------------------


@pytest.mark.parametrize(
    ("errors", "status", "terminal_class"),
    [
        # A budget stop names the run wherever it sits among the drafts.
        ((TRANSPORT, TURNS, BUDGET), "budget_exhausted", "budget_exhausted"),
        ((BUDGET, TURNS, TRANSPORT), "budget_exhausted", "budget_exhausted"),
        # Another terminal that is not a ruling outranks a lost call.
        ((EVIDENCE, TRANSPORT, TRANSPORT), "planner_protocol_failure", "inconclusive"),
        ((TURNS, TRANSPORT, EVIDENCE), "planner_protocol_failure", "inconclusive"),
        ((TURNS, CONTENT, TRANSPORT_FINAL), "planner_protocol_failure", "inconclusive"),
        # A turn limit beside lost calls does not make the run judged.
        ((TURNS, TRANSPORT, TRANSPORT), "planner_infrastructure_failure", "inconclusive"),
    ],
    ids=[
        "budget_last",
        "budget_first",
        "terminal_before_lost_calls",
        "terminal_after_lost_call",
        "terminal_beside_permanent_stops",
        "lost_calls_beside_a_turn_limit",
    ],
)
def test_no_planner_reply_is_named_by_its_loss_in_any_draft_order(
    tmp_path, monkeypatch, errors, status, terminal_class
):
    run = _compete(
        tmp_path,
        monkeypatch,
        faults={name: _lost(error) for name, error in zip(DRAFTS, errors, strict=True)},
    )

    assert (run.status, run.terminal_class) == (status, terminal_class), run.reason

    from assay.primitives.plan import _planner_backend_error_status

    _is_named(
        run,
        status,
        terminal_class,
        "no planner returned a draft: "
        + ", ".join(
            f"{name} {_planner_backend_error_status(error)}"
            for name, error in zip(DRAFTS, errors, strict=True)
        ),
        DRAFT_CALLS,
    )


# --- drafts came back and none reached the comparison ---------------------------


@pytest.mark.parametrize(
    ("refused", "unreviewed", "unreviewed_calls"),
    [
        (
            _each(DRAFTS, _says("not JSON")),
            {
                "plan_1": _says("not JSON"),
                "plan_2": _says("not JSON"),
                "plan_3_critic": _lost(),
            },
            [*DRAFT_CALLS, "plan_3_critic"],
        ),
        (
            _each(DRAFTS, _says(OVERSIZE)),
            {
                "plan_1": _says(OVERSIZE),
                "plan_2": _says(OVERSIZE),
                "plan_3_critic": _lost(),
            },
            [*DRAFT_CALLS, "plan_3_critic"],
        ),
        (
            _each(DRAFTS, _cites_a_missing_file),
            {
                "plan_1": _cites_a_missing_file,
                "plan_2": _cites_a_missing_file,
                "plan_3_critic": _lost(),
            },
            [*DRAFT_CALLS, "plan_3_critic"],
        ),
        (
            {
                "plan_1": _lost(),
                "plan_2": _lost(TURNS),
                "plan_3": _cites_a_missing_file,
            },
            {"plan_1": _lost(), "plan_2": _lost(TURNS), "plan_3_critic": _lost()},
            [*DRAFT_CALLS, "plan_3_critic"],
        ),
    ],
    ids=["decoder", "size_limit", "gate", "beside_lost_planners"],
)
def test_refused_drafts_are_a_ruling_and_an_unreviewed_sibling_is_not(
    tmp_path, monkeypatch, refused, unreviewed, unreviewed_calls
):
    """Every draft that came back was refused; then the last one lost its Critic."""
    judged = _compete(tmp_path / "refused", monkeypatch, faults=refused)

    _is_kept(judged, NO_CANDIDATE, DRAFT_CALLS)

    open_candidate = _compete(tmp_path / "unreviewed", monkeypatch, faults=unreviewed)

    _is_named(
        open_candidate,
        "critic_infrastructure_failure",
        "inconclusive",
        "candidate 'plan_3' was not reviewed (critic_infrastructure_failure) and no "
        "candidate reached the comparison",
        unreviewed_calls,
    )


@pytest.mark.parametrize(
    ("fault", "status", "terminal_class", "calls_per_critic"),
    [
        (_lost(), "critic_infrastructure_failure", "inconclusive", 1),
        (_lost(BUDGET), "budget_exhausted", "budget_exhausted", 1),
        # Bounded response repair: three replies outside the review protocol.
        (_says("not a review"), "critic_protocol_failure", "inconclusive", 3),
    ],
    ids=["lost", "budget_stop", "outside_the_protocol"],
)
def test_drafts_that_all_lost_their_critic_name_the_run(
    tmp_path, monkeypatch, fault, status, terminal_class, calls_per_critic
):
    run = _compete(tmp_path, monkeypatch, faults=_each(CRITICS, fault))

    _is_named(
        run,
        status,
        terminal_class,
        f"candidate 'plan_1' was not reviewed ({status}) and no candidate reached "
        "the comparison",
        [*DRAFT_CALLS, *(label for label in CRITICS for _ in range(calls_per_critic))],
    )
    assert set(run.rows.values()) == {status}


# --- a draft lost its Judge -------------------------------------------------------


@pytest.mark.parametrize(
    ("faults", "status", "terminal_class", "named", "judge_calls"),
    [
        (
            {"plan_2_judge": _lost()},
            "judge_infrastructure_failure",
            "inconclusive",
            "plan_2",
            1,
        ),
        (
            _each(JUDGES, _lost()),
            "judge_infrastructure_failure",
            "inconclusive",
            "plan_1",
            1,
        ),
        (
            {"plan_2_judge": _lost(BUDGET)},
            "budget_exhausted",
            "budget_exhausted",
            "plan_2",
            1,
        ),
        (
            {"plan_2_judge": _says("not a review")},
            "judge_protocol_failure",
            "inconclusive",
            "plan_2",
            3,
        ),
    ],
    ids=["one_lost", "all_lost", "budget_stop", "outside_the_protocol"],
)
def test_a_lost_judge_beside_open_candidates_names_the_run(
    tmp_path, monkeypatch, faults, status, terminal_class, named, judge_calls
):
    """The run still ends there; the comparison is not asked."""
    run = _compete(tmp_path, monkeypatch, faults=faults)

    calls = [
        label
        for label in REVIEW_CALLS
        for _ in range(judge_calls if label in faults else 1)
    ]
    _is_named(
        run,
        status,
        terminal_class,
        f"the comparison was not asked: a reviewer of candidate {named!r} gave no "
        f"verdict ({status}); {BUNDLE_REFUSED}",
        [*DRAFT_CALLS, *calls],
    )


def test_lost_judge_beside_refused_siblings_follows_the_critic(tmp_path, monkeypatch):
    """A Critic ``reject`` is a ruling whatever the Judge would have answered."""
    siblings = {"plan_1_judge": "reject", "plan_3_judge": "reject"}
    lost_judge = {"plan_2_judge": _lost()}
    calls = [*DRAFT_CALLS, *REVIEW_CALLS]

    critic_rejected = _compete(
        tmp_path / "critic_rejected",
        monkeypatch,
        faults=lost_judge,
        rulings={**siblings, "plan_2_critic": "reject"},
    )

    # Every candidate was ruled against: the judged refusal it was before.
    _is_kept(critic_rejected, BUNDLE_REFUSED, calls)
    assert critic_rejected.rows == {
        "plan_1": "candidate_rejected",
        "plan_2": "judge_infrastructure_failure",
        "plan_3": "candidate_rejected",
    }

    for critic_decision in ("pass", "revise"):
        open_candidate = _compete(
            tmp_path / f"critic_{critic_decision}",
            monkeypatch,
            faults=lost_judge,
            rulings=(
                siblings
                if critic_decision == "pass"
                else {**siblings, "plan_2_critic": critic_decision}
            ),
        )

        # Nothing ruled against plan_2, and its Judge gave no verdict.
        _is_named(
            open_candidate,
            "judge_infrastructure_failure",
            "inconclusive",
            "the comparison was not asked: a reviewer of candidate 'plan_2' gave no "
            "verdict (judge_infrastructure_failure); ",
            calls,
        )

    beside_accepted = _compete(
        tmp_path / "critic_rejected_beside_accepted",
        monkeypatch,
        faults=lost_judge,
        rulings={"plan_2_critic": "reject"},
    )

    # Accepted candidates were waiting for a comparison that was not asked.
    _is_named(
        beside_accepted,
        "judge_infrastructure_failure",
        "inconclusive",
        "the comparison was not asked: a reviewer of candidate 'plan_2' gave no "
        "verdict (judge_infrastructure_failure); ",
        calls,
    )


def test_rule_set_refusal_after_a_lost_judge_is_read_from_the_verdicts(
    tmp_path, monkeypatch
):
    """The recheck overwrites that draft's status with a permanent one."""
    calls = [*DRAFT_CALLS, *REVIEW_CALLS]

    judged = _compete(
        tmp_path / "siblings_rejected",
        monkeypatch,
        faults={"plan_2_judge": _lost()},
        rulings={"plan_1_judge": "reject", "plan_3_judge": "reject"},
        rule_set_refuses="plan_2",
    )

    assert judged.rows["plan_2"] == "validate_blocked"
    _is_kept(judged, BUNDLE_REFUSED, calls)

    open_candidates = _compete(
        tmp_path / "siblings_accepted",
        monkeypatch,
        faults={"plan_2_judge": _lost()},
        rule_set_refuses="plan_2",
    )

    assert open_candidates.rows["plan_2"] == "validate_blocked"
    _is_named(
        open_candidates,
        "judge_infrastructure_failure",
        "inconclusive",
        "the comparison was not asked: a reviewer of candidate 'plan_2' gave no "
        "verdict (judge_infrastructure_failure); ",
        calls,
    )


# --- the comparison ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("faults", "rulings", "review_calls"),
    [
        ({}, {}, REVIEW_CALLS),
        ({}, {"plan_2_judge": "reject", "plan_3_judge": "reject"}, REVIEW_CALLS),
        (
            # A lost planner, and a lost Critic on a draft outside the bundle.
            {"plan_1": _lost(), "plan_2_critic": _lost()},
            {"plan_3_judge": "reject"},
            ["plan_2_critic", "plan_3_critic", "plan_3_judge"],
        ),
        ({}, _each(JUDGES, "inconclusive"), REVIEW_CALLS),
    ],
    ids=[
        "accepted",
        "one_accepted_two_rejected",
        "unreviewed_sibling_and_a_rejected_draft",
        "judges_answer_inconclusive",
    ],
)
def test_an_abstaining_comparison_is_a_ruling_and_a_lost_one_is_not(
    tmp_path, monkeypatch, faults, rulings, review_calls
):
    calls = [*DRAFT_CALLS, *review_calls, "comparison"]

    abstained = _compete(
        tmp_path / "abstained",
        monkeypatch,
        faults=faults,
        rulings=rulings,
        comparison=ABSTAIN,
    )

    _is_kept(abstained, ABSTAINED, calls)

    lost = _compete(
        tmp_path / "lost",
        monkeypatch,
        faults={**faults, "comparison": _lost()},
        rulings=rulings,
    )

    _is_named(
        lost,
        "judge_infrastructure_failure",
        "inconclusive",
        "the comparison gave no answer (judge_infrastructure_failure); "
        + SELECTOR_LOST,
        calls,
    )


@pytest.mark.parametrize(
    ("planners", "comparison_fault", "status", "terminal_class", "comparison_calls"),
    [
        (
            {"plan_1": _lost()},
            _lost(),
            "judge_infrastructure_failure",
            "inconclusive",
            1,
        ),
        (
            # A draft the decoder refused is ruled against, as a rejected one is.
            {"plan_1": _says("not JSON")},
            _lost(),
            "judge_infrastructure_failure",
            "inconclusive",
            1,
        ),
        ({}, _says("not a review"), "judge_protocol_failure", "inconclusive", 3),
        ({}, _lost(BUDGET), "budget_exhausted", "budget_exhausted", 1),
    ],
    ids=[
        "lost_beside_a_lost_planner",
        "lost_beside_a_decoder_refusal",
        "outside_its_contract",
        "budget_stop",
    ],
)
def test_every_candidate_ruled_against_stays_judged_without_a_comparison_answer(
    tmp_path, monkeypatch, planners, comparison_fault, status, terminal_class,
    comparison_calls,
):
    faults = {**planners, "comparison": comparison_fault}
    reviewed = [name for name in DRAFTS if name not in planners]
    calls = [
        *DRAFT_CALLS,
        *(label for label in REVIEW_CALLS if label.rsplit("_", 1)[0] in reviewed),
        *["comparison"] * comparison_calls,
    ]

    rejected = _compete(
        tmp_path / "rejected",
        monkeypatch,
        faults=faults,
        rulings={f"{name}_judge": "reject" for name in reviewed},
    )

    # Every candidate that existed was ruled against; the comparison adds no
    # ruling.
    _is_kept(rejected, SELECTOR_LOST, calls)

    accepted = _compete(tmp_path / "accepted", monkeypatch, faults=faults)

    _is_named(
        accepted,
        status,
        terminal_class,
        f"the comparison gave no answer ({status}); {SELECTOR_LOST}",
        calls,
    )


# --- a cancel request in a window without a typed stop ------------------------------


@pytest.mark.parametrize("window", ["before_comparison", "before_synthesis_review"])
def test_cancel_in_a_window_without_a_typed_stop_is_cancelled(
    tmp_path, monkeypatch, window
):
    cancel = threading.Event()
    if window == "before_comparison":
        # The request arrives while the last reviewed draft becomes eligible.
        candidate = _plan_selection.PlanSelectionCandidate

        def last_candidate(**fields):
            if fields["id"] == "plan_3":
                cancel.set()
            return candidate(**fields)

        monkeypatch.setattr(_plan_selection, "PlanSelectionCandidate", last_candidate)
    else:
        # The request arrives once the answered comparison's synthesis is taken up.
        select = _plan_selection.select_plan_candidate

        class Proposed:
            def __init__(self, selection):
                self._selection = selection

            def __getattr__(self, name):
                if name == "candidate_dag":
                    cancel.set()
                return getattr(self._selection, name)

        monkeypatch.setattr(
            _plan_selection,
            "select_plan_candidate",
            lambda *args, **kwargs: Proposed(select(*args, **kwargs)),
        )

    run = _compete(
        tmp_path,
        monkeypatch,
        cancel_event=cancel,
        synthesis=window == "before_synthesis_review",
    )

    _is_named(
        run,
        "cancelled",
        "cancelled",
        "cancelled before comparison"
        if window == "before_comparison"
        else "cancelled before synthesis review",
        [*DRAFT_CALLS, *REVIEW_CALLS]
        + (["comparison"] if window == "before_synthesis_review" else []),
    )


# --- typed stops and handed-on statuses stay what they are --------------------------


@contextmanager
def _rule_is_never_asked(monkeypatch):
    """Fail when the rule for a run without a ruling is consulted."""
    asked = []
    rule = competition._terminal_without_ruling
    with monkeypatch.context() as patch:
        patch.setattr(
            competition,
            "_terminal_without_ruling",
            lambda *args: asked.append(args) or rule(*args),
        )
        yield
    assert asked == []


class _NoSecondLease(harness.Pool):
    """The second planner's backend cannot be built."""

    def __init__(self):
        super().__init__()
        self.barrier = SimpleNamespace(wait=lambda **kwargs: None)
        self.finished["plan_2"].set()

    def new_exclusive(self, role, *, log_dir, cwd, toolset=None, **kwargs):
        path = Path(log_dir)
        if path.name == "plan_2" and path.parent.name == "independent_planners":
            raise RuntimeError("route construction failed")
        return super().new_exclusive(
            role, log_dir=log_dir, cwd=cwd, toolset=toolset, **kwargs
        )


def test_typed_draft_stops_are_decided_before_the_rule(tmp_path, monkeypatch):
    with _rule_is_never_asked(monkeypatch):
        cancellation = _compete(
            tmp_path / "cancellation_error",
            monkeypatch,
            faults={
                "plan_2": _lost(ExternalCancellation(cause="provider", message="stop"))
            },
        )
        raised = _compete(tmp_path / "raised", monkeypatch, faults={"plan_2": _raises})
        unbuilt = _compete(tmp_path / "unbuilt", monkeypatch, pool=_NoSecondLease())

    assert (cancellation.status, cancellation.reason, cancellation.calls) == (
        "cancelled",
        "independent planning was cancelled and joined",
        DRAFT_CALLS,
    )
    # Without the second planner the fixture does not order the other two.
    for run, calls in ((raised, DRAFTS), (unbuilt, ("plan_1", "plan_3"))):
        assert (run.status, run.reason, sorted(run.calls)) == (
            "budget_accounting_incomplete",
            "a draft lacks complete execution or accounting evidence",
            list(calls),
        )
        _nothing_landed(run)


def test_statuses_after_an_answered_comparison_are_decided_before_the_rule(
    tmp_path, monkeypatch
):
    reviewed = [*DRAFT_CALLS, *REVIEW_CALLS, "comparison"]
    with _rule_is_never_asked(monkeypatch):
        selected = _compete(
            tmp_path / "selected", monkeypatch, faults={"plan_2_critic": _lost()}
        )
        gate_refused = _compete(
            tmp_path / "synthesis_gate_refused",
            monkeypatch,
            comparison={
                "mode": "synthesize",
                "selected_id": None,
                "selected_digest": None,
                "synthesized_dag": _citing_a_missing_file(
                    _serialize_tier_dag(harness._dag("combined"))
                ),
            },
        )
        judge_lost = _compete(
            tmp_path / "synthesis_judge_lost",
            monkeypatch,
            faults={"synthesis_judge": _lost()},
            synthesis=True,
        )
        rejected = _compete(
            tmp_path / "synthesis_rejected",
            monkeypatch,
            rulings={"synthesis_judge": "reject"},
            synthesis=True,
        )
        with pytest.raises(ConnectionError):
            _compete(
                tmp_path / "reviewer_raises",
                monkeypatch,
                faults={"plan_2_critic": _raises},
            )

    # One lost Critic drops its draft; the comparison selects among the others.
    assert selected.outcome.accepted and selected.status == "accepted"
    assert selected.calls == [
        *DRAFT_CALLS,
        *(label for label in REVIEW_CALLS if label != "plan_2_judge"),
        "comparison",
    ]
    # A synthesis the deterministic gate refuses before any reviewer call.
    assert (gate_refused.status, gate_refused.calls) == ("budget_exhausted", reviewed)
    assert (judge_lost.status, judge_lost.terminal_class, judge_lost.calls) == (
        "judge_infrastructure_failure",
        "inconclusive",
        [*reviewed, "synthesis_critic", "synthesis_judge"],
    )
    assert (rejected.status, rejected.terminal_class, rejected.calls) == (
        "candidate_rejected",
        "contract_error",
        [*reviewed, "synthesis_critic", "synthesis_judge"],
    )
    for run in (gate_refused, judge_lost, rejected):
        _nothing_landed(run)
    report = json.loads(
        next((tmp_path / "reviewer_raises").rglob("plan_competition.json")).read_text()
    )
    assert report["status"] == KEPT


def test_synthesis_reviewer_that_answers_inconclusive_is_filed_as_not_judged(
    tmp_path, monkeypatch
):
    run = _compete(
        tmp_path, monkeypatch, rulings={"synthesis_judge": "inconclusive"}, synthesis=True
    )

    assert run.status == "candidate_review_inconclusive"
    assert run.calls == [
        *DRAFT_CALLS, *REVIEW_CALLS, "comparison", "synthesis_critic", "synthesis_judge",
    ]
    _nothing_landed(run)
    # No reviewer ruled against the synthesis and none passed it.
    assert run.terminal_class == "inconclusive"


# --- one planner against three ------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "single_calls"),
    [(TRANSPORT, 5), (TRANSPORT_FINAL, 1), (EVIDENCE, 1), (BUDGET, 1)],
    ids=["retriable", "not_retriable", "incomplete_evidence", "budget_stop"],
)
def test_a_lost_planner_has_one_status_at_one_and_three_planners(
    tmp_path, monkeypatch, error, single_calls
):
    single = _plan_once(tmp_path / "one", monkeypatch, planner=lambda call: _lost(error))
    three = _compete(tmp_path / "three", monkeypatch, faults=_each(DRAFTS, _lost(error)))

    # One planner spends its round budget; three planners are asked once each.
    assert single.planner_calls == single_calls and three.calls == DRAFT_CALLS
    assert (three.status, three.terminal_class) == (
        single.status,
        single.terminal_class,
    ), three.reason
    _nothing_landed(three)

    from assay.primitives.plan import _planner_backend_error_status

    assert single.status == _planner_backend_error_status(error)
    assert three.reason == "no planner returned a draft: " + ", ".join(
        f"{name} {single.status}" for name in DRAFTS
    )


@pytest.mark.parametrize(
    ("role", "fault", "status"),
    [
        ("critic", _lost(), "critic_infrastructure_failure"),
        ("judge", _lost(), "judge_infrastructure_failure"),
        ("critic", _says("not a review"), "critic_protocol_failure"),
        ("judge", _says("not a review"), "judge_protocol_failure"),
        ("critic", _lost(BUDGET), "budget_exhausted"),
        ("judge", _lost(BUDGET), "budget_exhausted"),
    ],
    ids=[
        "critic_lost",
        "judge_lost",
        "critic_outside_the_protocol",
        "judge_outside_the_protocol",
        "critic_budget_stop",
        "judge_budget_stop",
    ],
)
def test_a_lost_reviewer_has_one_status_at_one_and_three_planners(
    tmp_path, monkeypatch, role, fault, status
):
    single = _plan_once(tmp_path / "one", monkeypatch, faults={f"single_{role}": fault})
    # One lost Critic only drops its draft, so every Critic is lost here; one
    # lost Judge ends the run.
    three = _compete(
        tmp_path / "three",
        monkeypatch,
        faults=_each(CRITICS, fault) if role == "critic" else {"plan_2_judge": fault},
    )

    assert single.status == status
    assert (three.status, three.terminal_class) == (status, single.terminal_class)
    _nothing_landed(three)


def test_where_one_planner_and_three_planners_still_differ(tmp_path, monkeypatch):
    """Pinned as differences; none of them is parity."""
    # Every planner on its turn limit or a content rejection: permanent with
    # one planner, the competition's own status with three.
    for name, error, status in (
        ("turn_limit", TURNS, "hard_fail_max_turns"),
        ("content_rejected", CONTENT, "hard_fail_content_rejected"),
    ):
        single = _plan_once(
            tmp_path / f"one_{name}", monkeypatch, planner=lambda call: _lost(error)
        )
        three = _compete(
            tmp_path / f"three_{name}", monkeypatch, faults=_each(DRAFTS, _lost(error))
        )
        assert (single.status, single.terminal_class) == (status, "permanent_failure")
        _is_kept(three, NO_CANDIDATE, DRAFT_CALLS)
    mixed = _compete(
        tmp_path / "three_mixed",
        monkeypatch,
        faults={"plan_1": _lost(TURNS), "plan_2": _lost(CONTENT), "plan_3": _lost(TURNS)},
    )
    _is_kept(mixed, NO_CANDIDATE, DRAFT_CALLS)

    # Lost calls beside a refused reply. One planner decides by which call came
    # last: here the lost calls. Three planners: the only candidate was refused.
    single = _plan_once(
        tmp_path / "one_refused_then_lost",
        monkeypatch,
        planner=lambda call: _cites_a_missing_file if call == 1 else _lost(),
    )
    three = _compete(
        tmp_path / "three_refused_beside_lost",
        monkeypatch,
        faults={"plan_1": _cites_a_missing_file, "plan_2": _lost(), "plan_3": _lost()},
    )
    assert (single.status, single.terminal_class) == (
        "planner_infrastructure_failure",
        "inconclusive",
    )
    _is_kept(three, NO_CANDIDATE, DRAFT_CALLS)

    # A planner that stopped on its budget beside refused replies. One planner
    # ends on the budget stop whatever came before it. Three planners: every
    # candidate that existed was refused.
    single = _plan_once(
        tmp_path / "one_refused_then_budget_stop",
        monkeypatch,
        planner=lambda call: _cites_a_missing_file if call == 1 else _lost(BUDGET),
    )
    three = _compete(
        tmp_path / "three_refused_beside_budget_stop",
        monkeypatch,
        faults={
            "plan_1": _lost(BUDGET),
            "plan_2": _says("not JSON"),
            "plan_3": _says("not JSON"),
        },
    )
    assert (single.status, single.terminal_class, single.planner_calls) == (
        "budget_exhausted",
        "budget_exhausted",
        2,
    )
    _is_kept(three, NO_CANDIDATE, DRAFT_CALLS)

    # A Critic ``reject`` followed by a lost Judge: the lost Judge with one
    # planner, a ruled-against draft with three.
    single = _plan_once(
        tmp_path / "one_critic_reject_judge_lost",
        monkeypatch,
        faults={"single_judge": _lost()},
        rulings={"single_critic": "reject"},
    )
    assert (single.status, single.terminal_class) == (
        "judge_infrastructure_failure",
        "inconclusive",
    )

    # Lost calls beside a turn limit. One planner ends on the turn limit as soon
    # as it occurs; with three planners the lost calls name the run.
    single = _plan_once(
        tmp_path / "one_lost_then_turn_limit",
        monkeypatch,
        planner=lambda call: _lost() if call == 1 else _lost(TURNS),
    )
    three = _compete(
        tmp_path / "three_lost_beside_turn_limit",
        monkeypatch,
        faults={"plan_1": _lost(), "plan_2": _lost(), "plan_3": _lost(TURNS)},
    )
    assert (single.status, single.terminal_class, single.planner_calls) == (
        "hard_fail_max_turns",
        "permanent_failure",
        2,
    )
    _is_named(
        three,
        "planner_infrastructure_failure",
        "inconclusive",
        "no planner returned a draft: ",
        DRAFT_CALLS,
    )


# --- a lost call with incomplete usage is an accounting terminal ----------------------


def _openai_compatible_result(
    tmp_path,
    monkeypatch,
    create,
    *,
    toolset=None,
    response_schema=None,
    max_turns=1,
    timeout_seconds=1.0,
):
    """What the OpenAI-compatible adapter returns for one single-attempt call."""
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "adapter-fixture:\n  input_per_mtok: 0\n  output_per_mtok: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CRUCIBLE_PRICING_CONFIG", str(pricing))
    backend = OpenAIBackend(
        model="adapter-fixture",
        base_url="http://127.0.0.1:9/v1",
        api_key="not-needed",
        capability="text" if toolset is None else "read_only",
    )
    if toolset is not None:
        backend.attach_toolset(toolset)
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return backend.invoke(
        Invocation(
            system="s",
            user="u",
            tools=None,
            max_turns=max_turns,
            budget_usd=None,
            response_schema=response_schema,
            policy=BackendPolicy(
                retry_budget=1,
                timeout_seconds=timeout_seconds,
                on_cancellation="raise",
                on_max_turns="raise",
            ),
        )
    )


def test_a_lost_planner_call_is_named_only_with_complete_usage(tmp_path, monkeypatch):
    def refuse_connection(**_kwargs):
        raise ConnectionError("connection refused")

    lost = _openai_compatible_result(tmp_path, monkeypatch, refuse_connection)

    # The adapter's classified connection error: retriable, usage unknown.
    assert (lost.error.kind, lost.error.retriable) == ("transport", True)
    assert not lost.completeness.token_usage_complete
    assert not lost.completeness.cost_complete

    single = _plan_once(tmp_path / "one", monkeypatch, planner=lambda call: lambda _: lost)
    three = _compete(
        tmp_path / "three", monkeypatch, faults=_each(DRAFTS, lambda _: lost)
    )

    # Unknown usage is an accounting terminal at both planner counts.
    assert (single.status, single.terminal_class) == (
        "incomplete_accounting",
        "inconclusive",
    )
    assert (three.status, three.reason, three.calls) == (
        "budget_accounting_incomplete",
        "a draft lacks complete execution or accounting evidence",
        DRAFT_CALLS,
    )
    _nothing_landed(three)

    # The same typed error with complete evidence names the run.
    named = _compete(
        tmp_path / "complete", monkeypatch, faults=_each(DRAFTS, _lost(lost.error))
    )

    _is_named(
        named,
        "planner_infrastructure_failure",
        "inconclusive",
        "no planner returned a draft: ",
        DRAFT_CALLS,
    )


def test_drafts_accounted_as_the_gateway_adapter_accounts_them_are_selected(
    tmp_path, monkeypatch
):
    # A planner call with source tools on the OpenAI-compatible route: a tool
    # round, a draft and the schema finalisation, each reply with its usage.
    replies = iter(
        [
            _stub_response("", tool_calls=[_tool_call("c1", "lookup", "{}")]),
            _stub_response("draft"),
            _stub_response('{"tiers": []}'),
        ]
    )
    paid = _openai_compatible_result(
        tmp_path,
        monkeypatch,
        lambda **_kwargs: next(replies),
        toolset=_StubToolSet(),
        response_schema={"type": "object"},
        max_turns=5,
        # Three requests, and no test of the wall: one the clock of a slow
        # runner does not reach.
        timeout_seconds=600.0,
    )

    assert paid.error is None and paid.completeness.complete
    assert paid.pricing_evidence is not None
    assert len(paid.pricing_evidence.segments) == 3

    def accounted_by_the_adapter(result):
        return replace(
            result,
            usage=paid.usage,
            completeness=paid.completeness,
            execution_trace=paid.execution_trace,
            pricing_evidence=paid.pricing_evidence,
        )

    with monkeypatch.context() as patch:
        _script(patch, _each(DRAFTS, accounted_by_the_adapter), {}, None)
        _pool, ledger, _corpus, _spec, outcome, report = harness._run(tmp_path / "three")

    # Three planners need complete receipts with pricing evidence for every
    # call; Architect then reconciles the outcome with the call ledger.
    assert outcome.accepted, (outcome.status, outcome.reason, report)
    architect._require_plan_outcome_pricing_reconciliation(
        outcome=outcome,
        usage_before=CostAccumulator().to_dict(),
        planning_usage=ledger,
        role="competition",
    )


# --- the rule over a closed domain ----------------------------------------------------


_LOST_VERDICT = SimpleNamespace(
    decision="inconclusive",
    findings=(
        SimpleNamespace(
            source="deterministic", category="backend_error", id="backend_transport"
        ),
    ),
)
_RULING_VERDICT = SimpleNamespace(decision="inconclusive", findings=())
_ENDS = (
    ("no_reply", "planner_infrastructure_failure"),
    ("no_reply", "planner_protocol_failure"),
    ("no_reply", "budget_exhausted"),
    ("no_reply", "hard_fail_max_turns"),
    ("refused", None),
    ("refused", "judge_infrastructure_failure"),
    ("open", None),
    ("open", "critic_infrastructure_failure"),
    ("open", "judge_protocol_failure"),
    ("open", "budget_exhausted"),
    # In no table: an unregistered status never names a run.
    ("open", "candidate_validation_failed"),
    # An accounting status has an unjudged class and would pass through.
    ("open", "incomplete_accounting"),
    ("unknown", None),
)
_SELECTIONS = {
    "not_asked": None,
    "abstained": SimpleNamespace(
        error=None, mode="none", candidate_dag=None, verdict=_RULING_VERDICT
    ),
    "synthesis_proposed": SimpleNamespace(
        error=None, mode="synthesize", candidate_dag=object(), verdict=_RULING_VERDICT
    ),
    "bundle_refused": SimpleNamespace(
        error=BUNDLE_REFUSED + "x", mode="none", candidate_dag=None,
        verdict=_RULING_VERDICT,
    ),
    "comparison_lost": SimpleNamespace(
        error=SELECTOR_LOST, mode="none", candidate_dag=None, verdict=_LOST_VERDICT
    ),
}


def test_rule_names_a_run_only_with_a_status_that_is_not_a_ruling():
    rule = competition._terminal_without_ruling
    not_a_ruling = {
        *_outcome.UNJUDGED_TERMINAL_CLASSES,
        _outcome.TerminalClass.BUDGET_EXHAUSTED,
    }
    returned = set()
    for combination in itertools.product(_ENDS, repeat=3):
        ends = [
            (f"plan_{index + 1}", kind, loss)
            for index, (kind, loss) in enumerate(combination)
        ]
        kinds = [kind for _name, kind, _loss in ends]
        # A draft is in the comparison only as a candidate: refused or open.
        candidates = frozenset(
            name for name, kind, _loss in ends if kind in ("refused", "open")
        )
        for compared in {frozenset(), candidates}:
            for answer, selection in _SELECTIONS.items():
                if selection is not None and not compared:
                    continue
                for cancelled in (False, True):
                    named = rule(
                        ends,
                        compared,
                        selection,
                        cancelled,
                        None,
                        lambda role, verdict, corpus: f"{role}_infrastructure_failure",
                    )
                    if named is None:
                        continue
                    status, reason = named
                    case = (ends, answer, cancelled, named)
                    assert reason, case
                    assert _outcome.classify_terminal_status(status) in not_a_ruling, case
                    assert "unknown" not in kinds, case
                    if status == "cancelled":
                        assert cancelled and answer in ("not_asked", "synthesis_proposed")
                        continue
                    # Never after an answered comparison, and never when every
                    # candidate that existed was ruled against.
                    assert answer not in ("abstained", "synthesis_proposed"), case
                    assert not candidates or "open" in kinds, case
                    returned.add(status)

    assert returned == {
        "planner_infrastructure_failure",
        "planner_protocol_failure",
        "critic_infrastructure_failure",
        "judge_infrastructure_failure",
        "judge_protocol_failure",
        "budget_exhausted",
        # Guaranteed is the class, not this list: see ``_ENDS``.
        "incomplete_accounting",
    }


# --- Architect ------------------------------------------------------------------------


def _architect(repo, monkeypatch, *, faults=None, rulings=None, comparison=None):
    """Run Architect's default three-planner planning on the fixture project."""
    _workspace, root, corpus, _spec = repo
    pools = []
    before = set(corpus.spec_dir.glob("runs/*_architect_*"))

    @contextmanager
    def backend_pool(*args, **kwargs):
        pools.append(harness.Pool())
        yield pools[-1]

    with monkeypatch.context() as patch:
        _script(patch, faults or {}, rulings or {}, comparison)
        patch.setattr(architect, "backend_pool", backend_pool)
        patch.setattr(
            architect, "run_tier", lambda **kwargs: pytest.fail("no plan was accepted")
        )
        exit_code = architect.run_architect(
            spec=corpus.spec_dir, cwd=str(root), budget_usd=10
        )
    (run_dir,) = set(corpus.spec_dir.glob("runs/*_architect_*")) - before
    outcome = json.loads((run_dir / "outcome.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    (pool,) = pools
    return SimpleNamespace(
        exit_code=exit_code,
        status=outcome["detail_status"],
        terminal_class=outcome["terminal_class"],
        failure=(summary["failure_phase"], summary["failure_kind"], summary["decision"]),
        calls=[label for label, _ in pool.order],
        published=corpus.latest_spec().tier_dag,
    )


def _is_planning_stop(run, exit_code, status, terminal_class):
    assert (run.exit_code, run.status, run.terminal_class) == (
        exit_code,
        status,
        terminal_class,
    )
    assert run.failure == ("architect_planning", status, "inconclusive")
    assert run.published is None


@pytest.mark.parametrize(
    ("faults", "exit_code", "status", "terminal_class"),
    [
        (_each(DRAFTS, _lost()), 2, "planner_infrastructure_failure", "inconclusive"),
        (_each(DRAFTS, _lost(EVIDENCE)), 2, "planner_protocol_failure", "inconclusive"),
        # Every planner stopped on its budget: exit 1, as with one planner.
        (_each(DRAFTS, _lost(BUDGET)), 1, "budget_exhausted", "budget_exhausted"),
        (_each(CRITICS, _lost()), 2, "critic_infrastructure_failure", "inconclusive"),
        ({"plan_2_judge": _lost()}, 2, "judge_infrastructure_failure", "inconclusive"),
        ({"plan_2_judge": _lost(BUDGET)}, 1, "budget_exhausted", "budget_exhausted"),
        ({"comparison": _lost()}, 2, "judge_infrastructure_failure", "inconclusive"),
        (
            {"comparison": _says("not a review")},
            2,
            "judge_protocol_failure",
            "inconclusive",
        ),
    ],
    ids=[
        "planners_lost",
        "planner_terminal",
        "planner_budget_stop",
        "critics_lost",
        "judge_lost",
        "judge_budget_stop",
        "comparison_lost",
        "comparison_outside_its_contract",
    ],
)
def test_architect_files_a_three_planner_planning_stop(
    tmp_path, monkeypatch, faults, exit_code, status, terminal_class
):
    run = _architect(harness._repo(tmp_path), monkeypatch, faults=faults)

    _is_planning_stop(run, exit_code, status, terminal_class)


def test_architect_keeps_a_judged_refusal_and_plans_again_after_a_lost_run(
    tmp_path, monkeypatch
):
    judged = {
        "decoder": dict(faults=_each(DRAFTS, _says("not JSON"))),
        "rejected_and_abstained": dict(
            rulings=_each(JUDGES, "reject"), comparison=ABSTAIN
        ),
        "rejected_and_comparison_lost": dict(
            faults={"comparison": _lost()}, rulings=_each(JUDGES, "reject")
        ),
    }
    for name, scenario in judged.items():
        run = _architect(harness._repo(tmp_path / name), monkeypatch, **scenario)

        # Exit code, status and class of a ruling against every candidate.
        _is_planning_stop(run, 2, KEPT, "contract_error")

    repo = harness._repo(tmp_path / "lost")
    for _attempt in range(2):
        run = _architect(repo, monkeypatch, faults=_each(DRAFTS, _lost()))

        # Nothing is cached for a lost run: the same Corpus plans again.
        _is_planning_stop(run, 2, "planner_infrastructure_failure", "inconclusive")
        assert run.calls == DRAFT_CALLS
