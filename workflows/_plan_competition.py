# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Independent read-only plan drafts, adversarial gates, and one selected DAG.

The source Spec never changes during the competition. Children publish only
content-addressed diagnostics; the caller alone publishes the returned staged
Spec. A comparison may propose a synthesis, but cannot approve it.
"""

from __future__ import annotations

import json
import hashlib
import math
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    ExternalCancellation,
    Invocation,
    InvocationResult,
    LifecycleEvent,
    PlanOutcome,
    Evidence,
    Finding,
    RuleContext,
    Usage,
    UsagePricingEvidence,
    ValidationRuleset,
    Verdict,
)
from assay.canonical import hash_record, serialize
from assay.primitives.plan import (
    _planner_backend_error_status,
    _planner_system_prompt,
    _planner_user_message,
    _tier_dag_response_schema,
)
from assay.primitives.validate import validate

from crucible.backends.features import FeatureRequirement
from crucible.rules import available as available_rules
from crucible.common.cost import CostAccumulator
from crucible.common.read_only_sources import SourceReferenceError, _read_file
from crucible.tools._builder import ToolSetBuilder
from crucible.tools.source import build_source_tools

from ._outcome import (
    UNJUDGED_TERMINAL_CLASSES,
    TerminalClass,
    classify_terminal_status,
    workspace_state_digest,
)
from ._backends import BackendCleanupError
from ._plan_provenance import _workspace_head_identity
from ._readonly_scheduler import (
    ReadOnlyChildRequest,
    ReadOnlyCleanupError,
    ReadOnlyIsolationError,
    _ManagedBackend,
    run_read_only_fanout,
)
from ._review import is_review_infrastructure_failure
from ._run import exec_ctx_now
from ._run_policy import OrchestrationPolicy


PLAN_COMPETITION_VERSION = "crucible.plan-competition.v1"
DEFAULT_PLANNER_COUNT = 3
MAX_PLANNER_COUNT = 8
_DRAFT_SHARE = 0.30
_REVIEW_SHARE = 0.30
_SELECTION_SHARE = 0.10
_SYNTHESIS_REVIEW_SHARE = 0.30


class _CompetitionCancelled(RuntimeError):
    pass


class PlanCompetitionSourceChanged(ValueError):
    """Publication cannot retain the exact source identity the plans used."""


def competition_publication_guard(*, corpus, base_spec, staged_spec, cwd):
    """Bind the immutable competition report to an under-lock source recheck."""
    expected = None
    for ref in reversed(
        staged_spec.lifecycle_event_refs[len(base_spec.lifecycle_event_refs) :]
    ):
        event = corpus.get_record(ref, as_type=LifecycleEvent)
        if event.kind == "plan_competition":
            binding = json.loads(event.reason)
            report = json.loads(corpus.get_artifact(binding["report_ref"]))
            if binding["source_spec"] != str(hash_record(base_spec)):
                raise PlanCompetitionSourceChanged(
                    "competition source Spec identity is inconsistent"
                )
            expected = report["source_identity"]
            break
    if expected is None:
        raise PlanCompetitionSourceChanged("competition source binding is missing")
    root = Path(cwd).resolve(strict=True)

    def check():
        current = {
            "head": _workspace_head_identity(base_spec, root),
            "state": workspace_state_digest(root, excluded_roots=(corpus.spec_dir,)),
            "spec": str(hash_record(base_spec)),
        }
        if current != expected:
            raise PlanCompetitionSourceChanged(
                "planning source identity changed before executable plan publication"
            )

    return check


def planner_count_arg(value: str) -> int:
    import argparse

    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("planner count must be an integer") from exc
    if not 1 <= count <= MAX_PLANNER_COUNT:
        raise argparse.ArgumentTypeError(
            f"planner count must be 1..{MAX_PLANNER_COUNT}"
        )
    return count


def validate_competition_admission(count: int, budget_usd: float | None) -> None:
    if type(count) is not int or not 1 <= count <= MAX_PLANNER_COUNT:
        raise ValueError(f"planner_count must be an integer in 1..{MAX_PLANNER_COUNT}")
    if count > 1 and (
        budget_usd is None
        or isinstance(budget_usd, bool)
        or not isinstance(budget_usd, (int, float))
        or not math.isfinite(budget_usd)
        or budget_usd <= 0
    ):
        raise ValueError(
            "parallel planning requires a positive finite total budget_usd"
        )


class _DraftBackend:
    """Render the existing Assay planner ABI for each exclusive routed lease."""

    def __init__(self, backend: Any, spec: Any, toolset: Any) -> None:
        self._backend, self._spec, self._toolset = backend, spec, toolset

    def invoke(self, invocation: Invocation):
        policy = self._spec.plan_policy
        return self._backend.invoke(
            replace(
                invocation,
                system=_planner_system_prompt(
                    self._backend, prompt_toolset=self._toolset
                ),
                user=_planner_user_message(
                    self._spec,
                    slice_cap=getattr(policy, "max_total_slices", None),
                    tier_cap=getattr(policy, "max_total_tiers", None),
                    planner_guidance=getattr(policy, "planner_guidance", None),
                )
                + (
                    "\n\nDerive one complete plan independently from the authoritative "
                    "goal and visible source. Other planners have separate contexts; "
                    "do not assume their conclusions or add requirements. Return only "
                    "the complete TierDAG JSON required by the response schema."
                ),
                response_schema=_tier_dag_response_schema(),
            )
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def _draft_ends(
    children: Any,
    rows: list[dict[str, Any]],
    outcomes: dict[str, PlanOutcome],
    compared: set[str],
    corpus: Any,
    review_status: Callable[[str, Any, Any], str],
) -> list[tuple[str, str, str | None]]:
    """Say how each draft ended: ``(id, kind, loss)`` in request order.

    ``no_reply``: the planner returned a typed backend error, so no candidate
    exists. ``refused``: the size limit, the decoder, a deterministic gate or
    the candidate's own review ruled against it; a Critic ``reject`` is such a
    ruling whatever became of the Judge. ``open``: a candidate exists and
    nothing ruled against it. ``unknown``: the records prove none of these.

    ``loss`` is the status single-track planning gives a lost call of the
    draft's planner or reviewers, else ``None``.
    """

    ends: list[tuple[str, str, str | None]] = []
    for child, row in zip(children, rows, strict=True):
        outcome = outcomes.get(child.child_id)
        if outcome is None:
            error = child.result.error if child.result is not None else None
            if error is not None:
                ends.append(
                    (child.child_id, "no_reply", _planner_backend_error_status(error))
                )
            elif (
                child.error_kind == "ChildOutputLimitExceeded"
                or row["status"] == "invalid_candidate"
            ):
                # The planner answered; the size limit or the decoder refused it.
                ends.append((child.child_id, "refused", None))
            else:
                ends.append((child.child_id, "unknown", None))
            continue
        loss = None
        if child.child_id in compared:
            # Read the verdicts, not the review status: the rule-set recheck
            # may have overwritten the status of a draft whose Judge was lost.
            lost = next(
                (
                    verdict
                    for verdict in (
                        corpus.get_record(ref, as_type=Verdict)
                        for ref in row["verdict_refs"]
                    )
                    if verdict.kind in {"critic", "judge"}
                    and is_review_infrastructure_failure(verdict)
                ),
                None,
            )
            if lost is not None:
                loss = review_status(lost.kind, lost, corpus)
        elif outcome.decision == "inconclusive":
            loss = outcome.status
        if outcome.decision == "rejected" or outcome.final_critic_decision == "reject":
            ends.append((child.child_id, "refused", loss))
        elif outcome.decision in ("accepted", "inconclusive"):
            ends.append((child.child_id, "open", loss))
        else:
            ends.append((child.child_id, "unknown", None))
    return ends


def _first_loss(losses: list[tuple[str, str | None]]) -> tuple[str, str] | None:
    """Choose the loss that names a run nothing ruled on, or ``None``.

    A budget stop first. Otherwise only a status the outcome table files as
    not a ruling can name the run, and a lost planner call names it only when
    no other loss does. A permanent or unregistered status never names a run.
    """

    classed = [
        (name, status, classify_terminal_status(status))
        for name, status in losses
        if status is not None
    ]
    for name, status, terminal in classed:
        if terminal is TerminalClass.BUDGET_EXHAUSTED:
            return name, status
    named = [
        (name, status)
        for name, status, terminal in classed
        if terminal in UNJUDGED_TERMINAL_CLASSES
    ]
    return next(
        (loss for loss in named if loss[1] != "planner_infrastructure_failure"),
        named[0] if named else None,
    )


def _terminal_without_ruling(
    ends: list[tuple[str, str, str | None]],
    compared: set[str],
    selection: Any,
    cancelled: bool,
    corpus: Any,
    review_status: Callable[[str, Any, Any], str],
) -> tuple[str, str] | None:
    """Name a run that ended without a plan and without a ruling against it.

    Returns ``None`` when a ruling ended the run (the comparison answered, or
    every candidate that existed was ruled against) and when the records do
    not prove lost work; the caller then keeps its status. Otherwise returns
    ``(status, reason)`` with the status single-track planning gives the loss.
    """

    if any(kind == "unknown" for _name, kind, _loss in ends):
        return None
    if selection is None and compared:
        # Reviewed drafts were waiting and the comparison was never asked.
        return ("cancelled", "cancelled before comparison") if cancelled else None
    if selection is not None and selection.error is None:
        if (
            cancelled
            and selection.mode == "synthesize"
            and selection.candidate_dag is not None
        ):
            return "cancelled", "cancelled before synthesis review"
        return None
    candidates = [end for end in ends if end[1] != "no_reply"]
    if candidates and all(kind == "refused" for _name, kind, _loss in candidates):
        return None
    if selection is not None:
        waiting = _first_loss(
            [(name, loss) for name, _kind, loss in ends if name in compared]
        )
        if waiting is not None:
            name, status = waiting
            return (
                status,
                f"the comparison was not asked: a reviewer of candidate {name!r} "
                f"gave no verdict ({status}); {selection.error}",
            )
        if is_review_infrastructure_failure(selection.verdict):
            answer = _first_loss(
                [("comparison", review_status("judge", selection.verdict, corpus))]
            )
            if answer is not None:
                return (
                    answer[1],
                    f"the comparison gave no answer ({answer[1]}); {selection.error}",
                )
        return None
    named = _first_loss(
        [(name, loss) for name, kind, loss in ends if kind == "open"]
        if candidates
        else [(name, loss) for name, _kind, loss in ends]
    )
    if named is None:
        return None
    name, status = named
    if candidates:
        return (
            status,
            f"candidate {name!r} was not reviewed ({status}) and no candidate "
            "reached the comparison",
        )
    return (
        status,
        "no planner returned a draft: "
        + ", ".join(f"{name} {loss}" for name, _kind, loss in ends),
    )


def run_plan_competition(
    spec: Any,
    *,
    pool: Any,
    corpus: Any,
    cwd: str,
    run_dir: Path,
    planner_role: str,
    critic_role: str,
    judge_role: str,
    critic_rubric: Any,
    judge_rubric: Any,
    planner_count: int = DEFAULT_PLANNER_COUNT,
    budget_usd: float | None,
    max_turns: int = 5,
    enable_source_tools: bool = False,
    candidate_transform: Callable | None = None,
    candidate_validator: Callable | None = None,
    wrap_backend: Callable[[str, Any, Any], Any] | None = None,
    source_toolset_factory: Callable[[Any], Any] | None = None,
    call_ledger: Any = None,
    cancel_event: threading.Event | None = None,
) -> PlanOutcome:
    """Stage one competition result; never advance Corpus manifest authority.

    USD shares are disjoint declarations, not a promise of a provider hard
    stop. Every overrun/incomplete call stops later transport and acceptance.
    Failed branches retain their incurred usage, transcripts and diagnostics.

    Each draft and each proposed synthesis is decoded with canonical scope
    paths against ``cwd`` before any caller transform, gate, review, comparison
    or record sees it; the raw draft stays stored as written.

    A run that ends without a plan and without a typed stop keeps
    ``plan_competition_inconclusive`` when a ruling ended it, and when the
    records do not prove that a lost call ended it. When a call was lost while
    a candidate had no ruling against it, or no planner returned a draft, it
    returns the status single-track planning gives that loss.
    """
    from . import plan as plan_workflow
    from ._plan_selection import (
        PlanSelectionCandidate,
        _decode_authored_dag,
        select_plan_candidate,
    )
    from ._plan_source_snapshot import create_plan_source_snapshot

    validate_competition_admission(planner_count, budget_usd)
    if planner_count < 2:
        raise ValueError("run_plan_competition requires at least two planners")
    assert budget_usd is not None
    root = Path(cwd).resolve(strict=True)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_identity = {
        "head": _workspace_head_identity(spec, root),
        "state": workspace_state_digest(root, excluded_roots=(corpus.spec_dir,)),
        "spec": str(hash_record(spec)),
    }
    validator = candidate_validator
    if validator is None:
        from ._plan_validation import build_pre_review_candidate_validator

        validator = build_pre_review_candidate_validator(spec=spec, project_root=root)

    def validate_candidate(dag):
        private_findings: list[Finding] = []
        for tier in dag.tiers:
            for slice_ in tier.slices:
                for reference in slice_.directive.read_only_references:
                    path = reference.path
                    try:
                        frozen = _read_file(snapshot_root, path, None)
                        admitted = (
                            "sha256:" + hashlib.sha256(frozen).hexdigest()
                            == reference.sha256
                        )
                    except (OSError, SourceReferenceError):
                        admitted = False
                    if not admitted:
                        private_findings.append(
                            Finding(
                                id=f"private_planning_reference_{len(private_findings)}",
                                severity="High",
                                category="private_planning_reference",
                                evidence=(
                                    Evidence(
                                        kind="source_reference_validation",
                                        location=path,
                                        excerpt="planning state cannot be source evidence",
                                    ),
                                ),
                                suggested_action="Ground the plan in project source, excluding sibling or Corpus execution state.",
                                source="deterministic",
                            )
                        )
        return (*private_findings, *validator(dag))

    policy = OrchestrationPolicy(
        mode="read_only",
        parent_roles=(planner_role,),
        child_roles=tuple(f"independent_plan_{i + 1}" for i in range(planner_count)),
        max_children=planner_count,
        max_concurrency=planner_count,
        max_child_turns=max_turns,
        max_total_cost_share=_DRAFT_SHARE,
        max_child_cost_share=_DRAFT_SHARE / planner_count,
        max_child_output_bytes=1024 * 1024,
        max_joined_output_bytes=planner_count * (1024 * 1024 + 256),
        fail_fast=False,
    )
    report: dict[str, Any] = {
        "schema_version": PLAN_COMPETITION_VERSION,
        "source_identity": source_identity,
        "implementation_sources": {
            name: "sha256:"
            + hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "_plan_competition.py",
                "_plan_selection.py",
                "_plan_source_snapshot.py",
            )
        },
        "planner_count": planner_count,
        "policy": asdict(policy),
        "policy_digest": str(policy.digest),
        "budget_usd": budget_usd,
        "budget_shares": {
            "drafts": _DRAFT_SHARE,
            "candidate_reviews": _REVIEW_SHARE,
            "comparison": _SELECTION_SHARE,
            "synthesis_review": _SYNTHESIS_REVIEW_SHARE,
        },
        "budget_enforcement": "shared_soft_ceiling_with_disjoint_call_allowances",
        "hard_cap_guaranteed": False,
        "candidates": [],
        "status": "running",
        "selected_id": None,
    }
    calls: list[tuple[Any, Any, Any]] = []
    physical_receipts: list[dict[str, Any]] = []
    initial_ledger = call_ledger.to_dict() if call_ledger is not None else {}
    initial_call_count = len(initial_ledger.get("pricing_evidence_by_call", ()))
    initial_evidence = call_ledger.pricing_evidence if call_ledger is not None else None
    initial_segment_count = (
        len(initial_evidence.segments) if initial_evidence is not None else 0
    )
    calls_lock = threading.Lock()
    raw_leases: list[Any] = []
    report["calls"] = []
    verdict_refs: list[Any] = list(spec.verdict_refs)
    outcomes: dict[str, PlanOutcome] = {}
    eligible = []
    diagnostic_dag = None
    diagnostic_ref = None
    winner: PlanOutcome | None = None
    selection: Any = None
    owned_reviewers: list[tuple[str, _ManagedBackend]] = []
    reason = "no independently reviewed candidate was selected"
    status = "plan_competition_inconclusive"

    def add_refs(refs) -> None:
        for ref in refs:
            if ref not in verdict_refs:
                verdict_refs.append(ref)

    def remaining(share: float) -> float:
        spent = math.fsum(call[0].cost_usd for call in calls)
        return max(0.0, min(budget_usd * share, budget_usd - spent))

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def source_unchanged() -> bool:
        return source_identity == {
            "head": _workspace_head_identity(spec, root),
            "state": workspace_state_digest(root, excluded_roots=(corpus.spec_dir,)),
            "spec": str(hash_record(spec)),
        }

    def complete(value: Any) -> bool:
        return value is not None and all(
            getattr(value, name, False)
            for name in (
                "token_usage_complete",
                "cost_complete",
                "terminal_complete",
                "trace_complete",
            )
        )

    def receipts_complete() -> bool:
        return all(
            complete(completeness) and evidence is not None and evidence.complete
            for _usage, completeness, evidence in calls
        ) and (call_ledger is None or not call_ledger.to_dict()["rollup_incomplete"])

    def save_progress(authoritative_cost: float | None = None) -> None:
        physical_total = math.fsum(call[0].cost_usd for call in calls)
        report["physical_receipts_cost_usd"] = physical_total
        report["cost_usd"] = (
            physical_total if authoritative_cost is None else authoritative_cost
        )
        (run_dir / "plan_competition.json").write_text(
            json.dumps(report, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def observed(role: str, backend: Any, toolset: Any = None) -> Any:
        # Prompt/accounting wrappers must not conceal a reused provider lease.
        with calls_lock:
            if any(prior is backend for prior in raw_leases):
                raise ReadOnlyIsolationError(
                    "independent planning reused a provider lease"
                )
            raw_leases.append(backend)

        class LedgerBackend:
            def invoke(self, invocation):
                invocation_ref = corpus.put_artifact(
                    serialize(invocation),
                    kind="plan_competition_prompts",
                )
                try:
                    if cancelled():
                        result = InvocationResult(
                            final_message="",
                            transcript=(),
                            usage=Usage(),
                            stop_reason="cancelled",
                            error=ExternalCancellation(
                                cause="workflow",
                                message="cancelled before provider transport",
                            ),
                            completeness=ExecutionCompleteness(
                                token_usage_complete=True,
                                cost_complete=True,
                                terminal_complete=True,
                                trace_complete=True,
                            ),
                            execution_trace=BackendExecutionTrace(
                                backend=str(getattr(backend, "name", "unknown")),
                                requested_model=str(
                                    getattr(backend, "model", "unknown")
                                ),
                                events=(
                                    BackendExecutionEvent(
                                        sequence=0,
                                        kind="terminal",
                                        name="cancelled_before_transport",
                                    ),
                                ),
                            ),
                            pricing_evidence=UsagePricingEvidence(
                                segments=(), complete=True
                            ),
                        )
                    else:
                        result = backend.invoke(invocation)
                except BaseException:
                    with calls_lock:
                        calls.append((Usage(), ExecutionCompleteness(), None))
                        report["calls"].append(
                            {
                                "role": role,
                                "usage_complete": False,
                                "invocation_ref": str(invocation_ref),
                            }
                        )
                    raise
                # Capture physical-call usage before a workflow wrapper or
                # provider cleanup can fail. Never double-add the aggregate
                # fixed-candidate/selection outcome to this call ledger.
                with calls_lock:
                    calls.append(
                        (result.usage, result.completeness, result.pricing_evidence)
                    )
                    physical_receipts.append(
                        {
                            "role": role,
                            "backend": str(getattr(backend, "name", "unknown")),
                            "model": getattr(backend, "model", None),
                            "usage": {
                                name: getattr(result.usage, name)
                                for name in (
                                    "input_tokens",
                                    "output_tokens",
                                    "cache_creation_tokens",
                                    "cache_read_tokens",
                                    "cost_usd",
                                )
                            },
                            "pricing_evidence": CostAccumulator._pricing_evidence_dict(
                                result.pricing_evidence
                            )
                            if result.pricing_evidence is not None
                            else None,
                        }
                    )
                result_ref = corpus.put_artifact(
                    serialize(result),
                    kind="plan_competition_calls",
                )
                with calls_lock:
                    report["calls"].append(
                        {
                            "role": role,
                            "result_ref": str(result_ref),
                            "invocation_ref": str(invocation_ref),
                        }
                    )
                return result

            def __getattr__(self, name):
                return getattr(backend, name)

        ledger_backend = LedgerBackend()
        return (
            wrap_backend(role, ledger_backend, toolset)
            if wrap_backend
            else ledger_backend
        )

    def cleanup_failure() -> BackendCleanupError:
        return BackendCleanupError(
            tuple(
                (label, managed.close_error or RuntimeError("reviewer cleanup pending"))
                for label, managed in owned_reviewers
                if not managed.closed
            ),
            retry_cleanup=settle_reviewers,
        )

    def settle_reviewers() -> None:
        # The retry callback retains the entire exclusive lease set. Nested
        # context unwinding must not hide an earlier Judge cleanup failure
        # when the enclosing Critic close fails as well.
        for _label, managed in owned_reviewers:
            managed.close()
        if any(not managed.closed for _label, managed in owned_reviewers):
            raise cleanup_failure()

    @contextmanager
    def reviewer(role: str, label: str) -> Iterator[Any]:
        if cancelled():
            raise _CompetitionCancelled(
                "cancelled before constructing another reviewer"
            )
        debug_dir = run_dir / "plan_competition_reviews" / label
        debug_dir.mkdir(parents=True, exist_ok=False)
        backend = pool.new_exclusive(
            role,
            toolset=None,
            log_dir=debug_dir,
            cwd=str(root),
            capability="text",
        )
        managed = _ManagedBackend(backend)
        owned_reviewers.append((label, managed))
        watcher_stop = threading.Event()

        def watch_cancel() -> None:
            while not watcher_stop.wait(0.025):
                if cancelled():
                    managed.close()
                    return

        watcher = threading.Thread(target=watch_cancel, name="plan-review-cancellation")
        if cancel_event is not None:
            watcher.start()
        try:
            yield observed(role, backend)
        finally:
            watcher_stop.set()
            if watcher.ident is not None:
                watcher.join()
            managed.close()
            if not managed.closed:
                raise cleanup_failure()

    def review_candidate(candidate_id: str, dag: Any, cap: float) -> PlanOutcome:
        # Both reviewers receive a fresh, independent session. Candidate
        # review never calls a Planner and therefore cannot rewrite a draft.
        with (
            reviewer(critic_role, candidate_id + "_critic") as critic,
            reviewer(
                judge_role,
                candidate_id + "_judge",
            ) as judge,
        ):
            outcome = plan_workflow.run_plan(
                spec,
                candidate_dag=dag,
                critic_backend=critic,
                judge_backend=judge,
                corpus=corpus,
                critic_rubric=critic_rubric,
                judge_rubric=judge_rubric,
                cwd=str(root),
                max_turns=max_turns,
                budget_usd=cap,
                candidate_transform=candidate_transform,
                candidate_validator=validate_candidate,
                max_candidate_validation_repairs=0,
            )
        add_refs(outcome.spec.verdict_refs[len(spec.verdict_refs) :])
        candidate = outcome.candidate_dag
        if candidate is not None:
            verdict = validate(
                replace(spec, tier_dag=candidate, implementation_refs=()),
                ruleset=ValidationRuleset(
                    name="competition_candidate", rule_ids=available_rules()
                ),
                ctx=RuleContext(workspace_root=root, submodule_path=None),
            )
            ref = corpus.put_record(verdict, kind="verdicts")
            add_refs((ref,))
            outcome = replace(
                outcome,
                spec=replace(
                    outcome.spec,
                    verdict_refs=(*outcome.spec.verdict_refs, ref),
                ),
            )
            if not verdict.passed:
                outcome = replace(
                    outcome,
                    decision="rejected",
                    status="validate_blocked",
                    reason="candidate deterministic validation failed",
                    spec=replace(outcome.spec, tier_dag=spec.tier_dag),
                )
        return outcome

    try:
        snapshot_root = create_plan_source_snapshot(
            root,
            run_dir / "plan_source_snapshot",
            excluded_roots=(Path(corpus.spec_dir), run_dir),
        )
        if not source_unchanged():
            raise ValueError(
                "source identity changed while capturing independent planning input"
            )
        requests = tuple(
            ReadOnlyChildRequest(
                child_id=f"plan_{i + 1}",
                role=f"independent_plan_{i + 1}",
                route_role=planner_role,
                cwd=snapshot_root,
                invocation=Invocation(
                    system="",
                    user="",
                    tools=None,
                    max_turns=max_turns,
                    budget_usd=None,
                ),
            )
            for i in range(planner_count)
        )

        def tools(_request, context):
            if not enable_source_tools:
                return ToolSetBuilder("plan_draft_empty").build(context)
            return (
                source_toolset_factory(context)
                if source_toolset_factory
                else build_source_tools(context)
            )

        def backend(request, toolset, debug_dir):
            lease = pool.new_exclusive(
                request.route_role,
                toolset=toolset,
                log_dir=debug_dir,
                cwd=str(request.cwd),
                context_budget=policy.max_child_context_bytes,
                feature_requirements=(
                    FeatureRequirement(
                        "cancellation",
                        frozenset({"interruptible", "boundary_only", "process_kill"}),
                    ),
                ),
            )
            return _DraftBackend(
                observed(request.route_role, lease, toolset), spec, toolset
            )

        try:
            drafted = run_read_only_fanout(
                requests,
                policy=policy,
                backend_factory=backend,
                toolset_factory=tools,
                debug_root=run_dir / "independent_planners",
                parent_budget_usd=budget_usd,
                cancel_event=cancel_event,
            )
        except ReadOnlyCleanupError as exc:
            report["drafts"] = exc.partial_result.to_summary()
            status, reason = (
                "cleanup_failed",
                "independent planner cleanup retains live ownership",
            )
            raise
        report["drafts"] = drafted.to_summary()
        if cancelled() or drafted.status == "cancelled":
            status, reason = (
                "cancelled",
                "independent planning was cancelled and joined",
            )
        elif any(
            child.error_kind == "ReadOnlyIsolationError" for child in drafted.children
        ):
            status, reason = (
                "plan_competition_isolation_failed",
                "independent drafts reused provider state",
            )
        elif (
            not all(child.evidence_complete for child in drafted.children)
            or not receipts_complete()
        ):
            status, reason = (
                "budget_accounting_incomplete",
                "a draft lacks complete execution or accounting evidence",
            )
        elif any(
            child.error_kind == "ChildBudgetShareExceeded" for child in drafted.children
        ):
            status, reason = (
                "budget_exhausted",
                "a draft exceeded its reserved total-budget share",
            )
        elif not source_unchanged():
            status, reason = (
                "validate_blocked",
                "source identity changed during independent drafting",
            )
        else:
            for child in drafted.children:
                row: dict[str, Any] = {"id": child.child_id, "status": child.status}
                report["candidates"].append(row)
                if not child.successful or child.result is None:
                    row["reason"] = child.error_message
                    continue
                raw_ref = corpus.put_artifact(
                    child.result.final_message.encode("utf-8"),
                    kind="plan_candidates",
                    name_hint=child.child_id + "_raw.json",
                )
                row["raw_ref"] = str(raw_ref)
                try:
                    dag = _decode_authored_dag(
                        child.result.final_message.encode("utf-8"), spec, str(root)
                    )
                except ValueError as exc:
                    row.update(status="invalid_candidate", reason=str(exc))
                    continue
                cap = remaining(_REVIEW_SHARE / planner_count)
                if cancelled() or cap <= 0:
                    status, reason = (
                        "cancelled" if cancelled() else "budget_exhausted",
                        "stopped before candidate review",
                    )
                    break
                calls_before_review = len(calls)
                outcome = review_candidate(child.child_id, dag, cap)
                diagnostic_dag, diagnostic_ref = (
                    outcome.candidate_dag,
                    outcome.candidate_ref,
                )
                outcomes[child.child_id] = outcome
                refs = outcome.spec.verdict_refs[len(spec.verdict_refs) :]
                row.update(
                    status=outcome.status,
                    accepted=outcome.accepted,
                    candidate_ref=str(outcome.candidate_ref or ""),
                    digest=str(hash_record(outcome.candidate_dag))
                    if outcome.candidate_dag
                    else None,
                    verdict_refs=[str(ref) for ref in refs],
                    reason=outcome.reason,
                )
                save_progress()
                if cancelled():
                    status, reason = "cancelled", "cancelled during candidate review"
                    break
                if (
                    len(calls) > calls_before_review
                    and not complete(outcome.execution_completeness)
                ) or not receipts_complete():
                    status, reason = (
                        "budget_accounting_incomplete",
                        "a candidate review lacks complete execution evidence",
                    )
                    break
                if outcome.cost_usd > cap + 1e-12:
                    status, reason = (
                        "budget_exhausted",
                        "a candidate review exceeded its reserved budget share",
                    )
                    break
                if outcome.candidate_dag is not None:
                    kinds = {
                        corpus.get_record(ref, as_type=Verdict).kind for ref in refs
                    }
                    if {"critic", "judge"} <= kinds:
                        eligible.append(
                            PlanSelectionCandidate(
                                id=child.child_id,
                                dag=outcome.candidate_dag,
                                accepted=outcome.accepted,
                                verdict_refs=tuple(str(ref) for ref in refs),
                                reason=outcome.reason,
                            )
                        )
            else:
                if eligible and not cancelled() and source_unchanged():
                    cap = remaining(_SELECTION_SHARE)
                    if cap > 0:
                        calls_before_comparison = len(calls)
                        with reviewer(judge_role, "comparison") as judge:
                            selection = select_plan_candidate(
                                spec,
                                tuple(eligible),
                                backend=judge,
                                corpus=corpus,
                                budget_usd=cap,
                                max_turns=max_turns,
                                project_root=str(root),
                            )
                        verdict = selection.verdict
                        add_refs((selection.verdict_ref,))
                        report["comparison"] = {
                            "mode": selection.mode,
                            "selected_id": selection.selected_id,
                            "verdict_ref": str(selection.verdict_ref),
                            "selection_ref": str(selection.selection_ref),
                            "bundle_ref": str(selection.bundle_ref),
                            "error": selection.error,
                        }
                        if cancelled():
                            status, reason = "cancelled", "cancelled during comparison"
                        elif len(calls) > calls_before_comparison and (
                            not complete(verdict.execution_completeness)
                            or not receipts_complete()
                        ):
                            status, reason = (
                                "budget_accounting_incomplete",
                                "comparison evidence is incomplete",
                            )
                        elif verdict.cost_usd > cap + 1e-12:
                            status, reason = (
                                "budget_exhausted",
                                "comparison exceeded its budget share",
                            )
                        elif selection.error is not None:
                            reason = selection.error
                        elif selection.mode == "select":
                            winner = (
                                outcomes.get(selection.selected_id)
                                if selection.selected_id is not None
                                else None
                            )
                            report["selected_id"] = selection.selected_id
                        elif (
                            selection.mode == "synthesize"
                            and selection.candidate_dag is not None
                        ):
                            cap = remaining(_SYNTHESIS_REVIEW_SHARE)
                            if cap > 0 and not cancelled():
                                winner = review_candidate(
                                    "synthesis", selection.candidate_dag, cap
                                )
                                report["synthesis"] = {
                                    "status": winner.status,
                                    "accepted": winner.accepted,
                                    "candidate_ref": str(winner.candidate_ref or ""),
                                    "verdict_refs": [
                                        str(ref)
                                        for ref in winner.spec.verdict_refs[
                                            len(spec.verdict_refs) :
                                        ]
                                    ],
                                }
                                if cancelled():
                                    status, reason, winner = (
                                        "cancelled",
                                        "cancelled during synthesis review",
                                        None,
                                    )
                                elif (
                                    not complete(winner.execution_completeness)
                                    or not receipts_complete()
                                    or winner.cost_usd > cap + 1e-12
                                ):
                                    status, reason, winner = (
                                        "budget_exhausted",
                                        "synthesis review exceeded its evidence or budget allowance",
                                        None,
                                    )
                        else:
                            reason = "comparison selected no acceptable plan"
                elif not source_unchanged():
                    status, reason = (
                        "validate_blocked",
                        "source identity changed before comparison",
                    )
        if winner is not None and winner.accepted:
            aggregate = plan_workflow._plan_failure_usage_fields(calls)
            if not complete(aggregate["execution_completeness"]):
                status, reason, winner = (
                    "budget_accounting_incomplete",
                    "complete competition accounting is required for acceptance",
                    None,
                )
            elif aggregate["cost_usd"] > budget_usd + 1e-12:
                status, reason, winner = (
                    "budget_exhausted",
                    "competition exceeded the shared total budget",
                    None,
                )
            elif cancelled():
                status, reason, winner = (
                    "cancelled",
                    "cancelled before final selection",
                    None,
                )
            elif not source_unchanged():
                status, reason, winner = (
                    "validate_blocked",
                    "source identity or cancellation changed before final selection",
                    None,
                )
            else:
                status, reason = (
                    "accepted",
                    "independent plans compared; selected candidate passed all gates",
                )
        if winner is not None and not winner.accepted:
            status, reason = winner.status, winner.reason
            diagnostic_dag, diagnostic_ref = winner.candidate_dag, winner.candidate_ref
            winner = None
        if winner is None and status == "plan_competition_inconclusive":
            # No typed stop fired and nothing was selected: name the lost call
            # when the records prove that one ended the run without a ruling,
            # else keep the status.
            compared = {candidate.id for candidate in eligible}
            named = _terminal_without_ruling(
                _draft_ends(
                    drafted.children,
                    report["candidates"],
                    outcomes,
                    compared,
                    corpus,
                    plan_workflow._review_control_failure_status,
                ),
                compared,
                selection,
                cancelled(),
                corpus,
                plan_workflow._review_control_failure_status,
            )
            if named is not None:
                status, reason = named
    except _CompetitionCancelled as exc:
        status, reason, winner = "cancelled", str(exc), None
    except BackendCleanupError:
        status, reason = (
            "cleanup_failed",
            "reviewer cleanup retains exclusive provider ownership",
        )
        raise
    finally:
        report.update(status=status, reason=reason)
        save_progress()

    fields = plan_workflow._plan_failure_usage_fields(calls)
    if (
        calls
        and not receipts_complete()
        and fields["execution_completeness"] is not None
    ):
        fields["execution_completeness"] = replace(
            fields["execution_completeness"], cost_complete=False
        )
    if call_ledger is not None and call_ledger.pricing_evidence is not None:
        ledger_now = call_ledger.to_dict()
        delta = ledger_now["pricing_evidence_by_call"][initial_call_count:]
        key = lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"))
        if Counter(map(key, delta)) != Counter(map(key, physical_receipts)):
            call_ledger.note_accounting_debt(
                role="plan_competition_reconciliation",
                missing=("cost_complete", "token_usage_complete"),
            )
        else:
            # The exact role/model/usage/pricing receipt multiset has been
            # independently reconciled. Normalize its float projection to the
            # authoritative serialized workflow ledger before publication;
            # fsum vs += alone must not manufacture post-publication debt.
            fields["cost_usd"] = ledger_now["total_usd"] - initial_ledger.get(
                "total_usd", 0.0
            )
        # Simultaneous responses have no source request order. Preserve the
        # workflow ledger's serialized receipt order, which the publication
        # owner independently reconciles against every physical response.
        ordered = call_ledger.pricing_evidence
        evidence = UsagePricingEvidence(
            segments=ordered.segments[initial_segment_count:],
            complete=ordered.complete,
        )
        if evidence.reconciles(
            Usage(
                input_tokens=fields["input_tokens"],
                output_tokens=fields["output_tokens"],
                cost_usd=fields["cost_usd"],
                cache_creation_tokens=fields["cache_creation_tokens"],
                cache_read_tokens=fields["cache_read_tokens"],
            )
        ):
            fields["pricing_evidence"] = evidence
    if winner is not None and (
        not complete(fields["execution_completeness"])
        or (
            fields["pricing_evidence"] is not None
            and not fields["pricing_evidence"].complete
        )
        or fields["cost_usd"] > budget_usd + 1e-12
        or (call_ledger is not None and call_ledger.to_dict()["rollup_incomplete"])
    ):
        winner = None
        status, reason = (
            "budget_accounting_incomplete",
            "physical-call receipts do not establish complete competition accounting",
        )
    report.update(status=status, reason=reason)
    report["cost_usd"] = fields["cost_usd"]
    report["accounting_complete"] = complete(fields["execution_completeness"])
    save_progress(fields["cost_usd"])
    report_ref = corpus.put_artifact(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        kind="plan_competitions",
        name_hint="plan_competition.json",
    )
    event = LifecycleEvent(
        kind="plan_competition",
        reason=json.dumps(
            {"report_ref": str(report_ref), "source_spec": str(hash_record(spec))},
            sort_keys=True,
        ),
        affected_versions=(spec.version + 1,),
        execution=exec_ctx_now("plan_competition"),
    )
    event_ref = corpus.put_record(event, kind="lifecycle_events")
    if winner is not None:
        staged = replace(
            winner.spec,
            verdict_refs=tuple(verdict_refs),
            lifecycle_event_refs=(*winner.spec.lifecycle_event_refs, event_ref),
        )
        return replace(winner, spec=staged, reason=reason, **fields)
    terminal_ref = corpus.put_record(
        LifecycleEvent(
            kind="plan_inconclusive",
            reason=reason,
            affected_versions=(spec.version + 1,),
            execution=exec_ctx_now("plan_competition"),
        ),
        kind="lifecycle_events",
    )
    staged = spec.with_appended_refs(
        verdict_refs=tuple(verdict_refs[len(spec.verdict_refs) :]),
        lifecycle_event_refs=(event_ref, terminal_ref),
    )
    return PlanOutcome(
        spec=staged,
        candidate_dag=diagnostic_dag,
        candidate_ref=diagnostic_ref,
        decision="inconclusive",
        status=status,
        rounds_used=planner_count,
        final_critic_decision=None,
        final_judge_decision=None,
        reason=reason,
        **fields,
    )
