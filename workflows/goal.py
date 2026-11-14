# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Iterative, corpus-native goal refinement: Red -> Blue -> Judge.

Blue emits one complete canonical goal replacement.  The workflow parses and
deterministically validates that replacement, lets the Judge adjudicate the
complete evidence chain, and stages valid ``revise`` candidates in memory for
the next round.  Only a candidate that passes both deterministic validation and
the independent Judge is promoted into the authoritative Spec lineage.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from assay import LifecycleEvent, Spec, ValidationRuleset, Verdict
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.validate import validate

from crucible.common.cost import CostAccumulator, CostLimitExceeded
from crucible.rules import available as available_rules
from crucible.rules import context_aware as context_aware_rules
from crucible.rubrics import load as load_rubric

from ._analysis_handoff import (
    attach_artifacts,
    decision_exit_code,
    enforce_budget_overrun,
    handoff_artifact,
    incomplete_adjudication_reason,
    incomplete_adjudication_rubric,
    persist_output,
    record_cost,
    remaining_budget,
    terminal_verdict,
)
from ._args import (
    add_backend_config_arg,
    add_budget_arg,
    add_max_turns_arg,
    add_spec_arg,
    positive_int_arg,
)
from ._backends import backend_pool
from ._run import exec_ctx_now, open_run_with_capture, record_handled_terminal
from ._review import review

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crucible-workflow-goal")
    add_spec_arg(parser)
    parser.add_argument("--max-rounds", type=positive_int_arg, default=3)
    add_budget_arg(parser, kind="cumulative")
    add_backend_config_arg(parser, "goal_red", "goal_blue", "goal_judge")
    add_max_turns_arg(parser, default=10, scope="per review invocation")
    return parser.parse_args(argv)


def _plain(value: Any) -> Any:
    """Convert immutable Assay containers into YAML/JSON-native values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _dump_kinded(value) -> dict:
    return {
        field.name: _plain(getattr(value, field.name))
        for field in dataclasses.fields(value)
    }


def _project_payload(spec: Spec) -> dict[str, Any]:
    project = spec.project_context
    return {
        "project": project.project,
        "repo_url": project.repo_url,
        "base_commit": project.base_commit,
        "branch": project.branch,
        "submodule_path": project.submodule_path,
        "runners": _plain(project.runners),
    }


def _goal_payload(spec: Spec) -> dict[str, Any]:
    return {
        "intent": spec.intent,
        "constraints": [_dump_kinded(item) for item in spec.constraints],
        "done_criteria": [_dump_kinded(item) for item in spec.done_criteria],
    }


def _render_goal(spec: Spec) -> bytes:
    payload = {**_goal_payload(spec), "project_context": _project_payload(spec)}
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def _parse_canonical_revision(spec: Spec, blue: Verdict) -> Spec:
    proposals = [
        finding.proposal.replacement
        for finding in blue.findings
        if finding.proposal is not None
        and finding.proposal.replacement is not None
        and finding.category == "canonical_revision"
    ]
    if len(proposals) != 1:
        raise ValueError(
            "Blue must emit exactly one canonical_revision replacement"
        )
    raw = json.loads(proposals[0])
    required = {"intent", "constraints", "done_criteria"}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError(
            "canonical revision must contain exactly intent, constraints, "
            "and done_criteria"
        )
    if not isinstance(raw["intent"], str) or not raw["intent"].strip():
        raise ValueError("canonical revision intent must be a non-empty string")
    if not isinstance(raw["constraints"], list):
        raise ValueError("canonical revision constraints must be an array")
    if not isinstance(raw["done_criteria"], list):
        raise ValueError("canonical revision done_criteria must be an array")

    # Reuse Spec's authoritative kind registry/parser, then transplant only the
    # authored goal fields into a preview.  Lineage advances only after Judge.
    parsed = Spec.from_yaml(yaml.safe_dump({
        "schema_version": spec.schema_version,
        "project_context": _project_payload(spec),
        **raw,
    }, sort_keys=False))
    canonical = _goal_payload(parsed)
    if canonical != raw:
        raise ValueError(
            "canonical revision contains unknown, omitted, or non-canonical "
            "constraint/criterion fields"
        )
    return dataclasses.replace(
        spec,
        intent=parsed.intent,
        constraints=parsed.constraints,
        done_criteria=parsed.done_criteria,
    )


def _promote_goal(spec: Spec, candidate: Spec) -> Spec:
    return spec._next_version(  # noqa: SLF001 - Spec exposes no goal-specific fork
        intent=candidate.intent,
        constraints=candidate.constraints,
        done_criteria=candidate.done_criteria,
    )


@dataclasses.dataclass
class _Round:
    index: int
    red: Verdict
    blue: Verdict | None = None
    validation: Verdict | None = None
    judge: Verdict | None = None
    staged: bool = False
    promoted: bool = False
    validation_error: str | None = None


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    corpus = Corpus(args.spec)
    spec = corpus.latest_spec()
    if spec is None:
        message = f"no spec manifest found under {args.spec / 'versions'}"
        record_handled_terminal(
            corpus, "goal", status="setup_error",
            message=message, error_kind="missing_spec",
            provenance_root=None,
        )
        return 2

    with open_run_with_capture(
        corpus,
        "goal",
        provenance_root=None,
    ) as handle, \
            backend_pool(args, handle.run_dir) as pool:
        red_backend = pool.get("goal_red")
        blue_backend = pool.get("goal_blue")
        judge_backend = pool.get("goal_judge")
        red_rubric = load_rubric("goal_red")
        blue_rubric = load_rubric("goal_blue")
        judge_rubric = load_rubric("goal_judge")
        costs = CostAccumulator(max_cost_usd=args.budget_usd)
        rounds: list[_Round] = []
        history: list[tuple[str, Verdict]] = []
        final_judge: Verdict | None = None
        draft_spec = spec
        approved_candidate: Spec | None = None

        for index in range(args.max_rounds):
            goal_artifact = handoff_artifact(
                _render_goal(draft_spec), history,
                heading="PRIOR GOAL-REFINEMENT ROUNDS",
            )
            try:
                red = review(
                    goal_artifact, red_rubric, backend=red_backend,
                    corpus=corpus, max_turns=args.max_turns,
                    budget_usd=remaining_budget(costs), kind="critic",
                )
            except CostLimitExceeded:
                final_judge = terminal_verdict(
                    corpus,
                    goal_artifact,
                    rubric=incomplete_adjudication_rubric(
                        costs,
                        workflow="goal",
                    ),
                    decision="inconclusive",
                    reasoning=incomplete_adjudication_reason(
                        costs,
                        budget_reason=(
                            "cumulative budget exhausted before Red review"
                        ),
                    ),
                )
                break
            current = _Round(index=index, red=red)
            rounds.append(current)
            history.append((f"round_{index}.red", red))
            if not record_cost(
                costs, role="goal_red", usage=red, backend=red_backend,
            ):
                final_judge = terminal_verdict(
                    corpus,
                    goal_artifact,
                    rubric=incomplete_adjudication_rubric(
                        costs,
                        workflow="goal",
                    ),
                    decision="inconclusive",
                    reasoning=incomplete_adjudication_reason(
                        costs,
                        budget_reason=(
                            "cumulative budget exhausted after Red review"
                        ),
                    ),
                )
                break

            blue_artifact = handoff_artifact(
                _render_goal(draft_spec), (("red", red),),
            )
            blue = review(
                blue_artifact, blue_rubric, backend=blue_backend,
                corpus=corpus, max_turns=args.max_turns,
                budget_usd=remaining_budget(costs), kind="critic",
            )
            current.blue = blue
            history.append((f"round_{index}.blue", blue))
            if not record_cost(
                costs, role="goal_blue", usage=blue, backend=blue_backend,
            ):
                final_judge = terminal_verdict(
                    corpus,
                    blue_artifact,
                    rubric=incomplete_adjudication_rubric(
                        costs,
                        workflow="goal",
                    ),
                    decision="inconclusive",
                    reasoning=incomplete_adjudication_reason(
                        costs,
                        budget_reason=(
                            "cumulative budget exhausted after Blue review"
                        ),
                    ),
                )
                break

            try:
                candidate = _parse_canonical_revision(draft_spec, blue)
                validation = validate(
                    candidate,
                    ruleset=ValidationRuleset(
                        name="goal_revision_gate",
                        rule_ids=tuple(
                            rule_id for rule_id in available_rules()
                            if rule_id not in set(context_aware_rules())
                        ),
                    ),
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                current.validation_error = str(exc)
                validation = terminal_verdict(
                    corpus, blue_artifact, rubric="goal_revision_gate",
                    decision="reject",
                    reasoning=f"canonical revision failed validation: {exc}",
                    kind="validate",
                )
                candidate = draft_spec
            current.validation = validation
            history.append((f"round_{index}.validation", validation))

            judge_artifact = handoff_artifact(
                _render_goal(candidate),
                history,
                heading="COMPLETE GOAL-REFINEMENT HISTORY",
            )
            judge = review(
                judge_artifact, judge_rubric, backend=judge_backend,
                corpus=corpus, max_turns=args.max_turns,
                budget_usd=remaining_budget(costs), kind="judge",
            )
            current.judge = judge
            final_judge = judge
            history.append((f"round_{index}.judge", judge))
            record_cost(
                costs, role="goal_judge", usage=judge, backend=judge_backend,
            )
            judge = enforce_budget_overrun(judge, costs)
            if validation.decision != "pass" and judge.decision == "pass":
                reason = (
                    "deterministic goal validation did not pass; a Judge pass "
                    "cannot promote an invalid candidate"
                )
                judge = dataclasses.replace(
                    judge,
                    decision="inconclusive",
                    reasoning=f"{reason}. {judge.reasoning}",
                    coverage=dataclasses.replace(
                        judge.coverage,
                        complete=False,
                        omissions=judge.coverage.omissions + (reason,),
                    ),
                )
            current.judge = judge
            final_judge = judge
            history[-1] = (f"round_{index}.judge", judge)

            if validation.decision == "pass" and judge.decision == "pass":
                # Stage approval until the final evidence set is known.  The
                # candidate and every review record are published in one Corpus
                # commit below, so a crash cannot expose a revised goal without
                # the Judge/validation records that authorized it.
                approved_candidate = candidate
                current.promoted = True
                draft_spec = candidate
            elif validation.decision == "pass" and judge.decision == "revise":
                # Carry a valid intermediate candidate into the next debate
                # round without publishing it as authoritative corpus state.
                draft_spec = candidate
                current.staged = True

            handle.write_event({
                "round": index,
                "red_decision": red.decision,
                "blue_decision": blue.decision,
                "validation_decision": validation.decision,
                "judge_decision": judge.decision,
                "staged": current.staged,
                "promoted": current.promoted,
                "cost_usd": costs.total_cost_usd,
            })
            if judge.decision != "revise":
                break

        if final_judge is None:
            final_judge = terminal_verdict(
                corpus, _render_goal(spec), rubric="goal_round_limit",
                decision="inconclusive",
                reasoning="round limit reached without an adjudicated decision",
            )

        final_goal = _render_goal(approved_candidate or spec)
        goal_ref = persist_output(
            corpus, handle.run_dir / "final_goal.yaml", final_goal,
            kind="goal_outputs",
        )
        final_judge = attach_artifacts(final_judge, final_goal=goal_ref)
        if rounds and rounds[-1].judge is not None:
            rounds[-1].judge = final_judge

        # Attach every actual review/validation record and any approved goal
        # replacement in one visibility commit. Content-addressed blobs may be
        # left unreachable by a process death, but partial authority cannot be
        # published in the Spec lineage.
        persisted_ids: set[int] = set()
        verdicts_to_commit: list[Verdict] = []
        for round_state in rounds:
            for verdict in (
                round_state.red,
                round_state.blue,
                round_state.validation,
                round_state.judge,
            ):
                if verdict is not None and id(verdict) not in persisted_ids:
                    verdicts_to_commit.append(verdict)
                    persisted_ids.add(id(verdict))
        if id(final_judge) not in persisted_ids:
            verdicts_to_commit.append(final_judge)

        def _publish_goal(candidate_spec: Spec) -> Spec:
            if approved_candidate is None:
                return candidate_spec
            return dataclasses.replace(
                candidate_spec,
                intent=approved_candidate.intent,
                constraints=approved_candidate.constraints,
                done_criteria=approved_candidate.done_criteria,
            )

        decision = final_judge.decision
        spec = corpus.commit(
            verdicts=tuple(verdicts_to_commit),
            transform=_publish_goal,
            lifecycle_event_factory=lambda committed_version: LifecycleEvent(
                kind="landed" if decision == "pass" else "abandoned",
                reason=(
                    f"goal refinement {decision}; final_goal_ref={goal_ref}"
                ),
                affected_versions=(committed_version,),
                execution=exec_ctx_now(
                    "goal", run_id=handle.run_dir.name,
                ),
            ),
        ).spec

        summary = {
            "status": decision,
            "decision": decision,
            "rounds_run": len(rounds),
            "max_rounds": args.max_rounds,
            "promotions": sum(item.promoted for item in rounds),
            "staged_revisions": sum(item.staged for item in rounds),
            "final_goal_ref": goal_ref,
            "cost_usd": costs.total_cost_usd,
            "costs": costs.to_dict(),
            "version": spec.version,
            "rounds": [
                {
                    "round": item.index,
                    "red_decision": item.red.decision,
                    "blue_decision": item.blue.decision if item.blue else None,
                    "validation_decision": (
                        item.validation.decision if item.validation else None
                    ),
                    "judge_decision": item.judge.decision if item.judge else None,
                    "staged": item.staged,
                    "promoted": item.promoted,
                    "validation_error": item.validation_error,
                }
                for item in rounds
            ],
        }
        handle.write_summary(summary)
        handle.finalize_elapsed(RunResult(
            status=decision, cost_usd=costs.total_cost_usd,
        ))
        return decision_exit_code(decision)


if __name__ == "__main__":
    sys.exit(main())
