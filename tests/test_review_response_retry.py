# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded repair retries for malformed review responses.

These tests exercise the Crucible adapter through the real Assay review
parser.  Only the paid backend is faked, which keeps prompt, transcript, and
corpus provenance assertions representative of production behavior.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
)
from assay.corpus.store import Corpus
from assay.primitives.review import _MAX_ARTIFACT_CHARS

from crucible.rubrics import load
from crucible.tools._dispatch_budget import ProgressBoundedMutatingToolSet
from crucible.workflows._review import (
    _compact_planning_history_context,
    _context_prior_findings,
    _deterministic_terminal_reason,
    _obligations_rubric,
    _retryable_output_length_error,
    _tool_source_provider,
    _verified_plan_gate_closures,
    review,
)


_ARTIFACT = b"ORIGINAL REVIEW INPUT -- retain this exact source"
_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="review-response-retry-stub",
    requested_model="review-response-retry-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


def test_progress_snapshot_is_not_duck_typed_as_review_source_evidence():
    class _MutatingToolSet:
        read_only_names = {"read"}
        write_names = {"write"}

        def to_anthropic_tools(self):
            return [
                {
                    "name": name,
                    "description": name,
                    "input_schema": {"type": "object"},
                }
                for name in ("read", "write")
            ]

        def to_openai_tools(self):
            return [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": name,
                        "parameters": {"type": "object"},
                    },
                }
                for name in ("read", "write")
            ]

        def dispatch(self, name, arguments):
            return "ok"

    attached = ProgressBoundedMutatingToolSet(
        _MutatingToolSet(),
        max_consecutive_read_calls=1,
    )
    backend = SimpleNamespace(attached_toolset=attached)

    assert _tool_source_provider(backend)() == {}


def _pass_response() -> dict[str, object]:
    return {
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The complete original input satisfies the rubric.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
    }


def _schema_failure_response(key: str, private_value: str) -> str:
    payload = _pass_response()
    payload[key] = private_value
    return json.dumps(payload)


def _semantic_inconclusive_response() -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "inconclusive",
        "reasoning": "The evidence supports no decisive outcome.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
    })


def _crucible_contract_failure_response() -> str:
    """Valid review_v2, but invalid for the spec_clarity vocabulary."""

    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "PRIVATE_BAD_REASONING",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [{
            "id": "outside-rubric-vocabulary",
            "criterion_id": "scope.in_bounds",
            "severity": "High",
            "confidence": 1.0,
            "disposition": "confirmed",
            "category": "scope",
            "targets": [],
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "artifact",
                "excerpt": "ORIGINAL REVIEW INPUT",
            }],
            "proposal": {
                "action": "Use only the declared rubric vocabulary.",
                "replacement": None,
            },
            "attack_spec": None,
        }],
    })


def _workflow_canonicalization_response() -> str:
    """Valid rubric response with a source-exact but overbroad quotation."""

    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "The input needs one clarification.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [{
            "id": "clarify-input",
            "criterion_id": "intent_clarity",
            "severity": "High",
            "confidence": 1.0,
            "disposition": "confirmed",
            "category": "ambiguity",
            "targets": [],
            "evidence": [{
                "source_id": "artifact",
                "kind": "source_quote",
                "location": "artifact",
                "excerpt": _ARTIFACT.decode(),
            }],
            "proposal": {
                "action": "Clarify the retained input.",
                "replacement": None,
            },
            "attack_spec": None,
        }],
    })


def _canonicalize_workflow_excerpt(candidate):
    finding = candidate.findings[0]
    evidence = finding.evidence[0]
    replacement = dataclasses.replace(
        candidate,
        findings=(dataclasses.replace(
            finding,
            evidence=(dataclasses.replace(
                evidence,
                excerpt="ORIGINAL REVIEW INPUT",
            ),),
        ),),
    )
    return replacement, ({
        "operation": "canonicalize_test_excerpt",
        "path": "findings[0].evidence[0].excerpt",
    },)


_PLAN_ARTIFACT = json.dumps({
    "tiers": [{
        "tier_id": "t0",
        "depends_on": [],
        "slices": [
            {
                "id": "config-load",
                "directive": {
                    "slice_id": "config-load",
                    "intent": "Load configuration values",
                    "files_in_scope": ["astropy/table/tests/test_mixin.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            },
            {
                "id": "alias-validation",
                "directive": {
                    "slice_id": "alias-validation",
                    "intent": "Validate alias mappings",
                    "files_in_scope": ["src/aliases.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            },
        ],
    }],
}, separators=(",", ":")).encode()
_CANONICAL_PLAN_ARTIFACT = json.dumps({
    "tiers": [{
        "tier_id": "t0",
        "depends_on": [],
        "slices": [
            {
                "id": "config-load",
                "directive": {
                    "slice_id": "config-load",
                    "intent": "load configuration",
                    "files_in_scope": ["src/config.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            },
            {
                "id": "alias-validation",
                "directive": {
                    "slice_id": "alias-validation",
                    "intent": "validate aliases",
                    "files_in_scope": ["src/aliases.py"],
                    "creates_paths": [],
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 1},
                },
            },
        ],
    }],
}, indent=2).encode()
_BREAKER_ARTIFACT = b"""diff --git a/astropy/table/table.py b/astropy/table/table.py
@@ -1243,3 +1243,4 @@
+            warnings.warn(
             data = data.view(NdarrayMixin)
"""
_BREAKER_INTENT = (
    "Add a FutureWarning here and change the behavior in 5.2 by removing "
    "this clause."
)
_DEGENERATE_BREAKER_RESPONSE = (
    '{"schema_version":"assay.review.v2","decision":"inconclusive",'
    '"reasoning":"...","coverage":{ "complete": true, '
    '"sources_reviewed": ["artifact","context:spec_intent"], '
    '"omissions": [] }, "findings": []}'
)


def _attack_finding(
    *,
    finding_id: str,
    severity: str,
    issue_location: str = "proposal",
    issue_excerpt: str = _BREAKER_INTENT,
    artifact_excerpt: str = "+            warnings.warn(",
) -> dict[str, object]:
    return {
        "id": finding_id,
        "criterion_id": "meaning",
        "severity": severity,
        "confidence": 0.9,
        "disposition": "confirmed" if severity in {"High", "Critical"} else "suspected",
        "category": "broken_meaning",
        "targets": [],
        "evidence": [
            {
                "source_id": "context:spec_intent",
                "kind": "issue_text",
                "location": issue_location,
                "excerpt": issue_excerpt,
            },
            {
                "source_id": "artifact",
                "kind": "diff",
                "location": "astropy/table/table.py",
                "excerpt": artifact_excerpt,
            },
        ],
        "proposal": None,
        "attack_spec": {
            "setup": "Construct a Table from a structured ndarray.",
            "stimulus": "Add the structured array as a column.",
            "oracle": "The 5.2 behavior adds a Column without NdarrayMixin conversion.",
        },
    }


def _breaker_response(
    *, decision: str, findings: list[dict[str, object]], reasoning: str,
) -> str:
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": decision,
        "reasoning": reasoning,
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact", "context:spec_intent"],
            "omissions": [],
        },
        "findings": findings,
    })


def _impl_breaker_attack_response(
    *,
    finding_id: str,
    setup: str,
    stimulus: str,
    oracle: str,
) -> str:
    return _breaker_response(
        decision="revise",
        reasoning="A grounded correctness attack should be executed.",
        findings=[{
            "id": finding_id,
            "criterion_id": "correctness",
            "severity": "High",
            "confidence": 0.95,
            "disposition": "confirmed",
            "category": "correctness",
            "targets": [],
            "evidence": [
                {
                    "source_id": "context:spec_intent",
                    "kind": "required_behavior",
                    "location": "spec_intent",
                    "excerpt": _BREAKER_INTENT,
                },
                {
                    "source_id": "artifact",
                    "kind": "diff",
                    "location": "artifact",
                    "excerpt": _BREAKER_ARTIFACT.decode(),
                },
            ],
            "proposal": None,
            "attack_spec": {
                "setup": setup,
                "stimulus": stimulus,
                "oracle": oracle,
            },
        }],
    )


_ARTIFACT_FINDING_ID_LEDGER_SCHEMA = (
    "crucible.review-artifact-finding-id-ledger.v1"
)
_ARTIFACT_FINDING_IDS = (
    "correctness-preserve-boolean-threshold",
    "correctness-preserve-integer-threshold",
)
_ARTIFACT_FINDING_ROOT = (
    "A boolean threshold enables the default early-exit behavior, while an "
    "integer threshold exits after exactly that many failures."
)
_ARTIFACT_FINDING_ORACLES = {
    _ARTIFACT_FINDING_IDS[0]: (
        "With a boolean threshold enabled, the default early-exit behavior "
        "remains active."
    ),
    _ARTIFACT_FINDING_IDS[1]: (
        "With an integer threshold of two, execution exits after exactly two "
        "failures."
    ),
}


def _artifact_finding_id_case(
    rubric_name: str,
) -> tuple[bytes, tuple[str, ...]]:
    if rubric_name == "breach_attack_authority":
        payload = {
            "schema_version": "crucible.breach-authority-admission.v1",
            "authority": {
                "schema_version": "crucible.breach-root-authority.v1",
                "authority_digest": "sha256:" + "a" * 64,
                "tier_id": "t0",
                "issue_text": "Preserve threshold early-exit semantics.",
                "slices": [{
                    "slice_id": "thresholds",
                    "requirement_refs": [_ARTIFACT_FINDING_ROOT],
                }],
            },
            "generated_attacks": [
                {
                    "attack_id": attack_id,
                    "severity": "High",
                    "category": "correctness",
                    "generated_non_evidentiary": {
                        "setup": "Configure the threshold.",
                        "stimulus": "Record failures until execution exits.",
                        "oracle": _ARTIFACT_FINDING_ORACLES[attack_id],
                    },
                }
                for attack_id in _ARTIFACT_FINDING_IDS
            ],
        }
    elif rubric_name == "breach_attack_entailment":
        payload = {
            "schema_version": "crucible.breach-attack-entailment.v1",
            "authority_digest": "sha256:" + "a" * 64,
            "attacks": [
                {
                    "attack_id": attack_id,
                    "oracle": _ARTIFACT_FINDING_ORACLES[attack_id],
                    "root_requirements": [{
                        "location": "requirement:thresholds:0",
                        "requirement_ref": _ARTIFACT_FINDING_ROOT,
                        "requirement_index": 0,
                        "slice_id": "thresholds",
                    }],
                }
                for attack_id in _ARTIFACT_FINDING_IDS
            ],
        }
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported artifact-id case {rubric_name!r}")
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode(),
        _ARTIFACT_FINDING_IDS,
    )


def _artifact_finding_id_response(
    rubric_name: str,
    finding_ids: tuple[str, ...],
) -> str:
    if rubric_name == "breach_attack_authority":
        criterion_id = "attack_authority"
        category = "compatible"

        def evidence(attack_id: str) -> list[dict[str, str]]:
            del attack_id
            return [{
                "source_id": "artifact",
                "kind": "root_requirement",
                "location": "slice:thresholds",
                "excerpt": _ARTIFACT_FINDING_ROOT,
            }]
    elif rubric_name == "breach_attack_entailment":
        criterion_id = "attack_entailment"
        category = "entailed"

        def evidence(attack_id: str) -> list[dict[str, str]]:
            oracle = _ARTIFACT_FINDING_ORACLES.get(
                attack_id,
                _ARTIFACT_FINDING_ORACLES[_ARTIFACT_FINDING_IDS[0]],
            )
            return [
                {
                    "source_id": "artifact",
                    "kind": "attack_claim",
                    "location": f"attack:{attack_id}:oracle",
                    "excerpt": oracle,
                },
                {
                    "source_id": "artifact",
                    "kind": "root_requirement",
                    "location": "requirement:thresholds:0",
                    "excerpt": _ARTIFACT_FINDING_ROOT,
                },
            ]
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported artifact-id case {rubric_name!r}")

    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "Every supplied attack was classified independently.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [
            {
                "id": attack_id,
                "criterion_id": criterion_id,
                "severity": "High",
                "confidence": 0.99,
                "disposition": "addressed",
                "category": category,
                "targets": [],
                "evidence": evidence(attack_id),
                "proposal": None,
                "attack_spec": None,
            }
            for attack_id in finding_ids
        ],
    })


def _artifact_false_incomplete_response(rubric_name: str) -> str:
    """Return the live failure shape: complete packet, false omission claim."""

    payload = json.loads(_artifact_finding_id_response(
        rubric_name,
        _ARTIFACT_FINDING_IDS,
    ))
    payload["decision"] = "inconclusive"
    payload["reasoning"] = (
        "Every attack is inconclusive because required packet data is absent."
    )
    payload["coverage"] = {
        "complete": False,
        "sources_reviewed": ["artifact"],
        "omissions": ["selected root clauses were not supplied"],
    }
    for finding in payload["findings"]:
        finding["category"] = "inconclusive"
        finding["disposition"] = "suspected"
    return json.dumps(payload)


def _artifact_semantic_inconclusive_response(rubric_name: str) -> str:
    """Return complete closure with honest per-attack semantic uncertainty."""

    payload = json.loads(_artifact_finding_id_response(
        rubric_name,
        _ARTIFACT_FINDING_IDS,
    ))
    payload["decision"] = "inconclusive"
    payload["reasoning"] = "The selected clauses remain genuinely ambiguous."
    for finding in payload["findings"]:
        finding["category"] = "inconclusive"
        finding["disposition"] = "suspected"
    return json.dumps(payload)


def _artifact_finding_id_ledger(invocation: Invocation) -> dict[str, object]:
    """Extract the flat harness ledger without depending on prompt markers."""

    decoder = json.JSONDecoder()
    prompt = invocation.system + "\n" + invocation.user
    for index, char in enumerate(prompt):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(prompt, index)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("schema_version")
            == _ARTIFACT_FINDING_ID_LEDGER_SCHEMA
        ):
            return candidate
    raise AssertionError("artifact finding-id repair ledger was not rendered")


def _plan_judge_context() -> dict[str, str]:
    return {
        "spec_intent": "Change production behavior.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Production behavior changed.",
        "prior_critic_findings": json.dumps([{
            "id": "f1-test-edits",
            "criterion_id": "executability",
            "severity": "Critical",
            "confidence": 1.0,
            "disposition": "confirmed",
            "category": "other",
            "targets": [],
            "evidence": [],
            "proposal": None,
            "attack_spec": None,
        }]),
        "prior_judge_findings": "[]",
    }


def _plan_target_ledger_context() -> dict[str, str]:
    """Mirror the live round-four cardinality shape with 17 material priors."""

    context = _plan_judge_context()

    def prior(index: int) -> dict[str, object]:
        return {
            "id": f"historical-plan-finding-{index}",
            "criterion_id": "responsibility_size",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
            "evidence": [],
            "proposal": {
                "action": (
                    "Split configuration loading and alias validation into "
                    f"independent responsibilities ({index})."
                ),
                "replacement": None,
            },
            "attack_spec": None,
        }

    context["prior_critic_findings"] = json.dumps([
        prior(index) for index in range(9)
    ])
    context["prior_judge_findings"] = json.dumps([
        prior(index) for index in range(9, 17)
    ])
    return context


def _plan_judge_prior_ref() -> str:
    context = _plan_judge_context()
    return _context_prior_findings(context)[0]["ref"]


def _plan_judge_response(
    *,
    decision: str,
    disposition: str,
    sources_reviewed: list[str],
) -> str:
    prior_ref = _plan_judge_prior_ref()
    violated = decision in {"revise", "reject"}
    criterion_results = []
    for index, criterion in enumerate((
        "intent_coverage",
        "executability",
        "critic_disposition",
        "contradiction",
    )):
        criterion_results.append({
            "criterion_id": criterion,
            "outcome": "violated" if criterion == "executability" else "satisfied",
            "reasoning": "The current candidate was checked directly.",
            "evidence": [{
                "source_id": "artifact",
                "kind": "current_state",
                "location": "file",
                "excerpt": "astropy/table/tests/test_mixin.py",
            }],
            "headline": index == 1,
        })
    assert violated
    action = (
        "UNFIXABLE_PROOF: "
        '{"scope":"original_intent","bounded_plan_possible":false,'
        '"reason":"immutable API constraints conflict with required behavior"}'
        if disposition == "unfixable"
        else "Remove the repairable test-edit slice."
    )
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": decision,
        "reasoning": "The candidate needs a bounded source-only revision.",
        "coverage": {
            "complete": True,
            "sources_reviewed": sources_reviewed,
            "omissions": [],
        },
        "findings": [{
            "id": "f1-test-edits-adjudication",
            "criterion_id": "executability",
            "severity": "Critical",
            "confidence": 0.95,
            "disposition": disposition,
            "category": "other",
            "targets": [prior_ref],
            "evidence": [{
                "source_id": "artifact",
                "kind": "current_state",
                "location": "file",
                "excerpt": "astropy/table/tests/test_mixin.py",
            }],
            "proposal": {"action": action, "replacement": None},
            "attack_spec": None,
        }],
        "criterion_results": criterion_results,
    })


def _repaired_plan_gate_context(
    *, include_infrastructure_failure: bool = False,
) -> dict[str, str]:
    findings: list[dict[str, object]] = []
    records: list[dict[str, object]] = [
        {
            "round": 0,
            "decision": "inconclusive",
            "source": "planner_backend",
            "findings": [{
                "id": "planner_backend_error",
                "severity": "Critical",
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "source": "deterministic",
                "evidence": [{
                    "kind": "backend_error",
                    "excerpt": "transport: temporary planner failure",
                }],
            }],
        },
        {
            "round": 1,
            "decision": "revise",
            "source": "planner_parser",
            "findings": [{
                "id": "planner_parse_failure",
                "severity": "Critical",
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "source": "deterministic",
                "evidence": [{
                    "kind": "parse_error",
                    "excerpt": "previous response was not valid JSON",
                }],
            }],
        },
        {
            "round": 2,
            "decision": "revise",
            "source": "parallel_conflict_gate",
            "findings": [{
                "id": "parallel_conflict_t0_config_alias",
                "severity": "Critical",
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "source": "deterministic",
                "evidence": [{
                    "kind": "parallel_conflict",
                    "excerpt": "both write src/config.py",
                }],
            }],
        },
        {
            "round": 3,
            "decision": "revise",
            "source": "empty_scope_gate",
            "findings": [{
                "id": "empty_scope_filler_3_verify",
                "severity": "High",
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "source": "deterministic",
                "evidence": [{
                    "kind": "empty_plan_scope",
                    "excerpt": (
                        "the product permits only 0 verification-only "
                        "empty-scope slice(s)"
                    ),
                }],
            }],
        },
        {
            "round": 4,
            "decision": "revise",
            "source": "plan_size_bound_gate",
            "findings": [{
                "id": "plan_size_bound_exceeded",
                "severity": "High",
                "disposition": "confirmed",
                "category": "other",
                "targets": [],
                "source": "deterministic",
                "evidence": [{
                    "kind": "plan_size_bound",
                    "excerpt": (
                        "17 slices exceeds cap 8; 9 tiers exceeds cap 6; "
                        "anchor=1"
                    ),
                }],
            }],
        },
    ]
    if include_infrastructure_failure:
        findings.append({
            "id": "rubric_contract_failure",
            "criterion_id": "harness.rubric_contract",
            "severity": "Critical",
            "disposition": "confirmed",
            "category": "review_contract",
            "targets": [],
            "source": "deterministic",
            "evidence": [{
                "kind": "contract_error",
                "excerpt": "review response violated its schema",
            }],
        })
        records.append({
            "round": 5,
            "decision": "inconclusive",
            "source": "critic",
            "findings": findings,
        })
    return {
        "spec_intent": "Load configuration and validate aliases.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Both increments pass their tests.",
        "prior_critic_findings": json.dumps(records),
        "prior_judge_findings": "[]",
    }


def _plan_judge_empty_frontier_pass_response() -> str:
    evidence = [{
        "source_id": "artifact",
        "kind": "current_state",
        "location": "tier:t0",
        "excerpt": '"tier_id": "t0"',
    }]
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "All current plan checks and criteria pass.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [],
        "criterion_results": [
            {
                "criterion_id": criterion,
                "outcome": "satisfied",
                "reasoning": "The current plan directly satisfies this check.",
                "evidence": evidence,
                "headline": index == 0,
            }
            for index, criterion in enumerate((
                "intent_coverage",
                "executability",
                "critic_disposition",
                "contradiction",
            ))
        ],
    })


def _large_planning_history_context(
    closed_rounds: int,
) -> tuple[dict[str, str], str]:
    records = [
        {
            "round": round_index,
            "decision": "pass",
            "source": "critic",
            "findings": [{
                "id": f"closed-{round_index}",
                "criterion_id": "intent_coverage",
                "severity": "High",
                "disposition": "addressed",
                "category": "other",
                "targets": [],
                "evidence": [],
                "proposal": None,
                "attack_spec": None,
            }],
        }
        for round_index in range(closed_rounds)
    ]
    records.append({
        "round": closed_rounds,
        "decision": "revise",
        "source": "critic",
        "findings": [{
            "id": "active-blocker",
            "criterion_id": "intent_coverage",
            "severity": "High",
            "disposition": "confirmed",
            "category": "other",
            "targets": [],
            "evidence": [],
            "proposal": None,
            "attack_spec": None,
        }],
    })
    context = {
        "spec_intent": "Load configuration and validate aliases.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Both increments pass their tests.",
        "prior_critic_findings": json.dumps(records),
        "prior_judge_findings": "[]",
    }
    parsed = _context_prior_findings(dict(context))
    active_ref = next(
        item["ref"] for item in parsed if item["id"] == "active-blocker"
    )
    return context, active_ref


def _plan_judge_active_frontier_revise_response(prior_ref: str) -> str:
    evidence = [{
        "source_id": "artifact",
        "kind": "current_state",
        "location": "tier:t0",
        "excerpt": '"tier_id": "t0"',
    }]
    criteria = (
        "intent_coverage",
        "executability",
        "critic_disposition",
        "contradiction",
    )
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "The one active frontier blocker remains unresolved.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact"],
            "omissions": [],
        },
        "findings": [{
            "id": "active-blocker-adjudication",
            "criterion_id": "critic_disposition",
            "severity": "High",
            "confidence": 0.95,
            "disposition": "confirmed",
            "category": "unresolved_critic",
            "targets": [prior_ref],
            "evidence": evidence,
            "proposal": {
                "action": "Revise the current plan to close the active gap.",
                "replacement": None,
            },
            "attack_spec": None,
        }],
        "criterion_results": [
            {
                "criterion_id": criterion,
                "outcome": (
                    "violated"
                    if criterion == "critic_disposition"
                    else "satisfied"
                ),
                "reasoning": "The current candidate was checked directly.",
                "evidence": evidence,
                "headline": criterion == "critic_disposition",
            }
            for criterion in criteria
        ],
    })


@dataclass(frozen=True)
class _Reply:
    final: str
    cost_usd: float
    error: object | None = None
    completeness: ExecutionCompleteness | None = None
    execution_trace: BackendExecutionTrace | None = None


class _SequenceBackend:
    name = "review-response-retry-stub"
    model = "review-response-retry-model"
    capability = "text"

    def __init__(self, *replies: _Reply) -> None:
        self.replies = replies
        self.invocations: list[Invocation] = []

    def invoke(self, invocation: Invocation) -> InvocationResult:
        index = len(self.invocations)
        if index >= len(self.replies):
            raise AssertionError("review made an unexpected extra paid call")
        self.invocations.append(invocation)
        reply = self.replies[index]
        completeness = reply.completeness or ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        )
        execution_trace = reply.execution_trace
        if execution_trace is None and completeness.trace_complete:
            execution_trace = BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name=(
                            getattr(reply.error, "kind", None)
                            or "end_turn"
                        ),
                    ),
                ),
            )
        return InvocationResult(
            final_message=reply.final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=reply.final),
            ),
            usage=Usage(
                input_tokens=index + 1,
                output_tokens=index + 2,
                cost_usd=reply.cost_usd,
            ),
            stop_reason="end_turn",
            error=reply.error,
            completeness=completeness,
            execution_trace=execution_trace,
        )


class _AnchorOnlyGlmPlanJudgeBackend:
    """Repeat a live GLM JSON-shape failure, then obey the safe fallback."""

    name = "openai_compat"
    model = "nvidia/zai-org/glm-5.2"
    capability = "text"

    def __init__(
        self,
        *,
        anchor_only_from_start: bool = False,
        fail_first_anchor_response: bool = False,
        nonverbatim_first_response: bool = False,
    ) -> None:
        self.invocations: list[Invocation] = []
        self.anchor_only_from_start = anchor_only_from_start
        self.fail_first_anchor_response = fail_first_anchor_response
        self.nonverbatim_first_response = nonverbatim_first_response

    def invoke(self, invocation: Invocation) -> InvocationResult:
        attempt = len(self.invocations)
        self.invocations.append(invocation)
        if self.anchor_only_from_start or attempt > 0:
            assert "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in (
                invocation.system
            )
            assert "JSON-path evidence locations are disabled" in (
                invocation.system
            )
            assert "Do not emit a location that" in invocation.system
        else:
            assert "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" not in (
                invocation.system
            )
        artifact_frame = invocation.user.split(
            "BEGIN_UNTRUSTED_ARTIFACT\n", 1,
        )[1].split("\nEND_UNTRUSTED_ARTIFACT", 1)[0]
        match = re.search(
            r'"location":"(anchor:sha256:[0-9a-f]{64})"',
            artifact_frame,
        )
        assert match is not None
        anchor = match.group(1)
        payload = json.loads(_plan_judge_response(
            decision="revise",
            disposition="confirmed",
            sources_reviewed=[
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ],
        ))
        for collection in ("findings", "criterion_results"):
            for item in payload[collection]:
                for evidence in item["evidence"]:
                    evidence.update({
                        "source_id": "artifact",
                        "location": anchor if attempt else "json:$",
                        "excerpt": (
                            "model placeholder"
                            if attempt
                            else _PLAN_ARTIFACT.decode()
                        ),
                    })
                    if self.nonverbatim_first_response and attempt == 0:
                        evidence.update({
                            "location": "artifact",
                            "excerpt": "not retained source text",
                        })
        if self.fail_first_anchor_response and attempt == 1:
            payload.pop("reasoning")
        response = json.dumps(payload)
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _TargetLedgerGlmPlanJudgeBackend:
    """Exercise duplicate/missing repair across the live 17-alias shape."""

    name = "openai_compat"
    model = "nvidia/zai-org/glm-5.2"
    capability = "text"

    def __init__(
        self,
        *,
        cardinality_attempt: int = 0,
        ledger_repair_failures: int = 0,
        ledger_repair_assay_failure: bool = False,
        output_length_attempt: int | None = None,
        decision_mismatch_after_ledger: bool = False,
    ) -> None:
        assert cardinality_attempt >= 0
        assert ledger_repair_failures >= 0
        self.invocations: list[Invocation] = []
        self.aliases: list[str] = []
        self.ledger_snapshots: list[dict[str, object]] = []
        self.ledger_current_state_rule_seen = False
        self.cardinality_attempt = cardinality_attempt
        self.ledger_repair_failures = ledger_repair_failures
        self.ledger_repair_assay_failure = ledger_repair_assay_failure
        self.output_length_attempt = output_length_attempt
        self.decision_mismatch_after_ledger = decision_mismatch_after_ledger

    @staticmethod
    def _user_json_block(user: str, marker: str) -> dict[str, object]:
        encoded = user.split(f"BEGIN_{marker}\n", 1)[1].split(
            f"\nEND_{marker}", 1,
        )[0]
        return json.loads(encoded)

    def invoke(self, invocation: Invocation) -> InvocationResult:
        attempt = len(self.invocations)
        self.invocations.append(invocation)
        obligations = self._user_json_block(
            invocation.user,
            "CRUCIBLE_REVIEW_OBLIGATIONS",
        )
        aliases = [
            item["alias"]
            for item in obligations["obligations"]["material_prior_aliases"]
        ]
        assert len(aliases) == 17
        ledger_attempt = self.cardinality_attempt + 1
        if attempt == 0:
            self.aliases = aliases
        else:
            assert aliases == self.aliases
        defect: tuple[int, int] | None = None
        if attempt == self.cardinality_attempt:
            defect = (0, 1)
        elif (
            ledger_attempt <= attempt
            < ledger_attempt + self.ledger_repair_failures
        ):
            repair_index = attempt - ledger_attempt
            defect = (2 + 2 * repair_index, 3 + 2 * repair_index)
        invalid_targets = defect is not None
        targets = (
            [
                alias for index, alias in enumerate(aliases)
                if defect is None or index != defect[1]
            ]
            if invalid_targets else aliases
        )
        has_ledger_prompt = (
            "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in invocation.system
        )
        if not has_ledger_prompt:
            assert "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" not in (
                invocation.system
            )
        else:
            assert "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in (
                invocation.system
            )
            assert "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in (
                invocation.system
            )
            assert "JSON-path evidence locations are disabled" in (
                invocation.system
            )
            assert "Never copy, move, merge, or invent a disposition" in (
                invocation.system
            )
            assert "never cite the ledger as evidence" in invocation.system
            assert "copy an exact supplied" in invocation.system
            assert "Do not handwrite" in invocation.system
            self.ledger_current_state_rule_seen = all(
                rule in invocation.system
                for rule in (
                    "Every finding that contains one or more required aliases",
                    "independent current-state evidence item from the artifact",
                    "context, or a retained tool result",
                    "source evidence must never be the only evidence",
                    "Apply this requirement separately to every alias-carrying "
                    "finding",
                )
            )
            ledger = self._user_json_block(
                invocation.user,
                "CRUCIBLE_MATERIAL_PRIOR_TARGET_REPAIR_LEDGER",
            )
            digest = ledger.pop("ledger_sha256")
            encoded = json.dumps(
                ledger,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            assert digest == (
                "sha256:" + hashlib.sha256(encoded).hexdigest()
            )
            self.ledger_snapshots.append({
                **ledger,
                "ledger_sha256": digest,
            })
            previous_defect = (
                (0, 1)
                if attempt == ledger_attempt
                else (
                    2 + 2 * (attempt - ledger_attempt - 1),
                    3 + 2 * (attempt - ledger_attempt - 1),
                )
            )
            duplicate_index, missing_index = previous_defect
            primary_indexes = [
                index for index in range(len(aliases))
                if index != missing_index
            ]
            duplicate_path_index = primary_indexes.index(duplicate_index)
            assert ledger["required_aliases"] == aliases
            assert ledger["missing_aliases"] == [aliases[missing_index]]
            assert ledger["duplicate_aliases"] == [aliases[duplicate_index]]
            assert ledger["duplicate_observations"] == [{
                "alias": aliases[duplicate_index],
                "occurrences": [
                    {
                        "path": (
                            "findings[0].targets"
                            f"[{duplicate_path_index}]"
                        ),
                        "disposition": "addressed",
                    },
                    {
                        "path": "findings[1].targets[0]",
                        "disposition": "addressed",
                    },
                ],
                "same_terminal_disposition": True,
            }]
            assert {
                item["alias"]: item["count"]
                for item in ledger["observed_counts"]
            } == {
                alias: (
                    2 if index == duplicate_index
                    else 0 if index == missing_index
                    else 1
                )
                for index, alias in enumerate(aliases)
            }

        if attempt == self.output_length_attempt:
            return InvocationResult(
                final_message="",
                transcript=(Message(role="user", content=invocation.user),),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="max_turns",
                error=MaxTurnsExceeded(
                    turns_used=1,
                    turns_limit=1,
                    message="server returned finish_reason=length",
                ),
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )

        artifact_frame = invocation.user.split(
            "BEGIN_UNTRUSTED_ARTIFACT\n", 1,
        )[1].split("\nEND_UNTRUSTED_ARTIFACT", 1)[0]
        anchor_match = re.search(
            r'"location":"(anchor:sha256:[0-9a-f]{64})"',
            artifact_frame,
        )
        assert anchor_match is not None
        anchor = anchor_match.group(1)
        prior_frame = invocation.user.split(
            "BEGIN_UNTRUSTED_CONTEXT_PRIOR_CRITIC_FINDINGS\n", 1,
        )[1].split(
            "\nEND_UNTRUSTED_CONTEXT_PRIOR_CRITIC_FINDINGS", 1,
        )[0]
        prior_anchor = json.loads(prior_frame)["anchors"][0]["location"]
        payload = json.loads(_plan_judge_response(
            decision="revise",
            disposition="addressed",
            sources_reviewed=[
                "artifact",
                "context:spec_intent",
                "context:spec_constraints",
                "context:spec_done_criteria",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ],
        ))
        payload["decision"] = "pass"
        if self.decision_mismatch_after_ledger and attempt >= ledger_attempt:
            payload["decision"] = "revise"
        payload["reasoning"] = (
            "Every distinct historical plan finding was independently "
            "adjudicated against the current candidate."
        )
        if self.ledger_repair_assay_failure and attempt == ledger_attempt:
            payload.pop("reasoning")
        finding = payload["findings"][0]
        finding["id"] = "historical-plan-adjudications"
        finding["disposition"] = "addressed"
        finding["targets"] = targets
        if self.cardinality_attempt > 0 and attempt == 0:
            evidence_location = "anchor:sha256:" + "0" * 64
            evidence_excerpt = "model placeholder"
        elif self.cardinality_attempt > 1 and attempt == 1:
            evidence_location = "chars:0-1"
            evidence_excerpt = '"'
        else:
            evidence_location = anchor
            evidence_excerpt = "model placeholder"
        finding_evidence_source = "artifact"
        if attempt == ledger_attempt and not self.ledger_current_state_rule_seen:
            # Model the terminal-3 failure exactly: absent an explicit
            # per-finding repair instruction, the replacement uses the bound
            # historical finding as its only evidence. The response is valid
            # only when the repair prompt asks for independent current-state
            # evidence and the fake backend consequently selects the artifact.
            finding_evidence_source = "context:prior_critic_findings"
            evidence_location = prior_anchor
            evidence_excerpt = "model placeholder"
        finding["evidence"] = [{
            "source_id": finding_evidence_source,
            "kind": "anchor",
            "location": evidence_location,
            "excerpt": evidence_excerpt,
        }]
        finding["proposal"] = {
            "action": (
                "REPARTITION_PROOF: "
                + json.dumps({
                    "config-load": "Load configuration values",
                    "alias-validation": "Validate alias mappings",
                }, separators=(",", ":"))
            ),
            "replacement": None,
        }
        if invalid_targets:
            assert defect is not None
            duplicate_finding = json.loads(json.dumps(finding))
            duplicate_finding["id"] = "duplicate-historical-adjudication"
            duplicate_finding["targets"] = [aliases[defect[0]]]
            payload["findings"].append(duplicate_finding)
        for result in payload["criterion_results"]:
            result["outcome"] = "satisfied"
            result["evidence"] = [{
                "source_id": "artifact",
                "kind": "anchor",
                "location": evidence_location,
                "excerpt": evidence_excerpt,
            }]
        response = json.dumps(payload)
        return InvocationResult(
            final_message=response,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=response),
            ),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


def _glm_sequence_backend(*replies: _Reply) -> _SequenceBackend:
    backend = _SequenceBackend(*replies)
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"
    return backend


def _retry_manifest(corpus: Corpus, verdict) -> dict[str, object]:
    ref = verdict.artifacts["review_retry_manifest"]
    return json.loads(corpus.get_artifact(ref))


def _call_telemetry(corpus: Corpus, verdict) -> dict[str, object]:
    ref = verdict.artifacts["review_call_telemetry"]
    return json.loads(corpus.get_artifact(ref))


def _assert_manifest_shape(manifest: dict[str, object], attempts: int) -> None:
    assert set(manifest) == {
        "accounting_debt",
        "execution_completeness",
        "schema_version",
        "attempts",
        "terminal_reason",
        "total_cost_usd",
        "total_input_tokens",
        "total_output_tokens",
        "total_cache_creation_tokens",
        "total_cache_read_tokens",
    }
    assert manifest["schema_version"] == "crucible.review-response-retry.v1"
    records = manifest["attempts"]
    assert isinstance(records, list)
    assert len(records) == attempts
    for index, record in enumerate(records):
        assert set(record) == {
            "attempt",
            "cost_usd",
            "execution_completeness",
            "accounting_debt",
            "coverage_normalization_ref",
            "contract_canonicalization_ref",
            "response_repair_ref",
            "raw_response_ref",
            "target_alias_expansion_ref",
            "decision",
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "validation_error",
            "transcript_ref",
            "rendered_prompt_ref",
            "rubric_ref",
        }
        assert record["attempt"] == index
        for key in ("transcript_ref", "rendered_prompt_ref", "rubric_ref"):
            assert record[key].startswith("sha256:")
    cache_creation_values = [
        record["cache_creation_tokens"] for record in records
    ]
    cache_read_values = [record["cache_read_tokens"] for record in records]
    assert all(
        (creation is None) == (read is None)
        for creation, read in zip(
            cache_creation_values,
            cache_read_values,
            strict=True,
        )
    )
    expected_creation = (
        None
        if any(value is None for value in cache_creation_values)
        else sum(cache_creation_values)
    )
    expected_read = (
        None
        if any(value is None for value in cache_read_values)
        else sum(cache_read_values)
    )
    assert manifest["total_cache_creation_tokens"] == expected_creation
    assert manifest["total_cache_read_tokens"] == expected_read


def _assert_aggregate_transcript(corpus: Corpus, verdict, manifest) -> None:
    """The aggregate must retain each attempt transcript byte-for-byte."""

    records = manifest["attempts"]
    attempt_messages = [
        json.loads(corpus.get_artifact(record["transcript_ref"]))
        for record in records
    ]
    aggregate = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert isinstance(aggregate, list)
    cursor = 0
    for record, transcript in zip(records, attempt_messages):
        separator = aggregate[cursor]
        assert separator["role"] == "system"
        assert json.loads(separator["content"]) == {
            "schema_version": "crucible.review-response-retry.v1",
            "attempt": record["attempt"],
            "validation_error": record["validation_error"],
        }
        cursor += 1
        assert aggregate[cursor:cursor + len(transcript)] == transcript
        cursor += len(transcript)
    assert cursor == len(aggregate)


def _reply_with_accounting_debt(final: str, cost_usd: float = 0.25) -> _Reply:
    return _Reply(
        final=final,
        cost_usd=cost_usd,
        completeness=ExecutionCompleteness(
            token_usage_complete=False,
            cost_complete=False,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="review-response-retry-stub",
            requested_model="review-response-retry-model",
            events=(
                BackendExecutionEvent(
                    sequence=0,
                    kind="terminal",
                    name="end_turn",
                ),
            ),
        ),
    )


def test_semantic_success_preserves_accounting_debt(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _SequenceBackend(
        _reply_with_accounting_debt(json.dumps(_pass_response())),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert len(backend.invocations) == 1


def test_accounting_debt_blocks_additional_paid_response_repair(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _SequenceBackend(
        _reply_with_accounting_debt(
            _crucible_contract_failure_response(),
        ),
        _Reply(json.dumps(_pass_response()), 0.4),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert len(backend.invocations) == 1
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["terminal_reason"] == "accounting_debt_before_retry"
    assert manifest["accounting_debt"] == [
        "token_usage_complete",
        "cost_complete",
    ]
    assert len(manifest["attempts"]) == 1


def test_accounting_debt_blocks_response_repair_with_incomplete_trace(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _SequenceBackend(
        _Reply(
            _schema_failure_response("unexpected", "PRIVATE"),
            0.25,
            completeness=ExecutionCompleteness(
                token_usage_complete=False,
                cost_complete=False,
                terminal_complete=True,
                trace_complete=False,
            ),
        ),
        _Reply(json.dumps(_pass_response()), 0.4),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert len(backend.invocations) == 1
    assert _retry_manifest(corpus, verdict)["terminal_reason"] == (
        "accounting_debt_before_retry"
    )
    telemetry = _call_telemetry(corpus, verdict)
    assert telemetry["terminal_reason"] == "accounting_debt_before_retry"
    assert len(telemetry["attempts"]) == 1
    assert telemetry["accounting_debt"] == [
        "token_usage_complete",
        "cost_complete",
    ]
    assert telemetry["execution_completeness"] == {
        "token_usage_complete": False,
        "cost_complete": False,
        "terminal_complete": True,
        "trace_complete": False,
    }


def test_obligation_prompt_distinguishes_normalized_and_tool_source_coverage():
    envelope = {
        "schema_version": "crucible.review-obligations.v1",
        "obligations_sha256": "sha256:" + "0" * 64,
        "obligations": {
            "required_source_ids": ["artifact"],
            "material_prior_bindings": [],
        },
    }

    # Coverage ownership is an explicit adapter input, independent of which
    # bundled reviewer supplies the base prompt.
    rubric = load("spec_clarity")
    tool_free = _obligations_rubric(
        rubric, envelope, normalize_coverage=True,
    ).system_prompt
    read_only = _obligations_rubric(
        rubric, envelope, normalize_coverage=False,
    ).system_prompt

    assert tool_free.startswith(rubric.system_prompt)
    assert read_only.startswith(rubric.system_prompt)
    assert "replaces only that field" in tool_free
    assert "every retained ``tool:`` source actually reviewed" in read_only
    assert "replaces only that field" not in read_only


def test_assay_schema_failure_retries_same_input_with_exact_error_and_provenance(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    private_bad_output = "DO_NOT_COPY_PRIOR_BAD_RESPONSE"
    malformed = _schema_failure_response(
        "model_private_echo", private_bad_output,
    )
    backend = _SequenceBackend(
        _Reply(malformed, 0.25),
        _Reply(json.dumps(_pass_response()), 0.40),
    )
    rubric = load("spec_clarity")
    original_system = rubric.system_prompt

    verdict = review(
        _ARTIFACT,
        rubric,
        backend=backend,
        corpus=corpus,
        budget_usd=1.0,
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert verdict.cost_usd == pytest.approx(0.65)
    assert corpus.get_artifact(verdict.artifact_ref) == _ARTIFACT
    assert len(backend.invocations) == 2
    first, second = backend.invocations
    assert first.budget_usd == pytest.approx(1.0)
    assert second.budget_usd == pytest.approx(0.75)
    assert second.user != first.user
    assert first.user in second.user
    assert _ARTIFACT.decode() in second.user
    assert second.system.startswith(original_system)
    assert rubric.system_prompt == original_system

    exact_error = (
        "invalid review_v2 response: response: "
        "unexpected=['model_private_echo']"
    )
    repair_suffix = second.system[len(original_system):]
    assert json.dumps(exact_error) not in repair_suffix
    assert json.dumps(exact_error) in second.user
    assert "original" in repair_suffix.lower()
    assert any(
        word in repair_suffix.lower() for word in ("full", "complete")
    )
    assert "replacement" in repair_suffix.lower()
    assert "Recheck every evidence excerpt" in repair_suffix
    assert "never reflow or" in repair_suffix
    encoded_envelope = second.user.split(
        "BEGIN_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE\n", 1,
    )[1].split("\nEND_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE", 1)[0]
    envelope = json.loads(encoded_envelope)
    assert envelope == {
        "schema_version": "crucible.review-response-repair.v1",
        "attempt": 1,
        "validation_error": exact_error,
        "validation_error_sha256": (
            "sha256:" + hashlib.sha256(exact_error.encode()).hexdigest()
        ),
    }
    assert private_bad_output not in second.system
    assert private_bad_output not in second.user

    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first_attempt, second_attempt = manifest["attempts"]
    assert first_attempt["cost_usd"] == pytest.approx(0.25)
    assert first_attempt["decision"] == "inconclusive"
    assert first_attempt["validation_error"] == exact_error
    assert second_attempt["cost_usd"] == pytest.approx(0.40)
    assert second_attempt["decision"] == "pass"
    assert second_attempt["validation_error"] is None
    assert manifest["total_cost_usd"] == pytest.approx(0.65)
    assert first_attempt["rubric_ref"] != second_attempt["rubric_ref"]

    for invocation, attempt in zip(backend.invocations, manifest["attempts"]):
        rendered = json.loads(
            corpus.get_artifact(attempt["rendered_prompt_ref"])
        )
        assert rendered == {
            "system": invocation.system,
            "user": invocation.user,
            "response_repair_degradation": None,
        }
        # Each manifest ref must resolve independently; the final transcript
        # is a distinct aggregate, not an alias for the successful attempt.
        assert json.loads(corpus.get_artifact(attempt["transcript_ref"]))
    assert verdict.transcript_ref not in {
        attempt["transcript_ref"] for attempt in manifest["attempts"]
    }
    _assert_aggregate_transcript(corpus, verdict, manifest)
    telemetry = _call_telemetry(corpus, verdict)
    assert telemetry["terminal_reason"] == "repaired"
    assert telemetry["total_cost_usd"] == pytest.approx(0.65)
    for attempt, expected_response in zip(
        telemetry["attempts"],
        (malformed, json.dumps(_pass_response())),
        strict=True,
    ):
        raw_ref = attempt["raw_response_ref"]
        assert isinstance(raw_ref, str)
        assert corpus.get_artifact(raw_ref) == expected_response.encode("utf-8")
        assert attempt["raw_response_bytes"] == len(
            expected_response.encode("utf-8")
        )
    telemetry_blob = corpus.get_artifact(
        verdict.artifacts["review_call_telemetry"]
    )
    assert private_bad_output.encode("utf-8") not in telemetry_blob


@pytest.mark.parametrize(
    ("response", "expected_decision"),
    [
        (json.dumps(_pass_response()), "pass"),
        ("not a JSON response", "inconclusive"),
    ],
)
def test_call_telemetry_materializes_exact_transcript_assistant_response(
    tmp_path,
    response,
    expected_decision,
):
    corpus = Corpus(tmp_path / "corpus")
    completeness = ExecutionCompleteness(
        token_usage_complete=True,
        cost_complete=True,
        terminal_complete=True,
        trace_complete=True,
    )
    trace = BackendExecutionTrace(
        backend="review-response-retry-stub",
        requested_model="review-response-retry-model",
        events=(
            BackendExecutionEvent(
                sequence=0,
                kind="terminal",
                name="end_turn",
            ),
        ),
    )
    backend = _SequenceBackend(
        _Reply(
            response,
            0.25,
            completeness=completeness,
            execution_trace=trace,
        )
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
    )

    assert verdict.decision == expected_decision
    telemetry = _call_telemetry(corpus, verdict)
    attempt = telemetry["attempts"][0]
    raw_ref = attempt["raw_response_ref"]
    assert isinstance(raw_ref, str)
    assert corpus.get_artifact(raw_ref) == response.encode("utf-8")
    assert attempt["raw_response_bytes"] == len(response.encode("utf-8"))


def test_response_repair_retains_deduplicated_validation_error_history(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    private_values = (
        "DO_NOT_REPLAY_HISTORY_A",
        "DO_NOT_REPLAY_HISTORY_B",
        "DO_NOT_REPLAY_HISTORY_C",
        "DO_NOT_REPLAY_HISTORY_B_AGAIN",
    )
    keys = ("history_a", "history_b", "history_c", "history_b")
    backend = _SequenceBackend(
        *(
            _Reply(_schema_failure_response(key, private_value), 0.0)
            for key, private_value in zip(keys, private_values)
        ),
        _Reply(json.dumps(_pass_response()), 0.0),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=4,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 5
    errors = tuple(
        f"invalid review_v2 response: response: unexpected=['{key}']" for key in keys
    )

    def repair_envelope(index: int) -> dict[str, object]:
        encoded = (
            backend.invocations[index]
            .user.split(
                "BEGIN_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE\n",
                1,
            )[1]
            .split("\nEND_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE", 1)[0]
        )
        return json.loads(encoded)

    def history_record(error: str) -> dict[str, str]:
        return {
            "validation_error": error,
            "validation_error_sha256": (
                "sha256:" + hashlib.sha256(error.encode()).hexdigest()
            ),
        }

    first_repair = repair_envelope(1)
    assert first_repair["validation_error"] == errors[0]
    assert "prior_validation_errors" not in first_repair
    assert repair_envelope(2)["prior_validation_errors"] == [
        history_record(errors[0]),
    ]
    assert repair_envelope(3)["prior_validation_errors"] == [
        history_record(errors[0]),
        history_record(errors[1]),
    ]
    final_repair = repair_envelope(4)
    assert final_repair["validation_error"] == errors[3]
    # The repeated latest error is represented by the compatibility field,
    # while distinct earlier failures retain deterministic first-seen order.
    assert final_repair["prior_validation_errors"] == [
        history_record(errors[0]),
        history_record(errors[2]),
    ]
    repair_instructions = " ".join(backend.invocations[4].system.split())
    assert "ordered, deduplicated validation history only" in (repair_instructions)
    assert "not prior response content or proof" in repair_instructions
    repair_prompts = "\n".join(
        invocation.system + invocation.user for invocation in backend.invocations[1:]
    )
    assert all(value not in repair_prompts for value in private_values)
    assert all(
        _ARTIFACT.decode() in invocation.user for invocation in backend.invocations
    )

    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=5)
    assert [attempt["validation_error"] for attempt in manifest["attempts"]] == [
        *errors,
        None,
    ]
    assert manifest["terminal_reason"] == "repaired"


def test_crucible_contract_failure_is_repaired_but_bad_response_is_not_replayed(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    invalid = _crucible_contract_failure_response()
    backend = _SequenceBackend(
        _Reply(invalid, 0.10),
        _Reply(json.dumps(_pass_response()), 0.20),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 2
    exact_error = (
        "Rubric-specific response contract failed: "
        "spec_clarity: finding[0] has undeclared criterion_id "
        "'scope.in_bounds'; spec_clarity: finding[0] has undeclared "
        "category 'scope'"
    )
    repair_suffix = backend.invocations[1].system[
        len(backend.invocations[0].system):
    ]
    assert json.dumps(exact_error) not in repair_suffix
    assert json.dumps(exact_error) in backend.invocations[1].user
    assert "PRIVATE_BAD_REASONING" not in backend.invocations[1].system
    assert "PRIVATE_BAD_REASONING" not in backend.invocations[1].user

    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["attempts"][0]["validation_error"] == exact_error
    assert manifest["attempts"][1]["validation_error"] is None
    assert manifest["total_cost_usd"] == pytest.approx(0.30)


def test_workflow_response_contract_validator_retries_without_replaying_model(
    tmp_path,
):
    rejected = _pass_response()
    rejected["reasoning"] = "PRIVATE_WORKFLOW_REJECT_MARKER"
    backend = _SequenceBackend(
        _Reply(json.dumps(rejected), 0.10),
        _Reply(json.dumps(_pass_response()), 0.20),
    )
    corpus = Corpus(tmp_path / "corpus")

    def validator(candidate):
        if "PRIVATE_WORKFLOW_REJECT_MARKER" in candidate.reasoning:
            return ("workflow citation binding is invalid",)
        return ()

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
        response_contract_validator=validator,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 2
    assert "PRIVATE_WORKFLOW_REJECT_MARKER" not in backend.invocations[1].user
    assert "workflow citation binding is invalid" in backend.invocations[1].user
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["terminal_reason"] == "repaired"
    assert manifest["attempts"][0]["validation_error"] == (
        "Rubric-specific response contract failed: "
        "workflow citation binding is invalid"
    )


def test_workflow_response_contract_validator_exhausts_fail_closed(tmp_path):
    backend = _SequenceBackend(*(
        _Reply(json.dumps(_pass_response()), 0.10) for _ in range(3)
    ))
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
        response_contract_validator=lambda candidate: (
            "workflow citation binding is invalid",
        ),
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["terminal_reason"] == "retries_exhausted"
    assert len(manifest["attempts"]) == 3


def test_workflow_response_contract_validator_cannot_suppress_rubric_errors(
    tmp_path,
):
    backend = _SequenceBackend(
        _Reply(_crucible_contract_failure_response(), 0.10),
        _Reply(json.dumps(_pass_response()), 0.20),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        response_retry_budget=1,
        response_contract_validator=lambda candidate: (),
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 2


@pytest.mark.parametrize(
    ("validator", "error"),
    [
        pytest.param(
            lambda candidate: ["not a tuple"],
            TypeError,
            id="malformed-return",
        ),
        pytest.param(
            lambda candidate: ("x" * 2_001,),
            TypeError,
            id="oversized-error",
        ),
        pytest.param(
            lambda candidate: (_ for _ in ()).throw(RuntimeError("boom")),
            RuntimeError,
            id="callback-exception",
        ),
    ],
)
def test_workflow_response_contract_validator_defects_halt(
    tmp_path,
    validator,
    error,
):
    backend = _SequenceBackend(_Reply(json.dumps(_pass_response()), 0.10))

    with pytest.raises(error):
        review(
            _ARTIFACT,
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            response_retry_budget=1,
            response_contract_validator=validator,
        )


def test_workflow_response_contract_validator_cannot_mutate_verdict(tmp_path):
    backend = _SequenceBackend(_Reply(json.dumps(_pass_response()), 0.10))

    def mutating_validator(candidate):
        object.__setattr__(candidate, "decision", "reject")
        return ()

    with pytest.raises(
        RuntimeError,
        match="response_contract_validator mutated",
    ):
        review(
            _ARTIFACT,
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            response_retry_budget=1,
            response_contract_validator=mutating_validator,
        )


def test_workflow_contract_canonicalizer_repairs_without_retry(tmp_path):
    backend = _SequenceBackend(
        _Reply(_workflow_canonicalization_response(), 0.10),
    )
    corpus = Corpus(tmp_path / "corpus")

    def validator(candidate):
        if candidate.findings[0].evidence[0].excerpt != "ORIGINAL REVIEW INPUT":
            return ("workflow citation includes JSON framing",)
        return ()

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
        response_contract_validator=validator,
        response_contract_canonicalizer=_canonicalize_workflow_excerpt,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    assert verdict.findings[0].evidence[0].excerpt == "ORIGINAL REVIEW INPUT"
    assert "review_retry_manifest" not in verdict.artifacts
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_contract_canonicalization"]
    ))
    assert report["canonicalizer"] == "workflow_response_contract"
    assert report["accepted"] is True
    assert report["acceptance_reason"] == "valid"
    assert report["rubric_errors_before"] == []
    assert report["rubric_errors_after"] == []
    assert report["workflow_errors_before"] == [
        "workflow citation includes JSON framing"
    ]
    assert report["workflow_errors_after"] == []
    assert "raw_response_ref" in report
    assert report["transcript_ref"] is not None
    assert report["input_verdict_sha256"] != report["output_verdict_sha256"]


def test_workflow_contract_canonicalizer_cannot_launder_rubric_error(tmp_path):
    backend = _SequenceBackend(
        _Reply(_crucible_contract_failure_response(), 0.10),
    )
    callback_calls = 0

    def canonicalizer(candidate):
        nonlocal callback_calls
        callback_calls += 1
        return _canonicalize_workflow_excerpt(candidate)

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        response_retry_budget=0,
        response_contract_canonicalizer=canonicalizer,
    )

    assert verdict.decision == "inconclusive"
    assert callback_calls == 0
    assert len(backend.invocations) == 1
    assert "review_contract_canonicalization" not in verdict.artifacts


@pytest.mark.parametrize(
    ("canonicalizer", "error"),
    [
        pytest.param(
            lambda candidate: (
                _canonicalize_workflow_excerpt(candidate)[0],
                (),
            ),
            "actions must identify every and only changed",
            id="missing-action",
        ),
        pytest.param(
            lambda candidate: (
                dataclasses.replace(candidate, decision="pass"),
                ({
                    "operation": "forbidden_semantic_change",
                    "path": "decision",
                },),
            ),
            "changed a field other than evidence.excerpt",
            id="semantic-field-change",
        ),
        pytest.param(
            lambda candidate: (
                _canonicalize_workflow_excerpt(candidate)[0],
                ({
                    "operation": "oversized_action",
                    "path": "findings[0].evidence[0].excerpt",
                    "diagnostic": "x" * (64 * 1024),
                },),
            ),
            "actions exceed the byte limit",
            id="oversized-audit",
        ),
    ],
)
def test_workflow_contract_canonicalizer_defects_halt(
    tmp_path,
    canonicalizer,
    error,
):
    backend = _SequenceBackend(
        _Reply(_workflow_canonicalization_response(), 0.10),
    )

    with pytest.raises((RuntimeError, ValueError), match=error):
        review(
            _ARTIFACT,
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            response_retry_budget=1,
            response_contract_validator=lambda candidate: (
                "workflow citation includes JSON framing",
            ),
            response_contract_canonicalizer=canonicalizer,
        )

    assert len(backend.invocations) == 1


def test_workflow_contract_canonicalizer_cannot_mutate_input(tmp_path):
    backend = _SequenceBackend(
        _Reply(_workflow_canonicalization_response(), 0.10),
    )

    def mutating_canonicalizer(candidate):
        object.__setattr__(candidate, "decision", "pass")
        return candidate, ()

    with pytest.raises(
        RuntimeError,
        match="response_contract_canonicalizer mutated",
    ):
        review(
            _ARTIFACT,
            load("spec_clarity"),
            backend=backend,
            corpus=Corpus(tmp_path / "corpus"),
            response_retry_budget=1,
            response_contract_validator=lambda candidate: (
                "workflow citation includes JSON framing",
            ),
            response_contract_canonicalizer=mutating_canonicalizer,
        )

    assert len(backend.invocations) == 1


def test_workflow_contract_canonicalizer_rejects_new_workflow_error(tmp_path):
    backend = _SequenceBackend(
        _Reply(_workflow_canonicalization_response(), 0.10),
    )
    corpus = Corpus(tmp_path / "corpus")

    def validator(candidate):
        excerpt = candidate.findings[0].evidence[0].excerpt
        if excerpt == _ARTIFACT.decode():
            return ("old workflow citation error",)
        return ("new workflow citation error",)

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=0,
        response_contract_validator=validator,
        response_contract_canonicalizer=_canonicalize_workflow_excerpt,
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 1
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_contract_canonicalization"]
    ))
    assert report["accepted"] is False
    assert report["acceptance_reason"] == "rejected_new_workflow_errors"
    assert report["workflow_errors_before"] == [
        "old workflow citation error"
    ]
    assert report["workflow_errors_after"] == [
        "new workflow citation error"
    ]


@pytest.mark.parametrize(
    "rubric_name",
    ["breach_attack_authority", "breach_attack_entailment"],
)
def test_artifact_finding_id_omission_retries_against_trusted_ledger(
    tmp_path,
    rubric_name,
):
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    backend = _SequenceBackend(
        _Reply(
            _artifact_finding_id_response(rubric_name, finding_ids[:1]),
            0.10,
        ),
        _Reply(
            _artifact_finding_id_response(rubric_name, finding_ids),
            0.20,
        ),
    )
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert len(backend.invocations) == 2
    assert verdict.decision == "pass"
    assert tuple(finding.id for finding in verdict.findings) == finding_ids
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first_error = manifest["attempts"][0]["validation_error"]
    assert isinstance(first_error, str)
    assert "missing" in first_error.lower()
    assert finding_ids[1] in first_error
    assert manifest["attempts"][1]["validation_error"] is None

    assert _ARTIFACT_FINDING_ID_LEDGER_SCHEMA not in (
        backend.invocations[0].system + backend.invocations[0].user
    )
    ledger = _artifact_finding_id_ledger(backend.invocations[1])
    unsigned_ledger = {
        "schema_version": _ARTIFACT_FINDING_ID_LEDGER_SCHEMA,
        "rubric": rubric_name,
        "artifact_sha256": (
            "sha256:" + hashlib.sha256(artifact).hexdigest()
        ),
        "ordered_finding_ids": list(finding_ids),
    }
    ledger_digest = "sha256:" + hashlib.sha256(json.dumps(
        unsigned_ledger,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()).hexdigest()
    assert ledger == {
        **unsigned_ledger,
        "ledger_sha256": ledger_digest,
    }


@pytest.mark.parametrize(
    "rubric_name",
    ["breach_attack_authority", "breach_attack_entailment"],
)
def test_artifact_finding_id_omission_exhaustion_fails_closed(
    tmp_path,
    rubric_name,
):
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    omitted = _artifact_finding_id_response(rubric_name, finding_ids[:1])
    backend = _SequenceBackend(*(
        _Reply(omitted, 0.10) for _ in range(3)
    ))
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 3
    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    for attempt in manifest["attempts"]:
        error = attempt["validation_error"]
        assert isinstance(error, str)
        assert "missing" in error.lower()
        assert finding_ids[1] in error
    for invocation in backend.invocations[1:]:
        ledger = _artifact_finding_id_ledger(invocation)
        assert ledger["ordered_finding_ids"] == list(finding_ids)
        assert ledger["artifact_sha256"] == (
            "sha256:" + hashlib.sha256(artifact).hexdigest()
        )


@pytest.mark.parametrize(
    "rubric_name",
    ["breach_attack_authority", "breach_attack_entailment"],
)
def test_artifact_finding_id_unexpected_value_retries_to_exact_closure(
    tmp_path,
    rubric_name,
):
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    unexpected_id = "correctness-not-present-in-trusted-artifact"
    backend = _SequenceBackend(
        _Reply(_artifact_finding_id_response(
            rubric_name,
            (*finding_ids, unexpected_id),
        ), 0.10),
        _Reply(
            _artifact_finding_id_response(rubric_name, finding_ids),
            0.20,
        ),
    )
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert tuple(finding.id for finding in verdict.findings) == finding_ids
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first_error = manifest["attempts"][0]["validation_error"]
    assert isinstance(first_error, str)
    assert "unexpected" in first_error.lower()
    assert unexpected_id in first_error
    assert manifest["attempts"][1]["validation_error"] is None
    ledger = _artifact_finding_id_ledger(backend.invocations[1])
    assert ledger["ordered_finding_ids"] == list(finding_ids)
    assert unexpected_id not in ledger["ordered_finding_ids"]


@pytest.mark.parametrize(
    "rubric_name",
    ["breach_attack_authority", "breach_attack_entailment"],
)
def test_false_incomplete_all_inconclusive_response_repairs_with_id_ledger(
    tmp_path,
    rubric_name,
):
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    backend = _SequenceBackend(
        _Reply(_artifact_false_incomplete_response(rubric_name), 0.10),
        _Reply(
            _artifact_finding_id_response(rubric_name, finding_ids),
            0.20,
        ),
    )
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert tuple(finding.id for finding in verdict.findings) == finding_ids
    assert len(backend.invocations) == 2
    assert "coverage_attestation" not in json.loads(artifact)
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first_error = manifest["attempts"][0]["validation_error"]
    assert isinstance(first_error, str)
    assert "harness-attested complete artifact requires" in first_error
    assert "complete=False" in first_error
    assert "selected root clauses were not supplied" in first_error
    assert "missing=" not in first_error
    assert manifest["attempts"][1]["validation_error"] is None

    ledger = _artifact_finding_id_ledger(backend.invocations[1])
    assert ledger["ordered_finding_ids"] == list(finding_ids)
    assert ledger["artifact_sha256"] == (
        "sha256:" + hashlib.sha256(artifact).hexdigest()
    )
    first_obligations = _obligations_from_invocation(backend.invocations[0])
    second_obligations = _obligations_from_invocation(backend.invocations[1])
    assert first_obligations == second_obligations
    attestation = first_obligations["obligations"][
        "artifact_coverage_attestation"
    ]
    expected_artifact_kinds = {
        "breach_attack_authority": "hash_bound_attack_authority_bundle",
        "breach_attack_entailment": "hash_bound_attack_entailment_bundle",
    }
    assert attestation["rubric"] == rubric_name
    assert attestation["artifact_kind"] == expected_artifact_kinds[rubric_name]
    assert attestation["artifact_sha256"] == (
        "sha256:" + hashlib.sha256(artifact).hexdigest()
    )
    assert attestation["complete"] is True
    assert attestation["omissions"] == []
    encoded_obligations = json.dumps(
        first_obligations["obligations"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert first_obligations["obligations_sha256"] == (
        "sha256:" + hashlib.sha256(encoded_obligations).hexdigest()
    )


@pytest.mark.parametrize(
    "rubric_name",
    ["breach_attack_authority", "breach_attack_entailment"],
)
def test_repeated_false_incomplete_all_inconclusive_response_exhausts(
    tmp_path,
    rubric_name,
):
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    invalid = _artifact_false_incomplete_response(rubric_name)
    backend = _SequenceBackend(*(
        _Reply(invalid, 0.10) for _ in range(3)
    ))
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    assert len(backend.invocations) == 3
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    for attempt in manifest["attempts"]:
        error = attempt["validation_error"]
        assert isinstance(error, str)
        assert "harness-attested complete artifact requires" in error
        assert "complete=False" in error
    for invocation in backend.invocations[1:]:
        ledger = _artifact_finding_id_ledger(invocation)
        assert ledger["ordered_finding_ids"] == list(finding_ids)
    obligations = [
        _obligations_from_invocation(invocation)
        for invocation in backend.invocations
    ]
    assert obligations[1:] == obligations[:-1]
    attestation = obligations[0]["obligations"][
        "artifact_coverage_attestation"
    ]
    assert attestation["artifact_sha256"] == (
        "sha256:" + hashlib.sha256(artifact).hexdigest()
    )
    assert attestation["complete"] is True
    assert attestation["omissions"] == []


def test_entailment_semantic_inconclusive_repairs_with_exact_id_ledger(
    tmp_path,
):
    rubric_name = "breach_attack_entailment"
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    backend = _SequenceBackend(
        _Reply(_artifact_semantic_inconclusive_response(rubric_name), 0.10),
        _Reply(
            _artifact_finding_id_response(rubric_name, finding_ids),
            0.20,
        ),
    )
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert tuple(finding.id for finding in verdict.findings) == finding_ids
    assert len(backend.invocations) == 2
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    assert manifest["attempts"][0]["validation_error"] == (
        "Rubric-specific response contract failed: "
        "breach_attack_entailment: semantic inconclusive findings require "
        f"response repair; ids={list(finding_ids)!r}"
    )
    assert manifest["attempts"][1]["validation_error"] is None
    ledger = _artifact_finding_id_ledger(backend.invocations[1])
    assert ledger["ordered_finding_ids"] == list(finding_ids)
    assert ledger["artifact_sha256"] == (
        "sha256:" + hashlib.sha256(artifact).hexdigest()
    )


def test_entailment_repeated_semantic_inconclusive_exhausts_fail_closed(
    tmp_path,
):
    rubric_name = "breach_attack_entailment"
    artifact, finding_ids = _artifact_finding_id_case(rubric_name)
    response = _artifact_semantic_inconclusive_response(rubric_name)
    backend = _SequenceBackend(*(
        _Reply(response, 0.10) for _ in range(3)
    ))
    corpus = Corpus(tmp_path / rubric_name)

    verdict = review(
        artifact,
        load(rubric_name),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    assert len(backend.invocations) == 3
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    for attempt in manifest["attempts"]:
        error = attempt["validation_error"]
        assert isinstance(error, str)
        assert "semantic inconclusive findings require response repair" in error
        assert all(finding_id in error for finding_id in finding_ids)
    for invocation in backend.invocations[1:]:
        ledger = _artifact_finding_id_ledger(invocation)
        assert ledger["ordered_finding_ids"] == list(finding_ids)


def test_glm_output_length_gets_separate_audited_retry_after_schema_budget(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    private_partial = "PRIVATE_TRUNCATED_RESPONSE_MUST_NOT_BE_REPLAYED"
    backend = _glm_sequence_backend(
        _Reply(_schema_failure_response("bad_attempt_0", "x"), 0.10),
        _Reply(_schema_failure_response("bad_attempt_1", "y"), 0.10),
        _Reply(
            private_partial,
            0.10,
            MaxTurnsExceeded(
                turns_used=1,
                turns_limit=1,
                message="server returned finish_reason=length",
            ),
        ),
        _Reply(json.dumps(_pass_response()), 0.10),
    )

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=1.0,
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 4
    assert verdict.cost_usd == pytest.approx(0.40)
    assert verdict.input_tokens == 10
    assert verdict.output_tokens == 14
    assert [invocation.budget_usd for invocation in backend.invocations] == [
        pytest.approx(1.0),
        pytest.approx(0.9),
        pytest.approx(0.8),
        pytest.approx(0.7),
    ]
    assert private_partial not in backend.invocations[3].system
    assert private_partial not in backend.invocations[3].user

    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=4)
    assert manifest["terminal_reason"] == "repaired"
    assert manifest["attempts"][2]["validation_error"] == (
        "backend output ended with finish_reason=length before a complete "
        "structured response object; return one concise complete replacement "
        "object without trailing padding"
    )
    assert manifest["attempts"][3]["validation_error"] is None
    _assert_aggregate_transcript(corpus, verdict, manifest)


def test_glm_output_length_accounting_debt_blocks_fresh_call(tmp_path):
    debt_reply = _reply_with_accounting_debt("truncated", cost_usd=0.10)
    backend = _glm_sequence_backend(
        _Reply(
            debt_reply.final,
            debt_reply.cost_usd,
            MaxTurnsExceeded(
                turns_used=1,
                turns_limit=1,
                message="server returned finish_reason=length",
            ),
            debt_reply.completeness,
            debt_reply.execution_trace,
        ),
        _Reply(json.dumps(_pass_response()), 0.20),
    )
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=1.0,
        response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert len(backend.invocations) == 1
    assert _retry_manifest(corpus, verdict)["terminal_reason"] == (
        "accounting_debt_before_retry"
    )
    telemetry = _call_telemetry(corpus, verdict)
    assert telemetry["terminal_reason"] == "accounting_debt_before_retry"
    assert len(telemetry["attempts"]) == 1
    assert telemetry["attempts"][0]["repair_cause"] == (
        "backend output ended with finish_reason=length before a complete "
        "structured response object; return one concise complete replacement "
        "object without trailing padding"
    )


def test_glm_output_length_retry_is_one_shot(tmp_path):
    backend = _glm_sequence_backend(*(
        _Reply(
            "truncated",
            0.10,
            MaxTurnsExceeded(
                turns_used=1,
                turns_limit=1,
                message="server returned finish_reason=length",
            ),
        )
        for _ in range(2)
    ))
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "backend_max_turns"
    ]
    assert len(backend.invocations) == 2
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "backend_error"
    assert manifest["attempts"][0]["validation_error"] is not None
    assert manifest["attempts"][1]["validation_error"] is None


def test_glm_output_length_preserves_default_one_shot_review(tmp_path):
    backend = _glm_sequence_backend(_Reply(
        "truncated",
        0.10,
        MaxTurnsExceeded(
            turns_used=1,
            turns_limit=1,
            message="server returned finish_reason=length",
        ),
    ))

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts


def test_glm_generic_max_turns_remains_terminal(tmp_path):
    backend = _glm_sequence_backend(_Reply(
        "",
        0.10,
        MaxTurnsExceeded(
            turns_used=5,
            turns_limit=5,
            message="review exhausted its configured turns",
        ),
    ))

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "backend_max_turns"
    ]
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts


def test_non_glm_output_length_remains_terminal(tmp_path):
    backend = _SequenceBackend(_Reply(
        "truncated",
        0.10,
        MaxTurnsExceeded(
            turns_used=1,
            turns_limit=1,
            message="server returned finish_reason=length",
        ),
    ))
    backend.name = "openai_compat"

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "backend_max_turns"
    ]
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts


@pytest.mark.parametrize(
    "finding_id",
    [
        "backend_max_turns",
        "chunk_backend_max_turns",
        "reducer_backend_max_turns",
    ],
)
def test_glm_output_length_detector_covers_all_review_phases(finding_id):
    backend = SimpleNamespace(
        name="openai_compat",
        model="nvidia/zai-org/glm-5.2",
    )
    verdict = SimpleNamespace(
        decision="inconclusive",
        findings=(SimpleNamespace(
            id=finding_id,
            source="deterministic",
            category="backend_error",
            evidence=(SimpleNamespace(
                excerpt="server returned finish_reason=length",
            ),),
        ),),
    )

    assert _retryable_output_length_error(
        backend=backend,
        verdict=verdict,
    ) is not None


def test_exhausted_contract_repair_keeps_only_harness_finding(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    invalid = _crucible_contract_failure_response()
    backend = _SequenceBackend(*(
        _Reply(invalid, 0.10) for _ in range(3)
    ))

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.30)
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["terminal_reason"] == "retries_exhausted"
    assert len(manifest["attempts"]) == 3


def test_response_retries_exhaust_after_n_plus_one_and_do_not_stack_repairs(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    responses = [
        _schema_failure_response("bad_zero", "PRIVATE_ZERO"),
        _schema_failure_response("bad_one", "PRIVATE_ONE"),
        _schema_failure_response("bad_two", "PRIVATE_TWO"),
    ]
    backend = _SequenceBackend(*(
        _Reply(response, 0.10) for response in responses
    ))

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=1.0,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"
    assert verdict.cost_usd == pytest.approx(0.30)
    assert len(backend.invocations) == 3
    assert [item.budget_usd for item in backend.invocations] == pytest.approx(
        [1.0, 0.9, 0.8]
    )

    errors = [
        "invalid review_v2 response: response: unexpected=['bad_zero']",
        "invalid review_v2 response: response: unexpected=['bad_one']",
        "invalid review_v2 response: response: unexpected=['bad_two']",
    ]
    assert json.dumps(errors[0]) in backend.invocations[1].user
    assert json.dumps(errors[1]) in backend.invocations[2].user
    final_repair_envelope = json.loads(
        backend.invocations[2]
        .user.split(
            "BEGIN_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE\n",
            1,
        )[1]
        .split("\nEND_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE", 1)[0]
    )
    assert final_repair_envelope["validation_error"] == errors[1]
    assert final_repair_envelope["prior_validation_errors"] == [
        {
            "validation_error": errors[0],
            "validation_error_sha256": (
                "sha256:" + hashlib.sha256(errors[0].encode()).hexdigest()
            ),
        }
    ]
    assert all(
        invocation.system.count("=== CRUCIBLE DETERMINISTIC RESPONSE REPAIR ===") == 1
        for invocation in backend.invocations[1:]
    )
    assert all(
        private not in invocation.system and private not in invocation.user
        for invocation in backend.invocations[1:]
        for private in ("PRIVATE_ZERO", "PRIVATE_ONE", "PRIVATE_TWO")
    )

    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    assert [
        item["validation_error"] for item in manifest["attempts"]
    ] == errors
    assert manifest["total_cost_usd"] == pytest.approx(0.30)
    _assert_aggregate_transcript(corpus, verdict, manifest)


def test_final_invalid_attempt_at_cap_records_budget_precedence(tmp_path):
    malformed = _schema_failure_response("unexpected", "PRIVATE")
    backend = _SequenceBackend(*(
        _Reply(malformed, 0.01) for _ in range(3)
    ))
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=0.03,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.03)
    assert len(backend.invocations) == 3
    manifest = _retry_manifest(corpus, verdict)
    assert manifest["terminal_reason"] == (
        "budget_exhausted_after_final_attempt"
    )


@pytest.mark.parametrize("case", ["backend_error", "semantic_inconclusive"])
def test_non_repairable_inconclusive_is_not_retried(tmp_path, case):
    if case == "backend_error":
        reply = _Reply(
            "",
            0.15,
            TransportError(http_status=503, message="upstream unavailable"),
        )
        expected_finding = "backend_transport"
    else:
        reply = _Reply(_semantic_inconclusive_response(), 0.15)
        expected_finding = None
    backend = _SequenceBackend(reply)
    corpus = Corpus(tmp_path / case)

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        response_retry_budget=3,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.15)
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts
    if expected_finding is None:
        assert verdict.findings == ()
    else:
        assert verdict.findings[0].id == expected_finding


def test_breaker_complete_coverage_inconclusive_retries_to_valid_pass(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    repaired = _breaker_response(
        decision="pass",
        findings=[],
        reasoning="Complete review found no attack at the severity floor.",
    )
    backend = _SequenceBackend(
        _Reply(_DEGENERATE_BREAKER_RESPONSE, 0.10),
        _Reply(repaired, 0.20),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert verdict.findings == ()
    assert verdict.cost_usd == pytest.approx(0.30)
    assert len(backend.invocations) == 2
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    expected_error = (
        "Rubric-specific response contract failed: impl_breaker: "
        "inconclusive requires coverage.complete=false"
    )
    assert [
        attempt["validation_error"] for attempt in manifest["attempts"]
    ] == [expected_error, None]
    first_transcript = json.loads(corpus.get_artifact(
        manifest["attempts"][0]["transcript_ref"]
    ))
    assert first_transcript[-1] == {
        "role": "assistant",
        "content": _DEGENERATE_BREAKER_RESPONSE,
    }
    _assert_aggregate_transcript(corpus, verdict, manifest)


def test_breaker_candidate_diagnosis_in_oracle_retries_to_clean_spec(tmp_path):
    corpus = Corpus(tmp_path / "corpus")

    def response(oracle: str) -> str:
        return _breaker_response(
            decision="revise",
            reasoning="A grounded correctness attack should be executed.",
            findings=[{
                "id": "correctness-idempotent-abort",
                "criterion_id": "correctness",
                "severity": "High",
                "confidence": 0.95,
                "disposition": "confirmed",
                "category": "correctness",
                "targets": [],
                "evidence": [
                    {
                        "source_id": "context:spec_intent",
                        "kind": "required_behavior",
                        "location": "spec_intent",
                        "excerpt": _BREAKER_INTENT,
                    },
                    {
                        "source_id": "artifact",
                        "kind": "diff",
                        "location": "artifact",
                        "excerpt": _BREAKER_ARTIFACT.decode(),
                    },
                ],
                "proposal": None,
                "attack_spec": {
                    "setup": "Construct the affected object.",
                    "stimulus": "Exercise the operation twice.",
                    "oracle": oracle,
                },
            }],
        )

    contaminated = (
        "The operation is observed exactly once across two calls. "
        "The current diff has no guard, so the second call repeats it."
    )
    clean = "The operation is observed exactly once across two calls."
    backend = _SequenceBackend(
        _Reply(response(contaminated), 0.10),
        _Reply(response(clean), 0.20),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 2
    assert verdict.decision == "revise"
    assert verdict.findings[0].attack_spec.oracle == clean
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    assert "expected observable behavior" in (
        manifest["attempts"][0]["validation_error"]
    )
    assert manifest["attempts"][1]["validation_error"] is None
    _assert_aggregate_transcript(corpus, verdict, manifest)


def test_breaker_placeholder_attack_spec_retries_to_same_id_repair(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    finding_id = "correctness-idempotent-abort"
    backend = _SequenceBackend(
        _Reply(_impl_breaker_attack_response(
            finding_id=finding_id,
            setup="...",
            stimulus="...",
            oracle="...",
        ), 0.10),
        _Reply(_impl_breaker_attack_response(
            finding_id=finding_id,
            setup="Construct the affected object.",
            stimulus="Exercise the operation twice.",
            oracle="The operation is observed exactly once across two calls.",
        ), 0.20),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
    )

    assert len(backend.invocations) == 2
    assert verdict.decision == "revise"
    assert [finding.id for finding in verdict.findings] == [finding_id]
    assert verdict.findings[0].attack_spec.setup == (
        "Construct the affected object."
    )
    assert verdict.findings[0].attack_spec.stimulus == (
        "Exercise the operation twice."
    )
    assert verdict.findings[0].attack_spec.oracle == (
        "The operation is observed exactly once across two calls."
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first_error = manifest["attempts"][0]["validation_error"]
    assert isinstance(first_error, str)
    for field_name in ("setup", "stimulus", "oracle"):
        assert f".attack_spec.{field_name}" in first_error
    assert "non-placeholder" in first_error
    assert manifest["attempts"][1]["validation_error"] is None


def test_breaker_placeholder_attack_spec_exhaustion_fails_closed(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    placeholder = _impl_breaker_attack_response(
        finding_id="correctness-idempotent-abort",
        setup="...",
        stimulus="...",
        oracle="...",
    )
    backend = _SequenceBackend(
        _Reply(placeholder, 0.10),
        _Reply(placeholder, 0.20),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
    )

    assert len(backend.invocations) == 2
    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "retries_exhausted"
    for attempt in manifest["attempts"]:
        error = attempt["validation_error"]
        assert isinstance(error, str)
        assert ".attack_spec.setup" in error
        assert ".attack_spec.stimulus" in error
        assert ".attack_spec.oracle" in error
        assert "non-placeholder" in error


def test_breaker_attack_spec_repair_identity_drift_fails_closed(tmp_path):
    """The attack-spec identity ledger forbids substitution during repair."""

    corpus = Corpus(tmp_path / "corpus")
    original_id = "correctness-idempotent-abort"
    substituted_id = "correctness-renamed-during-repair"
    backend = _SequenceBackend(
        _Reply(_impl_breaker_attack_response(
            finding_id=original_id,
            setup="...",
            stimulus="Exercise the operation twice.",
            oracle="The operation is observed exactly once across two calls.",
        ), 0.10),
        _Reply(_impl_breaker_attack_response(
            finding_id=substituted_id,
            setup="Construct the affected object.",
            stimulus="Exercise the operation twice.",
            oracle="The operation is observed exactly once across two calls.",
        ), 0.20),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
    )

    assert len(backend.invocations) == 2
    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "retries_exhausted"
    first_error = manifest["attempts"][0]["validation_error"]
    assert isinstance(first_error, str)
    assert ".attack_spec.setup" in first_error
    identity_error = manifest["attempts"][1]["validation_error"]
    assert isinstance(identity_error, str)
    assert "ordered finding ids" in identity_error.lower()
    assert original_id in identity_error
    assert substituted_id in identity_error


def test_breaker_complete_coverage_inconclusive_exhaustion_fails_closed(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _SequenceBackend(*(
        _Reply(_DEGENERATE_BREAKER_RESPONSE, 0.10) for _ in range(3)
    ))

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert verdict.findings[0].source == "deterministic"
    assert verdict.cost_usd == pytest.approx(0.30)
    assert len(backend.invocations) == 3
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    expected_error = (
        "Rubric-specific response contract failed: impl_breaker: "
        "inconclusive requires coverage.complete=false"
    )
    assert {
        attempt["validation_error"] for attempt in manifest["attempts"]
    } == {expected_error}
    _assert_aggregate_transcript(corpus, verdict, manifest)


@pytest.mark.parametrize(
    "finding_id",
    ["backend_transport", "chunk_backend_transport", "reducer_backend_timeout"],
)
def test_backend_terminal_reason_uses_typed_category_not_id_prefix(finding_id):
    verdict = SimpleNamespace(findings=(SimpleNamespace(
        id=finding_id,
        category="backend_error",
    ),))

    assert _deterministic_terminal_reason(verdict) == "backend_error"


def test_response_retry_default_zero_preserves_single_attempt(tmp_path):
    backend = _SequenceBackend(_Reply(
        _schema_failure_response("unexpected", "PRIVATE"),
        0.05,
    ))

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
    )

    assert verdict.decision == "inconclusive"
    assert verdict.findings[0].id == "schema_failure"
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts


def test_response_retry_does_not_start_when_first_attempt_consumes_cap(tmp_path):
    backend = _SequenceBackend(_Reply(
        _schema_failure_response("unexpected", "PRIVATE"),
        0.10,
    ))
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=0.10,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.10)
    assert len(backend.invocations) == 1
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=1)
    assert manifest["terminal_reason"] == "budget_exhausted_before_retry"


def test_response_retry_overrun_stops_before_third_call(tmp_path):
    malformed = _schema_failure_response("unexpected", "PRIVATE")
    backend = _SequenceBackend(
        _Reply(malformed, 0.05),
        _Reply(malformed, 0.10),
    )
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=0.10,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.15)
    assert len(backend.invocations) == 2
    assert [item.budget_usd for item in backend.invocations] == pytest.approx(
        [0.10, 0.05]
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "budget_exhausted_before_retry"


def test_successful_repair_that_overruns_aggregate_cap_fails_closed(tmp_path):
    backend = _SequenceBackend(
        _Reply(_schema_failure_response("unexpected", "PRIVATE"), 0.05),
        _Reply(json.dumps(_pass_response()), 0.10),
    )
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        _ARTIFACT,
        load("spec_clarity"),
        backend=backend,
        corpus=corpus,
        budget_usd=0.10,
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert verdict.cost_usd == pytest.approx(0.15)
    assert len(backend.invocations) == 2
    assert [finding.id for finding in verdict.findings] == [
        "review_response_retry_budget_exhausted",
    ]
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "budget_overrun_after_retry"


def _obligations_from_invocation(invocation: Invocation) -> dict[str, object]:
    encoded = invocation.user.split(
        "BEGIN_CRUCIBLE_REVIEW_OBLIGATIONS\n", 1,
    )[1].split("\nEND_CRUCIBLE_REVIEW_OBLIGATIONS", 1)[0]
    return json.loads(encoded)


def test_plan_judge_obligations_bind_slice_intents_and_shape_alias_guidance(
    tmp_path,
):
    plan = json.loads(_CANONICAL_PLAN_ARTIFACT)
    long_intent = (
        "Load configuration into an independently landable foundation. "
        + "Preserve every exact configuration invariant. " * 20
    )
    assert len(long_intent) > 512
    plan["tiers"][0]["slices"][0]["directive"]["intent"] = long_intent
    artifact = json.dumps(plan, ensure_ascii=False, indent=2).encode()
    context, prior_ref = _large_planning_history_context(2)
    backend = _glm_sequence_backend(_Reply(
        _plan_judge_active_frontier_revise_response(prior_ref), 0.0,
    ))

    verdict = review(
        artifact,
        load("plan_judge"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context=context,
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    invocation = backend.invocations[0]
    envelope = _obligations_from_invocation(invocation)
    obligations = envelope["obligations"]
    expected_identities = [
        {
            "slice_id": candidate_slice["id"],
            "intent_sha256": (
                "sha256:"
                + hashlib.sha256(
                    candidate_slice["directive"]["intent"].encode("utf-8")
                ).hexdigest()
            ),
        }
        for candidate_slice in sorted(
            plan["tiers"][0]["slices"], key=lambda item: item["id"]
        )
    ]
    assert obligations["current_plan_slice_identities"] == expected_identities
    assert long_intent not in json.dumps(obligations, ensure_ascii=False)
    encoded_obligations = json.dumps(
        obligations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert envelope["obligations_sha256"] == (
        "sha256:" + hashlib.sha256(encoded_obligations).hexdigest()
    )
    assert len(obligations["material_prior_aliases"]) == 1
    assert "sole entry in exactly one ``finding.targets`` array" in (
        invocation.system
    )
    assert "use ``confirmed``; never emit partial proof JSON" in (
        invocation.system
    )
    assert "current_plan_slice_identities" in invocation.system
    assert "copy the exact binding" in invocation.system


def test_repaired_plan_gates_leave_frontier_and_reach_judge(tmp_path):
    context = _repaired_plan_gate_context()
    audit_history = _context_prior_findings(dict(context))
    closures = _verified_plan_gate_closures(
        rubric_name="plan_judge",
        prompt_artifact_text=_CANONICAL_PLAN_ARTIFACT.decode(),
        prior_findings=audit_history,
    )
    assert {
        item["id"] for item in audit_history if item["ref"] in closures
    } == {
        "planner_backend_error",
        "planner_parse_failure",
        "parallel_conflict_t0_config_alias",
        "empty_scope_filler_3_verify",
        "plan_size_bound_exceeded",
    }

    backend = _SequenceBackend(_Reply(
        _plan_judge_empty_frontier_pass_response(), 0.0,
    ))
    verdict = review(
        _CANONICAL_PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context=context,
    )

    assert verdict.decision == "pass"
    obligations = _obligations_from_invocation(backend.invocations[0])[
        "obligations"
    ]
    assert obligations["material_prior_bindings"] == []
    assert set(obligations["verified_deterministic_closures"]) == closures


def test_plan_gate_clearance_keeps_infrastructure_and_current_defects_active(
    tmp_path,
):
    context = _repaired_plan_gate_context(
        include_infrastructure_failure=True,
    )
    audit_history = _context_prior_findings(dict(context))
    infrastructure_ref = next(
        item["ref"] for item in audit_history
        if item["id"] == "rubric_contract_failure"
    )
    closures = _verified_plan_gate_closures(
        rubric_name="plan_judge",
        prompt_artifact_text=_CANONICAL_PLAN_ARTIFACT.decode(),
        prior_findings=audit_history,
    )
    assert infrastructure_ref not in closures

    unsafe = json.loads(_CANONICAL_PLAN_ARTIFACT)
    unsafe["tiers"][0]["slices"][1]["directive"]["files_in_scope"] = [
        "src/config.py"
    ]
    unsafe["tiers"][0]["slices"].append({
        "id": "verify-only",
        "directive": {
            "slice_id": "verify-only",
            "intent": "verify the current plan",
            "files_in_scope": [],
            "creates_paths": [],
            "verify_mode": "tests",
            "blast_radius": {"max_files": 0},
        },
    })
    unsafe_closures = _verified_plan_gate_closures(
        rubric_name="plan_judge",
        prompt_artifact_text=json.dumps(unsafe),
        prior_findings=audit_history,
    )
    active_ids = {
        item["id"] for item in audit_history
        if item["ref"] not in unsafe_closures
    }
    assert "parallel_conflict_t0_config_alias" in active_ids
    assert "empty_scope_filler_3_verify" in active_ids

    backend = _SequenceBackend(_Reply(
        _plan_judge_empty_frontier_pass_response(), 0.0,
    ))
    verdict = review(
        _CANONICAL_PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context=context,
    )
    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure"
    ]


def test_planning_history_projects_oversized_evidence_without_mutating_audit():
    oversized_excerpt = "é" * 600
    short_excerpt = "retain this exact short excerpt"
    context = {
        "prior_critic_findings": json.dumps(
            [
                {
                    "round": 0,
                    "decision": "revise",
                    "source": "critic",
                    "findings": [
                        {
                            "id": "materialized-scalar",
                            "criterion_id": "intent_coverage",
                            "severity": "High",
                            "disposition": "confirmed",
                            "category": "other",
                            "targets": [],
                            "evidence": [
                                {
                                    "source_id": "context:spec_constraints",
                                    "kind": "current_state",
                                    "location": "json:$.constraints[0]",
                                    "excerpt": oversized_excerpt,
                                },
                                {
                                    "source_id": "artifact",
                                    "kind": "current_state",
                                    "location": "tier:t0",
                                    "excerpt": short_excerpt,
                                },
                            ],
                            "proposal": None,
                            "attack_spec": None,
                        }
                    ],
                }
            ]
        ),
        "prior_judge_findings": "[]",
    }
    audit_history = _context_prior_findings(context)
    audit_payload = audit_history[0]["payload"]
    audit_payload_before = json.loads(json.dumps(audit_payload))
    finding_ref = audit_history[0]["ref"]

    _compact_planning_history_context(
        context,
        target_closure="all_material",
        prior_findings=audit_history,
    )

    projected = json.loads(context["prior_critic_findings"])[0]["finding"]
    projected_oversized, projected_short = projected["evidence"]
    encoded_excerpt = oversized_excerpt.encode("utf-8")
    assert projected["finding_ref"] == finding_ref
    assert projected_oversized == {
        "source_id": "context:spec_constraints",
        "kind": "current_state",
        "location": "json:$.constraints[0]",
        "excerpt_binding": {
            "schema_version": "crucible.plan-history-evidence-excerpt.v1",
            "sha256": ("sha256:" + hashlib.sha256(encoded_excerpt).hexdigest()),
            "chars": len(oversized_excerpt),
            "bytes": len(encoded_excerpt),
            "omitted_from_prompt": True,
        },
    }
    assert projected_short["excerpt"] == short_excerpt
    assert "excerpt_binding" not in projected_short
    assert oversized_excerpt not in context["prior_critic_findings"]
    assert audit_history[0]["payload"] == audit_payload_before
    assert audit_history[0]["payload"]["evidence"][0]["excerpt"] == (oversized_excerpt)


def test_planning_prompt_size_depends_on_active_frontier_not_full_history(
    tmp_path,
):
    prompt_lengths: list[int] = []
    for closed_rounds in (2, 100):
        context, prior_ref = _large_planning_history_context(closed_rounds)
        backend = _SequenceBackend(_Reply(
            _plan_judge_active_frontier_revise_response(prior_ref), 0.0,
        ))
        verdict = review(
            _CANONICAL_PLAN_ARTIFACT,
            load("plan_judge"),
            backend=backend,
            corpus=Corpus(tmp_path / f"corpus-{closed_rounds}"),
            context=context,
        )

        assert verdict.decision == "revise"
        prompt = backend.invocations[0].user
        assert "active-blocker" in prompt
        assert '"closed-0"' not in prompt
        assert f'"closed-{closed_rounds - 1}"' not in prompt
        prompt_lengths.append(len(prompt))

    assert abs(prompt_lengths[1] - prompt_lengths[0]) < 128


def test_v24_sized_plan_judge_stays_one_shot_and_adjudicates_prior(
    tmp_path,
):
    plan = json.loads(_CANONICAL_PLAN_ARTIFACT)
    plan["tiers"][0]["slices"][0]["directive"]["intent"] = "A" * 66_000
    artifact = json.dumps(plan, ensure_ascii=False, indent=2).encode()

    context, _ = _large_planning_history_context(0)
    active_record = json.loads(context["prior_critic_findings"])
    oversized_excerpt = "E" * 25_000
    active_record[0]["findings"][0]["evidence"] = [
        {
            "source_id": "context:spec_constraints",
            "kind": "current_state",
            "location": "json:$.constraints[0]",
            "excerpt": oversized_excerpt,
        }
    ]
    active_record[0]["findings"][0]["proposal"] = {
        "action": "P" * 35_000,
        "replacement": None,
    }
    context.update(
        {
            "spec_intent": "I" * 38_133,
            "spec_constraints": oversized_excerpt,
            "spec_done_criteria": "D" * 1_075,
            "prior_critic_findings": json.dumps(active_record),
        }
    )
    prior_ref = _context_prior_findings(dict(context))[0]["ref"]

    retained_chars = len(artifact.decode()) + sum(map(len, context.values()))
    assert retained_chars < _MAX_ARTIFACT_CHARS

    backend = _SequenceBackend(
        _Reply(
            _plan_judge_active_frontier_revise_response(prior_ref),
            0.0,
        )
    )
    verdict = review(
        artifact,
        load("plan_judge"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context=context,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 1
    invocation = backend.invocations[0]
    assert "assay.review.chunk.v1" not in invocation.user
    assert retained_chars + len(invocation.system) > _MAX_ARTIFACT_CHARS
    assert len(invocation.system) + len(invocation.user) < _MAX_ARTIFACT_CHARS
    assert invocation.user.count(oversized_excerpt) == 1
    assert "crucible.plan-history-evidence-excerpt.v1" in invocation.user
    assert verdict.findings[0].targets == (prior_ref,)


def test_critic_prompt_context_omits_planning_debate_history():
    context, _ = _large_planning_history_context(20)
    audit_history = _context_prior_findings(context)

    _compact_planning_history_context(
        context,
        target_closure="none",
        prior_findings=audit_history,
    )

    assert "prior_critic_findings" not in context
    assert "prior_judge_findings" not in context


@pytest.mark.parametrize(
    ("decision", "disposition", "declared_sources", "expected_decision"),
    [
        # GLM 13236 attempt 0: the target and independent artifact evidence are
        # valid; reject still fails because this defect is repairable.
        ("reject", "confirmed", [
            "artifact",
            "context:spec_intent",
            "context:spec_constraints",
            "context:spec_done_criteria",
            "context:prior_critic_findings",
            "context:prior_judge_findings",
        ], "inconclusive"),
        # Attempt 1: missing model bookkeeping is normalized from the supplied
        # harness inventory rather than consuming another semantic call.
        ("revise", "confirmed", [
            "artifact",
            "context:spec_intent",
            "context:prior_critic_findings",
            "context:prior_judge_findings",
        ], "revise"),
        # Attempt 2: a complete model-authored inventory remains equivalent.
        ("revise", "confirmed", [
            "artifact",
            "context:spec_intent",
            "context:spec_constraints",
            "context:spec_done_criteria",
            "context:prior_critic_findings",
            "context:prior_judge_findings",
        ], "revise"),
    ],
)
def test_plan_judge_13236_response_modes_use_harness_owned_coverage(
    tmp_path,
    decision,
    disposition,
    declared_sources,
    expected_decision,
):
    corpus = Corpus(tmp_path / "corpus")
    response = _plan_judge_response(
        decision=decision,
        disposition=disposition,
        sources_reviewed=declared_sources,
    )
    backend = _SequenceBackend(_Reply(response, 0.0))

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
    )

    required_sources = (
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    )
    assert verdict.decision == expected_decision
    assert verdict.coverage.sources_reviewed == required_sources
    assert verdict.coverage.complete is (decision != "reject")
    if decision != "reject":
        assert verdict.coverage.omissions == ()
    normalization = json.loads(corpus.get_artifact(
        verdict.artifacts["review_coverage_normalization"]
    ))
    assert normalization["model_declared_sources_reviewed"] == declared_sources
    assert normalization["normalized_sources_reviewed"] == list(required_sources)
    # The verdict may replace its transcript with an aggregate on retries;
    # this one-shot case always points at the exact raw exchange.
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assistant = next(item for item in transcript if item["role"] == "assistant")
    assert json.loads(assistant["content"])["coverage"][
        "sources_reviewed"
    ] == declared_sources
    if decision == "reject":
        assert "bounded repair requires decision='revise'" in verdict.reasoning
        assert "linked prior evidence" not in verdict.reasoning


def test_glm_plan_judge_downgrades_failed_json_evidence_to_anchors_once(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorOnlyGlmPlanJudgeBackend()

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert all(
        evidence.location == f"chars:0-{len(_PLAN_ARTIFACT)}"
        and evidence.excerpt == _PLAN_ARTIFACT.decode()
        for finding in verdict.findings
        for evidence in finding.evidence
    )
    repair = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert repair["policy"]["materialize_source_anchors"] is True
    assert repair["policy"]["materialize_json_paths"] is False
    assert {
        action["operation"] for action in repair["actions"]
    } == {"materialize_source_anchor"}
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    assert "unresolved deterministic evidence reference" in (
        manifest["attempts"][0]["validation_error"]
    )
    assert manifest["attempts"][1]["validation_error"] is None


def test_initial_anchor_only_plan_review_rejects_json_path_and_repairs_once(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorOnlyGlmPlanJudgeBackend(anchor_only_from_start=True)

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
        review_anchor_policy="anchors_only",
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 2
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        for invocation in backend.invocations
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    assert "unresolved deterministic evidence reference" in (
        manifest["attempts"][0]["validation_error"]
    )
    assert manifest["attempts"][1]["validation_error"] is None


def test_glm_plan_judge_materializes_container_json_path_without_retry(
    tmp_path,
):
    artifact = json.dumps({
        "tiers": [
            {"tier_id": "t0", "depends_on": [], "slices": []},
            {"tier_id": "t1", "depends_on": ["t0"], "slices": []},
        ],
    }, indent=2).encode()
    context = {
        "spec_intent": "Change production behavior.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Production behavior changed.",
        "prior_critic_findings": "[]",
        "prior_judge_findings": "[]",
    }
    artifact_text = artifact.decode()
    criterion_results = [
        {
            "criterion_id": "intent_coverage",
            "outcome": "satisfied",
            "reasoning": "The root behavior remains covered.",
            "evidence": [{
                "source_id": "context:spec_intent",
                "kind": "root_requirement",
                "location": "spec_intent",
                "excerpt": context["spec_intent"],
            }],
            "headline": True,
        },
        {
            "criterion_id": "executability",
            "outcome": "satisfied",
            "reasoning": "The tier dependency is explicit.",
            "evidence": [{
                "source_id": "artifact",
                "kind": "tier_dependency",
                "location": "json:$.tiers[1].depends_on",
                "excerpt": "[t0]",
            }],
            "headline": False,
        },
        {
            "criterion_id": "critic_disposition",
            "outcome": "not_applicable",
            "reasoning": "No active critic finding exists.",
            "evidence": [{
                "source_id": "context:prior_critic_findings",
                "kind": "active_frontier",
                "location": "prior_critic_findings",
                "excerpt": "[]",
            }],
            "headline": False,
        },
        {
            "criterion_id": "contradiction",
            "outcome": "satisfied",
            "reasoning": "The tier declarations do not contradict.",
            "evidence": [{
                "source_id": "artifact",
                "kind": "tier_dag",
                "location": "tier_id",
                "excerpt": '"tier_id": "t0"',
            }],
            "headline": False,
        },
    ]
    response = json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The complete plan is covered and executable.",
        "coverage": {
            "complete": True,
            "sources_reviewed": [
                "artifact",
                *(f"context:{key}" for key in context),
            ],
            "omissions": [],
        },
        "findings": [],
        "criterion_results": criterion_results,
    })
    backend = _SequenceBackend(_Reply(response, 0.0))
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"
    corpus = Corpus(tmp_path / "corpus")

    verdict = review(
        artifact,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=context,
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 1
    executability = next(
        item for item in verdict.criterion_results
        if item.criterion_id == "executability"
    )
    (evidence,) = executability.evidence
    expected = '[\n        "t0"\n      ]'
    assert evidence.excerpt == expected
    start = artifact_text.index(expected)
    assert evidence.location == f"chars:{start}-{start + len(expected)}"
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_response_repair"]
    ))
    assert {
        action["operation"] for action in report["actions"]
    } == {"materialize_json_container_path"}
    assert "review_retry_manifest" not in verdict.artifacts


def test_glm_plan_judge_canonicalizes_unproved_closure_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    context = {
        "spec_intent": "Load configuration and validate aliases.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Both behaviors remain supported.",
        "prior_critic_findings": json.dumps([{
            "id": "split-responsibilities",
            "criterion_id": "responsibility_size",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
            "evidence": [],
            "proposal": {
                "action": "Split loading and validation into independent slices.",
                "replacement": None,
            },
            "attack_spec": None,
        }]),
        "prior_judge_findings": "[]",
    }
    prior_ref = _context_prior_findings(dict(context))[0]["ref"]
    required_sources = [
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    ]
    evidence = [{
        "source_id": "artifact",
        "kind": "current_state",
        "location": "artifact",
        "excerpt": _PLAN_ARTIFACT.decode(),
    }]
    response = json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "The model claimed the prior was atomic.",
        "coverage": {
            "complete": True,
            "sources_reviewed": required_sources,
            "omissions": [],
        },
        "findings": [{
            "id": "close-split-responsibilities",
            "criterion_id": "critic_disposition",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "dismissed",
            "category": "other",
            "targets": [prior_ref],
            "evidence": evidence,
            "proposal": {
                "action": "ATOMICITY_PROOF: {",
                "replacement": None,
            },
            "attack_spec": None,
        }],
        "criterion_results": [
            {
                "criterion_id": criterion,
                "outcome": "satisfied",
                "reasoning": "The model claimed this criterion was satisfied.",
                "evidence": evidence,
                "headline": index == 0,
            }
            for index, criterion in enumerate((
                "intent_coverage",
                "executability",
                "critic_disposition",
                "contradiction",
            ))
        ],
    })
    backend = _SequenceBackend(_Reply(response, 0.0))
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=context,
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 1
    assert verdict.decision == "revise"
    assert verdict.findings[0].category == "unresolved_critic"
    assert verdict.findings[0].disposition == "confirmed"
    assert verdict.findings[0].proposal.action == (
        "Split loading and validation into independent slices."
    )
    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_contract_canonicalization"]
    ))
    assert report["schema_version"] == (
        "crucible.review-contract-canonicalization.v1"
    )
    assert report["contract_errors_after"] == []
    assert report["accepted"] is True
    assert report["acceptance_reason"] == "valid"
    assert "raw_response_ref" in report
    assert report["transcript_ref"] is not None
    assert {
        action["operation"] for action in report["actions"]
    } == {
        "canonicalize_singleton_category",
        "retain_unproved_prior_as_confirmed",
        "restore_unproved_prior_proposal",
        "restore_unproved_suggested_action",
        "retain_unproved_criterion_as_violated",
        "record_unproved_criterion_reasoning",
        "force_fail_conservative_revision",
        "record_fail_conservative_reasoning",
    }


def test_glm_plan_judge_canonicalizes_grouped_unproved_closure_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    context = {
        "spec_intent": "Load configuration and validate aliases.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Both behaviors remain supported.",
        "prior_critic_findings": json.dumps([{
            "id": "split-responsibilities",
            "criterion_id": "responsibility_size",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
            "evidence": [],
            "proposal": {
                "action": "Split loading and validation into independent slices.",
                "replacement": None,
            },
            "attack_spec": None,
        }]),
        "prior_judge_findings": json.dumps([{
            "id": "retain-verification-boundary",
            "criterion_id": "critic_disposition",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "confirmed",
            "category": "unresolved_critic",
            "targets": [],
            "evidence": [],
            "proposal": {
                "action": "Retain the independent verification boundary.",
                "replacement": None,
            },
            "attack_spec": None,
        }]),
    }
    prior_refs = {
        prior["id"]: prior["ref"]
        for prior in _context_prior_findings(dict(context))
    }
    grouped_targets = [
        prior_refs["split-responsibilities"],
        prior_refs["retain-verification-boundary"],
    ]
    required_sources = [
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    ]
    evidence = [{
        "source_id": "artifact",
        "kind": "current_state",
        "location": "artifact",
        "excerpt": _PLAN_ARTIFACT.decode(),
    }]
    grouped_action = "REPARTITION_PROOF: {"
    response = json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "One independent execution conflict remains.",
        "coverage": {
            "complete": True,
            "sources_reviewed": required_sources,
            "omissions": [],
        },
        "findings": [
            {
                "id": "grouped-prior-closure",
                "criterion_id": "critic_disposition",
                "severity": "High",
                "confidence": 0.9,
                "disposition": "addressed",
                "category": "unresolved_critic",
                "targets": grouped_targets,
                "evidence": evidence,
                "proposal": {
                    "action": grouped_action,
                    "replacement": None,
                },
                "attack_spec": None,
            },
            {
                "id": "independent-execution-conflict",
                "criterion_id": "executability",
                "severity": "High",
                "confidence": 0.9,
                "disposition": "confirmed",
                "category": "parallel_conflict",
                "targets": [],
                "evidence": evidence,
                "proposal": {
                    "action": "Remove the overlapping write.",
                    "replacement": None,
                },
                "attack_spec": None,
            },
        ],
        "criterion_results": [
            {
                "criterion_id": criterion,
                "outcome": (
                    "violated" if criterion == "executability"
                    else "not_applicable" if criterion == "contradiction"
                    else "satisfied"
                ),
                "reasoning": "The current candidate was checked.",
                "evidence": evidence,
                "headline": criterion == "executability",
            }
            for criterion in (
                "intent_coverage",
                "executability",
                "critic_disposition",
                "contradiction",
            )
        ],
    })
    backend = _SequenceBackend(_Reply(response, 0.0))
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=context,
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 1
    assert verdict.decision == "revise"
    grouped = verdict.findings[0]
    assert grouped.disposition == "confirmed"
    assert grouped.targets == tuple(grouped_targets)
    assert [
        {
            "source_id": item.source_id,
            "kind": item.kind,
            "location": item.location,
            "excerpt": item.excerpt,
        }
        for item in grouped.evidence
    ] == evidence
    assert grouped.proposal.action == grouped_action
    assert grouped.proposal.replacement is None
    assert grouped.suggested_action == grouped_action

    report_ref = verdict.artifacts["review_contract_canonicalization"]
    report = json.loads(corpus.get_artifact(report_ref))
    assert report["accepted"] is True
    assert report["acceptance_reason"] == "valid"
    assert report["contract_errors_after"] == []
    assert len(report["contract_errors_before"]) == 1
    assert "requires REPARTITION_PROOF JSON" in (
        report["contract_errors_before"][0]
    )
    assert {action["path"] for action in report["actions"]} == {
        "findings[0].disposition",
        "criterion_results[2].outcome",
        "criterion_results[2].reasoning",
        "reasoning",
    }
    assert report["transcript_ref"] == verdict.transcript_ref
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript[-1] == {"role": "assistant", "content": response}
    assert "review_retry_manifest" not in verdict.artifacts


def test_glm_impl_breaker_canonicalizes_singleton_categories_without_retry(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    evidence = [
        {
            "source_id": "context:spec_intent",
            "kind": "issue_text",
            "location": "spec_intent",
            "excerpt": _BREAKER_INTENT,
        },
        {
            "source_id": "artifact",
            "kind": "diff",
            "location": "artifact",
            "excerpt": _BREAKER_ARTIFACT.decode(),
        },
    ]
    attack_specs = [
        {
            "setup": "Construct a table from a structured ndarray.",
            "stimulus": "Add the structured array as a column.",
            "oracle": "The future behavior avoids NdarrayMixin conversion.",
        },
        {
            "setup": "Construct a table with the compatibility clause active.",
            "stimulus": "Exercise the warning path before version 5.2.",
            "oracle": "The compatibility warning is emitted exactly once.",
        },
    ]
    raw_findings = [
        {
            "id": finding_id,
            "criterion_id": "edge_case",
            "severity": severity,
            "confidence": confidence,
            "disposition": "confirmed",
            # ``edge_case`` is the sole allowed category for this criterion.
            "category": "correctness",
            "targets": [],
            "evidence": evidence,
            "proposal": None,
            "attack_spec": attack_spec,
        }
        for finding_id, severity, confidence, attack_spec in (
            (
                "edge-case-structured-array-conversion",
                "Critical",
                0.95,
                attack_specs[0],
            ),
            (
                "edge-case-compatibility-warning",
                "High",
                0.9,
                attack_specs[1],
            ),
        )
    ]
    response = json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "Both grounded edge attacks remain active.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact", "context:spec_intent"],
            "omissions": [],
        },
        "findings": raw_findings,
    })
    backend = _SequenceBackend(_Reply(response, 0.0))
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"

    verdict = review(
        _BREAKER_ARTIFACT,
        load("impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 1
    assert verdict.decision == "revise"
    assert verdict.reasoning == "Both grounded edge attacks remain active."
    assert [finding.id for finding in verdict.findings] == [
        item["id"] for item in raw_findings
    ]
    assert [finding.category for finding in verdict.findings] == [
        "edge_case",
        "edge_case",
    ]
    assert [
        {
            "severity": finding.severity,
            "confidence": finding.confidence,
            "disposition": finding.disposition,
            "targets": list(finding.targets),
            "proposal": finding.proposal,
            "evidence": [
                {
                    "source_id": item.source_id,
                    "kind": item.kind,
                    "location": item.location,
                    "excerpt": item.excerpt,
                }
                for item in finding.evidence
            ],
            "attack_spec": {
                "setup": finding.attack_spec.setup,
                "stimulus": finding.attack_spec.stimulus,
                "oracle": finding.attack_spec.oracle,
            },
        }
        for finding in verdict.findings
    ] == [
        {
            "severity": item["severity"],
            "confidence": item["confidence"],
            "disposition": item["disposition"],
            "targets": item["targets"],
            "proposal": item["proposal"],
            "evidence": item["evidence"],
            "attack_spec": item["attack_spec"],
        }
        for item in raw_findings
    ]

    report_ref = verdict.artifacts["review_contract_canonicalization"]
    report = json.loads(corpus.get_artifact(report_ref))
    assert report["accepted"] is True
    assert report["acceptance_reason"] == "valid"
    assert len(report["contract_errors_before"]) == 2
    assert all(
        "uses category 'correctness'" in error
        and "expected ('edge_case',)" in error
        for error in report["contract_errors_before"]
    )
    assert report["contract_errors_after"] == []
    assert {
        (action["operation"], action["path"])
        for action in report["actions"]
    } == {
        ("canonicalize_singleton_category", "findings[0].category"),
        ("canonicalize_singleton_category", "findings[1].category"),
    }
    assert report["transcript_ref"] == verdict.transcript_ref
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript[-1] == {"role": "assistant", "content": response}
    assert "review_retry_manifest" not in verdict.artifacts


def test_glm_plan_critic_salvages_grounded_blocker_without_retry(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    directive = "Add alias support while preserving compatibility."
    paired_evidence = [
        {
            "source_id": "artifact",
            "kind": "tier_dag",
            "location": "artifact",
            "excerpt": _PLAN_ARTIFACT.decode(),
        },
        {
            "source_id": "context:spec_intent",
            "kind": "directive",
            "location": "spec_intent",
            "excerpt": directive,
        },
    ]
    raw_findings = [
        {
            "id": "independent-alias-responsibilities",
            "criterion_id": "responsibility_size",
            "severity": "High",
            "confidence": 0.95,
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
            "evidence": paired_evidence,
            "proposal": {
                "action": (
                    "Split alias parsing from independently landable "
                    "compatibility wiring."
                ),
                "replacement": None,
            },
            "attack_spec": None,
        },
        {
            "id": "sequence-alias-wiring",
            "criterion_id": "sequencing",
            "severity": "Medium",
            "confidence": 0.8,
            "disposition": "confirmed",
            "category": "missing_sequencing",
            "targets": [],
            # Deliberately malformed: the DeepSWE critic requires each
            # finding to pair artifact and directive evidence.
            "evidence": paired_evidence[:1],
            "proposal": {
                "action": (
                    "Make compatibility wiring depend on alias parsing."
                ),
                "replacement": None,
            },
            "attack_spec": None,
        },
    ]
    response = json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "revise",
        "reasoning": "The plan contains independently actionable problems.",
        "coverage": {
            "complete": True,
            "sources_reviewed": ["artifact", "context:spec_intent"],
            "omissions": [],
        },
        "findings": raw_findings,
    })
    backend = _SequenceBackend(_Reply(response, 0.0))
    backend.name = "openai_compat"
    backend.model = "nvidia/zai-org/glm-5.2"

    verdict = review(
        _PLAN_ARTIFACT,
        load("deepswe_plan_critic"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": directive},
        response_retry_budget=2,
    )

    assert len(backend.invocations) == 1
    assert verdict.decision == "revise"
    assert [finding.id for finding in verdict.findings] == [
        "independent-alias-responsibilities"
    ]
    retained = verdict.findings[0]
    assert retained.severity == "High"
    assert retained.disposition == "confirmed"
    assert [item.source_id for item in retained.evidence] == [
        "artifact",
        "context:spec_intent",
    ]

    report = json.loads(corpus.get_artifact(
        verdict.artifacts["review_contract_canonicalization"]
    ))
    assert report["accepted"] is True
    assert report["acceptance_reason"] == "valid"
    assert len(report["contract_errors_before"]) == 1
    assert "finding[1] requires paired evidence" in (
        report["contract_errors_before"][0]
    )
    assert report["contract_errors_after"] == []
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", report["input_verdict_sha256"]
    )
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", report["output_verdict_sha256"]
    )
    assert report["input_verdict_sha256"] != report["output_verdict_sha256"]
    assert len(report["actions"]) == 1
    (action,) = report["actions"]
    assert action["operation"] == "drop_contract_invalid_critic_finding"
    assert action["path"] == "findings[1]"
    assert action["finding_id"] == "sequence-alias-wiring"
    assert len(action["validation_errors"]) == 1
    assert "finding[0] requires paired evidence" in (
        action["validation_errors"][0]
    )
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", action["before_sha256"])
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}", action["validation_errors_sha256"]
    )
    assert report["transcript_ref"] == verdict.transcript_ref
    transcript = json.loads(corpus.get_artifact(verdict.transcript_ref))
    assert transcript[-1] == {"role": "assistant", "content": response}
    assert "review_retry_manifest" not in verdict.artifacts


def test_glm_plan_judge_rejects_canonicalization_with_new_contract_error(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    context = {
        "spec_intent": "Load configuration and validate aliases.",
        "spec_constraints": "[]",
        "spec_done_criteria": "Both behaviors remain supported.",
        "prior_critic_findings": json.dumps([{
            "id": "split-responsibilities",
            "criterion_id": "responsibility_size",
            "severity": "High",
            "confidence": 0.9,
            "disposition": "confirmed",
            "category": "under_decomposed",
            "targets": [],
            "evidence": [],
            "proposal": {
                "action": "Split loading and validation into independent slices.",
                # The retained prior is valid history, but this replacement is
                # forbidden on a new plan_judge response. Restoring it makes
                # the canonical candidate strictly no safer than the input.
                "replacement": "retained prior replacement",
            },
            "attack_spec": None,
        }]),
        "prior_judge_findings": "[]",
    }
    prior_ref = _context_prior_findings(dict(context))[0]["ref"]
    required_sources = [
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    ]
    evidence = [{
        "source_id": "artifact",
        "kind": "current_state",
        "location": "artifact",
        "excerpt": _PLAN_ARTIFACT.decode(),
    }]

    def response(
        *,
        decision: str,
        reasoning: str,
        disposition: str,
        category: str,
        action: str,
        critic_outcome: str,
    ) -> str:
        return json.dumps({
            "schema_version": "assay.review.v2",
            "decision": decision,
            "reasoning": reasoning,
            "coverage": {
                "complete": True,
                "sources_reviewed": required_sources,
                "omissions": [],
            },
            "findings": [{
                "id": "close-split-responsibilities",
                "criterion_id": "critic_disposition",
                "severity": "High",
                "confidence": 0.9,
                "disposition": disposition,
                "category": category,
                "targets": [prior_ref],
                "evidence": evidence,
                "proposal": {"action": action, "replacement": None},
                "attack_spec": None,
            }],
            "criterion_results": [
                {
                    "criterion_id": criterion,
                    "outcome": (
                        critic_outcome
                        if criterion == "critic_disposition"
                        else "satisfied"
                    ),
                    "reasoning": "The current candidate was checked directly.",
                    "evidence": evidence,
                    "headline": criterion == "critic_disposition",
                }
                for criterion in (
                    "intent_coverage",
                    "executability",
                    "critic_disposition",
                    "contradiction",
                )
            ],
        })

    original_reasoning = "ORIGINAL_PASS_RETRY_CONTENT"
    invalid = response(
        decision="pass",
        reasoning=original_reasoning,
        disposition="dismissed",
        category="other",
        action="ATOMICITY_PROOF: {",
        critic_outcome="satisfied",
    )
    valid = response(
        decision="revise",
        reasoning="The under-decomposition prior remains active.",
        disposition="confirmed",
        category="unresolved_critic",
        action="Split loading and validation into independent slices.",
        critic_outcome="violated",
    )
    backend = _glm_sequence_backend(
        _Reply(invalid, 0.0),
        _Reply(valid, 0.0),
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=context,
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert verdict.reasoning == "The under-decomposition prior remains active."
    assert len(backend.invocations) == 2
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    first, second = manifest["attempts"]
    assert first["decision"] == "pass"
    assert second["decision"] == "revise"
    assert second["validation_error"] is None

    report_ref = first["contract_canonicalization_ref"]
    assert report_ref is not None
    report = json.loads(corpus.get_artifact(report_ref))
    assert report["accepted"] is False
    assert report["acceptance_reason"] == (
        "rejected_new_or_unreduced_errors"
    )
    assert report["contract_errors_before"]
    assert report["contract_errors_after"] == [
        "plan_judge: finding[0] forbids proposal.replacement"
    ]
    assert report["input_verdict_sha256"] != report["output_verdict_sha256"]
    assert report["raw_response_ref"] is None
    attempt_transcript = json.loads(corpus.get_artifact(
        report["transcript_ref"]
    ))
    assert attempt_transcript[-1] == {
        "role": "assistant",
        "content": invalid,
    }

    original_error = (
        "Rubric-specific response contract failed: "
        + "; ".join(report["contract_errors_before"])
    )
    assert first["validation_error"] == original_error
    repair_envelope = backend.invocations[1].user.split(
        "BEGIN_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE\n", 1,
    )[1].split("\nEND_UNTRUSTED_RESPONSE_REPAIR_ENVELOPE", 1)[0]
    assert json.loads(repair_envelope)["validation_error"] == original_error
    assert report["contract_errors_after"][0] not in original_error
    assert original_reasoning not in backend.invocations[1].user


def test_glm_plan_anchor_only_fallback_is_latched_across_schema_repair(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorOnlyGlmPlanJudgeBackend(
        fail_first_anchor_response=True,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 3
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        for invocation in backend.invocations[1:]
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "repaired"
    assert "unresolved deterministic evidence reference" in (
        manifest["attempts"][0]["validation_error"]
    )
    assert "missing=['reasoning']" in (
        manifest["attempts"][1]["validation_error"]
    )
    assert manifest["attempts"][2]["validation_error"] is None
    final_repair = json.loads(corpus.get_artifact(
        manifest["attempts"][2]["response_repair_ref"]
    ))
    assert final_repair["policy"]["materialize_json_paths"] is False


def test_glm_plan_nonverbatim_evidence_latches_anchor_only_fallback(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _AnchorOnlyGlmPlanJudgeBackend(
        fail_first_anchor_response=True,
        nonverbatim_first_response=True,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "revise"
    assert len(backend.invocations) == 3
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        for invocation in backend.invocations[1:]
    )
    manifest = _retry_manifest(corpus, verdict)
    first_error = manifest["attempts"][0]["validation_error"]
    assert first_error == (
        "invalid review_v2 response: findings[0].evidence[0].excerpt: "
        "is not a verbatim substring of 'artifact'"
    )
    final_repair = json.loads(corpus.get_artifact(
        manifest["attempts"][2]["response_repair_ref"]
    ))
    assert final_repair["policy"]["materialize_json_paths"] is False


def test_glm_plan_target_cardinality_retry_uses_fail_closed_ledger(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend()

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=1,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 2
    assert backend.ledger_current_state_rule_seen is True
    assert len(verdict.findings[0].targets) == 17
    assert len(set(verdict.findings[0].targets)) == 17
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=2)
    assert manifest["terminal_reason"] == "repaired"
    assert manifest["attempts"][0]["coverage_normalization_ref"] is not None
    first_error = manifest["attempts"][0]["validation_error"]
    assert "requires exactly one adjudication, got 2" in first_error
    assert "requires exactly one adjudication, got 0" in first_error
    assert manifest["attempts"][1]["validation_error"] is None
    repair = json.loads(corpus.get_artifact(
        manifest["attempts"][1]["response_repair_ref"]
    ))
    assert repair["policy"]["materialize_source_anchors"] is True
    assert repair["policy"]["materialize_json_paths"] is False
    assert {
        action["operation"] for action in repair["actions"]
    } == {"materialize_source_anchor"}


def test_glm_plan_gets_one_target_ledger_extension_at_normal_retry_cap(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend(
        cardinality_attempt=2,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 4
    assert all(
        "CRUCIBLE ANCHOR-ONLY EVIDENCE FALLBACK" in invocation.system
        for invocation in backend.invocations[1:]
    )
    assert all(
        "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" not in invocation.system
        for invocation in backend.invocations[:3]
    )
    assert "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in (
        backend.invocations[3].system
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=4)
    assert manifest["terminal_reason"] == "repaired"
    assert manifest["attempts"][0]["validation_error"].endswith(
        "unresolved deterministic source anchor"
    )
    assert manifest["attempts"][1]["validation_error"].endswith(
        "declared character span does not select the exact excerpt"
    )
    assert "requires exactly one adjudication, got 2" in (
        manifest["attempts"][2]["validation_error"]
    )
    assert "requires exactly one adjudication, got 0" in (
        manifest["attempts"][2]["validation_error"]
    )
    assert manifest["attempts"][3]["validation_error"] is None


def test_glm_plan_final_sequence_recovers_only_decision_scalar_after_closure(
    tmp_path,
):
    """Reproduce JSON failure -> length -> cardinality -> scalar mismatch."""

    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend(
        cardinality_attempt=2,
        output_length_attempt=1,
        decision_mismatch_after_ledger=True,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 5
    assert sum(
        "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in invocation.system
        for invocation in backend.invocations
    ) == 1
    assert "CRUCIBLE BOUNDED DECISION-CONSISTENCY RECOVERY" in (
        backend.invocations[4].system
    )
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=5)
    assert manifest["attempts"][0]["validation_error"].endswith(
        "unresolved deterministic source anchor"
    )
    assert manifest["attempts"][1]["validation_error"] == (
        "backend output ended with finish_reason=length before a complete "
        "structured response object; return one concise complete replacement "
        "object without trailing padding"
    )
    assert "requires exactly one adjudication, got 2" in (
        manifest["attempts"][2]["validation_error"]
    )
    assert manifest["attempts"][3]["validation_error"] == (
        "invalid review_v2 response: criterion_results: revise requires a "
        "violated result"
    )
    assert manifest["attempts"][4]["validation_error"] is None
    repair = json.loads(corpus.get_artifact(
        manifest["attempts"][4]["response_repair_ref"]
    ))
    assert repair["decision_repair_mode"] == (
        "bounded_fully_satisfied_revise_to_pass"
    )
    assert sum(
        action["operation"] == "recover_fully_satisfied_revise_to_pass"
        for action in repair["actions"]
    ) == 1


def test_glm_plan_extends_failed_early_ledger_repair_with_current_ledger(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend(
        cardinality_attempt=1,
        ledger_repair_failures=1,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "pass"
    assert len(backend.invocations) == 4
    assert sum(
        "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in invocation.system
        for invocation in backend.invocations
    ) == 2
    assert len(backend.ledger_snapshots) == 2
    first_ledger, second_ledger = backend.ledger_snapshots
    assert first_ledger["missing_aliases"] == [backend.aliases[1]]
    assert first_ledger["duplicate_aliases"] == [backend.aliases[0]]
    assert second_ledger["missing_aliases"] == [backend.aliases[3]]
    assert second_ledger["duplicate_aliases"] == [backend.aliases[2]]
    assert first_ledger["ledger_sha256"] != second_ledger["ledger_sha256"]
    second_payload = dict(second_ledger)
    second_digest = second_payload.pop("ledger_sha256")
    encoded = json.dumps(
        second_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert second_digest == "sha256:" + hashlib.sha256(encoded).hexdigest()
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=4)
    assert manifest["terminal_reason"] == "repaired"
    assert "requires exactly one adjudication, got 2" in (
        manifest["attempts"][1]["validation_error"]
    )
    assert "requires exactly one adjudication, got 2" in (
        manifest["attempts"][2]["validation_error"]
    )
    assert manifest["attempts"][3]["validation_error"] is None


def test_glm_plan_target_ledger_extension_is_not_repeated_after_failure(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend(
        cardinality_attempt=1,
        ledger_repair_failures=2,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == [
        "rubric_contract_failure",
    ]
    assert len(backend.invocations) == 4
    assert sum(
        "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in invocation.system
        for invocation in backend.invocations
    ) == 2
    assert len(backend.ledger_snapshots) == 2
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=4)
    assert manifest["terminal_reason"] == "retries_exhausted"
    assert "requires exactly one adjudication, got 2" in (
        manifest["attempts"][3]["validation_error"]
    )


def test_glm_plan_stale_target_ledger_does_not_extend_assay_failure(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend(
        cardinality_attempt=1,
        ledger_repair_assay_failure=True,
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=2,
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 3
    assert sum(
        "CRUCIBLE MATERIAL-PRIOR TARGET REPAIR" in invocation.system
        for invocation in backend.invocations
    ) == 1
    manifest = _retry_manifest(corpus, verdict)
    _assert_manifest_shape(manifest, attempts=3)
    assert manifest["terminal_reason"] == "retries_exhausted"
    assert manifest["attempts"][2]["validation_error"].startswith(
        "invalid review_v2 response:"
    )


def test_glm_plan_target_ledger_preserves_zero_retry_one_shot(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    backend = _TargetLedgerGlmPlanJudgeBackend()

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_target_ledger_context(),
        response_retry_budget=0,
    )

    assert verdict.decision == "inconclusive"
    assert len(backend.invocations) == 1
    assert "review_retry_manifest" not in verdict.artifacts


def test_review_obligations_are_hash_bound_and_stable_across_repair(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    all_sources = [
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    ]
    backend = _SequenceBackend(
        _Reply(_plan_judge_response(
            decision="reject",
            disposition="confirmed",
            sources_reviewed=all_sources,
        ), 0.0),
        _Reply(_plan_judge_response(
            decision="revise",
            disposition="confirmed",
            sources_reviewed=[
                "artifact",
                "context:spec_intent",
                "context:prior_critic_findings",
                "context:prior_judge_findings",
            ],
        ), 0.0),
    )

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    first = _obligations_from_invocation(backend.invocations[0])
    second = _obligations_from_invocation(backend.invocations[1])
    assert first == second
    assert first["schema_version"] == "crucible.review-obligations.v1"
    encoded = json.dumps(
        first["obligations"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert first["obligations_sha256"] == (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    assert first["obligations"]["material_prior_bindings"] == [{
        "finding_ref": _plan_judge_prior_ref(),
        "source_id": "context:prior_critic_findings",
    }]
    stored = json.loads(corpus.get_artifact(
        verdict.artifacts["review_obligations"]
    ))
    assert stored == first
    manifest = _retry_manifest(corpus, verdict)
    assert all(
        attempt["coverage_normalization_ref"] is not None
        for attempt in manifest["attempts"]
    )


def test_plan_judge_reject_requires_unfixable_blocker(tmp_path):
    corpus = Corpus(tmp_path / "corpus")
    all_sources = [
        "artifact",
        "context:spec_intent",
        "context:spec_constraints",
        "context:spec_done_criteria",
        "context:prior_critic_findings",
        "context:prior_judge_findings",
    ]
    backend = _SequenceBackend(_Reply(_plan_judge_response(
        decision="reject",
        disposition="unfixable",
        sources_reviewed=all_sources,
    ), 0.0))

    verdict = review(
        _PLAN_ARTIFACT,
        load("plan_judge"),
        backend=backend,
        corpus=corpus,
        context=_plan_judge_context(),
    )

    assert verdict.decision == "reject"


def test_breaker_preserves_actionable_finding_across_bad_citation_repairs(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    central = _attack_finding(
        finding_id="future-warning-no-behavior-change",
        severity="Critical",
    )
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        artifact_excerpt=(
            "'converting a structured array to a NdarrayMixin will be "
            "deprecated in the future'"
        ),
    )
    bad_pass = _attack_finding(
        finding_id="warning-message-exact-text",
        severity="Medium",
        issue_location="chars:0-0",
    )
    bad_rewrite = _attack_finding(
        finding_id="warning-message-text-mismatch",
        severity="Medium",
        issue_location="chars:10-20",
    )
    backend = _SequenceBackend(
        _Reply(_breaker_response(
            decision="revise",
            findings=[central, malformed_medium],
            reasoning=(
                "The warning-only patch leaves the required 5.2 conversion "
                "behavior unchanged."
            ),
        ), 0.10),
        _Reply(_breaker_response(
            decision="pass",
            findings=[bad_pass],
            reasoning="The warning appears adequate.",
        ), 0.10),
        _Reply(_breaker_response(
            decision="revise",
            findings=[bad_rewrite],
            reasoning="Only wording risk remains.",
        ), 0.10),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=2,
    )

    assert verdict.decision == "revise"
    assert [finding.id for finding in verdict.findings] == [
        "future-warning-no-behavior-change",
    ]
    assert verdict.findings[0].attack_spec is not None
    assert verdict.cost_usd == pytest.approx(0.30)
    assert verdict.input_tokens == 6
    assert verdict.output_tokens == 9
    assert len(backend.invocations) == 3
    salvage = json.loads(corpus.get_artifact(
        verdict.artifacts["review_partial_salvage"]
    ))
    assert salvage["schema_version"] == "crucible.review-partial-salvage.v1"
    assert salvage["preserved_finding_ids"] == [
        "future-warning-no-behavior-change",
    ]
    assert salvage["sources"][0]["attempt"] == 0
    retry = _retry_manifest(corpus, verdict)
    assert retry["terminal_reason"] == "retries_exhausted"
    assert [item["input_tokens"] for item in retry["attempts"]] == [1, 2, 3]


def test_breaker_valid_repair_cannot_erase_preserved_actionable_finding(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    central = _attack_finding(
        finding_id="future-warning-no-behavior-change",
        severity="Critical",
    )
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        artifact_excerpt="fabricated warning text",
    )
    backend = _SequenceBackend(
        _Reply(_breaker_response(
            decision="revise",
            findings=[central, malformed_medium],
            reasoning=(
                "The warning-only patch leaves the required 5.2 conversion "
                "behavior unchanged."
            ),
        ), 0.10),
        _Reply(json.dumps(_pass_response()), 0.10),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert [finding.id for finding in verdict.findings] == [
        "future-warning-no-behavior-change",
    ]
    salvage = json.loads(corpus.get_artifact(
        verdict.artifacts["review_partial_salvage"]
    ))
    assert salvage["disposition"] == (
        "valid_repair_merged_with_preserved_revise"
    )
    retry = _retry_manifest(corpus, verdict)
    assert retry["terminal_reason"] == "repaired_with_preserved_findings"


def test_workflow_validator_disables_partial_salvage_without_extra_callback(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    central = _attack_finding(
        finding_id="future-warning-no-behavior-change",
        severity="Critical",
    )
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        artifact_excerpt="fabricated warning text",
    )
    backend = _SequenceBackend(
        _Reply(_breaker_response(
            decision="revise",
            findings=[central, malformed_medium],
            reasoning="One grounded blocker survives a malformed sibling.",
        ), 0.10),
        _Reply(json.dumps(_pass_response()), 0.10),
    )
    validated_decisions = []

    def validator(candidate):
        validated_decisions.append(candidate.decision)
        return ()

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
        response_contract_validator=validator,
    )

    assert verdict.decision == "pass"
    assert validated_decisions == ["pass"]
    assert "review_partial_salvage" not in verdict.artifacts
    assert _retry_manifest(corpus, verdict)["terminal_reason"] == "repaired"


def test_breaker_backend_terminal_cannot_erase_preserved_actionable_finding(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    central = _attack_finding(
        finding_id="future-warning-no-behavior-change",
        severity="Critical",
    )
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        artifact_excerpt="fabricated warning text",
    )
    backend = _SequenceBackend(
        _Reply(_breaker_response(
            decision="revise",
            findings=[central, malformed_medium],
            reasoning=(
                "The warning-only patch leaves the required 5.2 conversion "
                "behavior unchanged."
            ),
        ), 0.10),
        _Reply(
            "",
            0.10,
            MaxTurnsExceeded(
                turns_used=5,
                turns_limit=5,
                message="review repair exhausted its turn limit",
            ),
        ),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=1,
    )

    assert verdict.decision == "revise"
    assert [finding.id for finding in verdict.findings] == [
        "future-warning-no-behavior-change",
    ]
    assert verdict.cost_usd == pytest.approx(0.20)
    assert verdict.input_tokens == 3
    assert verdict.output_tokens == 5
    salvage = json.loads(corpus.get_artifact(
        verdict.artifacts["review_partial_salvage"]
    ))
    assert salvage["disposition"] == "deterministic_terminal_preserved_revise"
    assert salvage["deterministic_terminal"] == {
        "attempt": 1,
        "decision": "inconclusive",
        "finding_categories": ["backend_error"],
        "finding_ids": ["backend_max_turns"],
        "reason": "backend_error",
        "transcript_ref": salvage["deterministic_terminal"]["transcript_ref"],
    }
    assert salvage["deterministic_terminal"]["transcript_ref"].startswith(
        "sha256:"
    )
    retry = _retry_manifest(corpus, verdict)
    assert retry["terminal_reason"] == "backend_error_with_preserved_findings"
    assert [item["input_tokens"] for item in retry["attempts"]] == [1, 2]
    assert [item["output_tokens"] for item in retry["attempts"]] == [2, 3]


def test_breaker_budget_precedes_preserved_candidate_on_backend_terminal(
    tmp_path,
):
    corpus = Corpus(tmp_path / "corpus")
    central = _attack_finding(
        finding_id="future-warning-no-behavior-change",
        severity="Critical",
    )
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        artifact_excerpt="fabricated warning text",
    )
    backend = _SequenceBackend(
        _Reply(_breaker_response(
            decision="revise",
            findings=[central, malformed_medium],
            reasoning="A grounded blocker survives response repair.",
        ), 0.10),
        _Reply(
            "",
            0.10,
            MaxTurnsExceeded(turns_used=5, turns_limit=5),
        ),
    )

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=corpus,
        context={"spec_intent": _BREAKER_INTENT},
        budget_usd=0.20,
        response_retry_budget=1,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == ["backend_max_turns"]
    assert "review_partial_salvage" not in verdict.artifacts
    assert verdict.cost_usd == pytest.approx(0.20)
    retry = _retry_manifest(corpus, verdict)
    assert retry["terminal_reason"] == (
        "budget_exhausted_after_deterministic_terminal"
    )


def test_breaker_partial_recovery_fails_closed_without_actionable_evidence(
    tmp_path,
):
    malformed_medium = _attack_finding(
        finding_id="warning-message-wording",
        severity="Medium",
        issue_location="chars:0-0",
    )
    backend = _SequenceBackend(_Reply(_breaker_response(
        decision="revise",
        findings=[malformed_medium],
        reasoning="Only an advisory wording concern was attempted.",
    ), 0.0))

    verdict = review(
        _BREAKER_ARTIFACT,
        load("swebench_impl_breaker"),
        backend=backend,
        corpus=Corpus(tmp_path / "corpus"),
        context={"spec_intent": _BREAKER_INTENT},
        response_retry_budget=0,
    )

    assert verdict.decision == "inconclusive"
    assert [finding.id for finding in verdict.findings] == ["schema_failure"]
    assert "review_partial_salvage" not in verdict.artifacts
