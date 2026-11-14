# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Lossless handoff and persistence helpers for analytical workflows.

Analytical recipes pass complete ``assay.review.v2`` records between roles.
Findings are deliberately not projected into a smaller, prompt-specific shape:
evidence identity, dispositions, targets, proposals, attack specifications, and
coverage are all part of the next role's input and the corpus record.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from assay import ContentHash, Evidence, Finding, Proposal, ReviewCoverage, Verdict
from assay.corpus.store import Corpus

from crucible.common.cost import CostAccumulator, CostLimitExceeded

from ._atomic_persistence import persist_and_attach
from ._run import write_atomic_bytes


DECISION_EXIT_CODES = {
    "pass": 0,
    "revise": 1,
    "reject": 2,
    "inconclusive": 3,
}

_HANDOFF_SCHEMA = "crucible.review-handoff.v2"
_SOURCE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_FINDING_REF_RE = re.compile(
    r"\Aprior:[A-Za-z0-9_.-]+:[0-9]+:sha256-[0-9a-f]{64}:"
    r"[A-Za-z0-9_.-]+\Z"
)


@dataclass(frozen=True)
class ParsedReviewHandoff:
    """Validated handoff split into prompt data and evidentiary sources."""

    original_artifact: bytes
    prompt_text: str
    source_texts: Mapping[str, str]
    prior_findings: tuple[Mapping[str, Any], ...]


def _source_token(value: object) -> str:
    raw = str(value).strip()
    token = _SOURCE_TOKEN_RE.sub("-", raw).strip("-.")
    if token and token == raw:
        return token
    digest = hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:12]
    return f"encoded-{digest}"


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qualified_finding_ref(
    *, role: object, round_index: int, verdict_digest: str, finding_id: object,
) -> str:
    """Return a collision-proof reference to one finding in one verdict."""

    if round_index < 0:
        raise ValueError("finding ref round_index must be non-negative")
    if re.fullmatch(r"[0-9a-f]{64}", verdict_digest) is None:
        raise ValueError("finding ref verdict_digest must be a sha256 hex digest")
    result = (
        f"prior:{_source_token(role)}:{round_index}:sha256-{verdict_digest}:"
        f"{_source_token(finding_id)}"
    )
    if _FINDING_REF_RE.fullmatch(result) is None:  # pragma: no cover - invariant
        raise ValueError("could not construct a canonical finding ref")
    return result


def finding_ref_for_payload(
    *, role: object, round_index: int, payload: object, finding_id: object,
) -> str:
    """Qualify a finding against the exact canonical payload containing it."""

    return qualified_finding_ref(
        role=role,
        round_index=round_index,
        verdict_digest=_canonical_digest(payload),
        finding_id=finding_id,
    )


def _prior_evidence_packet(finding: Mapping[str, Any]) -> dict[str, Any]:
    """Return only prior observation/linkage fields, never generated content."""

    return {
        key: finding.get(key)
        for key in (
            "id",
            "criterion_id",
            "severity",
            "confidence",
            "disposition",
            "category",
            "targets",
            "evidence",
            "source",
        )
    }


def verdict_payload(verdict: Verdict) -> dict:
    """Return the complete JSON-safe review record without lossy aliases."""

    return dataclasses.asdict(verdict)


def handoff_artifact(
    artifact: bytes,
    history: Iterable[tuple[str, Verdict]],
    *,
    heading: str = "REVIEW HISTORY",
) -> bytes:
    """Encode an exact artifact and complete prior Verdicts as a typed envelope.

    Unlike the former delimiter-based concatenation, arbitrary bytes in the
    artifact cannot terminate or forge the history section.  The envelope is
    decoded only by the strict Crucible review adapter.
    """

    records = [
        {"role": role, "verdict": verdict_payload(verdict)}
        for role, verdict in history
    ]
    if not records:
        return artifact
    payload = {
        "schema_version": _HANDOFF_SCHEMA,
        "heading": heading,
        "artifact": {
            "encoding": "base64",
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "data": base64.b64encode(artifact).decode("ascii"),
        },
        "records": records,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def parse_review_handoff(blob: bytes) -> ParsedReviewHandoff | None:
    """Decode a v2 handoff, returning ``None`` for an ordinary artifact.

    A blob that claims the reserved handoff schema is rejected on any shape,
    hash, source-id, or history error.  This prevents a malformed envelope from
    silently falling back to ordinary artifact semantics.
    """

    try:
        payload = json.loads(blob)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != _HANDOFF_SCHEMA:
        return None
    if set(payload) != {"schema_version", "heading", "artifact", "records"}:
        raise ValueError("review handoff has unexpected or missing top-level fields")
    if not isinstance(payload["heading"], str) or not payload["heading"].strip():
        raise ValueError("review handoff heading must be a non-empty string")
    encoded = payload["artifact"]
    if not isinstance(encoded, dict) or set(encoded) != {
        "encoding", "sha256", "data",
    }:
        raise ValueError("review handoff artifact has invalid fields")
    if encoded["encoding"] != "base64":
        raise ValueError("review handoff artifact encoding must be base64")
    try:
        original = base64.b64decode(encoded["data"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("review handoff artifact is not valid base64") from exc
    if hashlib.sha256(original).hexdigest() != encoded["sha256"]:
        raise ValueError("review handoff artifact sha256 does not match its bytes")
    records = payload["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("review handoff records must be a non-empty array")

    artifact_text = original.decode("utf-8", "replace")
    source_texts: dict[str, str] = {"artifact": artifact_text}
    prior_findings: list[Mapping[str, Any]] = []
    rendered_records: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"role", "verdict"}:
            raise ValueError(f"review handoff record[{record_index}] has invalid fields")
        role = record["role"]
        verdict = record["verdict"]
        if not isinstance(role, str) or not role.strip() or not isinstance(verdict, dict):
            raise ValueError(f"review handoff record[{record_index}] is malformed")
        findings = verdict.get("findings")
        if not isinstance(findings, (list, tuple)):
            raise ValueError(
                f"review handoff record[{record_index}].verdict.findings must be an array"
            )
        rendered_findings: list[dict[str, Any]] = []
        verdict_digest = _canonical_digest(verdict)
        for finding_index, finding in enumerate(findings):
            if not isinstance(finding, dict):
                raise ValueError(
                    f"review handoff finding[{record_index}][{finding_index}] is malformed"
                )
            finding_id = finding.get("id")
            if not isinstance(finding_id, str) or not finding_id.strip():
                raise ValueError("review handoff prior finding has no stable id")
            source_id = qualified_finding_ref(
                role=role,
                round_index=record_index,
                verdict_digest=verdict_digest,
                finding_id=finding_id,
            )
            if source_id in seen_source_ids:
                raise ValueError(f"review handoff has duplicate source id {source_id!r}")
            seen_source_ids.add(source_id)
            packet = _prior_evidence_packet(finding)
            packet["finding_ref"] = source_id
            source_text = json.dumps(
                packet, ensure_ascii=False, sort_keys=True, default=str,
            )
            source_texts[source_id] = source_text
            prior_findings.append({
                "id": finding_id,
                "ref": source_id,
                "role": role,
                "round_index": record_index,
                "verdict_digest": verdict_digest,
                "source_id": source_id,
                "severity": finding.get("severity"),
                "disposition": finding.get("disposition"),
                "category": finding.get("category"),
                "targets": tuple(finding.get("targets", ())) if isinstance(
                    finding.get("targets", ()), (list, tuple)
                ) else (),
                "source": finding.get("source"),
            })
            rendered_findings.append({
                "evidence_source_id": source_id,
                "evidence_packet": packet,
                "non_evidentiary_generated_content": {
                    "proposal": finding.get("proposal"),
                    "attack_spec": finding.get("attack_spec"),
                    "suggested_action": finding.get("suggested_action"),
                },
            })
        rendered_records.append({
            "role": role,
            "decision": verdict.get("decision"),
            "coverage": verdict.get("coverage"),
            "findings": rendered_findings,
            "non_evidentiary_reasoning": verdict.get("reasoning"),
        })

    prompt = json.dumps(
        {
            "schema_version": "crucible.review-input.v2",
            "heading": payload["heading"],
            "artifact": {"source_id": "artifact", "text": artifact_text},
            "prior_records": rendered_records,
            "evidence_rule": (
                "Only artifact and evidence_source_id payloads are citable; "
                "reasoning and generated content are context, never evidence."
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=str,
    )
    return ParsedReviewHandoff(
        original_artifact=original,
        prompt_text=prompt,
        source_texts=source_texts,
        prior_findings=tuple(prior_findings),
    )


def attach_artifacts(verdict: Verdict, **artifacts: ContentHash) -> Verdict:
    """Return ``verdict`` with output refs merged into its artifact map."""

    return dataclasses.replace(
        verdict,
        artifacts={**verdict.artifacts, **artifacts},
    )


def persist_verdict(corpus: Corpus, spec, verdict: Verdict):
    """Persist one verdict and attach the exact same record to ``spec``."""

    del spec  # attachment always rebases onto the authoritative latest Spec
    return persist_and_attach(
        corpus, verdict, kind="verdicts", attachment="verdict",
    )


def persist_output(
    corpus: Corpus,
    path: Path,
    blob: bytes,
    *,
    kind: str,
) -> ContentHash:
    """Write a run output and store the identical bytes content-addressably."""

    write_atomic_bytes(path, blob)
    return corpus.put_artifact(blob, kind=kind, name_hint=path.name)


def remaining_budget(costs: CostAccumulator) -> float | None:
    """Return a usable per-call cap, raising when the cumulative cap is spent."""

    remaining = costs.remaining()
    if remaining is not None and remaining <= 0.0:
        raise CostLimitExceeded(
            accounting_debt_reason(costs)
            or "cumulative analysis budget exhausted"
        )
    return remaining


def record_cost(
    costs: CostAccumulator,
    *,
    role: str,
    usage,
    backend,
) -> bool:
    """Record a call and report whether another capped call may start.

    ``CostAccumulator`` records before raising on an overrun.  Analytical
    workflows use a soft terminal boundary: the completed call is retained,
    then no subsequent role is invoked.
    """

    try:
        costs.record(
            role=role,
            usage=usage,
            backend_name=backend.name,
            model=backend.model,
        )
    except CostLimitExceeded:
        return False
    remaining = costs.remaining()
    return remaining is None or remaining > 0.0


def record_invocation_cost(
    costs: CostAccumulator,
    *,
    role: str,
    result,
    backend,
) -> bool:
    """Record one raw backend result without dropping completeness evidence.

    ``InvocationResult`` keeps numeric usage under ``result.usage`` and
    provider completeness under ``result.completeness``. Recording only the
    former would make an observed lower bound look authoritative, so direct
    backend callers use this helper instead of projecting the result to bare
    ``Usage`` first.
    """

    can_continue = record_cost(
        costs,
        role=role,
        usage=result.usage,
        backend=backend,
    )
    completeness = result.completeness
    missing = tuple(
        field_name
        for field_name in ("token_usage_complete", "cost_complete")
        if not getattr(completeness, field_name)
    )
    if missing:
        costs.note_accounting_debt(role=role, missing=missing)
        return False
    return can_continue


def decision_exit_code(decision: str) -> int:
    """Fail closed for unknown decisions."""

    return DECISION_EXIT_CODES.get(decision, DECISION_EXIT_CODES["inconclusive"])


def accounting_debt_reason(costs: CostAccumulator) -> str | None:
    """Explain why an analysis ledger no longer authorizes a pass or paid call.

    The accumulated numeric total remains a useful observed lower bound.  It
    cannot establish remaining spend authority when a provider did not prove
    token or cost completeness, however, so every analytical workflow uses
    this shared diagnostic at its final decision boundary.
    """

    summary = costs.to_dict()
    if not summary["accounting_debt"]:
        return None
    missing_by_role = summary["accounting_debt_by_role"]
    details = "; ".join(
        f"{role}: {', '.join(missing)}"
        for role, missing in missing_by_role.items()
    )
    return (
        "incomplete backend accounting; observed cost "
        f"${costs.total_cost_usd:.4f} is only a lower bound"
        + (f" ({details})" if details else "")
    )


def incomplete_adjudication_reason(
    costs: CostAccumulator,
    *,
    budget_reason: str = "cumulative budget exhausted before adjudication",
) -> str:
    """Return the authoritative reason an analytical debate stopped early."""

    return accounting_debt_reason(costs) or budget_reason


def incomplete_adjudication_rubric(
    costs: CostAccumulator,
    *,
    workflow: str,
) -> str:
    """Name an early terminal after its authoritative stopping condition."""

    suffix = (
        "accounting"
        if accounting_debt_reason(costs) is not None
        else "budget"
    )
    return f"{workflow}_{suffix}"


def enforce_budget_overrun(
    verdict: Verdict,
    costs: CostAccumulator,
) -> Verdict:
    """Enforce the shared accounting/cost authority boundary on a verdict.

    Accounting debt takes precedence over the numeric cap: the recorded cost
    is then a lower bound, so calling the condition ordinary budget exhaustion
    would be misleading.  Either condition makes a nominal pass inconclusive.
    """

    reason = accounting_debt_reason(costs)
    cap = costs.max_cost_usd
    if reason is None and (cap is None or costs.total_cost_usd <= cap):
        return verdict
    if reason is None:
        reason = (
            f"cumulative cost ${costs.total_cost_usd:.4f} exceeded "
            f"cap ${cap:.4f}"
        )
    omissions = verdict.coverage.omissions
    if reason not in omissions:
        omissions = omissions + (reason,)
    reasoning = verdict.reasoning
    if reason not in reasoning:
        reasoning = f"{reason}; {reasoning}"
    return dataclasses.replace(
        verdict,
        decision="inconclusive",
        reasoning=reasoning,
        coverage=dataclasses.replace(
            verdict.coverage,
            complete=False,
            omissions=omissions,
        ),
    )


def terminal_verdict(
    corpus: Corpus,
    artifact: bytes,
    *,
    rubric: str,
    decision: str,
    reasoning: str,
    kind: str = "judge",
    complete: bool = False,
) -> Verdict:
    """Create a typed, corpus-grounded workflow terminal verdict."""

    if decision not in DECISION_EXIT_CODES:
        raise ValueError(f"unsupported terminal decision: {decision!r}")
    artifact_ref = corpus.put_artifact(
        artifact, kind="analysis_inputs", name_hint=f"{rubric}.bin",
    )
    findings = ()
    if decision in {"revise", "reject"}:
        excerpt = artifact.decode("utf-8", "replace").strip()[:500] or reasoning
        findings = (Finding(
            id=f"{rubric}_{decision}",
            criterion_id="workflow.terminal",
            severity="Critical" if decision == "reject" else "High",
            confidence=1.0,
            disposition="confirmed",
            category="workflow_terminal",
            targets=(),
            evidence=(Evidence(
                source_id="artifact",
                kind="workflow_input",
                location="artifact",
                excerpt=excerpt,
            ),),
            proposal=Proposal(action=reasoning, replacement=None),
            attack_spec=None,
            suggested_action=reasoning,
            source="deterministic",
        ),)
    return Verdict(
        rubric=rubric,
        decision=decision,  # type: ignore[arg-type]
        findings=findings,
        artifact_ref=artifact_ref,
        artifact_commit=None,
        reasoning=reasoning,
        coverage=ReviewCoverage(
            complete=complete,
            sources_reviewed=("artifact",),
            omissions=() if complete else (reasoning,),
            artifact_chars_total=len(artifact.decode("utf-8", "replace")),
            artifact_chars_reviewed=(
                len(artifact.decode("utf-8", "replace")) if complete else 0
            ),
        ),
        backend_name=None,
        transcript_ref=None,
        cost_usd=0.0,
        execution=None,
        kind=kind,  # type: ignore[arg-type]
    )
