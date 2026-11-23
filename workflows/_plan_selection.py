# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Compare reviewed plans without granting or publishing plan authority."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

from assay import ReviewCoverage, Spec, TierDAG, Verdict
from assay.canonical import hash_record, serialize
from assay.corpus.store import Corpus
from assay.primitives.plan import (
    _canonicalize_scope_path,
    _serialize_tier_dag,
    _validated_project_root,
)

from crucible.rubrics import load

from ._review import is_review_infrastructure_failure, review


_SCHEMA = "crucible.plan-selection.v1"
_BUNDLE_SCHEMA = "crucible.plan-selection-bundle.v1"
_MAX_RESPONSE_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_RESPONSE_FIELDS = frozenset({
    "schema_version", "source_spec_hash", "bundle_ref", "mode", "selected_id",
    "selected_digest", "synthesized_dag", "reason",
})
_CHAR_SPAN = re.compile(r"(?:^|;)chars:([0-9]+)-([0-9]+)\Z")


@dataclass(frozen=True)
class PlanSelectionCandidate:
    id: str
    dag: TierDAG
    accepted: bool
    verdict_refs: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True)
class PlanSelectionResult:
    mode: Literal["select", "synthesize", "none"]
    selected_id: str | None
    candidate_dag: TierDAG | None
    verdict: Verdict
    verdict_ref: str
    selection_ref: str
    bundle_ref: str
    error: str | None

    @property
    def cost_usd(self) -> float:
        return self.verdict.cost_usd


def _decode_dag(raw: bytes) -> TierDAG:
    # Plan imports this module only on the competition path. Keep its strict
    # candidate decoder as the single schema/topology authority.
    from .plan import _decode_review_candidate

    return _decode_review_candidate(raw)


def _decode_authored_dag(
    raw: bytes, spec: Spec, project_root: str | None,
) -> TierDAG:
    """Strictly decode a plan a model authored in this run; canonicalize its scope.

    Single-track planning canonicalizes scope paths while Assay parses the
    Planner reply. A competition draft and a proposed synthesis are decoded
    here instead, so Assay's rule runs here, on ``files_in_scope`` and
    ``creates_paths``, against the absolute ``project_root`` and never the
    process working directory. ``None`` proves no same-named directory, so a
    leading project/submodule-name segment is always removed.

    Use this only where model output becomes a plan. The rule is not
    idempotent: an operator candidate and stored plan bytes keep their content
    and digest through ``_decode_dag``. A synthesis is new model text, so a
    path it copies from a judged candidate is canonicalized again.
    """
    dag = _decode_dag(raw)
    context = spec.project_context
    prefix = context.submodule_path or context.project or ""

    def canonical(slice_id: str, paths: tuple[str, ...]) -> tuple[str, ...]:
        result = []
        for written in paths:
            path = _canonicalize_scope_path(written, prefix, project_root)
            if not path:
                # The closed candidate schema has no empty scope path.
                raise ValueError(
                    f"slice {slice_id!r} scope path {written!r} is empty after "
                    "canonicalization"
                )
            result.append(path)
        return tuple(result)

    return replace(dag, tiers=tuple(
        replace(tier, slices=tuple(
            replace(slice_, directive=replace(
                slice_.directive,
                files_in_scope=canonical(slice_.id, slice_.directive.files_in_scope),
                creates_paths=canonical(slice_.id, slice_.directive.creates_paths),
            ))
            for slice_ in tier.slices
        ))
        for tier in dag.tiers
    ))


def _candidate_payload(candidate: PlanSelectionCandidate) -> dict[str, Any]:
    if not isinstance(candidate, PlanSelectionCandidate):
        raise TypeError("selection candidates must be PlanSelectionCandidate values")
    if not isinstance(candidate.id, str) or not _IDENTIFIER.fullmatch(candidate.id):
        raise ValueError("candidate id must be a nonempty bounded identifier")
    if type(candidate.accepted) is not bool or not isinstance(candidate.reason, str):
        raise ValueError("candidate accepted/reason fields have invalid types")
    if not isinstance(candidate.dag, TierDAG):
        raise TypeError("candidate dag must be a TierDAG")
    if type(candidate.verdict_refs) is not tuple or any(
        not isinstance(ref, str) for ref in candidate.verdict_refs
    ):
        raise ValueError("candidate verdict_refs must be a tuple of record refs")
    encoded_dag = _serialize_tier_dag(candidate.dag)
    decoded = _decode_dag(encoded_dag)
    if hash_record(decoded) != hash_record(candidate.dag):
        raise ValueError("candidate DAG is not losslessly representable")
    return {
        "id": candidate.id,
        "digest": hash_record(candidate.dag),
        "dag": json.loads(encoded_dag),
        "accepted": candidate.accepted,
        "verdict_refs": candidate.verdict_refs,
        "reason": candidate.reason,
    }


def _review_evidence(
    candidate: PlanSelectionCandidate, corpus: Corpus,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    latest: dict[str, Verdict] = {}
    for ref in candidate.verdict_refs:
        verdict = cast(Verdict, corpus.get_record(ref, as_type=Verdict))
        evidence.append({"ref": ref, "verdict": verdict})
        if verdict.kind in {"critic", "judge", "validate"}:
            latest[verdict.kind] = verdict
    for role in ("critic", "judge"):
        if role not in latest:
            raise ValueError(f"candidate {candidate.id!r} lacks {role} evidence")
        verdict = latest[role]
        artifact = corpus.get_artifact(verdict.artifact_ref)
        if hash_record(_decode_dag(artifact)) != hash_record(candidate.dag):
            raise ValueError(
                f"candidate {candidate.id!r} final {role} reviewed another DAG"
            )
        if not verdict.coverage.complete or verdict.accounting_debt:
            raise ValueError(f"candidate {candidate.id!r} has incomplete {role} evidence")
        completeness = verdict.execution_completeness
        if completeness is None or not (
            completeness.token_usage_complete and completeness.cost_complete
            and completeness.terminal_complete
        ):
            raise ValueError(f"candidate {candidate.id!r} has incomplete {role} execution")
        if is_review_infrastructure_failure(verdict):
            raise ValueError(f"candidate {candidate.id!r} has failed {role} evidence")
    if candidate.accepted and any(
        role not in latest or latest[role].decision != "pass"
        or not latest[role].coverage.complete
        for role in ("critic", "judge", "validate")
    ):
        raise ValueError(
            f"candidate {candidate.id!r} is marked accepted without current pass gates"
        )
    return evidence


def _unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate selection JSON key {key!r}")
        result[key] = value
    return result


def _json_range(payload: Any, path: tuple[str | int, ...]) -> tuple[int, int]:
    """Locate an exact subtree in the trusted canonical comparison packet."""

    def width(value: Any) -> int:
        return len(serialize(value).decode("utf-8"))

    start = 0
    node = payload
    for component in path:
        start += 1  # Opening object/array delimiter.
        if isinstance(component, str):
            for key in sorted(node):
                start += width(key) + 1  # Key and colon.
                if key == component:
                    node = node[key]
                    break
                start += width(node[key]) + 1  # Value and comma.
            else:
                raise ValueError("comparison evidence path does not exist")
        else:
            start += sum(width(value) + 1 for value in node[:component])
            node = node[component]
    return start, start + width(node)


def _parse_selection(
    verdict: Verdict,
    *,
    spec: Spec,
    project_root: str | None,
    source_spec_hash: str,
    bundle_ref: str,
    candidates: dict[str, PlanSelectionCandidate],
    required_evidence: tuple[tuple[str, tuple[tuple[int, int], ...]], ...],
) -> tuple[dict[str, Any], TierDAG | None]:
    if len(verdict.findings) != 1:
        raise ValueError("selection requires exactly one typed finding")
    finding = verdict.findings[0]
    spans = tuple(
        (int(match.group(1)), int(match.group(2)))
        for item in finding.evidence if item.source_id == "artifact"
        if (match := _CHAR_SPAN.search(item.location)) is not None
    )
    for label, ranges in required_evidence:
        if not any(
            lower <= start < end <= upper
            for lower, upper in ranges for start, end in spans
        ):
            raise ValueError(
                "selection evidence must ground root intent, every DAG, and its "
                f"final Critic/Judge text; missing {label}"
            )
    replacement = finding.proposal.replacement if finding.proposal is not None else None
    if not isinstance(replacement, str) or not replacement:
        raise ValueError("selection requires proposal.replacement JSON")
    if len(replacement.encode("utf-8")) > _MAX_RESPONSE_BYTES:
        raise ValueError("selection response exceeds the candidate byte limit")
    try:
        payload = json.loads(replacement, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("selection replacement must be strict JSON") from exc
    if type(payload) is not dict or set(payload) != _RESPONSE_FIELDS:
        raise ValueError("selection JSON has missing or unknown fields")
    if (
        payload["schema_version"] != _SCHEMA
        or payload["source_spec_hash"] != source_spec_hash
        or payload["bundle_ref"] != bundle_ref
    ):
        raise ValueError("selection is not bound to this Spec and candidate bundle")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("selection reason must be nonempty text")
    mode = payload["mode"]
    if mode == "select":
        selected_id = payload["selected_id"]
        if not isinstance(selected_id, str) or selected_id not in candidates:
            raise ValueError("selection names an unknown candidate")
        candidate = candidates[selected_id]
        if not candidate.accepted:
            raise ValueError("selection cannot approve a rejected candidate")
        if payload["selected_digest"] != hash_record(candidate.dag):
            raise ValueError("selection candidate digest does not match")
        if payload["synthesized_dag"] is not None:
            raise ValueError("exact selection must not contain a synthesized DAG")
        dag = candidate.dag
    elif mode == "synthesize":
        if payload["selected_id"] is not None or payload["selected_digest"] is not None:
            raise ValueError("synthesis must not select a candidate identity")
        dag = _decode_authored_dag(
            serialize(payload["synthesized_dag"]), spec, project_root,
        )
    elif mode == "none":
        if any(payload[key] is not None for key in (
            "selected_id", "selected_digest", "synthesized_dag",
        )):
            raise ValueError("abstention must not contain a candidate")
        dag = None
    else:
        raise ValueError("selection mode must be select, synthesize, or none")
    expected_category = {
        "select": "selected", "synthesize": "synthesized", "none": "no_selection",
    }[mode]
    if finding.id != "plan_selection" or finding.category != expected_category:
        raise ValueError("selection finding does not match its typed mode")
    if verdict.decision != ("inconclusive" if mode == "none" else "pass"):
        raise ValueError("selection decision does not match its typed mode")
    return payload, dag


def select_plan_candidate(
    spec: Spec,
    candidates: tuple[PlanSelectionCandidate, ...],
    *,
    backend: Any,
    corpus: Corpus,
    budget_usd: float | None = None,
    max_turns: int = 5,
    project_root: str | None = None,
) -> PlanSelectionResult:
    """Return an exact reviewed candidate or an unaccepted synthesis proposal.

    The caller owns a separate selector backend/session and must give every
    synthesized DAG fresh fixed-candidate Critic/Judge and deterministic/source
    gates. This helper writes evidence only; it never updates the Corpus Spec.

    ``project_root`` is the existing absolute directory a proposed synthesis's
    scope paths are canonicalized against, as for a draft; ``None`` always
    removes a leading project/submodule-name segment. An invalid root raises
    before any selector call.
    """

    if not isinstance(spec, Spec):
        raise TypeError("selection requires a Spec")
    if type(max_turns) is not int or max_turns <= 0:
        raise ValueError("max_turns must be a positive integer")
    if budget_usd is not None and (
        type(budget_usd) not in (int, float)
        or not math.isfinite(budget_usd) or budget_usd < 0
    ):
        raise ValueError("budget_usd must be finite and nonnegative")
    project_root = _validated_project_root(project_root)
    payloads = [_candidate_payload(candidate) for candidate in candidates]
    by_id = {candidate.id: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidate ids must be unique")
    source_spec_hash = hash_record(spec)
    bundle: dict[str, Any] = {
        "schema_version": _BUNDLE_SCHEMA,
        "source_spec_hash": source_spec_hash,
        "spec": spec,
        "candidates": payloads,
    }
    evidence_error: str | None = None
    try:
        for candidate, payload in zip(candidates, payloads, strict=True):
            payload["reviews"] = _review_evidence(candidate, corpus)
    except (ValueError, TypeError, OSError) as exc:
        evidence_error = f"candidate evidence is invalid: {exc}"
        bundle["evidence_error"] = evidence_error
    bundle_ref = corpus.put_artifact(serialize(bundle), kind="plan_selection")
    comparison_bytes = serialize({"bundle_ref": bundle_ref, "bundle": bundle})
    comparison = json.loads(comparison_bytes)
    evidence_paths: list[tuple[str, tuple[tuple[str | int, ...], ...]]] = [(
        "root Spec", (
            ("bundle", "spec", "intent"), ("bundle", "spec", "constraints"),
            ("bundle", "spec", "done_criteria"),
        ),
    )]
    if evidence_error is None:
        for index, payload in enumerate(payloads):
            prefix: tuple[str | int, ...] = ("bundle", "candidates", index)
            evidence_paths.append((f"candidate {index} DAG", ((*prefix, "dag"),)))
            for role in ("critic", "judge"):
                final_index = max(
                    i for i, item in enumerate(payload["reviews"])
                    if item["verdict"].kind == role
                )
                review_prefix = (*prefix, "reviews", final_index, "verdict")
                evidence_paths.append((
                    f"candidate {index} final {role}", (
                        (*review_prefix, "reasoning"), (*review_prefix, "findings"),
                    ),
                ))
    required_evidence = tuple(
        (label, tuple(_json_range(comparison, path) for path in alternatives))
        for label, alternatives in evidence_paths
    )

    def response_errors(verdict: Verdict) -> tuple[str, ...]:
        try:
            _parse_selection(
                verdict, spec=spec, project_root=project_root,
                source_spec_hash=source_spec_hash,
                bundle_ref=bundle_ref, candidates=by_id,
                required_evidence=required_evidence,
            )
        except (ValueError, TypeError, RecursionError) as exc:
            return (str(exc)[:1500],)
        return ()

    if evidence_error is not None or not candidates:
        verdict = Verdict(
            rubric="plan_selection", decision="inconclusive", findings=(),
            artifact_ref=bundle_ref, artifact_commit=spec.project_context.base_commit,
            reasoning=evidence_error or "No reviewed candidates are available.",
            coverage=ReviewCoverage(
                complete=evidence_error is None, sources_reviewed=("artifact",),
                omissions=(evidence_error,) if evidence_error else (),
            ),
            backend_name=None, transcript_ref=None, cost_usd=0.0, execution=None,
            kind="judge",
        )
    else:
        verdict = review(
            comparison_bytes,
            load("plan_selection"), backend=backend, corpus=corpus,
            max_turns=max_turns, budget_usd=budget_usd, kind="judge",
            response_retry_budget=2, response_contract_validator=response_errors,
            review_anchor_policy="anchors",
        )
    verdict = replace(
        verdict,
        artifacts={**verdict.artifacts, "plan_selection_bundle": bundle_ref},
    )
    verdict_ref = corpus.put_record(verdict, kind="verdicts")
    error = evidence_error
    mode: Literal["select", "synthesize", "none"] = "none"
    selected_id = None
    dag = None
    response = None
    if candidates and evidence_error is None:
        try:
            if is_review_infrastructure_failure(verdict) or verdict.accounting_debt:
                raise ValueError("selector review failed or its accounting is incomplete")
            completeness = verdict.execution_completeness
            if completeness is None or not (
                completeness.token_usage_complete and completeness.cost_complete
                and completeness.terminal_complete
            ):
                raise ValueError("selector execution evidence is incomplete")
            response, dag = _parse_selection(
                verdict, spec=spec, project_root=project_root,
                source_spec_hash=source_spec_hash,
                bundle_ref=bundle_ref, candidates=by_id,
                required_evidence=required_evidence,
            )
            mode = cast(Literal["select", "synthesize", "none"], response["mode"])
            selected_id = response["selected_id"]
        except (ValueError, TypeError, RecursionError) as exc:
            error = str(exc)
            dag = None
    selection_ref = corpus.put_artifact(serialize({
        "schema_version": "crucible.plan-selection-result.v1",
        "source_spec_hash": source_spec_hash,
        "bundle_ref": bundle_ref,
        "verdict_ref": verdict_ref,
        "mode": mode,
        "selected_id": selected_id,
        "candidate_digest": hash_record(dag) if dag is not None else None,
        "candidate_dag": dag,
        "response": response,
        "error": error,
    }), kind="plan_selection")
    return PlanSelectionResult(
        mode=mode, selected_id=selected_id, candidate_dag=dag, verdict=verdict,
        verdict_ref=verdict_ref, selection_ref=selection_ref, bundle_ref=bundle_ref,
        error=error,
    )
