# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adapt static workflow fixtures to the strict runtime-generated v2 schema.

Workflow tests intentionally script semantic responses. Production reviewers
see per-run collision-proof prior refs and role-specific criterion schemas, so
the fixture adapter fills those runtime identities without weakening product
validation.
"""

from __future__ import annotations

import json
from typing import Any, Mapping


def _source_texts(user: str) -> dict[str, str]:
    """Return the first exact text envelope for each runtime source ID."""

    sources: dict[str, str] = {}
    decoder = json.JSONDecoder()
    for index, char in enumerate(user):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(user[index:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "assay.untrusted-source.v1"
            and isinstance(value.get("source_id"), str)
            and isinstance(value.get("text"), str)
        ):
            sources.setdefault(value["source_id"], value["text"])
        if (
            isinstance(value, dict)
            and value.get("schema_version")
            == "assay.untrusted-source.anchored.v1"
            and isinstance(value.get("source_id"), str)
            and isinstance(value.get("anchors"), list)
        ):
            text = next(
                (
                    text
                    for anchor in value["anchors"]
                    if isinstance(anchor, dict)
                    and isinstance((text := anchor.get("text")), str)
                    and text
                ),
                "",
            )
            if text:
                sources.setdefault(value["source_id"], text)
    return sources


def _artifact_text(user: str) -> str:
    return _source_texts(user).get("artifact", "")


def _schema_criteria(schema: Mapping[str, Any] | None) -> tuple[str, ...]:
    try:
        values = schema["properties"]["criterion_results"]["items"][
            "properties"
        ]["criterion_id"]["enum"]
    except (KeyError, TypeError):
        return ()
    return tuple(value for value in values if isinstance(value, str))


def _material_prior_refs(user: str) -> tuple[str, ...]:
    """Read the harness-owned active closure frontier from the prompt."""

    decoder = json.JSONDecoder()
    for index, char in enumerate(user):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(user[index:])
        except json.JSONDecodeError:
            continue
        if not (
            isinstance(value, dict)
            and value.get("schema_version") == "crucible.review-obligations.v1"
        ):
            continue
        obligations = value.get("obligations")
        if not isinstance(obligations, dict):
            return ()
        bindings = obligations.get("material_prior_bindings", ())
        if not isinstance(bindings, list):
            return ()
        return tuple(dict.fromkeys(
            finding_ref
            for item in bindings
            if isinstance(item, dict)
            and isinstance(finding_ref := item.get("finding_ref"), str)
        ))
    return ()


def _source_for_ref(ref: str) -> str:
    if ref.startswith("prior:prior_critic_findings:"):
        return "context:prior_critic_findings"
    if ref.startswith("prior:prior_judge_findings:"):
        return "context:prior_judge_findings"
    return ref


def adapt_scripted_review_response(final: str, invocation: Any) -> str:
    """Bind a canned response to runtime refs and required criterion cards."""

    try:
        payload = json.loads(final)
    except (TypeError, json.JSONDecodeError):
        return final
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "assay.review.v2"
    ):
        return final

    refs = _material_prior_refs(invocation.user)
    ref_by_finding_id: dict[str, list[str]] = {}
    for ref in refs:
        ref_by_finding_id.setdefault(ref.rsplit(":", 1)[-1], []).append(ref)

    coverage = payload.get("coverage")
    if isinstance(coverage, dict) and isinstance(coverage.get("sources_reviewed"), list):
        retained = [
            item
            for item in coverage["sources_reviewed"]
            if not (isinstance(item, str) and item.startswith("prior:"))
        ]
        coverage["sources_reviewed"] = list(dict.fromkeys((
            *retained, *(_source_for_ref(ref) for ref in refs),
        )))

    findings = payload.get("findings")
    if not isinstance(findings, list):
        return final
    assigned: set[str] = set()
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        targets = finding.get("targets")
        bound: list[str] = []
        if isinstance(targets, list):
            for target in targets:
                if target in refs and target not in assigned:
                    bound.append(target)
                    assigned.add(target)
                    continue
                matches = ref_by_finding_id.get(str(target), [])
                match = next((item for item in matches if item not in assigned), None)
                if match is not None:
                    bound.append(match)
                    assigned.add(match)
        finding["targets"] = bound

    # Closure roles must account for every supplied material prior once. The
    # semantic workflow fixtures use a single adjudicating finding, so attach
    # any ref whose old short id was omitted to that finding.
    if findings:
        first = findings[0]
        if isinstance(first, dict):
            first_targets = first.setdefault("targets", [])
            if isinstance(first_targets, list):
                first_targets.extend(ref for ref in refs if ref not in assigned)
            evidence = first.setdefault("evidence", [])
            if isinstance(evidence, list):
                for ref in refs:
                    source_id = _source_for_ref(ref)
                    if not any(
                        isinstance(item, dict) and item.get("source_id") == source_id
                        for item in evidence
                    ):
                        evidence.append({
                            "source_id": source_id,
                            "kind": "prior_finding_ref",
                            "location": "finding_ref",
                            "excerpt": ref,
                        })

    criteria = _schema_criteria(getattr(invocation, "response_schema", None))
    if criteria and "criterion_results" not in payload:
        source_texts = _source_texts(invocation.user)
        first_finding = next(
            (item for item in findings if isinstance(item, dict)), {}
        )
        evidence = first_finding.get("evidence", [])
        exemplar = next(
            (
                item
                for item in evidence
                if isinstance(item, dict)
                and isinstance(item.get("source_id"), str)
                and isinstance(item.get("excerpt"), str)
            ),
            None,
        )
        if exemplar is None:
            artifact = _artifact_text(invocation.user)
            excerpt = artifact.strip()[:120] or "artifact"
            exemplar = {
                "source_id": "artifact",
                "kind": "source_excerpt",
                "location": "artifact",
                "excerpt": excerpt,
            }
        decision = payload.get("decision")
        finding_criterion = first_finding.get("criterion_id")
        headline_index = (
            criteria.index(finding_criterion)
            if finding_criterion in criteria
            else 0
        )
        criterion_evidence = [dict(exemplar)]
        represented_sources = {exemplar["source_id"]}
        for source_id in (
            "context:spec_intent",
            "context:spec_constraints",
            "context:spec_done_criteria",
        ):
            source_text = source_texts.get(source_id, "")
            if not source_text or source_id in represented_sources:
                continue
            criterion_evidence.append({
                "source_id": source_id,
                "kind": "source_excerpt",
                "location": source_id,
                "excerpt": source_text.strip()[:120],
            })
            represented_sources.add(source_id)
        payload["criterion_results"] = [
            {
                "criterion_id": criterion,
                "outcome": (
                    "violated"
                    if decision in {"revise", "reject"} and index == headline_index
                    else "inconclusive"
                    if decision == "inconclusive" and index == headline_index
                    else "satisfied"
                ),
                "reasoning": "fixture-backed criterion adjudication",
                "evidence": [dict(item) for item in criterion_evidence],
                "headline": index == headline_index,
            }
            for index, criterion in enumerate(criteria)
        ]
    return json.dumps(payload)


__all__ = ["adapt_scripted_review_response"]
