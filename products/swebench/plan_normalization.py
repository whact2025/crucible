# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Deterministic SWE-bench plan normalization.

SWE-bench's authoring contract treats repository tests as read-only: agents
may inspect and execute them, while the official evaluator owns the hidden
``test_patch``.  Planner models nevertheless commonly add a sibling
"write regression tests" slice to an otherwise correct source fix.  Rejecting
the entire TierDAG for that mechanical mistake discards useful source work.

This module removes test-path writes and rewrites test-only citations to
neutral evidence prose.  It never invents source work or changes source-fix
semantics, and the normalized DAG must still pass the normal Critic and Judge
gates before implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import logging
import re
from typing import Any

from assay import Slice, Tier, TierDAG

from .paths import is_standard_test_path


logger = logging.getLogger(__name__)


class TestOnlyPlanError(ValueError):
    """Raised when normalization leaves no source-bearing implementation."""


_TEST_EVIDENCE_REPLACEMENT = "existing repository test evidence"
_FILE_CITATION_RE = re.compile(
    r"(?P<quote>[`'\"]?)"
    r"(?P<path>\b[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,12}\b)"
    r"(?P=quote)"
)
_TEST_SYMBOL_RE = re.compile(
    r"(?P<quote>[`'\"]?)"
    r"(?P<symbol>\b(?:test_[A-Za-z0-9_]+|[A-Za-z0-9_]+_tests?|"
    r"Test[A-Z][A-Za-z0-9_]*)\b(?:\(\))?)"
    r"(?P=quote)"
)


@dataclass(frozen=True)
class IntentCitationRewrite:
    """One exact token removed from a slice's original intent."""

    tier_id: str
    slice_id: str
    kind: str
    original_token: str
    replacement: str
    original_start: int
    original_end: int


@dataclass(frozen=True)
class PlanNormalization:
    """One auditable raw-to-normalized TierDAG transformation."""

    dag: TierDAG
    stripped_paths: tuple[str, ...]
    dropped_slices: tuple[str, ...]
    dropped_tiers: tuple[str, ...]
    rewired_dependencies: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    intent_citation_rewrites: tuple[IntentCitationRewrite, ...]

    @property
    def changed(self) -> bool:
        return bool(
            self.stripped_paths
            or self.dropped_slices
            or self.dropped_tiers
            or self.rewired_dependencies
            or self.intent_citation_rewrites
        )

    def manifest(self) -> dict[str, Any]:
        """Return the stable JSON-compatible audit manifest body."""

        return {
            "schema_version": "crucible.swebench-plan-normalization.v2",
            "policy": "remove_test_writes_and_citations_then_re_review",
            "stripped_paths": list(self.stripped_paths),
            "dropped_slices": list(self.dropped_slices),
            "dropped_tiers": list(self.dropped_tiers),
            "rewired_dependencies": [
                {
                    "tier_id": tier_id,
                    "before": list(before),
                    "after": list(after),
                }
                for tier_id, before, after in self.rewired_dependencies
            ],
            "rewritten_intent_citations": [
                {
                    "tier_id": rewrite.tier_id,
                    "slice_id": rewrite.slice_id,
                    "kind": rewrite.kind,
                    "original_token": rewrite.original_token,
                    "replacement": rewrite.replacement,
                    "original_span": [
                        rewrite.original_start,
                        rewrite.original_end,
                    ],
                }
                for rewrite in self.intent_citation_rewrites
            ],
        }


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _dag_payload(dag: TierDAG) -> dict[str, Any]:
    return {
        "tiers": [
            {
                "tier_id": tier.tier_id,
                "depends_on": list(tier.depends_on),
                "slices": [
                    {
                        "id": slice_.id,
                        "directive": {
                            "slice_id": slice_.directive.slice_id,
                            "requirement_refs": list(
                                slice_.directive.requirement_refs
                            ),
                            "intent": slice_.directive.intent,
                            "files_in_scope": list(
                                slice_.directive.files_in_scope
                            ),
                            "creates_paths": list(
                                slice_.directive.creates_paths
                            ),
                            "verify_mode": slice_.directive.verify_mode,
                            "blast_radius": dict(
                                slice_.directive.blast_radius
                            ),
                            **({
                                "read_only_references": [
                                    asdict(reference)
                                    for reference in slice_.directive.read_only_references
                                ],
                            } if slice_.directive.read_only_references else {}),
                        },
                    }
                    for slice_ in tier.slices
                ],
            }
            for tier in dag.tiers
        ],
    }


def _rewrite_test_citations(
    intent: str,
    *,
    tier_id: str,
    slice_id: str,
) -> tuple[str, tuple[IntentCitationRewrite, ...]]:
    """Replace test paths/symbols with neutral, path-free evidence prose."""

    if not intent:
        return intent, ()

    # (start, end, priority, kind). Path matches win overlap with a test-like
    # basename such as ``tests/test_separable.py``.
    matches: list[tuple[int, int, int, str]] = []
    for match in _FILE_CITATION_RE.finditer(intent):
        path = match.group("path")
        if not is_standard_test_path(path):
            continue
        matches.append((match.start(), match.end(), 0, "test_path"))
    matches.extend(
        (match.start(), match.end(), 1, "test_symbol")
        for match in _TEST_SYMBOL_RE.finditer(intent)
    )
    if not matches:
        return intent, ()

    selected: list[tuple[int, int, int, str]] = []
    for candidate in sorted(matches, key=lambda item: (item[0], item[2], -item[1])):
        start, end, _priority, _kind = candidate
        if any(start < kept_end and end > kept_start for kept_start, kept_end, *_ in selected):
            continue
        selected.append(candidate)
    selected.sort(key=lambda item: item[0])

    rendered: list[str] = []
    rewrites: list[IntentCitationRewrite] = []
    cursor = 0
    for start, end, _priority, kind in selected:
        rendered.append(intent[cursor:start])
        rendered.append(_TEST_EVIDENCE_REPLACEMENT)
        rewrites.append(IntentCitationRewrite(
            tier_id=tier_id,
            slice_id=slice_id,
            kind=kind,
            original_token=intent[start:end],
            replacement=_TEST_EVIDENCE_REPLACEMENT,
            original_start=start,
            original_end=end,
        ))
        cursor = end
    rendered.append(intent[cursor:])
    return "".join(rendered), tuple(rewrites)


def normalize_swebench_plan(dag: TierDAG) -> PlanNormalization:
    """Remove test writes/citations and return a source DAG for re-review.

    Mixed source/test slices retain their source paths.  Slices containing
    only test paths are removed.  Empty tiers are removed and dependencies on
    them are replaced transitively by their own dependencies. Test paths and
    individual test symbols in retained intent prose become neutral test
    evidence references. If every slice was test-only, fail closed instead of
    manufacturing an executable plan.
    """

    stripped_paths: list[str] = []
    dropped_slices: list[str] = []
    citation_rewrites: list[IntentCitationRewrite] = []
    filtered_tiers: list[Tier] = []

    for tier in dag.tiers:
        kept_slices: list[Slice] = []
        for slice_ in tier.slices:
            directive = slice_.directive
            normalized_intent, rewrites = _rewrite_test_citations(
                directive.intent,
                tier_id=tier.tier_id,
                slice_id=directive.slice_id,
            )
            citation_rewrites.extend(rewrites)
            original_paths = (
                *directive.files_in_scope,
                *directive.creates_paths,
            )
            files = tuple(
                path
                for path in directive.files_in_scope
                if not is_standard_test_path(path)
            )
            creates = tuple(
                path
                for path in directive.creates_paths
                if not is_standard_test_path(path)
            )
            stripped_paths.extend(
                path for path in original_paths if is_standard_test_path(path)
            )

            # Only drop a slice because of this policy when it actually had
            # write paths and every one was a test path.  A malformed empty-
            # scope slice is preserved for the ordinary deterministic plan
            # validators to reject with their more precise diagnostic.
            if original_paths and not files and not creates:
                dropped_slices.append(slice_.id)
                continue
            if (
                files != directive.files_in_scope
                or creates != directive.creates_paths
                or normalized_intent != directive.intent
            ):
                directive = replace(
                    directive,
                    intent=normalized_intent,
                    files_in_scope=files,
                    creates_paths=creates,
                )
                slice_ = replace(slice_, directive=directive)
            kept_slices.append(slice_)
        filtered_tiers.append(replace(tier, slices=tuple(kept_slices)))

    tier_by_id = {tier.tier_id: tier for tier in filtered_tiers}
    dropped_tiers = tuple(
        tier.tier_id for tier in filtered_tiers if not tier.slices
    )
    kept_tiers = [tier for tier in filtered_tiers if tier.slices]
    if not kept_tiers:
        raise TestOnlyPlanError(
            "SWE-bench candidate contains no source-bearing slice after "
            "test-write normalization"
        )

    dropped = frozenset(dropped_tiers)

    def expand_dependency(tier_id: str, trail: frozenset[str]) -> list[str]:
        if tier_id not in dropped:
            return [tier_id]
        if tier_id in trail:
            raise ValueError(
                "cycle while rewiring removed SWE-bench test-only tiers: "
                f"{tier_id}"
            )
        tier = tier_by_id[tier_id]
        expanded: list[str] = []
        for dependency in tier.depends_on:
            expanded.extend(
                expand_dependency(dependency, trail | {tier_id})
            )
        return expanded

    rewired: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    normalized_tiers: list[Tier] = []
    for tier in kept_tiers:
        dependencies: list[str] = []
        for dependency in tier.depends_on:
            dependencies.extend(expand_dependency(dependency, frozenset()))
        after = _unique(dependencies)
        if after != tier.depends_on:
            rewired.append((tier.tier_id, tier.depends_on, after))
            tier = replace(tier, depends_on=after)
        normalized_tiers.append(tier)

    return PlanNormalization(
        dag=TierDAG(tiers=tuple(normalized_tiers)),
        stripped_paths=_unique(stripped_paths),
        dropped_slices=_unique(dropped_slices),
        dropped_tiers=dropped_tiers,
        rewired_dependencies=tuple(rewired),
        intent_citation_rewrites=tuple(citation_rewrites),
    )


class AuditedSwebenchPlanTransform:
    """Callable Assay candidate hook that persists every changed mapping.

    The planner's exact raw response already lives in its transcript.  These
    additional artifacts make the typed raw DAG, normalized DAG, and the
    deterministic transformation manifest independently addressable.
    """

    def __init__(self, corpus: Any) -> None:
        self._corpus = corpus
        self.artifact_refs: list[str] = []

    def __call__(self, dag: TierDAG) -> TierDAG:
        try:
            result = normalize_swebench_plan(dag)
        except TestOnlyPlanError:
            raw_ref = self._corpus.put_artifact(
                json.dumps(
                    _dag_payload(dag),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8"),
                kind="plan_candidates_raw",
                name_hint="swebench_raw_test_only_plan.json",
            )
            manifest_ref = self._corpus.put_artifact(
                json.dumps(
                    {
                        "schema_version": (
                            "crucible.swebench-plan-normalization.v2"
                        ),
                        "policy": (
                            "remove_test_writes_and_citations_then_re_review"
                        ),
                        "terminal_reason": "test_only_plan",
                        "raw_candidate_ref": str(raw_ref),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ).encode("utf-8"),
                kind="plan_normalizations",
                name_hint="swebench_test_only_plan.json",
            )
            self.artifact_refs.append(str(manifest_ref))
            logger.warning(
                "SWE-bench planner candidate contained only test writes; "
                "manifest=%s",
                manifest_ref,
            )
            raise
        if not result.changed:
            return dag
        raw_ref = self._corpus.put_artifact(
            json.dumps(
                _dag_payload(dag), ensure_ascii=False, sort_keys=True, indent=2,
            ).encode("utf-8"),
            kind="plan_candidates_raw",
            name_hint="swebench_raw_plan.json",
        )
        normalized_ref = self._corpus.put_artifact(
            json.dumps(
                _dag_payload(result.dag),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8"),
            kind="plan_candidates_normalized",
            name_hint="swebench_normalized_plan.json",
        )
        manifest = {
            **result.manifest(),
            "raw_candidate_ref": str(raw_ref),
            "normalized_candidate_ref": str(normalized_ref),
        }
        manifest_ref = self._corpus.put_artifact(
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2,
            ).encode("utf-8"),
            kind="plan_normalizations",
            name_hint="swebench_plan_normalization.json",
        )
        self.artifact_refs.append(str(manifest_ref))
        logger.info(
            "SWE-bench normalized planner candidate: stripped=%d "
            "rewritten_citations=%d dropped_slices=%d dropped_tiers=%d "
            "manifest=%s",
            len(result.stripped_paths),
            len(result.intent_citation_rewrites),
            len(result.dropped_slices),
            len(result.dropped_tiers),
            manifest_ref,
        )
        return result.dag


__all__ = [
    "AuditedSwebenchPlanTransform",
    "IntentCitationRewrite",
    "PlanNormalization",
    "TestOnlyPlanError",
    "normalize_swebench_plan",
]
