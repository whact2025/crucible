# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Reproducible, provider-neutral rubric calibration.

Offline calibration is deliberately split into three parts:

* a versioned corpus containing the exact artifact, context, and prior inputs;
* a provider-neutral prediction generator; and
* a deterministic scorer.

Every retained source has its own SHA-256 and every prediction is bound to an
aggregate input digest.  A stale prediction therefore cannot be scored against
changed review material, even when its ``case_id`` happens to match.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Protocol


_CASE_SCHEMA = "crucible.rubric-calibration-case.v2"
_PREDICTION_SCHEMA = "crucible.rubric-calibration-prediction.v2"
_DECISIONS = frozenset({"pass", "revise", "reject", "inconclusive"})


@dataclass(frozen=True)
class CalibrationSource:
    """One exact, hash-verifiable input supplied to a reviewer."""

    source_id: str
    text: str
    sha256: str


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    rubric: str
    rubric_digest: str
    input_digest: str
    artifact_input: CalibrationSource
    context_inputs: tuple[CalibrationSource, ...]
    prior_inputs: tuple[CalibrationSource, ...]
    expected_decisions: tuple[str, ...]
    required_criteria: tuple[str, ...]
    expected_evidence_tokens: tuple[str, ...]
    forbidden_output_tokens: tuple[str, ...]
    tags: tuple[str, ...]
    replicate_group: str | None = None

    @property
    def context(self) -> dict[str, str]:
        """Return retained ``context:<key>`` inputs in review-call shape."""

        return {
            source.source_id.removeprefix("context:"): source.text
            for source in self.context_inputs
        }


@dataclass(frozen=True)
class CalibrationRequest:
    """Provider-neutral request passed to a prediction generator."""

    case_id: str
    rubric: str
    rubric_digest: str
    input_digest: str
    artifact: str
    context: Mapping[str, str]
    prior_inputs: tuple[CalibrationSource, ...]


@dataclass(frozen=True)
class CalibrationPrediction:
    case_id: str
    rubric_digest: str
    input_digest: str
    decision: str
    criterion_ids: tuple[str, ...]
    evidence_excerpts: tuple[str, ...]
    rendered_output: str
    provider: str
    model: str
    cost_usd: float
    latency_seconds: float


@dataclass(frozen=True)
class CalibrationMetrics:
    total: int
    pass_cases: int
    nonpass_cases: int
    decision_accuracy: float
    false_pass_rate: float
    false_reject_rate: float
    criterion_coverage_rate: float
    evidence_recall: float
    injection_resistance_rate: float
    stability_rate: float
    mean_cost_usd: float
    p95_cost_usd: float
    mean_latency_seconds: float
    p95_latency_seconds: float


class PredictionGenerator(Protocol):
    """Minimal interface implemented by local, hosted, or fixture runners."""

    def predict(
        self,
        request: CalibrationRequest,
    ) -> CalibrationPrediction | Mapping[str, Any]: ...


BackendAdapter = Callable[
    [object, CalibrationRequest], CalibrationPrediction | Mapping[str, Any]
]


def source_sha256(text: str) -> str:
    """Return Crucible's canonical textual SHA-256 representation."""

    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_source(source_id: str, text: str) -> CalibrationSource:
    """Build a source with a matching content hash."""

    return CalibrationSource(source_id, text, source_sha256(text))


@lru_cache(maxsize=None)
def rubric_digest_for(name: str) -> str:
    """Hash the fully rendered rubric contract used for review."""

    from . import load

    encoded = json.dumps(
        asdict(load(name)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def input_digest_for(
    rubric_digest: str,
    artifact_input: CalibrationSource,
    context_inputs: Iterable[CalibrationSource],
    prior_inputs: Iterable[CalibrationSource],
) -> str:
    """Hash the complete, ordered review input envelope."""

    def payload(source: CalibrationSource) -> dict[str, str]:
        return {
            "source_id": source.source_id,
            "text": source.text,
            "sha256": source.sha256,
        }

    encoded = json.dumps(
        {
            "rubric_digest": rubric_digest,
            "artifact_input": payload(artifact_input),
            "context_inputs": [payload(item) for item in context_inputs],
            "prior_inputs": [payload(item) for item in prior_inputs],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return tuple(rows)


def _strings(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(isinstance(item, str) and item for item in value)
    ):
        qualifier = " (may be empty)" if allow_empty else ""
        raise ValueError(
            f"{field} must be an array of unique non-empty strings{qualifier}"
        )
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _source(value: object, *, field: str, prefix: str | None) -> CalibrationSource:
    if not isinstance(value, dict) or set(value) != {"source_id", "text", "sha256"}:
        raise ValueError(f"{field} must be a source object")
    source_id = value["source_id"]
    text = value["text"]
    digest = value["sha256"]
    if not isinstance(source_id, str) or not source_id:
        raise ValueError(f"{field}.source_id must be non-empty")
    if prefix is not None and not source_id.startswith(prefix):
        raise ValueError(f"{field}.source_id must start with {prefix!r}")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{field}.text must be non-empty")
    if not isinstance(digest, str) or digest != source_sha256(text):
        raise ValueError(f"{field}.sha256 does not match retained text")
    return CalibrationSource(source_id, text, digest)


def _sources(
    value: object, *, field: str, prefix: str
) -> tuple[CalibrationSource, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    result = tuple(
        _source(item, field=f"{field}[{index}]", prefix=prefix)
        for index, item in enumerate(value)
    )
    ids = [item.source_id for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{field} source ids must be unique")
    return result


def _case_from_row(row: Mapping[str, Any], *, index: int) -> CalibrationCase:
    required = {
        "schema_version",
        "case_id",
        "rubric",
        "rubric_digest",
        "input_digest",
        "artifact_input",
        "context_inputs",
        "prior_inputs",
        "expected_decisions",
        "required_criteria",
        "expected_evidence_tokens",
        "forbidden_output_tokens",
        "tags",
        "replicate_group",
    }
    if set(row) != required or row.get("schema_version") != _CASE_SCHEMA:
        raise ValueError(f"case[{index}] has an invalid v2 shape")
    case_id = row["case_id"]
    rubric = row["rubric"]
    group = row["replicate_group"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"case[{index}].case_id must be non-empty")
    if not isinstance(rubric, str) or not rubric:
        raise ValueError(f"case[{index}].rubric must be non-empty")
    current_rubric_digest = rubric_digest_for(rubric)
    if row["rubric_digest"] != current_rubric_digest:
        raise ValueError(
            f"case[{index}].rubric_digest does not match the loaded rubric"
        )
    if group is not None and (not isinstance(group, str) or not group):
        raise ValueError(f"case[{index}].replicate_group is invalid")
    artifact = _source(
        row["artifact_input"],
        field=f"case[{index}].artifact_input",
        prefix=None,
    )
    if artifact.source_id != "artifact":
        raise ValueError(f"case[{index}].artifact_input source_id must be 'artifact'")
    contexts = _sources(
        row["context_inputs"],
        field=f"case[{index}].context_inputs",
        prefix="context:",
    )
    priors = _sources(
        row["prior_inputs"],
        field=f"case[{index}].prior_inputs",
        prefix="prior:",
    )
    all_source_ids = [
        artifact.source_id,
        *(item.source_id for item in contexts),
        *(item.source_id for item in priors),
    ]
    if len(all_source_ids) != len(set(all_source_ids)):
        raise ValueError(f"case[{index}] source ids must be globally unique")
    expected_digest = input_digest_for(
        current_rubric_digest,
        artifact,
        contexts,
        priors,
    )
    if row["input_digest"] != expected_digest:
        raise ValueError(f"case[{index}].input_digest does not match retained inputs")
    expected_decisions = _strings(
        row["expected_decisions"],
        field=f"case[{index}].expected_decisions",
    )
    if not set(expected_decisions) <= _DECISIONS:
        raise ValueError(f"case[{index}].expected_decisions contains an invalid value")
    evidence_tokens = _strings(
        row["expected_evidence_tokens"],
        field=f"case[{index}].expected_evidence_tokens",
    )
    forbidden_tokens = _strings(
        row["forbidden_output_tokens"],
        field=f"case[{index}].forbidden_output_tokens",
    )
    retained_text = "\n".join(item.text for item in (artifact, *contexts, *priors))
    if any(token not in retained_text for token in evidence_tokens):
        raise ValueError(
            f"case[{index}] expected evidence token is absent from retained inputs"
        )
    if any(token not in retained_text for token in forbidden_tokens):
        raise ValueError(
            f"case[{index}] forbidden output token is absent from retained inputs"
        )
    return CalibrationCase(
        case_id=case_id,
        rubric=rubric,
        rubric_digest=current_rubric_digest,
        input_digest=expected_digest,
        artifact_input=artifact,
        context_inputs=contexts,
        prior_inputs=priors,
        expected_decisions=expected_decisions,
        required_criteria=_strings(
            row["required_criteria"],
            field=f"case[{index}].required_criteria",
        ),
        expected_evidence_tokens=evidence_tokens,
        forbidden_output_tokens=forbidden_tokens,
        tags=_strings(row["tags"], field=f"case[{index}].tags"),
        replicate_group=group,
    )


def validate_inventory_coverage(cases: Iterable[CalibrationCase]) -> None:
    """Fail unless cases exercise every active rubric and key metric cohort."""

    from . import contracts

    case_list = tuple(cases)
    active = {item.name: item for item in contracts() if item.status == "active"}
    seen = {item.rubric for item in case_list}
    if unknown := seen - set(active):
        raise ValueError(
            f"calibration cases reference unknown rubrics: {sorted(unknown)!r}"
        )
    if missing := set(active) - seen:
        raise ValueError(
            f"calibration corpus is missing active rubrics: {sorted(missing)!r}"
        )
    for case in case_list:
        rubric = active[case.rubric]
        if set(case.required_criteria) != set(rubric.criteria):
            raise ValueError(
                f"case {case.case_id!r} criteria do not match rubric {case.rubric!r}"
            )
        context_ids = {item.source_id for item in case.context_inputs}
        missing_context = {
            f"context:{key}" for key in rubric.required_context
        } - context_ids
        if missing_context:
            raise ValueError(
                f"case {case.case_id!r} is missing required context inputs: "
                f"{sorted(missing_context)!r}"
            )
    if not any(item.expected_decisions == ("pass",) for item in case_list):
        raise ValueError("calibration corpus requires an unambiguous pass case")
    if not any("pass" not in item.expected_decisions for item in case_list):
        raise ValueError("calibration corpus requires a non-pass case")
    if not any(item.forbidden_output_tokens for item in case_list):
        raise ValueError("calibration corpus requires an injection-resistance case")
    groups: dict[str, list[CalibrationCase]] = {}
    for case in case_list:
        if case.replicate_group is not None:
            groups.setdefault(case.replicate_group, []).append(case)
    if not groups or any(len(items) < 2 for items in groups.values()):
        raise ValueError(
            "every calibration replicate group must contain at least two cases"
        )
    for name, items in groups.items():
        if len({item.input_digest for item in items}) != 1:
            raise ValueError(f"replicate group {name!r} does not bind identical inputs")


def load_cases(
    path: Path,
    *,
    require_active_inventory: bool = True,
) -> tuple[CalibrationCase, ...]:
    cases: list[CalibrationCase] = []
    seen: set[str] = set()
    for index, row in enumerate(_load_jsonl(path)):
        case = _case_from_row(row, index=index)
        if case.case_id in seen:
            raise ValueError(f"case[{index}].case_id must be unique")
        seen.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("calibration corpus cannot be empty")
    if require_active_inventory:
        validate_inventory_coverage(cases)
    return tuple(cases)


def _prediction_from_row(
    row: Mapping[str, Any],
    *,
    index: int,
) -> CalibrationPrediction:
    required = {
        "schema_version",
        "case_id",
        "rubric_digest",
        "input_digest",
        "decision",
        "criterion_ids",
        "evidence_excerpts",
        "rendered_output",
        "provider",
        "model",
        "cost_usd",
        "latency_seconds",
    }
    if set(row) != required or row.get("schema_version") != _PREDICTION_SCHEMA:
        raise ValueError(f"prediction[{index}] has an invalid v2 shape")
    case_id = row["case_id"]
    decision = row["decision"]
    digest = row["input_digest"]
    rubric_digest = row["rubric_digest"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError(f"prediction[{index}].case_id must be non-empty")
    if decision not in _DECISIONS:
        raise ValueError(f"prediction[{index}].decision is invalid")
    if (
        not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or len(digest) != 71
    ):
        raise ValueError(f"prediction[{index}].input_digest is invalid")
    if (
        not isinstance(rubric_digest, str)
        or not rubric_digest.startswith("sha256:")
        or len(rubric_digest) != 71
    ):
        raise ValueError(f"prediction[{index}].rubric_digest is invalid")
    rendered = row["rendered_output"]
    provider = row["provider"]
    model = row["model"]
    if not isinstance(rendered, str):
        raise ValueError(f"prediction[{index}].rendered_output must be a string")
    if not isinstance(provider, str) or not provider:
        raise ValueError(f"prediction[{index}].provider must be non-empty")
    if not isinstance(model, str) or not model:
        raise ValueError(f"prediction[{index}].model must be non-empty")
    cost = row["cost_usd"]
    latency = row["latency_seconds"]
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(float(cost))
        or float(cost) < 0
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise ValueError(
            f"prediction[{index}] cost/latency must be finite and non-negative"
        )
    return CalibrationPrediction(
        case_id=case_id,
        rubric_digest=rubric_digest,
        input_digest=digest,
        decision=decision,
        criterion_ids=_strings(
            row["criterion_ids"],
            field=f"prediction[{index}].criterion_ids",
            allow_empty=True,
        ),
        evidence_excerpts=_strings(
            row["evidence_excerpts"],
            field=f"prediction[{index}].evidence_excerpts",
            allow_empty=True,
        ),
        rendered_output=rendered,
        provider=provider,
        model=model,
        cost_usd=float(cost),
        latency_seconds=float(latency),
    )


def load_predictions(path: Path) -> tuple[CalibrationPrediction, ...]:
    predictions: list[CalibrationPrediction] = []
    seen: set[str] = set()
    for index, row in enumerate(_load_jsonl(path)):
        prediction = _prediction_from_row(row, index=index)
        if prediction.case_id in seen:
            raise ValueError(f"prediction[{index}].case_id must be unique")
        seen.add(prediction.case_id)
        predictions.append(prediction)
    return tuple(predictions)


def _prediction_mapping(prediction: CalibrationPrediction) -> dict[str, Any]:
    return {
        "schema_version": _PREDICTION_SCHEMA,
        "case_id": prediction.case_id,
        "rubric_digest": prediction.rubric_digest,
        "input_digest": prediction.input_digest,
        "decision": prediction.decision,
        "criterion_ids": list(prediction.criterion_ids),
        "evidence_excerpts": list(prediction.evidence_excerpts),
        "rendered_output": prediction.rendered_output,
        "provider": prediction.provider,
        "model": prediction.model,
        "cost_usd": prediction.cost_usd,
        "latency_seconds": prediction.latency_seconds,
    }


def write_predictions(
    path: Path,
    predictions: Iterable[CalibrationPrediction],
) -> None:
    """Write predictions in stable JSONL order."""

    rows = [
        json.dumps(_prediction_mapping(item), ensure_ascii=False, sort_keys=True)
        for item in predictions
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _request(case: CalibrationCase) -> CalibrationRequest:
    return CalibrationRequest(
        case_id=case.case_id,
        rubric=case.rubric,
        rubric_digest=case.rubric_digest,
        input_digest=case.input_digest,
        artifact=case.artifact_input.text,
        context=case.context,
        prior_inputs=case.prior_inputs,
    )


def generate_predictions(
    cases: Iterable[CalibrationCase],
    generator: PredictionGenerator,
    *,
    output_path: Path | None = None,
) -> tuple[CalibrationPrediction, ...]:
    """Invoke a supplied generator without provider or credential coupling."""

    predictions: list[CalibrationPrediction] = []
    for index, case in enumerate(cases):
        raw = generator.predict(_request(case))
        prediction = (
            raw
            if isinstance(raw, CalibrationPrediction)
            else _prediction_from_row(raw, index=index)
        )
        if prediction.case_id != case.case_id:
            raise ValueError(
                f"generator returned case_id {prediction.case_id!r} for "
                f"{case.case_id!r}"
            )
        if prediction.input_digest != case.input_digest:
            raise ValueError(
                f"generator returned a stale input digest for {case.case_id!r}"
            )
        if prediction.rubric_digest != case.rubric_digest:
            raise ValueError(
                f"generator returned a stale rubric digest for {case.case_id!r}"
            )
        predictions.append(prediction)
    result = tuple(predictions)
    if output_path is not None:
        write_predictions(output_path, result)
    return result


class _BackendGenerator:
    def __init__(self, backend: object, adapter: BackendAdapter) -> None:
        self._backend = backend
        self._adapter = adapter

    def predict(
        self,
        request: CalibrationRequest,
    ) -> CalibrationPrediction | Mapping[str, Any]:
        return self._adapter(self._backend, request)


def generate_with_backend(
    cases: Iterable[CalibrationCase],
    *,
    backend: object,
    adapter: BackendAdapter,
    output_path: Path | None = None,
) -> tuple[CalibrationPrediction, ...]:
    """Generate predictions through a caller-supplied backend adapter.

    Crucible never discovers credentials or assumes a provider here.  The
    adapter owns transport and converts the backend response to the v2
    prediction mapping; this function enforces identity before persistence.
    """

    return generate_predictions(
        cases,
        _BackendGenerator(backend, adapter),
        output_path=output_path,
    )


def _nearest_rank_p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def score(
    cases: Iterable[CalibrationCase],
    predictions: Iterable[CalibrationPrediction],
) -> CalibrationMetrics:
    case_list = tuple(cases)
    prediction_list = tuple(predictions)
    prediction_by_id = {item.case_id: item for item in prediction_list}
    if len(prediction_by_id) != len(prediction_list):
        raise ValueError("predictions contain duplicate case ids")
    case_ids = {item.case_id for item in case_list}
    if set(prediction_by_id) != case_ids:
        raise ValueError(
            "predictions must cover the calibration cases exactly; "
            f"missing={sorted(case_ids - set(prediction_by_id))!r}, "
            f"extra={sorted(set(prediction_by_id) - case_ids)!r}"
        )
    if not case_list:
        raise ValueError("cannot score an empty calibration corpus")

    correct = false_pass = false_reject = 0
    criterion_hits = criterion_total = evidence_hits = evidence_total = 0
    injection_safe = injection_total = 0
    pass_cases = nonpass_cases = 0
    grouped: dict[str, list[str]] = {}
    for case in case_list:
        prediction = prediction_by_id[case.case_id]
        if prediction.input_digest != case.input_digest:
            raise ValueError(
                f"prediction {case.case_id!r} was generated for different inputs"
            )
        if prediction.rubric_digest != case.rubric_digest:
            raise ValueError(
                f"prediction {case.case_id!r} was generated for a different rubric"
            )
        expected = set(case.expected_decisions)
        correct += prediction.decision in expected
        if expected == {"pass"}:
            pass_cases += 1
            false_reject += prediction.decision != "pass"
        elif "pass" not in expected:
            nonpass_cases += 1
            false_pass += prediction.decision == "pass"
        required = set(case.required_criteria)
        criterion_total += len(required)
        criterion_hits += len(required & set(prediction.criterion_ids))
        joined_evidence = "\n".join(prediction.evidence_excerpts)
        for token in case.expected_evidence_tokens:
            evidence_total += 1
            evidence_hits += token in joined_evidence
        if case.forbidden_output_tokens:
            injection_total += 1
            injection_safe += not any(
                token in prediction.rendered_output
                for token in case.forbidden_output_tokens
            )
        if case.replicate_group is not None:
            grouped.setdefault(case.replicate_group, []).append(prediction.decision)

    stable_groups = sum(len(set(decisions)) == 1 for decisions in grouped.values())
    costs = [prediction_by_id[item.case_id].cost_usd for item in case_list]
    latencies = [prediction_by_id[item.case_id].latency_seconds for item in case_list]
    total = len(case_list)
    return CalibrationMetrics(
        total=total,
        pass_cases=pass_cases,
        nonpass_cases=nonpass_cases,
        decision_accuracy=correct / total,
        false_pass_rate=(false_pass / nonpass_cases if nonpass_cases else 0.0),
        false_reject_rate=(false_reject / pass_cases if pass_cases else 0.0),
        criterion_coverage_rate=(
            criterion_hits / criterion_total if criterion_total else 1.0
        ),
        evidence_recall=(evidence_hits / evidence_total if evidence_total else 1.0),
        injection_resistance_rate=(
            injection_safe / injection_total if injection_total else 1.0
        ),
        stability_rate=(stable_groups / len(grouped) if grouped else 1.0),
        mean_cost_usd=fmean(costs),
        p95_cost_usd=_nearest_rank_p95(costs),
        mean_latency_seconds=fmean(latencies),
        p95_latency_seconds=_nearest_rank_p95(latencies),
    )


__all__ = [
    "BackendAdapter",
    "CalibrationCase",
    "CalibrationMetrics",
    "CalibrationPrediction",
    "CalibrationRequest",
    "CalibrationSource",
    "PredictionGenerator",
    "generate_predictions",
    "generate_with_backend",
    "input_digest_for",
    "load_cases",
    "load_predictions",
    "make_source",
    "rubric_digest_for",
    "score",
    "source_sha256",
    "validate_inventory_coverage",
    "write_predictions",
]
