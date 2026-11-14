# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Closed, provider-free SWE-bench backend convergence matrix planning.

The matrix in :mod:`crucible.products.swebench` compares two complete
workflow routes, not individual backend calls.  Loading and expanding it is
provider-free: the code authenticates the dataset bytes, backend routing
files, cohort selection, counterbalancing, state-root separation, and nominal
configured budget before an operator can review a proposed cell.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml  # type: ignore[import-untyped]

from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.backend_convergence_catalog import (
    BackendCatalogContract,
    BackendLaneCatalog,
    BackendLaneCatalogError,
    BackendLaneDefinition,
    PromotionAuthorityValidator,
    generate_reference_arm_pairs,
    load_backend_profile_catalog,
    validate_backend_lane_catalog,
)
from crucible.products.swebench.dataset import SwebenchInstance, load_dataset_bytes


MATRIX_SCHEMA = "crucible.swebench-backend-convergence-matrix.v1"
PLAN_SCHEMA = "crucible.swebench-backend-convergence-plan.v1"
MATRIX_SCHEMA_V2 = "crucible.swebench-backend-convergence-matrix.v2"
PLAN_SCHEMA_V2 = "crucible.swebench-backend-convergence-plan.v2"

_ARMS = ("A", "B")
_PHASES = ("calibration", "pilot", "heldout")
_SELECTION_ALGORITHM = "sha256-ranked-and-block-interleaved-by-stratum-v1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_INSTANCE_RE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-[0-9]+$")
SWEBENCH_BACKEND_CONVERGENCE_ROLES = (
    "architect_planner",
    "architect_critic",
    "architect_judge",
    "implementer",
    "breach_defender",
    "swebench_impl_breaker",
    "breach_judge",
    "diff_scope",
    "tier_completeness",
    "review_conflict_authority",
)
SWEBENCH_BACKEND_CATALOG_CONTRACT = BackendCatalogContract(
    product="swebench",
    roles=SWEBENCH_BACKEND_CONVERGENCE_ROLES,
)
_EXPECTED_ROLES = SWEBENCH_BACKEND_CONVERGENCE_ROLES
_CALIBRATION_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "django__django-10097",
    "django__django-10554",
    "pytest-dev__pytest-10051",
    "pytest-dev__pytest-10081",
    "scikit-learn__scikit-learn-10297",
    "scikit-learn__scikit-learn-10844",
    "sympy__sympy-11618",
    "sympy__sympy-12096",
)
_EXPECTED_STRATA: Mapping[str, str] = {
    "astropy": "astropy/astropy",
    "django": "django/django",
    "matplotlib": "matplotlib/matplotlib",
    "requests": "psf/requests",
    "xarray": "pydata/xarray",
    "pylint": "pylint-dev/pylint",
    "pytest": "pytest-dev/pytest",
    "sklearn": "scikit-learn/scikit-learn",
    "sphinx": "sphinx-doc/sphinx",
    "sympy": "sympy/sympy",
}
_EXPECTED_HELDOUT_QUOTAS: Mapping[str, int] = {
    "astropy": 11,
    "django": 11,
    "matplotlib": 11,
    "requests": 6,
    "xarray": 11,
    "pylint": 8,
    "pytest": 11,
    "sklearn": 11,
    "sphinx": 10,
    "sympy": 10,
}
_EXPECTED_DATASET_PROVENANCE: Mapping[str, str] = {
    "id": "princeton-nlp/SWE-bench_Verified:test",
    "split": "test",
    "revision": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "source_file": "data/test-00000-of-00001.parquet",
    "source_sha256": (
        "sha256:a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd"
    ),
    "export_algorithm": "pyarrow-read-table-to-pylist-json-dumps-indent2-v1",
}
_EXPECTED_DATASET_DIGEST = (
    "sha256:1001dcfee9f23df1c084b17ec5f2440d4608fb6972c60303ed1d0c60e3c9493a"
)
_PLAN_SCHEMA_BY_MATRIX: Mapping[str, str] = {
    MATRIX_SCHEMA: PLAN_SCHEMA,
    MATRIX_SCHEMA_V2: PLAN_SCHEMA_V2,
}
_MATRIX_LABELS: Mapping[str, str] = {
    MATRIX_SCHEMA: "Astra/Fable",
    MATRIX_SCHEMA_V2: "Astra/Mythos 5.1",
}
_EXPECTED_ANALYSIS_IMPLEMENTATIONS: Mapping[str, tuple[str, str, str]] = {
    MATRIX_SCHEMA: (
        "crucible.swebench-backend-convergence-analysis.v1",
        "backend_convergence_analysis.py",
        "sha256:4973d24d7537305350e9930fda269573a083030a6a77ed8b03b4d6b6d656e14d",
    ),
    MATRIX_SCHEMA_V2: (
        "crucible.swebench-backend-convergence-analysis.v2",
        "backend_convergence_analysis_astra_mythos51.py",
        "sha256:4209cee86a7676014e6ad5e0a05cfef23364edabd7cce71c202ccdd80427bd70",
    ),
}
_EXPECTED_ANALYSIS_SEEDS: Mapping[str, str] = {
    MATRIX_SCHEMA: "swebench-astra-fable51-analysis-20260912-v1",
    MATRIX_SCHEMA_V2: "swebench-astra-mythos51-analysis-20260914-v2",
}
_EXPECTED_SUCCESSOR_SELECTION_SEEDS: Mapping[str, str] = {
    MATRIX_SCHEMA_V2: "swebench-astra-mythos51-convergence-20260914-v2",
}
_OUTPUT_NAMESPACE_BY_MATRIX: Mapping[str, str] = {
    MATRIX_SCHEMA_V2: "swebench-astra-mythos51-v2",
}
_EXPECTED_LANES: Mapping[str, Mapping[str, tuple[str, str, str, str]]] = {
    MATRIX_SCHEMA: {
        "A": ("codex-astra", "codex", "gpt-6-astra", "max"),
        "B": ("claude-fable51", "claude", "claude-fable-5-1", "high"),
    },
    MATRIX_SCHEMA_V2: {
        "A": ("codex-astra", "codex", "gpt-6-astra", "max"),
        "B": ("claude-mythos51", "claude", "claude-mythos-5-1", "high"),
    },
}
_EXPECTED_TRANSPORTS: Mapping[str, str] = {
    "A": "legacy_app_server",
    "B": "claude_agent_sdk",
}
_EXPECTED_SESSION_EXTRAS: Mapping[str, Mapping[str, str]] = {
    "A": {"credential_source": "chatgpt_session"},
    "B": {
        "credential_source": "claude_session",
        "expected_org_id": "76f2dd92-7d56-4550-afaf-abd363e3235c",
        "expected_org_name": "NVIDIA Corporation - Power Users",
    },
}
_EXPECTED_METRICS = (
    "primary_resolved_pass_at_1_by_arm",
    "primary_paired_resolution_delta",
    "primary_paired_both_resolve_rate",
    "primary_paired_either_resolves_rate",
    "primary_paired_disagreement_rate",
    "all_attempt_failure_funnel_by_arm",
    "all_attempt_submitted_patch_rate_by_arm",
    "all_attempt_safety_gate_block_rate_by_arm",
    "all_attempt_cost_usd_by_arm",
    "all_attempt_wall_time_seconds_by_arm",
    "within_task_replicate_agreement_by_arm",
)


class BackendConvergenceMatrixError(ValueError):
    """Raised when a matrix, route, dataset, or expansion drifts."""


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise BackendConvergenceMatrixError(f"{field} must be an exact mapping")
    return value


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field: str,
) -> None:
    optional = optional or set()
    keys = set(value)
    if keys != required | (keys & optional):
        missing = sorted(required - keys)
        extra = sorted(keys - required - optional)
        raise BackendConvergenceMatrixError(
            f"{field} fields drifted; missing={missing}, extra={extra}"
        )


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise BackendConvergenceMatrixError(f"{field} must be exact non-empty text")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise BackendConvergenceMatrixError(f"{field} must be a positive integer")
    return value


def _bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise BackendConvergenceMatrixError(f"{field} must be an exact boolean")
    return value


def _money(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise BackendConvergenceMatrixError(f"{field} must be finite positive money")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise BackendConvergenceMatrixError(
            f"{field} must be finite positive money"
        ) from exc
    if not result.is_finite() or result <= 0:
        raise BackendConvergenceMatrixError(f"{field} must be finite positive money")
    return result


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


@dataclasses.dataclass(frozen=True)
class ConvergenceArm:
    key: str
    arm_id: str
    backend: str
    model: str
    effort: str
    routing_path: Path
    routing_digest: str
    routing_fingerprint: str

    def identity(self) -> dict[str, str]:
        return {
            "key": self.key,
            "id": self.arm_id,
            "backend": self.backend,
            "model": self.model,
            "effort": self.effort,
            "routing_path": str(self.routing_path),
            "routing_digest": self.routing_digest,
            "routing_fingerprint": self.routing_fingerprint,
        }


@dataclasses.dataclass(frozen=True)
class PhaseDefinition:
    name: str
    purpose: str
    prerequisite_phase: str | None
    task_ids: tuple[str, ...]
    per_stratum: Mapping[str, int]
    replicates: int
    stability_replicates: int | None
    stability_per_stratum: int
    adaptive_feedback: Literal["allowed", "forbidden"]
    promotion_evidence: bool
    evaluate_inline: bool


@dataclasses.dataclass(frozen=True)
class ExecutionPolicy:
    max_parallel_cells: int
    max_parallel_per_backend: int
    forbidden_environment_variables: tuple[str, ...]
    dependency_condition: str
    ready_queue_policy: str
    pair_order_algorithm: str
    max_turns: int
    planner_count: int
    budget_usd_per_instance: Decimal
    eval_timeout_seconds: int
    include_breach: bool
    include_review: bool
    build_verify: bool


@dataclasses.dataclass(frozen=True)
class AnalysisPolicy:
    implementation_schema: str
    implementation_path: Path
    implementation_digest: str
    comparison_unit: str
    provider_sampling_seed: str
    primary_endpoint: str
    primary_success_predicate: str
    primary_delta_direction: str
    primary_publication_rule: str
    primary_replicate: int
    primary_task_weighting: str
    primary_stratum_aggregation: str
    primary_cell_denominator: str
    secondary_attempt_denominator: str
    result_set_policy: str
    duplicate_result_treatment: str
    unplanned_result_treatment: str
    unknown_terminal_status_treatment: str
    failure_funnel_policy: str
    stability_estimand: str
    interval_method: str
    interval_quantile_method: str
    bootstrap_task_order: str
    bootstrap_index_derivation: str
    bootstrap_stratum_sample_size: str
    bootstrap_stratum_weighting: str
    analysis_seed: str
    bootstrap_resamples: int
    confidence_level: str
    terminal_cell_requirement: str
    infrastructure_failure_treatment: str
    cost_or_trace_incomplete_treatment: str
    missing_result_treatment: str
    identity_drift_treatment: str
    safety_violation_treatment: str
    phase_completion_requirement: str
    complete_cost_evidence_required: bool
    complete_trace_evidence_required: bool
    safety_gate: str
    latency_interpretation: str
    metrics: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "implementation": {
                "schema": self.implementation_schema,
                "path": str(self.implementation_path),
                "sha256": self.implementation_digest,
            },
            "comparison_unit": self.comparison_unit,
            "provider_sampling_seed": self.provider_sampling_seed,
            "primary_endpoint": self.primary_endpoint,
            "primary_success_predicate": self.primary_success_predicate,
            "primary_delta_direction": self.primary_delta_direction,
            "primary_publication_rule": self.primary_publication_rule,
            "primary_replicate": self.primary_replicate,
            "primary_task_weighting": self.primary_task_weighting,
            "primary_stratum_aggregation": self.primary_stratum_aggregation,
            "primary_cell_denominator": self.primary_cell_denominator,
            "secondary_attempt_denominator": self.secondary_attempt_denominator,
            "result_set_policy": self.result_set_policy,
            "duplicate_result_treatment": self.duplicate_result_treatment,
            "unplanned_result_treatment": self.unplanned_result_treatment,
            "unknown_terminal_status_treatment": (
                self.unknown_terminal_status_treatment
            ),
            "failure_funnel_policy": self.failure_funnel_policy,
            "stability_estimand": self.stability_estimand,
            "interval_method": self.interval_method,
            "interval_quantile_method": self.interval_quantile_method,
            "bootstrap_task_order": self.bootstrap_task_order,
            "bootstrap_index_derivation": self.bootstrap_index_derivation,
            "bootstrap_stratum_sample_size": self.bootstrap_stratum_sample_size,
            "bootstrap_stratum_weighting": self.bootstrap_stratum_weighting,
            "analysis_seed": self.analysis_seed,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
            "terminal_cell_requirement": self.terminal_cell_requirement,
            "infrastructure_failure_treatment": (self.infrastructure_failure_treatment),
            "cost_or_trace_incomplete_treatment": (
                self.cost_or_trace_incomplete_treatment
            ),
            "missing_result_treatment": self.missing_result_treatment,
            "identity_drift_treatment": self.identity_drift_treatment,
            "safety_violation_treatment": self.safety_violation_treatment,
            "phase_completion_requirement": self.phase_completion_requirement,
            "complete_cost_evidence_required": (self.complete_cost_evidence_required),
            "complete_trace_evidence_required": (self.complete_trace_evidence_required),
            "safety_gate": self.safety_gate,
            "latency_interpretation": self.latency_interpretation,
            "metrics": list(self.metrics),
        }


@dataclasses.dataclass(frozen=True)
class DatasetIdentity:
    dataset_id: str
    split: str
    revision: str
    source_file: str
    source_digest: str
    export_algorithm: str
    row_count: int
    digest: str

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.dataset_id,
            "split": self.split,
            "revision": self.revision,
            "source_file": self.source_file,
            "source_sha256": self.source_digest,
            "export_algorithm": self.export_algorithm,
            "row_count": self.row_count,
            "sha256": self.digest,
        }


@dataclasses.dataclass(frozen=True)
class BackendConvergenceMatrix:
    source_path: Path
    source_digest: str
    dataset: DatasetIdentity
    selection_seed: str
    strata: Mapping[str, str]
    phases: Mapping[str, PhaseDefinition]
    arms: Mapping[str, ConvergenceArm]
    execution: ExecutionPolicy
    analysis: AnalysisPolicy

    @property
    def schema_version(self) -> str:
        observed = (
            self.analysis.implementation_schema,
            self.analysis.implementation_path.name,
            self.analysis.implementation_digest,
        )
        for schema, implementation in _EXPECTED_ANALYSIS_IMPLEMENTATIONS.items():
            if observed == implementation:
                return schema
        raise BackendConvergenceMatrixError(
            "matrix carries an unsupported analysis implementation schema"
        )

    @property
    def plan_schema(self) -> str:
        return _PLAN_SCHEMA_BY_MATRIX[self.schema_version]


@dataclasses.dataclass(frozen=True)
class SelectedTask:
    instance_id: str
    repo: str
    base_commit: str
    stratum: str
    replicates: int

    def identity(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ConvergenceCell:
    cell_id: str
    phase: str
    task_id: str
    stratum: str
    replicate: int
    assignment: str
    order_index: int
    arm: str
    backend: str
    model: str
    depends_on: str | None
    root: Path
    argv: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "phase": self.phase,
            "task_id": self.task_id,
            "stratum": self.stratum,
            "replicate": self.replicate,
            "assignment": self.assignment,
            "order_index": self.order_index,
            "arm": self.arm,
            "backend": self.backend,
            "model": self.model,
            "depends_on": self.depends_on,
            "root": str(self.root),
            "argv": list(self.argv),
        }


@dataclasses.dataclass(frozen=True)
class BackendConvergencePlan:
    matrix: BackendConvergenceMatrix
    phase: PhaseDefinition
    mode: Literal["dry-run", "implement"]
    dataset_path: Path
    output_root: Path
    repo_cache_dir: Path
    selected_tasks: tuple[SelectedTask, ...]
    stability_task_ids: tuple[str, ...]
    cells: tuple[ConvergenceCell, ...]

    @property
    def nominal_configured_budget_usd(self) -> Decimal:
        if self.mode == "dry-run":
            return Decimal("0")
        return self.matrix.execution.budget_usd_per_instance * len(self.cells)

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.matrix.plan_schema,
            "execution_authority": "absent-provider-free-plan-only",
            "matrix_path": str(self.matrix.source_path),
            "matrix_digest": self.matrix.source_digest,
            "dataset": {
                **self.matrix.dataset.identity(),
                "path": str(self.dataset_path),
            },
            "phase": self.phase.name,
            "purpose": self.phase.purpose,
            "prerequisite_phase": self.phase.prerequisite_phase,
            "mode": self.mode,
            "adaptive_feedback": self.phase.adaptive_feedback,
            "promotion_evidence": self.phase.promotion_evidence,
            "analysis": self.matrix.analysis.identity(),
            "arms": {key: self.matrix.arms[key].identity() for key in _ARMS},
            "parallelism": {
                "max_parallel_cells": self.matrix.execution.max_parallel_cells,
                "max_parallel_per_backend": (
                    self.matrix.execution.max_parallel_per_backend
                ),
                "dependency_condition": (self.matrix.execution.dependency_condition),
                "ready_queue_policy": self.matrix.execution.ready_queue_policy,
                "pair_order_algorithm": (self.matrix.execution.pair_order_algorithm),
            },
            "dispatch_environment": {
                "must_be_absent": list(
                    self.matrix.execution.forbidden_environment_variables
                ),
            },
            "budget_usd_per_instance": str(
                self.matrix.execution.budget_usd_per_instance
            ),
            "planner_count": self.matrix.execution.planner_count,
            "planning_budget_share": "0.40",
            "nominal_configured_budget_usd": str(self.nominal_configured_budget_usd),
            "output_root": str(self.output_root),
            "repo_cache_dir": str(self.repo_cache_dir),
            "selected_tasks": [task.identity() for task in self.selected_tasks],
            "stability_task_ids": list(self.stability_task_ids),
            "cells": [cell.identity() for cell in self.cells],
        }
        output_namespace = _OUTPUT_NAMESPACE_BY_MATRIX.get(self.matrix.schema_version)
        if output_namespace is not None:
            payload["output_namespace"] = output_namespace
        if include_digest:
            payload["plan_digest"] = _digest_bytes(_canonical_bytes(payload))
        return payload

    def to_json_bytes(self) -> bytes:
        return _canonical_bytes(self.payload())


def validate_swebench_backend_catalog(
    *,
    route_root: Path | str,
    lanes: tuple[BackendLaneDefinition, ...],
    require_full_registry: bool = False,
    promotion_authority_validator: PromotionAuthorityValidator | None = None,
) -> BackendLaneCatalog:
    """Validate lanes against SWE-bench's complete ordered role set."""

    return validate_backend_lane_catalog(
        SWEBENCH_BACKEND_CATALOG_CONTRACT,
        route_root=route_root,
        lanes=lanes,
        require_full_registry=require_full_registry,
        promotion_authority_validator=promotion_authority_validator,
    )


def load_swebench_backend_profile_catalog(
    *,
    catalog_path: Path | str | None = None,
    expected_catalog_digest: str | None = None,
    require_full_registry: bool = True,
) -> BackendLaneCatalog:
    """Bind every checked-in SWE-bench profile as a diagnostic N-lane arm."""

    source = (
        Path(__file__).resolve().with_name("backend_profile_catalog.yaml")
        if catalog_path is None
        else catalog_path
    )
    return load_backend_profile_catalog(
        SWEBENCH_BACKEND_CATALOG_CONTRACT,
        catalog_path=source,
        expected_catalog_digest=expected_catalog_digest,
        require_full_registry=require_full_registry,
    )


def _confined_route(matrix_path: Path, value: Any, *, field: str) -> Path:
    relative = Path(_text(value, field=field))
    if relative.is_absolute() or ".." in relative.parts:
        raise BackendConvergenceMatrixError(f"{field} must be a confined path")
    route = (matrix_path.parent / relative).resolve()
    if matrix_path.parent not in route.parents:
        raise BackendConvergenceMatrixError(f"{field} escapes the matrix directory")
    return route


def _load_arm(
    key: str,
    value: Any,
    *,
    matrix_path: Path,
    matrix_schema: str,
    roles: tuple[str, ...],
) -> ConvergenceArm:
    raw = _mapping(value, field=f"arms.{key}")
    _fields(
        raw,
        required={
            "id",
            "backend",
            "model",
            "effort",
            "routing",
            "routing_sha256",
            "routing_fingerprint",
        },
        field=f"arms.{key}",
    )
    expected = _EXPECTED_LANES[matrix_schema][key]
    observed = tuple(
        _text(raw[name], field=f"arms.{key}.{name}")
        for name in ("id", "backend", "model", "effort")
    )
    if observed != expected:
        raise BackendConvergenceMatrixError(
            f"arm {key} must be exact convergence lane {expected!r}"
        )
    route_path = _confined_route(
        matrix_path,
        raw["routing"],
        field=f"arms.{key}.routing",
    )
    try:
        route_bytes = route_path.read_bytes()
        routing = lint_routing_config(route_bytes)
    except (OSError, ValueError) as exc:
        raise BackendConvergenceMatrixError(
            f"arm {key} routing file is invalid: {exc}"
        ) from exc
    declared_digest = _text(raw["routing_sha256"], field=f"arms.{key}.routing_sha256")
    if not _DIGEST_RE.fullmatch(declared_digest):
        raise BackendConvergenceMatrixError(
            f"arms.{key}.routing_sha256 must be tagged SHA-256"
        )
    observed_digest = _digest_bytes(route_bytes)
    if observed_digest != declared_digest:
        raise BackendConvergenceMatrixError(
            f"arm {key} routing bytes do not match the matrix"
        )
    declared_fingerprint = _text(
        raw["routing_fingerprint"],
        field=f"arms.{key}.routing_fingerprint",
    )
    if not _DIGEST_RE.fullmatch(declared_fingerprint):
        raise BackendConvergenceMatrixError(
            f"arms.{key}.routing_fingerprint must be tagged SHA-256"
        )
    observed_fingerprint = routing_fingerprint(routing)
    if observed_fingerprint != declared_fingerprint:
        raise BackendConvergenceMatrixError(
            f"arm {key} effective routing fingerprint does not match the matrix"
        )
    expected_extra = dict(_EXPECTED_SESSION_EXTRAS[key])
    if tuple(roles) != _EXPECTED_ROLES or set(routing.roles) != set(_EXPECTED_ROLES):
        raise BackendConvergenceMatrixError(
            f"arm {key} must explicitly pin the complete SWE-bench role set"
        )
    effective_routes = (("default", routing.default),) + tuple(
        (role, routing.for_role(role)) for role in _EXPECTED_ROLES
    )
    for role, config in effective_routes:
        if (config.backend, config.model, config.effort) != observed[1:]:
            raise BackendConvergenceMatrixError(
                f"arm {key} role {role!r} drifts from its exact lane"
            )
        if config.feature_admission != "enforce":
            raise BackendConvergenceMatrixError(
                f"arm {key} role {role!r} must enforce feature admission"
            )
        if config.base_url is not None or config.api_key is not None:
            raise BackendConvergenceMatrixError(
                f"arm {key} role {role!r} must use its first-party session"
            )
        if config.extra != expected_extra:
            raise BackendConvergenceMatrixError(
                f"arm {key} role {role!r} session authority drifted"
            )
        if any("fallback" in name.casefold() for name in config.extra):
            raise BackendConvergenceMatrixError(
                f"arm {key} role {role!r} configures fallback"
            )
    return ConvergenceArm(
        key=key,
        arm_id=observed[0],
        backend=observed[1],
        model=observed[2],
        effort=observed[3],
        routing_path=route_path,
        routing_digest=observed_digest,
        routing_fingerprint=observed_fingerprint,
    )


def _load_phase(name: str, value: Any, *, strata: Mapping[str, str]) -> PhaseDefinition:
    raw = _mapping(value, field=f"phases.{name}")
    _fields(
        raw,
        required={
            "purpose",
            "prerequisite_phase",
            "task_ids",
            "per_stratum",
            "replicates",
            "stability_replicates",
            "stability_per_stratum",
            "adaptive_feedback",
            "promotion_evidence",
            "evaluate_inline",
        },
        field=f"phases.{name}",
    )
    task_ids_raw = raw["task_ids"]
    if type(task_ids_raw) is not list or any(
        type(item) is not str or not _INSTANCE_RE.fullmatch(item)
        for item in task_ids_raw
    ):
        raise BackendConvergenceMatrixError(
            f"phases.{name}.task_ids must be exact instance identifiers"
        )
    task_ids = tuple(task_ids_raw)
    if len(task_ids) != len(set(task_ids)):
        raise BackendConvergenceMatrixError(
            f"phases.{name}.task_ids contains duplicates"
        )
    per_stratum_raw = _mapping(raw["per_stratum"], field=f"phases.{name}.per_stratum")
    if set(per_stratum_raw) - set(strata):
        raise BackendConvergenceMatrixError(
            f"phases.{name}.per_stratum contains an unknown stratum"
        )
    per_stratum: dict[str, int] = {}
    for stratum, count in per_stratum_raw.items():
        per_stratum[stratum] = _positive_int(
            count, field=f"phases.{name}.per_stratum.{stratum}"
        )
    if bool(task_ids) == bool(per_stratum):
        raise BackendConvergenceMatrixError(
            f"phases.{name} requires exactly one selection form"
        )
    stability_raw = raw["stability_replicates"]
    stability_replicates = (
        None
        if stability_raw is None
        else _positive_int(stability_raw, field=f"phases.{name}.stability_replicates")
    )
    stability_per_stratum = raw["stability_per_stratum"]
    if type(stability_per_stratum) is not int or stability_per_stratum < 0:
        raise BackendConvergenceMatrixError(
            f"phases.{name}.stability_per_stratum must be non-negative"
        )
    if (stability_replicates is None) != (stability_per_stratum == 0):
        raise BackendConvergenceMatrixError(
            f"phases.{name} stability fields must be enabled together"
        )
    adaptive_feedback = raw["adaptive_feedback"]
    if adaptive_feedback not in {"allowed", "forbidden"}:
        raise BackendConvergenceMatrixError(
            f"phases.{name}.adaptive_feedback is unsupported"
        )
    prerequisite_raw = raw["prerequisite_phase"]
    if prerequisite_raw is not None and prerequisite_raw not in _PHASES:
        raise BackendConvergenceMatrixError(
            f"phases.{name}.prerequisite_phase is unsupported"
        )
    return PhaseDefinition(
        name=name,
        purpose=_text(raw["purpose"], field=f"phases.{name}.purpose"),
        prerequisite_phase=prerequisite_raw,
        task_ids=task_ids,
        per_stratum=per_stratum,
        replicates=_positive_int(raw["replicates"], field=f"phases.{name}.replicates"),
        stability_replicates=stability_replicates,
        stability_per_stratum=stability_per_stratum,
        adaptive_feedback=adaptive_feedback,
        promotion_evidence=_bool(
            raw["promotion_evidence"],
            field=f"phases.{name}.promotion_evidence",
        ),
        evaluate_inline=_bool(
            raw["evaluate_inline"], field=f"phases.{name}.evaluate_inline"
        ),
    )


def _validate_phase_contract(
    phases: Mapping[str, PhaseDefinition],
    *,
    matrix_schema: str,
) -> None:
    generation = matrix_schema.rsplit(".", 1)[-1]
    calibration = phases["calibration"]
    if (
        calibration.task_ids != _CALIBRATION_IDS
        or calibration.prerequisite_phase is not None
        or calibration.per_stratum
        or calibration.replicates != 1
        or calibration.stability_replicates is not None
        or calibration.stability_per_stratum != 0
        or calibration.adaptive_feedback != "allowed"
        or calibration.promotion_evidence
        or not calibration.evaluate_inline
    ):
        message = (
            "v1 calibration must be the exact historical broad-10 diagnostic"
            if matrix_schema == MATRIX_SCHEMA
            else f"{generation} calibration must be the exact historical broad-10 "
            "diagnostic"
        )
        raise BackendConvergenceMatrixError(message)

    pilot = phases["pilot"]
    if (
        pilot.task_ids
        or pilot.prerequisite_phase != "calibration"
        or dict(pilot.per_stratum) != {name: 2 for name in _EXPECTED_STRATA}
        or pilot.replicates != 3
        or pilot.stability_replicates is not None
        or pilot.stability_per_stratum != 0
        or pilot.adaptive_feedback != "forbidden"
        or pilot.promotion_evidence
        or not pilot.evaluate_inline
    ):
        message = (
            "v1 pilot must be a sealed 20-task, three-replicate diagnostic"
            if matrix_schema == MATRIX_SCHEMA
            else f"{generation} pilot must be a sealed 20-task, three-replicate "
            "diagnostic"
        )
        raise BackendConvergenceMatrixError(message)

    heldout = phases["heldout"]
    if (
        heldout.task_ids
        or heldout.prerequisite_phase != "pilot"
        or dict(heldout.per_stratum) != dict(_EXPECTED_HELDOUT_QUOTAS)
        or heldout.replicates != 1
        or heldout.stability_replicates != 3
        or heldout.stability_per_stratum != 2
        or heldout.adaptive_feedback != "forbidden"
        or heldout.promotion_evidence
        or not heldout.evaluate_inline
    ):
        message = (
            "v1 heldout must be the sealed 100-task diagnostic with "
            "20 preselected three-replicate stability tasks"
            if matrix_schema == MATRIX_SCHEMA
            else f"{generation} heldout must be the sealed 100-task diagnostic "
            "with 20 preselected three-replicate stability tasks"
        )
        raise BackendConvergenceMatrixError(message)


def load_backend_convergence_matrix(
    path: Path | str,
) -> BackendConvergenceMatrix:
    """Load and authenticate a supported checked-in convergence matrix."""

    source = Path(path).resolve()
    try:
        source_bytes = source.read_bytes()
        decoded = source_bytes.decode("utf-8")
        raw = yaml.safe_load(decoded)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BackendConvergenceMatrixError(f"invalid matrix {source}: {exc}") from exc
    root = _mapping(raw, field="matrix")
    _fields(
        root,
        required={
            "schema_version",
            "workflow",
            "dataset",
            "selection",
            "roles",
            "arms",
            "phases",
            "execution",
            "analysis",
        },
        field="matrix",
    )
    matrix_schema = root["schema_version"]
    if (
        type(matrix_schema) is not str
        or matrix_schema not in _PLAN_SCHEMA_BY_MATRIX
        or root["workflow"] != "swebench"
    ):
        raise BackendConvergenceMatrixError("unsupported matrix schema or workflow")
    roles_raw = root["roles"]
    if (
        type(roles_raw) is not list
        or not roles_raw
        or any(
            type(role) is not str or not _IDENTIFIER_RE.fullmatch(role)
            for role in roles_raw
        )
    ):
        raise BackendConvergenceMatrixError("roles must be unique identifiers")
    roles = tuple(roles_raw)
    if len(roles) != len(set(roles)):
        raise BackendConvergenceMatrixError("roles must not contain duplicates")
    if roles != _EXPECTED_ROLES:
        raise BackendConvergenceMatrixError(
            "matrix must name the complete ordered SWE-bench role set"
        )

    dataset = _mapping(root["dataset"], field="dataset")
    _fields(
        dataset,
        required={
            "id",
            "split",
            "revision",
            "source_file",
            "source_sha256",
            "export_algorithm",
            "row_count",
            "sha256",
        },
        field="dataset",
    )
    for field, expected in _EXPECTED_DATASET_PROVENANCE.items():
        if dataset[field] != expected:
            raise BackendConvergenceMatrixError(f"dataset.{field} must be {expected!r}")
    source_digest = _text(dataset["source_sha256"], field="dataset.source_sha256")
    if not _DIGEST_RE.fullmatch(source_digest):
        raise BackendConvergenceMatrixError(
            "dataset.source_sha256 must be tagged SHA-256"
        )
    dataset_digest = _text(dataset["sha256"], field="dataset.sha256")
    if not _DIGEST_RE.fullmatch(dataset_digest):
        raise BackendConvergenceMatrixError("dataset.sha256 must be tagged SHA-256")
    if dataset_digest != _EXPECTED_DATASET_DIGEST:
        message = (
            "dataset.sha256 must bind the exact v1 canonical export"
            if matrix_schema == MATRIX_SCHEMA
            else "dataset.sha256 must bind the exact canonical export"
        )
        raise BackendConvergenceMatrixError(message)
    dataset_identity = DatasetIdentity(
        dataset_id=_EXPECTED_DATASET_PROVENANCE["id"],
        split=_EXPECTED_DATASET_PROVENANCE["split"],
        revision=_EXPECTED_DATASET_PROVENANCE["revision"],
        source_file=_EXPECTED_DATASET_PROVENANCE["source_file"],
        source_digest=source_digest,
        export_algorithm=_EXPECTED_DATASET_PROVENANCE["export_algorithm"],
        row_count=_positive_int(dataset["row_count"], field="dataset.row_count"),
        digest=dataset_digest,
    )
    if dataset_identity.row_count != 500:
        message = (
            "v1 dataset must contain 500 rows"
            if matrix_schema == MATRIX_SCHEMA
            else "dataset must contain 500 rows"
        )
        raise BackendConvergenceMatrixError(message)

    selection = _mapping(root["selection"], field="selection")
    _fields(selection, required={"algorithm", "seed", "strata"}, field="selection")
    if selection["algorithm"] != _SELECTION_ALGORITHM:
        raise BackendConvergenceMatrixError("unsupported selection algorithm")
    selection_seed = _text(selection["seed"], field="selection.seed")
    expected_selection_seed = _EXPECTED_SUCCESSOR_SELECTION_SEEDS.get(matrix_schema)
    if (
        expected_selection_seed is not None
        and selection_seed != expected_selection_seed
    ):
        raise BackendConvergenceMatrixError(
            f"selection.seed must be {expected_selection_seed!r}"
        )
    strata_raw = _mapping(selection["strata"], field="selection.strata")
    if not strata_raw:
        raise BackendConvergenceMatrixError("selection.strata must not be empty")
    strata: dict[str, str] = {}
    for name, repo in strata_raw.items():
        if type(name) is not str or not _IDENTIFIER_RE.fullmatch(name):
            raise BackendConvergenceMatrixError("stratum names must be identifiers")
        repo_text = _text(repo, field=f"selection.strata.{name}")
        if repo_text.count("/") != 1:
            raise BackendConvergenceMatrixError(
                f"selection.strata.{name} must be owner/repository"
            )
        strata[name] = repo_text
    if len(set(strata.values())) != len(strata):
        raise BackendConvergenceMatrixError("each stratum must name a distinct repo")
    if strata != _EXPECTED_STRATA:
        message = (
            "matrix must use the exact v1 primary repository strata"
            if matrix_schema == MATRIX_SCHEMA
            else "matrix must use the exact primary repository strata"
        )
        raise BackendConvergenceMatrixError(message)

    arms_raw = _mapping(root["arms"], field="arms")
    if tuple(arms_raw) != _ARMS:
        raise BackendConvergenceMatrixError("matrix requires ordered arms A then B")
    arms = {
        key: _load_arm(
            key,
            arms_raw[key],
            matrix_path=source,
            matrix_schema=matrix_schema,
            roles=roles,
        )
        for key in _ARMS
    }
    try:
        catalog = validate_swebench_backend_catalog(
            route_root=source.parent,
            lanes=tuple(
                BackendLaneDefinition(
                    key=key,
                    lane_id=arms[key].arm_id,
                    backend=arms[key].backend,
                    runtime_backend=arms[key].backend,
                    transport=_EXPECTED_TRANSPORTS[key],
                    routing_path=arms[key].routing_path,
                    routing_digest=arms[key].routing_digest,
                    routing_fingerprint=arms[key].routing_fingerprint,
                    evidence_status="diagnostic",
                )
                for key in _ARMS
            ),
        )
    except BackendLaneCatalogError as exc:
        raise BackendConvergenceMatrixError(
            f"exact {_MATRIX_LABELS[matrix_schema]} adapter catalog is invalid: {exc}"
        ) from exc
    reference_pairs = generate_reference_arm_pairs(catalog, reference_arm="A")
    if (
        len(reference_pairs) != 1
        or reference_pairs[0].arm_a.key != "A"
        or reference_pairs[0].arm_b.key != "B"
        or reference_pairs[0].promotion_eligible
    ):
        raise BackendConvergenceMatrixError(
            f"exact {_MATRIX_LABELS[matrix_schema]} reference pair must remain "
            "diagnostic A versus B"
        )

    phases_raw = _mapping(root["phases"], field="phases")
    if tuple(phases_raw) != _PHASES:
        raise BackendConvergenceMatrixError(
            "matrix phases must be calibration, pilot, heldout in order"
        )
    phases = {
        name: _load_phase(name, phases_raw[name], strata=strata) for name in _PHASES
    }
    _validate_phase_contract(phases, matrix_schema=matrix_schema)

    execution_raw = _mapping(root["execution"], field="execution")
    execution_fields = {
        "max_parallel_cells",
        "max_parallel_per_backend",
        "forbidden_environment_variables",
        "dependency_condition",
        "ready_queue_policy",
        "pair_order_algorithm",
        "max_turns",
        "planner_count",
        "budget_usd_per_instance",
        "eval_timeout_seconds",
        "include_breach",
        "include_review",
        "build_verify",
    }
    output_namespace = _OUTPUT_NAMESPACE_BY_MATRIX.get(matrix_schema)
    if output_namespace is not None:
        execution_fields.add("output_namespace")
    _fields(
        execution_raw,
        required=execution_fields,
        field="execution",
    )
    if (
        output_namespace is not None
        and _text(execution_raw["output_namespace"], field="execution.output_namespace")
        != output_namespace
    ):
        raise BackendConvergenceMatrixError(
            f"execution.output_namespace must remain {output_namespace!r}"
        )
    forbidden_environment_raw = execution_raw["forbidden_environment_variables"]
    if forbidden_environment_raw != ["CRUCIBLE_BACKEND_CONFIG"]:
        raise BackendConvergenceMatrixError(
            "execution.forbidden_environment_variables must contain only "
            "CRUCIBLE_BACKEND_CONFIG"
        )
    execution = ExecutionPolicy(
        max_parallel_cells=_positive_int(
            execution_raw["max_parallel_cells"],
            field="execution.max_parallel_cells",
        ),
        max_parallel_per_backend=_positive_int(
            execution_raw["max_parallel_per_backend"],
            field="execution.max_parallel_per_backend",
        ),
        forbidden_environment_variables=("CRUCIBLE_BACKEND_CONFIG",),
        dependency_condition=_text(
            execution_raw["dependency_condition"],
            field="execution.dependency_condition",
        ),
        ready_queue_policy=_text(
            execution_raw["ready_queue_policy"],
            field="execution.ready_queue_policy",
        ),
        pair_order_algorithm=_text(
            execution_raw["pair_order_algorithm"],
            field="execution.pair_order_algorithm",
        ),
        max_turns=_positive_int(
            execution_raw["max_turns"], field="execution.max_turns"
        ),
        planner_count=_positive_int(
            execution_raw["planner_count"], field="execution.planner_count"
        ),
        budget_usd_per_instance=_money(
            execution_raw["budget_usd_per_instance"],
            field="execution.budget_usd_per_instance",
        ),
        eval_timeout_seconds=_positive_int(
            execution_raw["eval_timeout_seconds"],
            field="execution.eval_timeout_seconds",
        ),
        include_breach=_bool(
            execution_raw["include_breach"], field="execution.include_breach"
        ),
        include_review=_bool(
            execution_raw["include_review"], field="execution.include_review"
        ),
        build_verify=_bool(
            execution_raw["build_verify"], field="execution.build_verify"
        ),
    )
    expected_execution = ExecutionPolicy(
        max_parallel_cells=2,
        max_parallel_per_backend=1,
        forbidden_environment_variables=("CRUCIBLE_BACKEND_CONFIG",),
        dependency_condition="any-terminal",
        ready_queue_policy="paired-successor-first",
        pair_order_algorithm="within-stratum-alternating-balanced-v1",
        max_turns=90,
        planner_count=3,
        budget_usd_per_instance=Decimal("12"),
        eval_timeout_seconds=7200,
        include_breach=True,
        include_review=True,
        build_verify=True,
    )
    if execution != expected_execution:
        message = (
            "v1 convergence execution policy must remain exact"
            if matrix_schema == MATRIX_SCHEMA
            else "v2 convergence execution policy must remain exact"
        )
        raise BackendConvergenceMatrixError(message)

    analysis_raw = _mapping(root["analysis"], field="analysis")
    _fields(
        analysis_raw,
        required={
            "implementation",
            "comparison_unit",
            "provider_sampling_seed",
            "primary_endpoint",
            "primary_success_predicate",
            "primary_delta_direction",
            "primary_publication_rule",
            "primary_replicate",
            "primary_task_weighting",
            "primary_stratum_aggregation",
            "primary_cell_denominator",
            "secondary_attempt_denominator",
            "result_set_policy",
            "duplicate_result_treatment",
            "unplanned_result_treatment",
            "unknown_terminal_status_treatment",
            "failure_funnel_policy",
            "stability_estimand",
            "interval_method",
            "interval_quantile_method",
            "bootstrap_task_order",
            "bootstrap_index_derivation",
            "bootstrap_stratum_sample_size",
            "bootstrap_stratum_weighting",
            "analysis_seed",
            "bootstrap_resamples",
            "confidence_level",
            "terminal_cell_requirement",
            "infrastructure_failure_treatment",
            "cost_or_trace_incomplete_treatment",
            "missing_result_treatment",
            "identity_drift_treatment",
            "safety_violation_treatment",
            "phase_completion_requirement",
            "complete_cost_evidence_required",
            "complete_trace_evidence_required",
            "safety_gate",
            "latency_interpretation",
            "metrics",
        },
        field="analysis",
    )
    implementation_raw = _mapping(
        analysis_raw["implementation"], field="analysis.implementation"
    )
    _fields(
        implementation_raw,
        required={"schema", "path", "sha256"},
        field="analysis.implementation",
    )
    (
        expected_implementation_schema,
        expected_implementation_name,
        expected_implementation_digest,
    ) = _EXPECTED_ANALYSIS_IMPLEMENTATIONS[matrix_schema]
    if (
        implementation_raw["schema"] != expected_implementation_schema
        or implementation_raw["path"] != expected_implementation_name
    ):
        raise BackendConvergenceMatrixError(
            "analysis implementation schema and path must remain exact"
        )
    implementation_path = _confined_route(
        source,
        implementation_raw["path"],
        field="analysis.implementation.path",
    )
    implementation_digest = _text(
        implementation_raw["sha256"],
        field="analysis.implementation.sha256",
    )
    if (
        not _DIGEST_RE.fullmatch(implementation_digest)
        or implementation_digest != expected_implementation_digest
    ):
        message = (
            "analysis implementation digest must bind the exact v1 analyzer"
            if matrix_schema == MATRIX_SCHEMA
            else "analysis implementation digest must bind the exact v2 analyzer"
        )
        raise BackendConvergenceMatrixError(message)
    try:
        observed_implementation_digest = _digest_bytes(implementation_path.read_bytes())
    except OSError as exc:
        raise BackendConvergenceMatrixError(
            f"cannot read analysis implementation {implementation_path}: {exc}"
        ) from exc
    if observed_implementation_digest != implementation_digest:
        raise BackendConvergenceMatrixError(
            "analysis implementation bytes do not match the matrix"
        )
    expected_analysis = {
        "comparison_unit": "paired-task-at-primary-replicate",
        "provider_sampling_seed": "unavailable",
        "primary_endpoint": (
            "paired-intention-to-treat-resolved-pass-at-1-arm-a-minus-arm-b"
        ),
        "primary_success_predicate": "eval-status-exactly-resolved",
        "primary_delta_direction": "arm-a-minus-arm-b",
        "primary_publication_rule": "complete-valid-phase-only",
        "primary_task_weighting": "equal-per-selected-task",
        "primary_stratum_aggregation": "selected-task-count-weighted",
        "primary_cell_denominator": "all-replicate-1-cells",
        "secondary_attempt_denominator": "all-scheduled-cells",
        "result_set_policy": "exact-scheduled-cell-id-bijection",
        "duplicate_result_treatment": "phase-invalid-primary-suppressed",
        "unplanned_result_treatment": "phase-invalid-primary-suppressed",
        "unknown_terminal_status_treatment": "phase-invalid-primary-suppressed",
        "failure_funnel_policy": "exact-one-category-per-terminal-eval-status-v1",
        "stability_estimand": (
            "equal-task-mean-within-task-all-replicate-pair-binary-resolution-agreement"
        ),
        "interval_method": (
            "sha256-task-within-stratum-paired-percentile-bootstrap-v1"
        ),
        "interval_quantile_method": "nearest-rank-one-based",
        "bootstrap_task_order": "selected-task-order-filtered-by-stratum",
        "bootstrap_index_derivation": (
            "sha256-utf8-seed-nul-zero-based-sample-nul-stratum-nul-zero-based-"
            "draw-big-endian-modulo-stratum-size-v1"
        ),
        "bootstrap_stratum_sample_size": "original-selected-task-count",
        "bootstrap_stratum_weighting": "original-selected-task-count",
        "analysis_seed": _EXPECTED_ANALYSIS_SEEDS[matrix_schema],
        "confidence_level": "0.95",
        "terminal_cell_requirement": "exactly-one-per-scheduled-cell",
        "infrastructure_failure_treatment": (
            "accepted-terminal-status-scores-zero-unless-resolved"
        ),
        "cost_or_trace_incomplete_treatment": ("phase-incomplete-primary-suppressed"),
        "missing_result_treatment": "phase-incomplete-primary-suppressed",
        "identity_drift_treatment": "phase-invalid-primary-suppressed",
        "safety_violation_treatment": "phase-invalid-primary-suppressed",
        "phase_completion_requirement": (
            "all-cells-terminal-identity-valid-cost-trace-complete"
        ),
        "safety_gate": "zero-authority-or-unsettled-writer-violations",
        "latency_interpretation": "diagnostic-under-concurrency",
    }
    for field, expected in expected_analysis.items():
        if analysis_raw[field] != expected:
            raise BackendConvergenceMatrixError(
                f"analysis.{field} must be {expected!r}"
            )
    metrics_raw = analysis_raw["metrics"]
    if type(metrics_raw) is not list or tuple(metrics_raw) != _EXPECTED_METRICS:
        message = (
            "analysis.metrics must be the closed v1 convergence metric set"
            if matrix_schema == MATRIX_SCHEMA
            else "analysis.metrics must be the closed v2 convergence metric set"
        )
        raise BackendConvergenceMatrixError(message)
    analysis = AnalysisPolicy(
        implementation_schema=expected_implementation_schema,
        implementation_path=implementation_path,
        implementation_digest=implementation_digest,
        comparison_unit=expected_analysis["comparison_unit"],
        provider_sampling_seed=expected_analysis["provider_sampling_seed"],
        primary_endpoint=expected_analysis["primary_endpoint"],
        primary_success_predicate=expected_analysis["primary_success_predicate"],
        primary_delta_direction=expected_analysis["primary_delta_direction"],
        primary_publication_rule=expected_analysis["primary_publication_rule"],
        primary_replicate=_positive_int(
            analysis_raw["primary_replicate"], field="analysis.primary_replicate"
        ),
        primary_task_weighting=expected_analysis["primary_task_weighting"],
        primary_stratum_aggregation=expected_analysis["primary_stratum_aggregation"],
        primary_cell_denominator=expected_analysis["primary_cell_denominator"],
        secondary_attempt_denominator=expected_analysis[
            "secondary_attempt_denominator"
        ],
        result_set_policy=expected_analysis["result_set_policy"],
        duplicate_result_treatment=expected_analysis["duplicate_result_treatment"],
        unplanned_result_treatment=expected_analysis["unplanned_result_treatment"],
        unknown_terminal_status_treatment=expected_analysis[
            "unknown_terminal_status_treatment"
        ],
        failure_funnel_policy=expected_analysis["failure_funnel_policy"],
        stability_estimand=expected_analysis["stability_estimand"],
        interval_method=expected_analysis["interval_method"],
        interval_quantile_method=expected_analysis["interval_quantile_method"],
        bootstrap_task_order=expected_analysis["bootstrap_task_order"],
        bootstrap_index_derivation=expected_analysis["bootstrap_index_derivation"],
        bootstrap_stratum_sample_size=expected_analysis[
            "bootstrap_stratum_sample_size"
        ],
        bootstrap_stratum_weighting=expected_analysis["bootstrap_stratum_weighting"],
        analysis_seed=expected_analysis["analysis_seed"],
        bootstrap_resamples=_positive_int(
            analysis_raw["bootstrap_resamples"],
            field="analysis.bootstrap_resamples",
        ),
        confidence_level=expected_analysis["confidence_level"],
        terminal_cell_requirement=expected_analysis["terminal_cell_requirement"],
        infrastructure_failure_treatment=expected_analysis[
            "infrastructure_failure_treatment"
        ],
        cost_or_trace_incomplete_treatment=expected_analysis[
            "cost_or_trace_incomplete_treatment"
        ],
        missing_result_treatment=expected_analysis["missing_result_treatment"],
        identity_drift_treatment=expected_analysis["identity_drift_treatment"],
        safety_violation_treatment=expected_analysis["safety_violation_treatment"],
        phase_completion_requirement=expected_analysis["phase_completion_requirement"],
        complete_cost_evidence_required=_bool(
            analysis_raw["complete_cost_evidence_required"],
            field="analysis.complete_cost_evidence_required",
        ),
        complete_trace_evidence_required=_bool(
            analysis_raw["complete_trace_evidence_required"],
            field="analysis.complete_trace_evidence_required",
        ),
        safety_gate=expected_analysis["safety_gate"],
        latency_interpretation=expected_analysis["latency_interpretation"],
        metrics=_EXPECTED_METRICS,
    )
    if analysis.primary_replicate != 1 or analysis.bootstrap_resamples != 10_000:
        message = (
            "v1 analysis requires primary replicate 1 and 10,000 resamples"
            if matrix_schema == MATRIX_SCHEMA
            else "v2 analysis requires primary replicate 1 and 10,000 resamples"
        )
        raise BackendConvergenceMatrixError(message)
    if not (
        analysis.complete_cost_evidence_required
        and analysis.complete_trace_evidence_required
    ):
        message = (
            "v1 analysis requires complete cost and trace evidence"
            if matrix_schema == MATRIX_SCHEMA
            else "v2 analysis requires complete cost and trace evidence"
        )
        raise BackendConvergenceMatrixError(message)
    return BackendConvergenceMatrix(
        source_path=source,
        source_digest=_digest_bytes(source_bytes),
        dataset=dataset_identity,
        selection_seed=selection_seed,
        strata=strata,
        phases=phases,
        arms=arms,
        execution=execution,
        analysis=analysis,
    )


def _ranked(
    instances: list[SwebenchInstance],
    *,
    seed: str,
    phase: str,
) -> list[SwebenchInstance]:
    return sorted(
        instances,
        key=lambda item: (
            hashlib.sha256(
                f"{seed}\0{phase}\0{item.instance_id}".encode("utf-8")
            ).digest(),
            item.instance_id,
        ),
    )


def _block_interleaved(
    instances: list[SwebenchInstance],
    *,
    stratum_for_repo: Mapping[str, str],
    seed: str,
    phase: str,
) -> list[SwebenchInstance]:
    queues = {
        stratum: [item for item in instances if item.repo == repo]
        for repo, stratum in stratum_for_repo.items()
    }
    result: list[SwebenchInstance] = []
    wave = 0
    while any(queues.values()):
        active = [stratum for stratum, queue in queues.items() if queue]
        active.sort(
            key=lambda stratum: (
                hashlib.sha256(
                    f"{seed}\0{phase}\0{wave}\0{stratum}".encode("utf-8")
                ).digest(),
                stratum,
            )
        )
        for stratum in active:
            result.append(queues[stratum].pop(0))
        wave += 1
    return result


def _counterbalance_offsets(
    tasks: tuple[SelectedTask, ...],
    *,
    seed: str,
    phase: str,
) -> Mapping[str, int]:
    pair_totals: dict[str, int] = {}
    for task in tasks:
        pair_totals[task.stratum] = pair_totals.get(task.stratum, 0) + task.replicates
    odd_strata = [stratum for stratum, count in pair_totals.items() if count % 2]
    odd_strata.sort(
        key=lambda stratum: (
            hashlib.sha256(
                f"{seed}\0{phase}\0counterbalance\0{stratum}".encode("utf-8")
            ).digest(),
            stratum,
        )
    )
    if len(odd_strata) % 2:
        raise BackendConvergenceMatrixError(
            f"phase {phase} cannot balance an odd number of task pairs"
        )
    odd_offsets = {
        stratum: int(index >= len(odd_strata) // 2)
        for index, stratum in enumerate(odd_strata)
    }
    return {
        stratum: odd_offsets.get(
            stratum,
            hashlib.sha256(
                f"{seed}\0{phase}\0counterbalance-even\0{stratum}".encode("utf-8")
            ).digest()[0]
            % 2,
        )
        for stratum in pair_totals
    }


def select_matrix_tasks(
    matrix: BackendConvergenceMatrix,
    dataset_path: Path | str,
) -> Mapping[str, tuple[tuple[SelectedTask, ...], tuple[str, ...]]]:
    """Authenticate the dataset and select disjoint phase cohorts."""

    if type(matrix) is not BackendConvergenceMatrix:
        raise BackendConvergenceMatrixError("matrix must be loader-produced")
    canonical = load_backend_convergence_matrix(matrix.source_path)
    if matrix != canonical:
        raise BackendConvergenceMatrixError("caller matrix differs from source bytes")
    matrix = canonical
    path = Path(dataset_path).resolve()
    try:
        dataset_bytes = path.read_bytes()
    except OSError as exc:
        raise BackendConvergenceMatrixError(
            f"cannot read dataset {path}: {exc}"
        ) from exc
    if _digest_bytes(dataset_bytes) != matrix.dataset.digest:
        raise BackendConvergenceMatrixError("dataset bytes do not match the matrix")
    try:
        instances = load_dataset_bytes(dataset_bytes, source=path)
    except (OSError, ValueError) as exc:
        raise BackendConvergenceMatrixError(f"dataset is invalid: {exc}") from exc
    by_id = {item.instance_id: item for item in instances}
    if len(by_id) != len(instances):
        raise BackendConvergenceMatrixError("dataset instance ids are not unique")
    if len(instances) != matrix.dataset.row_count:
        raise BackendConvergenceMatrixError(
            f"dataset contains {len(instances)} rows; expected "
            f"{matrix.dataset.row_count}"
        )
    stratum_for_repo = {repo: name for name, repo in matrix.strata.items()}
    eligible = [item for item in instances if item.repo in stratum_for_repo]
    used: set[str] = set()
    result: dict[str, tuple[tuple[SelectedTask, ...], tuple[str, ...]]] = {}
    for phase_name in _PHASES:
        phase = matrix.phases[phase_name]
        selected_instances: list[SwebenchInstance] = []
        if phase.task_ids:
            missing = [task_id for task_id in phase.task_ids if task_id not in by_id]
            if missing:
                raise BackendConvergenceMatrixError(
                    f"phase {phase_name} tasks are absent from dataset: {missing}"
                )
            selected_instances = [by_id[task_id] for task_id in phase.task_ids]
            outside = [
                item.instance_id
                for item in selected_instances
                if item.repo not in stratum_for_repo
            ]
            if outside:
                raise BackendConvergenceMatrixError(
                    f"phase {phase_name} tasks are outside declared strata: {outside}"
                )
        else:
            for stratum, repo in matrix.strata.items():
                candidates = [
                    item
                    for item in eligible
                    if item.repo == repo and item.instance_id not in used
                ]
                ranked = _ranked(
                    candidates,
                    seed=matrix.selection_seed,
                    phase=phase_name,
                )
                count = phase.per_stratum.get(stratum, 0)
                if len(ranked) < count:
                    raise BackendConvergenceMatrixError(
                        f"phase {phase_name} stratum {stratum} has only "
                        f"{len(ranked)} unused tasks, requires {count}"
                    )
                selected_instances.extend(ranked[:count])
            selected_instances = _block_interleaved(
                selected_instances,
                stratum_for_repo=stratum_for_repo,
                seed=matrix.selection_seed,
                phase=f"{phase_name}-interleave",
            )
        overlap = sorted(
            item.instance_id for item in selected_instances if item.instance_id in used
        )
        if overlap:
            raise BackendConvergenceMatrixError(
                f"phase {phase_name} overlaps an earlier phase: {overlap}"
            )
        used.update(item.instance_id for item in selected_instances)

        stability: list[str] = []
        if phase.stability_replicates is not None:
            for stratum, repo in matrix.strata.items():
                candidates = [item for item in selected_instances if item.repo == repo]
                ranked = _ranked(
                    candidates,
                    seed=matrix.selection_seed,
                    phase=f"{phase_name}-stability",
                )
                if len(ranked) < phase.stability_per_stratum:
                    raise BackendConvergenceMatrixError(
                        f"phase {phase_name} stratum {stratum} lacks stability tasks"
                    )
                stability.extend(
                    item.instance_id for item in ranked[: phase.stability_per_stratum]
                )
        stability_set = set(stability)
        selected = tuple(
            SelectedTask(
                instance_id=item.instance_id,
                repo=item.repo,
                base_commit=item.base_commit,
                stratum=stratum_for_repo[item.repo],
                replicates=(
                    phase.stability_replicates
                    if item.instance_id in stability_set
                    and phase.stability_replicates is not None
                    else phase.replicates
                ),
            )
            for item in selected_instances
        )
        result[phase_name] = (selected, tuple(stability))
    return result


def _cell_argv(
    *,
    mode: Literal["dry-run", "implement"],
    matrix: BackendConvergenceMatrix,
    phase: PhaseDefinition,
    task: SelectedTask,
    arm: ConvergenceArm,
    dataset_path: Path,
    root: Path,
    repo_cache_dir: Path,
) -> tuple[str, ...]:
    argv = [
        str(Path(sys.executable).resolve()),
        "-m",
        "crucible.workflows.__main__",
        "swebench",
        "--spec",
        str(root / "spec"),
        "--dataset",
        str(dataset_path),
        "--instance-id",
        task.instance_id,
        "--backend-config",
        str(arm.routing_path),
        "--predictions-out",
        str(root / "predictions.jsonl"),
        f"--{mode}",
        "--planner-count",
        str(matrix.execution.planner_count),
    ]
    if mode == "implement":
        budget = str(matrix.execution.budget_usd_per_instance)
        argv.extend(
            [
                "--max-turns",
                str(matrix.execution.max_turns),
                "--budget-usd-per-instance",
                budget,
                "--budget-usd",
                budget,
                "--eval-timeout",
                str(matrix.execution.eval_timeout_seconds),
                "--repo-cache-dir",
                str(repo_cache_dir),
                "--max-concurrent-instances",
                "1",
            ]
        )
        if not phase.evaluate_inline:
            argv.append("--no-evaluate-inline")
        if not matrix.execution.include_breach:
            argv.append("--no-include-breach")
        if not matrix.execution.include_review:
            argv.append("--no-include-review")
        if not matrix.execution.build_verify:
            argv.append("--no-build-verify")
    return tuple(argv)


def build_backend_convergence_plan(
    matrix: BackendConvergenceMatrix,
    *,
    phase: str,
    dataset_path: Path | str,
    output_root: Path | str,
    repo_cache_dir: Path | str,
    mode: Literal["dry-run", "implement"] = "dry-run",
) -> BackendConvergencePlan:
    """Expand one frozen phase into isolated, dependency-ordered cells."""

    if type(matrix) is not BackendConvergenceMatrix:
        raise BackendConvergenceMatrixError("matrix must be loader-produced")
    canonical = load_backend_convergence_matrix(matrix.source_path)
    if matrix != canonical:
        raise BackendConvergenceMatrixError("caller matrix differs from source bytes")
    if phase not in _PHASES:
        raise BackendConvergenceMatrixError(f"unknown phase {phase!r}")
    if mode not in {"dry-run", "implement"}:
        raise BackendConvergenceMatrixError(f"unknown workflow mode {mode!r}")
    dataset = Path(dataset_path).resolve()
    root = Path(output_root).resolve()
    output_namespace = _OUTPUT_NAMESPACE_BY_MATRIX.get(canonical.schema_version)
    cell_root_base = root if output_namespace is None else root / output_namespace
    cache = Path(repo_cache_dir).resolve()
    selections = select_matrix_tasks(canonical, dataset)
    tasks, stability = selections[phase]
    cells: list[ConvergenceCell] = []
    counterbalance_offsets = _counterbalance_offsets(
        tasks,
        seed=canonical.selection_seed,
        phase=phase,
    )
    stratum_pair_indexes: dict[str, int] = {}
    for task in tasks:
        for replicate in range(1, task.replicates + 1):
            stratum_pair_index = stratum_pair_indexes.get(task.stratum, 0)
            assignment = (
                _ARMS
                if (stratum_pair_index + counterbalance_offsets[task.stratum])
                % len(_ARMS)
                == 0
                else tuple(reversed(_ARMS))
            )
            stratum_pair_indexes[task.stratum] = stratum_pair_index + 1
            predecessor: str | None = None
            for order_index, arm_key in enumerate(assignment, start=1):
                arm = canonical.arms[arm_key]
                cell_id = f"{phase}.{task.instance_id}.r{replicate:03d}.{arm.arm_id}"
                cell_root = (
                    cell_root_base
                    / phase
                    / task.instance_id
                    / f"replicate-{replicate:03d}"
                    / arm.arm_id
                )
                cells.append(
                    ConvergenceCell(
                        cell_id=cell_id,
                        phase=phase,
                        task_id=task.instance_id,
                        stratum=task.stratum,
                        replicate=replicate,
                        assignment="".join(assignment),
                        order_index=order_index,
                        arm=arm_key,
                        backend=arm.backend,
                        model=arm.model,
                        depends_on=predecessor,
                        root=cell_root,
                        argv=_cell_argv(
                            mode=mode,
                            matrix=canonical,
                            phase=canonical.phases[phase],
                            task=task,
                            arm=arm,
                            dataset_path=dataset,
                            root=cell_root,
                            repo_cache_dir=cache,
                        ),
                    )
                )
                predecessor = cell_id
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise BackendConvergenceMatrixError("expanded cell identifiers collide")
    if len({cell.root for cell in cells}) != len(cells):
        raise BackendConvergenceMatrixError("expanded cell roots collide")
    return BackendConvergencePlan(
        matrix=canonical,
        phase=canonical.phases[phase],
        mode=mode,
        dataset_path=dataset,
        output_root=root,
        repo_cache_dir=cache,
        selected_tasks=tasks,
        stability_task_ids=stability,
        cells=tuple(cells),
    )


__all__ = [
    "AnalysisPolicy",
    "BackendConvergenceMatrix",
    "BackendConvergenceMatrixError",
    "BackendConvergencePlan",
    "ConvergenceArm",
    "ConvergenceCell",
    "DatasetIdentity",
    "ExecutionPolicy",
    "MATRIX_SCHEMA",
    "MATRIX_SCHEMA_V2",
    "PLAN_SCHEMA",
    "PLAN_SCHEMA_V2",
    "PhaseDefinition",
    "SWEBENCH_BACKEND_CATALOG_CONTRACT",
    "SWEBENCH_BACKEND_CONVERGENCE_ROLES",
    "SelectedTask",
    "build_backend_convergence_plan",
    "load_backend_convergence_matrix",
    "load_swebench_backend_profile_catalog",
    "select_matrix_tasks",
    "validate_swebench_backend_catalog",
]
