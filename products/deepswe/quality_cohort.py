# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free planning for the bounded DeepSWE Astra/Mythos/Kimi pilot.

This module authenticates one checked-in matrix, the pinned 113-task corpus,
three static routing documents, and a byte-bound diagnostic analyzer.  It can
expand and validate a proposed 90-cell plan.  It contains no dispatch path,
does not construct a backend, and grants neither provider nor spend authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml  # type: ignore[import-untyped]

from crucible.backends.config import lint_routing_config, routing_fingerprint
from crucible.products.backend_convergence_catalog import (
    BackendLaneCatalogError,
    BackendLaneDefinition,
    validate_backend_lane_catalog,
)
from crucible.products.deepswe.backend_convergence import (
    DEEPSWE_BACKEND_CATALOG_CONTRACT,
    DEEPSWE_BACKEND_CONVERGENCE_ROLES,
)
from crucible.products.deepswe.dataset import (
    DeepSWEInstance,
    DeepSWEInstanceError,
    DeepSWETestsTreeIdentity,
    fingerprint_task_tests_tree,
    load_dataset,
)
from crucible.common.authority_io import (
    AuthorityIOError,
    read_exact_regular,
)

MATRIX_SCHEMA = "crucible.deepswe-backend-convergence-matrix.v1"
PLAN_SCHEMA = "crucible.deepswe-backend-convergence-plan.v1"
ANALYSIS_SCHEMA = "crucible.deepswe-backend-convergence-analysis.v1"
COHORT_ID = "deepswe-astra-mythos51-kimi-quality-pilot-20260914-v1"
SELECTION_SEED = COHORT_ID

ARMS = ("A", "B", "C")
ORDERS = ("ABC", "BCA", "CAB")
LANGUAGES = ("go", "javascript", "python", "rust", "typescript")
EXCLUDED_TASK_IDS = (
    "adaptix-name-mapping-aliases",
    "ts-pattern-match-each",
)
EXCLUDED_REPOS = ("prometheus/prometheus",)

_EXPECTED_CORPUS = {
    "id": "deep-swe-1-1-frontier-abc-20260802-v19",
    "git_commit": "e016041a6ccf8da29906afc9a3f5a8df940a1f78",
    "git_tree": "da6af978736343573c9f9560648c7a7f5e527a73",
    "clean_status_sha256": (
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
    "manifest": "tasks/manifest.json",
    "manifest_schema": "1.0",
    "manifest_sha256": (
        "sha256:f987b656fe75e20eeb7a050ec1b3c5d692cc8f102dfe7caca45906616b5ae879"
    ),
    "task_count": 113,
}
_EXPECTED_TASKS = (
    (
        "participle-grammar-conflict-analysis",
        "alecthomas/participle",
        "go",
        "feature_request",
        ("ABC", "BCA", "CAB"),
    ),
    (
        "kea-atomic-signal-selectors",
        "keajs/kea",
        "typescript",
        "feature_request",
        ("BCA", "CAB", "ABC"),
    ),
    (
        "abs-stepped-slices",
        "abs-lang/abs",
        "go",
        "feature_request",
        ("CAB", "ABC", "BCA"),
    ),
    (
        "obsidian-linter-scoped-ignore-markers",
        "platers/obsidian-linter",
        "typescript",
        "feature_request",
        ("ABC", "BCA", "CAB"),
    ),
    (
        "bandit-interprocedural-taint-checks",
        "PyCQA/bandit",
        "python",
        "feature_request",
        ("BCA", "CAB", "ABC"),
    ),
    (
        "yjs-map-conflict-detection",
        "yjs/yjs",
        "javascript",
        "feature_request",
        ("CAB", "ABC", "BCA"),
    ),
    (
        "boa-hierarchical-evaluation-cancellation",
        "boa-dev/boa",
        "rust",
        "feature_request",
        ("ABC", "BCA", "CAB"),
    ),
    (
        "vulture-persistent-analysis-cache",
        "jendrikseipp/vulture",
        "python",
        "feature_request",
        ("BCA", "CAB", "ABC"),
    ),
    (
        "testem-bail-on-test-failure",
        "testem/testem",
        "javascript",
        "feature_request",
        ("CAB", "ABC", "BCA"),
    ),
    (
        "fd-deterministic-multi-key-sorting",
        "sharkdp/fd",
        "rust",
        "feature_request",
        ("ABC", "BCA", "CAB"),
    ),
)
QUALITY_PILOT_TASK_IDS = tuple(item[0] for item in _EXPECTED_TASKS)

_EXPECTED_LANES: Mapping[str, tuple[str, str, str, str, str, str | None, str]] = {
    "A": (
        "codex-astra",
        "codex",
        "codex",
        "legacy_app_server",
        "gpt-6-astra",
        "max",
        "enforce",
    ),
    "B": (
        "claude-mythos51",
        "claude",
        "claude",
        "claude_agent_sdk",
        "claude-mythos-5-1",
        "high",
        "enforce",
    ),
    "C": (
        "kimi-k2.7-highspeed-diagnostic",
        "kimi",
        "kimi",
        "kimi_agent_sdk_worker",
        "kimi-k2.7-code-highspeed",
        None,
        "diagnostic",
    ),
}
_EXPECTED_ROUTE_IDENTITIES = {
    "A": (
        "backends_codex_astra.yaml",
        "sha256:3380b297a17c987f24a99de58fb7bb8cc6324d1df035cbbb18893bda50cf66e3",
        "sha256:1ca6960f903777f41f19252fbe2204b4a0b85dd9c92623f5bbb59e33b26c300b",
    ),
    "B": (
        "backends_claude_mythos51_org.yaml",
        "sha256:cd0f24b34a8fd8ada2a4feb7b5e8489986b3f641711e815ffd71dc14f0f02d74",
        "sha256:7182ce19381b6660b75f0c36f6fe309650c2d97f3aa20a2d7813a4fc8c4fab32",
    ),
    "C": (
        "backends_kimi_sdk_diagnostic.yaml",
        "sha256:0e772e66620c898592eb6ea48cbaf976a6de561363cfca8f370db01bf81a2c4b",
        "sha256:1f8d11259eceefe784a77ac87a1b7287dc474c92bc1c3a861c248170f40914b4",
    ),
}
_EXPECTED_ANALYSIS_DIGEST = (
    "sha256:1629b326ada7c7f375b148a9254ae979636a4eacc1264bc6f191a2bd503feeef"
)
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,191}$")
_MAX_CONTROL_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024


class DeepSWEQualityCohortError(ValueError):
    """Raised when the matrix, corpus, route, analyzer, or plan drifts."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise DeepSWEQualityCohortError(
                "matrix YAML contains an unhashable mapping key"
            ) from exc
        if duplicate:
            raise DeepSWEQualityCohortError(f"matrix YAML repeats mapping key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise DeepSWEQualityCohortError(f"{field} must be an exact mapping")
    return value


def _fields(value: Mapping[str, Any], *, required: set[str], field: str) -> None:
    if set(value) != required:
        raise DeepSWEQualityCohortError(
            f"{field} fields drifted; missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )


def _matches_exact_values(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    return set(observed) == set(expected) and all(
        type(observed[key]) is type(value) and observed[key] == value
        for key, value in expected.items()
    )


def _text(value: Any, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise DeepSWEQualityCohortError(f"{field} must be exact non-empty text")
    return value


def _bool(value: Any, *, field: str) -> bool:
    if type(value) is not bool:
        raise DeepSWEQualityCohortError(f"{field} must be an exact boolean")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise DeepSWEQualityCohortError(f"{field} must be a positive integer")
    return value


def _money(value: Any, *, field: str) -> Decimal:
    if type(value) is not str:
        raise DeepSWEQualityCohortError(f"{field} must be exact decimal text")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise DeepSWEQualityCohortError(f"{field} must be finite money") from exc
    if not result.is_finite() or result <= 0:
        raise DeepSWEQualityCohortError(f"{field} must be finite positive money")
    return result


def _tagged_digest(value: Any, *, field: str) -> str:
    value = _text(value, field=field)
    if _DIGEST_RE.fullmatch(value) is None:
        raise DeepSWEQualityCohortError(f"{field} must be tagged SHA-256")
    return value


def _read_regular(path: Path, *, maximum: int, field: str) -> bytes:
    try:
        payload, _metadata = read_exact_regular(path, maximum=maximum)
    except (AuthorityIOError, OSError) as exc:
        raise DeepSWEQualityCohortError(
            f"cannot read exact {field} {path}: {exc}"
        ) from exc
    return payload


def _confined_file(matrix_path: Path, value: Any, *, field: str) -> Path:
    name = _text(value, field=field)
    relative = Path(name)
    if relative.is_absolute() or relative.name != name or ".." in relative.parts:
        raise DeepSWEQualityCohortError(f"{field} must be one confined filename")
    return matrix_path.parent / relative


@dataclasses.dataclass(frozen=True)
class CorpusIdentity:
    corpus_id: str
    git_commit: str
    git_tree: str
    clean_status_digest: str
    manifest_relative: str
    manifest_schema: str
    manifest_digest: str
    task_count: int

    def identity(self) -> dict[str, Any]:
        return {
            "id": self.corpus_id,
            "git_commit": self.git_commit,
            "git_tree": self.git_tree,
            "clean_status_sha256": self.clean_status_digest,
            "manifest": self.manifest_relative,
            "manifest_schema": self.manifest_schema,
            "manifest_sha256": self.manifest_digest,
            "task_count": self.task_count,
        }


@dataclasses.dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    repo: str
    language: str
    category: str
    replicate_orders: tuple[str, str, str]

    def identity(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "language": self.language,
            "category": self.category,
            "replicate_orders": list(self.replicate_orders),
        }


@dataclasses.dataclass(frozen=True)
class QualityArm:
    key: str
    arm_id: str
    backend: str
    runtime_backend: str
    transport: str
    model: str
    effort: str | None
    feature_admission: str
    routing_path: Path
    routing_digest: str
    static_routing_fingerprint: str

    def identity(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "id": self.arm_id,
            "backend": self.backend,
            "runtime_backend": self.runtime_backend,
            "transport": self.transport,
            "model": self.model,
            "effort": self.effort,
            "feature_admission": self.feature_admission,
            "evidence_status": "diagnostic",
            "promotion_eligible": False,
            "routing_path": str(self.routing_path),
            "routing_digest": self.routing_digest,
            "static_routing_fingerprint": self.static_routing_fingerprint,
        }


@dataclasses.dataclass(frozen=True)
class ExecutionPolicy:
    max_parallel_cells: int
    max_parallel_per_backend: int
    first_block_serial: bool
    first_block_task_id: str
    first_block_replicate: int
    first_block_gate: str
    dependency_condition: str
    ready_queue_policy: str
    max_turns: int
    planner_count: int
    backend_limit_mode: str
    backend_limit_deadline_seconds: int
    planner_tool_call_limit: int
    planner_tool_result_bytes: int
    max_patch_bytes: int
    task_timeout_seconds: int
    outer_timeout_seconds: int
    storage_limit_bytes: int
    per_cell_soft_budget_usd: Decimal
    nominal_budget_usd: Decimal
    usd_enforcement: str


@dataclasses.dataclass(frozen=True)
class AnalysisPolicy:
    implementation_path: Path
    implementation_digest: str
    purpose: str
    result_set_policy: str
    publication_rule: str

    def identity(self) -> dict[str, Any]:
        return {
            "implementation": {
                "schema": ANALYSIS_SCHEMA,
                "path": str(self.implementation_path),
                "sha256": self.implementation_digest,
            },
            "purpose": self.purpose,
            "result_set_policy": self.result_set_policy,
            "publication_rule": self.publication_rule,
            "ranking_admission": False,
            "promotion_admission": False,
            "complete_cost_evidence_required": True,
            "complete_trace_evidence_required": True,
            "complete_settlement_evidence_required": True,
        }


@dataclasses.dataclass(frozen=True)
class QualityCohortMatrix:
    source_path: Path
    source_digest: str
    corpus: CorpusIdentity
    tasks: tuple[TaskDefinition, ...]
    arms: Mapping[str, QualityArm]
    execution: ExecutionPolicy
    analysis: AnalysisPolicy


@dataclasses.dataclass(frozen=True)
class BoundTask:
    task_id: str
    repo: str
    language: str
    category: str
    base_commit: str
    tests_identity: DeepSWETestsTreeIdentity
    replicate_orders: tuple[str, str, str]

    def identity(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "repo": self.repo,
            "language": self.language,
            "category": self.category,
            "base_commit": self.base_commit,
            "tests_identity": self.tests_identity.to_payload(),
            "replicate_orders": list(self.replicate_orders),
        }


@dataclasses.dataclass(frozen=True)
class CorpusBinding:
    root: Path
    manifest_path: Path
    manifest_digest: str
    git_commit: str
    git_tree: str
    clean_status_digest: str
    tasks: tuple[BoundTask, ...]
    # These records leave the same before/after Git authority as ``tasks``.
    # They stay out of the serialized plan, but let downstream provider-free
    # qualification bind full instances without a second unauthenticated load.
    instances: tuple[DeepSWEInstance, ...] = dataclasses.field(
        default=(), repr=False, compare=False
    )

    def identity(self, expected: CorpusIdentity) -> dict[str, Any]:
        return {
            **expected.identity(),
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "observed_manifest_sha256": self.manifest_digest,
            "observed_git_commit": self.git_commit,
            "observed_git_tree": self.git_tree,
            "observed_clean_status_sha256": self.clean_status_digest,
        }


@dataclasses.dataclass(frozen=True)
class QualityCell:
    cell_id: str
    block_id: str
    block_index: int
    task_id: str
    replicate: int
    assignment: str
    order_index: int
    arm: str
    backend: str
    model: str
    depends_on: tuple[str, ...]
    root: Path
    proposed_argv: tuple[str, ...]

    def identity(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "block_id": self.block_id,
            "block_index": self.block_index,
            "task_id": self.task_id,
            "replicate": self.replicate,
            "assignment": self.assignment,
            "order_index": self.order_index,
            "arm": self.arm,
            "backend": self.backend,
            "model": self.model,
            "depends_on": list(self.depends_on),
            "root": str(self.root),
            "route_authority": "static-planning-input-not-dispatch-route",
            "proposed_argv": list(self.proposed_argv),
        }


@dataclasses.dataclass(frozen=True)
class QualityCohortPlan:
    matrix: QualityCohortMatrix
    corpus: CorpusBinding
    output_root: Path
    cells: tuple[QualityCell, ...]

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        execution = self.matrix.execution
        payload: dict[str, Any] = {
            "schema_version": PLAN_SCHEMA,
            "cohort_id": COHORT_ID,
            "planner_mode": "provider-free-dry-run",
            "execution_authority": "absent-provider-free-plan-only",
            "dispatch": {
                "enabled": False,
                "entrypoint": None,
                "provider_calls": False,
            },
            "spend": {
                "authorized": False,
                "authorized_usd": "0",
                "per_cell_soft_usd": str(execution.per_cell_soft_budget_usd),
                "nominal_usd": str(execution.nominal_budget_usd),
                "usd_enforcement": execution.usd_enforcement,
                "hard_provider_spend_cap": False,
            },
            "claims": {
                "diagnostic_only": True,
                "ranking_admission": False,
                "promotion_admission": False,
                "formal_e12_execution_admission": False,
                "adaptive_feedback": False,
            },
            "matrix": {
                "path": str(self.matrix.source_path),
                "sha256": self.matrix.source_digest,
            },
            "corpus": self.corpus.identity(self.matrix.corpus),
            "analysis": self.matrix.analysis.identity(),
            "arms": {key: self.matrix.arms[key].identity() for key in ARMS},
            "selection": {
                "algorithm": "exact-audited-list-sha256-counterbalanced-v1",
                "seed": SELECTION_SEED,
                "exclusions": {
                    "task_ids": list(EXCLUDED_TASK_IDS),
                    "repos": list(EXCLUDED_REPOS),
                },
                "tasks": [task.identity() for task in self.corpus.tasks],
            },
            "parallelism": {
                "max_parallel_cells": execution.max_parallel_cells,
                "max_parallel_per_backend": execution.max_parallel_per_backend,
                "first_block_serial": execution.first_block_serial,
                "first_block_gate": execution.first_block_gate,
                "dependency_condition": execution.dependency_condition,
                "ready_queue_policy": execution.ready_queue_policy,
            },
            "limits": {
                "max_turns": execution.max_turns,
                "planner_count": execution.planner_count,
                "backend_limit_mode": execution.backend_limit_mode,
                "backend_limit_deadline_seconds": (
                    execution.backend_limit_deadline_seconds
                ),
                "planner_tool_call_limit": execution.planner_tool_call_limit,
                "planner_tool_result_bytes": execution.planner_tool_result_bytes,
                "max_patch_bytes": execution.max_patch_bytes,
                "task_timeout_seconds": execution.task_timeout_seconds,
                "outer_timeout_seconds": execution.outer_timeout_seconds,
                "storage_limit_bytes": execution.storage_limit_bytes,
            },
            "output_root": str(self.output_root),
            "scheduled_blocks": 30,
            "scheduled_cells": 90,
            "cells": [cell.identity() for cell in self.cells],
        }
        if include_digest:
            payload["plan_digest"] = _digest_bytes(_canonical_bytes(payload))
        return payload

    def to_json_bytes(self) -> bytes:
        return _canonical_bytes(self.payload())


def _parse_task(value: Any, *, index: int) -> TaskDefinition:
    raw = _mapping(value, field=f"selection.tasks[{index}]")
    _fields(
        raw,
        required={"task_id", "repo", "language", "category", "replicate_orders"},
        field=f"selection.tasks[{index}]",
    )
    orders_raw = raw["replicate_orders"]
    if type(orders_raw) is not list or len(orders_raw) != 3:
        raise DeepSWEQualityCohortError(
            f"selection.tasks[{index}].replicate_orders must contain three orders"
        )
    orders = tuple(orders_raw)
    if orders not in (
        ("ABC", "BCA", "CAB"),
        ("BCA", "CAB", "ABC"),
        ("CAB", "ABC", "BCA"),
    ):
        raise DeepSWEQualityCohortError(
            f"selection.tasks[{index}] does not contain one ABC/BCA/CAB rotation"
        )
    return TaskDefinition(
        task_id=_text(raw["task_id"], field=f"selection.tasks[{index}].task_id"),
        repo=_text(raw["repo"], field=f"selection.tasks[{index}].repo"),
        language=_text(raw["language"], field=f"selection.tasks[{index}].language"),
        category=_text(raw["category"], field=f"selection.tasks[{index}].category"),
        replicate_orders=orders,  # type: ignore[arg-type]
    )


def _load_arm(
    key: str,
    value: Any,
    *,
    matrix_path: Path,
) -> QualityArm:
    raw = _mapping(value, field=f"arms.{key}")
    _fields(
        raw,
        required={
            "id",
            "backend",
            "runtime_backend",
            "transport",
            "model",
            "effort",
            "feature_admission",
            "evidence_status",
            "routing",
            "routing_sha256",
            "static_routing_fingerprint",
        },
        field=f"arms.{key}",
    )
    observed_lane = (
        raw["id"],
        raw["backend"],
        raw["runtime_backend"],
        raw["transport"],
        raw["model"],
        raw["effort"],
        raw["feature_admission"],
    )
    if observed_lane != _EXPECTED_LANES[key]:
        raise DeepSWEQualityCohortError(
            f"arm {key} must be exact lane {_EXPECTED_LANES[key]!r}"
        )
    if raw["evidence_status"] != "diagnostic":
        raise DeepSWEQualityCohortError(f"arm {key} must remain diagnostic")
    route_name, expected_digest, expected_fingerprint = _EXPECTED_ROUTE_IDENTITIES[key]
    if raw["routing"] != route_name:
        raise DeepSWEQualityCohortError(f"arm {key} routing filename drifted")
    declared_digest = _tagged_digest(
        raw["routing_sha256"], field=f"arms.{key}.routing_sha256"
    )
    declared_fingerprint = _tagged_digest(
        raw["static_routing_fingerprint"],
        field=f"arms.{key}.static_routing_fingerprint",
    )
    if (
        declared_digest != expected_digest
        or declared_fingerprint != expected_fingerprint
    ):
        raise DeepSWEQualityCohortError(f"arm {key} static route identity drifted")
    route_path = _confined_file(matrix_path, route_name, field=f"arms.{key}.routing")
    route_bytes = _read_regular(
        route_path, maximum=_MAX_CONTROL_BYTES, field=f"arm {key} route"
    )
    if _digest_bytes(route_bytes) != expected_digest:
        raise DeepSWEQualityCohortError(f"arm {key} route bytes drifted")
    try:
        routing = lint_routing_config(route_bytes)
        static_fingerprint = routing_fingerprint(routing)
    except ValueError as exc:
        raise DeepSWEQualityCohortError(f"arm {key} route is invalid: {exc}") from exc
    if static_fingerprint != expected_fingerprint:
        raise DeepSWEQualityCohortError(f"arm {key} static routing fingerprint drifted")
    missing_roles = sorted(set(DEEPSWE_BACKEND_CONVERGENCE_ROLES) - set(routing.roles))
    if missing_roles:
        raise DeepSWEQualityCohortError(
            f"arm {key} does not explicitly pin all DeepSWE roles: {missing_roles}"
        )
    for role, config in (
        ("default", routing.default),
        *((role, routing.for_role(role)) for role in DEEPSWE_BACKEND_CONVERGENCE_ROLES),
    ):
        expected = _EXPECTED_LANES[key]
        if (config.backend, config.model, config.effort, config.feature_admission) != (
            expected[1],
            expected[4],
            expected[5],
            expected[6],
        ):
            raise DeepSWEQualityCohortError(
                f"arm {key} effective role {role!r} drifts from its exact lane"
            )
    return QualityArm(
        key=key,
        arm_id=observed_lane[0],
        backend=observed_lane[1],
        runtime_backend=observed_lane[2],
        transport=observed_lane[3],
        model=observed_lane[4],
        effort=observed_lane[5],
        feature_admission=observed_lane[6],
        routing_path=route_path,
        routing_digest=declared_digest,
        static_routing_fingerprint=static_fingerprint,
    )


def _load_execution(value: Any) -> ExecutionPolicy:
    raw = _mapping(value, field="execution")
    _fields(
        raw,
        required={
            "phase",
            "tasks",
            "replicates_per_task",
            "arms_per_block",
            "scheduled_blocks",
            "scheduled_cells",
            "max_parallel_cells",
            "max_parallel_per_backend",
            "first_block_serial",
            "first_block_task_id",
            "first_block_replicate",
            "first_block_gate",
            "dependency_condition",
            "ready_queue_policy",
            "max_turns",
            "planner_count",
            "backend_limit_mode",
            "backend_limit_deadline_seconds",
            "planner_tool_call_limit",
            "planner_tool_result_bytes",
            "max_patch_bytes",
            "task_timeout_seconds",
            "outer_timeout_seconds",
            "storage_limit_bytes",
            "budget",
        },
        field="execution",
    )
    expected_scalars: Mapping[str, Any] = {
        "phase": "quality_pilot",
        "tasks": 10,
        "replicates_per_task": 3,
        "arms_per_block": 3,
        "scheduled_blocks": 30,
        "scheduled_cells": 90,
        "max_parallel_cells": 3,
        "max_parallel_per_backend": 1,
        "first_block_serial": True,
        "first_block_task_id": QUALITY_PILOT_TASK_IDS[0],
        "first_block_replicate": 1,
        "first_block_gate": "all-three-terminal-settled-before-other-blocks",
        "dependency_condition": "predecessor-any-terminal-and-settled",
        "ready_queue_policy": "matrix-task-order-then-replicate-then-order-index",
        "max_turns": 90,
        "planner_count": 3,
        "backend_limit_mode": "diagnostic",
        "backend_limit_deadline_seconds": 1200,
        "planner_tool_call_limit": 32,
        "planner_tool_result_bytes": 262144,
        "max_patch_bytes": 67108864,
        "task_timeout_seconds": 5400,
        "outer_timeout_seconds": 5700,
        "storage_limit_bytes": 21474836480,
    }
    for field, expected in expected_scalars.items():
        if type(raw[field]) is not type(expected) or raw[field] != expected:
            raise DeepSWEQualityCohortError(
                f"execution.{field} must remain {expected!r}"
            )
    budget = _mapping(raw["budget"], field="execution.budget")
    _fields(
        budget,
        required={
            "per_cell_soft_usd",
            "nominal_usd",
            "usd_enforcement",
            "hard_provider_spend_cap",
            "spend_authorized",
        },
        field="execution.budget",
    )
    per_cell = _money(
        budget["per_cell_soft_usd"], field="execution.budget.per_cell_soft_usd"
    )
    nominal = _money(budget["nominal_usd"], field="execution.budget.nominal_usd")
    if (
        per_cell != Decimal("12")
        or nominal != Decimal("1080")
        or nominal != per_cell * 90
        or budget["usd_enforcement"] != "request-bound-post-hoc-lockout"
        or _bool(
            budget["hard_provider_spend_cap"],
            field="execution.budget.hard_provider_spend_cap",
        )
        or _bool(budget["spend_authorized"], field="execution.budget.spend_authorized")
    ):
        raise DeepSWEQualityCohortError(
            "quality-pilot budget must remain soft $12/cell, nominal $1080, "
            "post-hoc, and unauthorized"
        )
    return ExecutionPolicy(
        max_parallel_cells=_positive_int(
            raw["max_parallel_cells"], field="execution.max_parallel_cells"
        ),
        max_parallel_per_backend=_positive_int(
            raw["max_parallel_per_backend"],
            field="execution.max_parallel_per_backend",
        ),
        first_block_serial=_bool(
            raw["first_block_serial"], field="execution.first_block_serial"
        ),
        first_block_task_id=raw["first_block_task_id"],
        first_block_replicate=raw["first_block_replicate"],
        first_block_gate=raw["first_block_gate"],
        dependency_condition=raw["dependency_condition"],
        ready_queue_policy=raw["ready_queue_policy"],
        max_turns=raw["max_turns"],
        planner_count=raw["planner_count"],
        backend_limit_mode=raw["backend_limit_mode"],
        backend_limit_deadline_seconds=raw["backend_limit_deadline_seconds"],
        planner_tool_call_limit=raw["planner_tool_call_limit"],
        planner_tool_result_bytes=raw["planner_tool_result_bytes"],
        max_patch_bytes=raw["max_patch_bytes"],
        task_timeout_seconds=raw["task_timeout_seconds"],
        outer_timeout_seconds=raw["outer_timeout_seconds"],
        storage_limit_bytes=raw["storage_limit_bytes"],
        per_cell_soft_budget_usd=per_cell,
        nominal_budget_usd=nominal,
        usd_enforcement=budget["usd_enforcement"],
    )


def _load_analysis(value: Any, *, matrix_path: Path) -> AnalysisPolicy:
    raw = _mapping(value, field="analysis")
    _fields(
        raw,
        required={
            "implementation",
            "purpose",
            "result_set_policy",
            "publication_rule",
            "ranking_admission",
            "promotion_admission",
            "complete_cost_evidence_required",
            "complete_trace_evidence_required",
            "complete_settlement_evidence_required",
        },
        field="analysis",
    )
    expected = {
        "purpose": "descriptive-diagnostic-only",
        "result_set_policy": (
            "exact-scheduled-cell-and-closed-terminal-status-bijection"
        ),
        "publication_rule": "complete-valid-evidence-only",
        "ranking_admission": False,
        "promotion_admission": False,
        "complete_cost_evidence_required": True,
        "complete_trace_evidence_required": True,
        "complete_settlement_evidence_required": True,
    }
    for field, expected_value in expected.items():
        if type(raw[field]) is not type(expected_value) or raw[field] != expected_value:
            raise DeepSWEQualityCohortError(
                f"analysis.{field} must remain {expected_value!r}"
            )
    implementation = _mapping(raw["implementation"], field="analysis.implementation")
    _fields(
        implementation,
        required={"schema", "path", "sha256"},
        field="analysis.implementation",
    )
    if (
        implementation["schema"] != ANALYSIS_SCHEMA
        or implementation["path"] != "backend_convergence_analysis.py"
    ):
        raise DeepSWEQualityCohortError("analysis implementation identity drifted")
    declared_digest = _tagged_digest(
        implementation["sha256"], field="analysis.implementation.sha256"
    )
    if declared_digest != _EXPECTED_ANALYSIS_DIGEST:
        raise DeepSWEQualityCohortError("analysis implementation digest drifted")
    path = _confined_file(
        matrix_path,
        implementation["path"],
        field="analysis.implementation.path",
    )
    if (
        _digest_bytes(
            _read_regular(
                path, maximum=_MAX_CONTROL_BYTES, field="analysis implementation"
            )
        )
        != declared_digest
    ):
        raise DeepSWEQualityCohortError(
            "analysis implementation bytes do not match the matrix"
        )
    return AnalysisPolicy(
        implementation_path=path,
        implementation_digest=declared_digest,
        purpose="descriptive-diagnostic-only",
        result_set_policy=(
            "exact-scheduled-cell-and-closed-terminal-status-bijection"
        ),
        publication_rule="complete-valid-evidence-only",
    )


def load_quality_cohort_matrix(path: Path | str) -> QualityCohortMatrix:
    """Load and authenticate the exact checked-in quality-pilot matrix."""

    source = Path(os.path.abspath(Path(path)))
    source_bytes = _read_regular(source, maximum=_MAX_CONTROL_BYTES, field="matrix")
    try:
        raw = yaml.load(source_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DeepSWEQualityCohortError(f"matrix YAML is invalid: {exc}") from exc
    root = _mapping(raw, field="matrix")
    _fields(
        root,
        required={
            "schema_version",
            "workflow",
            "cohort_id",
            "authority",
            "corpus",
            "selection",
            "roles",
            "arms",
            "execution",
            "analysis",
        },
        field="matrix",
    )
    if (
        root["schema_version"] != MATRIX_SCHEMA
        or root["workflow"] != "deepswe"
        or root["cohort_id"] != COHORT_ID
    ):
        raise DeepSWEQualityCohortError("unsupported matrix identity")

    authority = _mapping(root["authority"], field="authority")
    expected_authority = {
        "provider_calls": False,
        "live_dispatch": False,
        "spend_authorized": False,
        "diagnostic_only": True,
        "ranking_admission": False,
        "promotion_admission": False,
        "formal_e12_execution_admission": False,
        "adaptive_feedback": False,
    }
    if not _matches_exact_values(authority, expected_authority):
        raise DeepSWEQualityCohortError(
            "matrix authority must remain provider-free, diagnostic, and non-ranking"
        )

    corpus_raw = _mapping(root["corpus"], field="corpus")
    _fields(corpus_raw, required=set(_EXPECTED_CORPUS), field="corpus")
    if not _matches_exact_values(corpus_raw, _EXPECTED_CORPUS):
        raise DeepSWEQualityCohortError("matrix corpus identity drifted")
    corpus = CorpusIdentity(
        corpus_id=corpus_raw["id"],
        git_commit=corpus_raw["git_commit"],
        git_tree=corpus_raw["git_tree"],
        clean_status_digest=corpus_raw["clean_status_sha256"],
        manifest_relative=corpus_raw["manifest"],
        manifest_schema=corpus_raw["manifest_schema"],
        manifest_digest=corpus_raw["manifest_sha256"],
        task_count=corpus_raw["task_count"],
    )

    selection = _mapping(root["selection"], field="selection")
    _fields(
        selection,
        required={
            "algorithm",
            "seed",
            "languages",
            "per_language",
            "exclusions",
            "tasks",
        },
        field="selection",
    )
    if (
        selection["algorithm"] != "exact-audited-list-sha256-counterbalanced-v1"
        or selection["seed"] != SELECTION_SEED
        or selection["languages"] != list(LANGUAGES)
        or type(selection["per_language"]) is not int
        or selection["per_language"] != 2
    ):
        raise DeepSWEQualityCohortError("quality-pilot selection policy drifted")
    exclusions = _mapping(selection["exclusions"], field="selection.exclusions")
    _fields(
        exclusions,
        required={"task_ids", "repos"},
        field="selection.exclusions",
    )
    if exclusions["task_ids"] != list(EXCLUDED_TASK_IDS) or exclusions["repos"] != list(
        EXCLUDED_REPOS
    ):
        raise DeepSWEQualityCohortError("quality-pilot exclusions drifted")
    tasks_raw = selection["tasks"]
    if type(tasks_raw) is not list or len(tasks_raw) != 10:
        raise DeepSWEQualityCohortError("quality pilot requires exactly ten tasks")
    tasks = tuple(
        _parse_task(value, index=index) for index, value in enumerate(tasks_raw)
    )
    observed_tasks = tuple(
        (task.task_id, task.repo, task.language, task.category, task.replicate_orders)
        for task in tasks
    )
    if observed_tasks != _EXPECTED_TASKS:
        raise DeepSWEQualityCohortError("audited quality-pilot task list drifted")

    roles = root["roles"]
    if type(roles) is not list or tuple(roles) != DEEPSWE_BACKEND_CONVERGENCE_ROLES:
        raise DeepSWEQualityCohortError(
            "matrix must name the complete ordered DeepSWE role set"
        )
    arms_raw = _mapping(root["arms"], field="arms")
    if tuple(arms_raw) != ARMS:
        raise DeepSWEQualityCohortError("matrix arms must be ordered A, B, C")
    arms = {key: _load_arm(key, arms_raw[key], matrix_path=source) for key in ARMS}
    try:
        catalog = validate_backend_lane_catalog(
            DEEPSWE_BACKEND_CATALOG_CONTRACT,
            route_root=source.parent,
            lanes=tuple(
                BackendLaneDefinition(
                    key=key,
                    lane_id=arms[key].arm_id,
                    backend=arms[key].backend,
                    runtime_backend=arms[key].runtime_backend,
                    transport=arms[key].transport,
                    routing_path=arms[key].routing_path,
                    routing_digest=arms[key].routing_digest,
                    routing_fingerprint=arms[key].static_routing_fingerprint,
                    evidence_status="diagnostic",
                )
                for key in ARMS
            ),
            allow_additional_roles=True,
        )
    except BackendLaneCatalogError as exc:
        raise DeepSWEQualityCohortError(
            f"quality-pilot adapter catalog is invalid: {exc}"
        ) from exc
    if any(lane.promotion_eligible for lane in catalog.lanes):
        raise DeepSWEQualityCohortError("quality-pilot lanes cannot be promotable")

    execution = _load_execution(root["execution"])
    analysis = _load_analysis(root["analysis"], matrix_path=source)
    return QualityCohortMatrix(
        source_path=source,
        source_digest=_digest_bytes(source_bytes),
        corpus=corpus,
        tasks=tasks,
        arms=arms,
        execution=execution,
        analysis=analysis,
    )


def _git_output(root: Path, *arguments: str, binary: bool = False) -> bytes | str:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
            },
            text=not binary,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DeepSWEQualityCohortError(
            f"cannot authenticate corpus Git identity with {arguments!r}: {exc}"
        ) from exc
    if completed.stderr:
        raise DeepSWEQualityCohortError(
            f"corpus Git identity command emitted stderr for {arguments!r}"
        )
    return completed.stdout


def _measure_git(root: Path) -> tuple[str, str, str]:
    top = str(_git_output(root, "rev-parse", "--show-toplevel")).strip()
    try:
        observed_top = Path(top).resolve(strict=True)
        expected_top = root.resolve(strict=True)
    except OSError as exc:
        raise DeepSWEQualityCohortError(
            f"cannot resolve corpus Git root: {exc}"
        ) from exc
    if observed_top != expected_top:
        raise DeepSWEQualityCohortError("dataset manifest is outside its Git root")
    commit = str(_git_output(root, "rev-parse", "--verify", "HEAD")).strip()
    tree = str(_git_output(root, "rev-parse", "--verify", "HEAD^{tree}")).strip()
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        binary=True,
    )
    assert isinstance(status, bytes)
    return commit, tree, _digest_bytes(status)


def _rank_key(task_id: str) -> tuple[bytes, str]:
    return hashlib.sha256(
        f"{SELECTION_SEED}\0{task_id}".encode("utf-8")
    ).digest(), task_id


def _validate_audited_selection(
    instances: tuple[DeepSWEInstance, ...],
    tasks: tuple[TaskDefinition, ...],
) -> None:
    eligible = tuple(
        item
        for item in instances
        if item.task_id not in EXCLUDED_TASK_IDS and item.repo not in EXCLUDED_REPOS
    )
    selected: list[DeepSWEInstance] = []
    for language in LANGUAGES:
        ranked = sorted(
            (item for item in eligible if item.language == language),
            key=lambda item: _rank_key(item.task_id),
        )
        if len(ranked) < 2:
            raise DeepSWEQualityCohortError(
                f"corpus lacks two eligible {language} quality-pilot tasks"
            )
        selected.extend(ranked[:2])
    selected.sort(key=lambda item: _rank_key(item.task_id))
    if tuple(item.task_id for item in selected) != tuple(
        task.task_id for task in tasks
    ):
        raise DeepSWEQualityCohortError(
            "corpus no longer produces the exact audited seeded selection"
        )
    for task_index, task in enumerate(tasks):
        expected_orders = tuple(
            ORDERS[(task_index + replicate - 1) % len(ORDERS)]
            for replicate in range(1, 4)
        )
        if task.replicate_orders != expected_orders:
            raise DeepSWEQualityCohortError(
                f"task {task.task_id!r} counterbalanced orders drifted"
            )


def validate_quality_corpus(
    matrix: QualityCohortMatrix,
    manifest_path: Path | str,
) -> CorpusBinding:
    """Authenticate the clean pinned corpus and bind the ten selected tasks."""

    if type(matrix) is not QualityCohortMatrix:
        raise DeepSWEQualityCohortError("matrix must be loader-produced")
    canonical = load_quality_cohort_matrix(matrix.source_path)
    if matrix != canonical:
        raise DeepSWEQualityCohortError("caller matrix differs from source bytes")
    matrix = canonical
    manifest = Path(os.path.abspath(Path(manifest_path)))
    root = manifest.parent.parent
    if manifest != root / matrix.corpus.manifest_relative:
        raise DeepSWEQualityCohortError(
            "dataset must be the pinned corpus root's tasks/manifest.json"
        )
    before_git = _measure_git(root)
    if before_git != (
        matrix.corpus.git_commit,
        matrix.corpus.git_tree,
        matrix.corpus.clean_status_digest,
    ):
        raise DeepSWEQualityCohortError(
            "corpus Git commit, tree, or clean state drifted"
        )
    manifest_bytes = _read_regular(
        manifest, maximum=_MAX_MANIFEST_BYTES, field="corpus manifest"
    )
    manifest_digest = _digest_bytes(manifest_bytes)
    if manifest_digest != matrix.corpus.manifest_digest:
        raise DeepSWEQualityCohortError("corpus manifest bytes drifted")
    try:
        manifest_root = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=lambda pairs: _unique_json_mapping(pairs),
            parse_constant=lambda value: _raise_invalid_json_constant(value),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSWEQualityCohortError(
            f"corpus manifest JSON is invalid: {exc}"
        ) from exc
    if (
        type(manifest_root) is not dict
        or manifest_root.get("schema_version") != matrix.corpus.manifest_schema
        or manifest_root.get("task_count") != matrix.corpus.task_count
    ):
        raise DeepSWEQualityCohortError("corpus manifest schema or task count drifted")
    try:
        instances = load_dataset(manifest)
    except (OSError, DeepSWEInstanceError) as exc:
        raise DeepSWEQualityCohortError(f"DeepSWE corpus is invalid: {exc}") from exc
    if len(instances) != matrix.corpus.task_count:
        raise DeepSWEQualityCohortError(
            f"corpus contains {len(instances)} tasks; expected {matrix.corpus.task_count}"
        )
    _validate_audited_selection(instances, matrix.tasks)
    by_id = {instance.task_id: instance for instance in instances}
    if len(by_id) != len(instances):
        raise DeepSWEQualityCohortError("corpus task ids are not unique")
    bound: list[BoundTask] = []
    for definition in matrix.tasks:
        instance = by_id[definition.task_id]
        if (
            instance.repo,
            instance.language,
            instance.category,
        ) != (definition.repo, definition.language, definition.category):
            raise DeepSWEQualityCohortError(
                f"selected task {definition.task_id!r} metadata drifted"
            )
        try:
            tests_identity = fingerprint_task_tests_tree(manifest, definition.task_id)
        except (OSError, DeepSWEInstanceError) as exc:
            raise DeepSWEQualityCohortError(
                f"selected task {definition.task_id!r} tests identity failed: {exc}"
            ) from exc
        bound.append(
            BoundTask(
                task_id=definition.task_id,
                repo=definition.repo,
                language=definition.language,
                category=definition.category,
                base_commit=instance.base_commit_hash,
                tests_identity=tests_identity,
                replicate_orders=definition.replicate_orders,
            )
        )
    after_git = _measure_git(root)
    if after_git != before_git:
        raise DeepSWEQualityCohortError("corpus Git identity changed during validation")
    return CorpusBinding(
        root=root.resolve(strict=True),
        manifest_path=manifest.resolve(strict=True),
        manifest_digest=manifest_digest,
        git_commit=before_git[0],
        git_tree=before_git[1],
        clean_status_digest=before_git[2],
        tasks=tuple(bound),
        instances=tuple(by_id[definition.task_id] for definition in matrix.tasks),
    )


def _unique_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeepSWEQualityCohortError(f"corpus manifest repeats JSON key {key!r}")
        result[key] = value
    return result


def _raise_invalid_json_constant(value: str) -> Any:
    raise DeepSWEQualityCohortError(
        f"corpus manifest contains invalid JSON constant {value!r}"
    )


def _resolve_with_nonexistent_tail(value: Path | str, *, field: str) -> Path:
    """Resolve every existing symlink while retaining a nonexistent tail."""

    try:
        return Path(os.path.abspath(Path(value))).resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DeepSWEQualityCohortError(
            f"cannot resolve {field} through its existing parent chain: {exc}"
        ) from exc


def _require_outside_root(candidate: Path, root: Path, *, field: str) -> None:
    if candidate == root or root in candidate.parents:
        raise DeepSWEQualityCohortError(f"{field} must remain outside {root}")


def validate_plan_output_path(
    plan: QualityCohortPlan,
    plan_path: Path | str,
) -> Path:
    """Resolve a future plan file outside corpus and proposed cell roots.

    The returned canonical path is the only path the CLI writes. Existing
    symlink prefixes are therefore consumed before directory creation, and a
    nonexistent suffix cannot hide that the destination is below a forbidden
    root.
    """

    if type(plan) is not QualityCohortPlan:
        raise DeepSWEQualityCohortError("plan must be builder-produced")
    resolved = _resolve_with_nonexistent_tail(plan_path, field="plan output")
    corpus_root = plan.corpus.root.resolve(strict=True)
    proposed_root = _resolve_with_nonexistent_tail(
        plan.output_root, field="proposed output root"
    )
    _require_outside_root(resolved, corpus_root, field="plan output")
    _require_outside_root(resolved, proposed_root, field="plan output")
    return resolved


def _directory_open_flags() -> int:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise DeepSWEQualityCohortError(
            "descriptor-safe plan freezing is unavailable on this platform"
        )
    return os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _open_directory_chain(
    path: Path,
    *,
    create: bool,
    field: str,
) -> tuple[list[int], tuple[str, ...]]:
    """Retain a no-follow descriptor for every component of an absolute path."""

    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise DeepSWEQualityCohortError(f"{field} must be an absolute normalized path")
    components = tuple(path.parts[1:])
    if any(not item or item in {".", ".."} or os.sep in item for item in components):
        raise DeepSWEQualityCohortError(f"{field} contains an unsafe path component")
    flags = _directory_open_flags()
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(os.sep, flags))
        for component in components:
            parent = descriptors[-1]
            created = False
            try:
                child = os.open(component, flags, dir_fd=parent)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=parent)
                    created = True
                except FileExistsError:
                    # Another actor won the name. It is still admitted only by
                    # the no-follow open and identity checks used for existing
                    # components.
                    pass
                child = os.open(component, flags, dir_fd=parent)
            descriptors.append(child)
            opened = os.fstat(child)
            named = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(opened.st_mode) or _directory_identity(
                opened
            ) != _directory_identity(named):
                raise DeepSWEQualityCohortError(
                    f"{field} directory identity changed while being bound"
                )
            if created:
                os.fchmod(child, 0o700)
                if stat.S_IMODE(os.fstat(child).st_mode) != 0o700:
                    raise DeepSWEQualityCohortError(
                        f"{field} created directory mode is not private"
                    )
                os.fsync(child)
                os.fsync(parent)
        return descriptors, components
    except BaseException:
        while descriptors:
            os.close(descriptors.pop())
        raise


def _open_optional_directory(path: Path, *, field: str) -> list[int] | None:
    try:
        descriptors, _ = _open_directory_chain(path, create=False, field=field)
    except FileNotFoundError:
        return None
    return descriptors


def _verify_directory_chain(
    descriptors: list[int],
    components: tuple[str, ...],
    *,
    field: str,
) -> None:
    if len(descriptors) != len(components) + 1:
        raise DeepSWEQualityCohortError(f"{field} descriptor chain is incomplete")
    for index, component in enumerate(components):
        named = os.stat(
            component,
            dir_fd=descriptors[index],
            follow_symlinks=False,
        )
        held = os.fstat(descriptors[index + 1])
        if not stat.S_ISDIR(held.st_mode) or _directory_identity(
            named
        ) != _directory_identity(held):
            raise DeepSWEQualityCohortError(
                f"{field} directory identity changed after binding"
            )


def _require_descriptor_outside_roots(
    descriptor: int,
    forbidden: frozenset[tuple[int, int, int]],
    *,
    field: str,
) -> None:
    """Prove the held directory's live ``..`` ancestry omits forbidden roots."""

    flags = _directory_open_flags()
    current = os.open(".", flags, dir_fd=descriptor)
    try:
        for _ in range(4096):
            current_identity = _directory_identity(os.fstat(current))
            if current_identity in forbidden:
                raise DeepSWEQualityCohortError(
                    f"{field} descriptor resolves inside a forbidden root"
                )
            parent = os.open("..", flags, dir_fd=current)
            parent_identity = _directory_identity(os.fstat(parent))
            if parent_identity == current_identity:
                os.close(parent)
                return
            os.close(current)
            current = parent
    finally:
        os.close(current)
    raise DeepSWEQualityCohortError(f"{field} ancestry exceeds its safety bound")


def _write_plan_exclusive_at(
    parent_descriptor: int,
    leaf: str,
    payload: bytes,
    *,
    forbidden: frozenset[tuple[int, int, int]],
    verify_parent: Callable[[], None],
) -> None:
    if not leaf or leaf in {".", ".."} or os.sep in leaf:
        raise DeepSWEQualityCohortError("plan output leaf is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    created_identity: tuple[int, int, int] | None = None
    primary_error: BaseException | None = None
    try:
        verify_parent()
        _require_descriptor_outside_roots(
            parent_descriptor, forbidden, field="plan output parent"
        )
        descriptor = os.open(leaf, flags, 0o600, dir_fd=parent_descriptor)
        created = os.fstat(descriptor)
        created_identity = _directory_identity(created)
        os.fchmod(descriptor, 0o600)
        verify_parent()
        _require_descriptor_outside_roots(
            parent_descriptor, forbidden, field="plan output parent"
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DeepSWEQualityCohortError("exclusive plan write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        named = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or final.st_nlink != 1
            or _directory_identity(created) != _directory_identity(final)
            or _directory_identity(final) != _directory_identity(named)
            or final.st_size != len(payload)
            or stat.S_IMODE(final.st_mode) != 0o600
        ):
            raise DeepSWEQualityCohortError(
                "exclusive plan output identity, size, or mode is not exact"
            )
        _require_descriptor_outside_roots(
            parent_descriptor, forbidden, field="plan output parent"
        )
        os.fsync(parent_descriptor)
        verify_parent()
    except BaseException as exc:
        primary_error = exc

    if descriptor is not None:
        closing = descriptor
        descriptor = None
        try:
            os.close(closing)
        except OSError as exc:
            if primary_error is None:
                primary_error = exc

    if primary_error is not None and created_identity is not None:
        try:
            current = os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or _directory_identity(current) != created_identity
            ):
                raise DeepSWEQualityCohortError(
                    "failed plan output no longer names the created inode"
                )
            os.unlink(leaf, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            try:
                os.stat(leaf, dir_fd=parent_descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise DeepSWEQualityCohortError(
                    "failed plan output cleanup did not remove the name"
                )
        except BaseException as cleanup_error:
            failure = DeepSWEQualityCohortError(
                "failed plan output cleanup is uncertain"
            )
            failure.add_note(
                f"primary publication error: {type(primary_error).__name__}"
            )
            raise failure from cleanup_error
    if primary_error is not None:
        raise primary_error


def freeze_quality_cohort_plan(
    plan: QualityCohortPlan,
    plan_path: Path | str,
) -> Path:
    """Durably create a frozen plan through held no-follow descriptors.

    Path validation is repeated inside this sole publication helper. Every
    parent lookup and creation remains relative to a held descriptor, while
    live descriptor ancestry is checked against held corpus and existing
    proposed-output roots immediately around the byte write.
    """

    resolved = validate_plan_output_path(plan, plan_path)
    corpus_root = plan.corpus.root.resolve(strict=True)
    output_root = _resolve_with_nonexistent_tail(
        plan.output_root, field="proposed output root"
    )
    corpus_chain: list[int] = []
    output_chain: list[int] | None = None
    parent_chain: list[int] = []
    try:
        corpus_chain, corpus_components = _open_directory_chain(
            corpus_root,
            create=False,
            field="authenticated corpus root",
        )
        _verify_directory_chain(
            corpus_chain,
            corpus_components,
            field="authenticated corpus root",
        )
        output_chain = _open_optional_directory(
            output_root,
            field="proposed output root",
        )
        parent_chain, parent_components = _open_directory_chain(
            resolved.parent,
            create=True,
            field="plan output parent",
        )
        _verify_directory_chain(
            parent_chain,
            parent_components,
            field="plan output parent",
        )
        forbidden_identities = {_directory_identity(os.fstat(corpus_chain[-1]))}
        if output_chain is not None:
            forbidden_identities.add(_directory_identity(os.fstat(output_chain[-1])))

        def verify_parent() -> None:
            _verify_directory_chain(
                parent_chain,
                parent_components,
                field="plan output parent",
            )

        _write_plan_exclusive_at(
            parent_chain[-1],
            resolved.name,
            plan.to_json_bytes(),
            forbidden=frozenset(forbidden_identities),
            verify_parent=verify_parent,
        )
        return resolved
    except FileExistsError:
        raise
    except DeepSWEQualityCohortError:
        raise
    except OSError as exc:
        raise DeepSWEQualityCohortError(
            f"cannot safely freeze plan output: {type(exc).__name__}"
        ) from exc
    finally:
        for chain in (parent_chain, output_chain or [], corpus_chain):
            while chain:
                os.close(chain.pop())


def _proposed_cell_argv(
    *,
    matrix: QualityCohortMatrix,
    corpus: CorpusBinding,
    task: BoundTask,
    arm: QualityArm,
    root: Path,
) -> tuple[str, ...]:
    execution = matrix.execution
    budget = str(execution.per_cell_soft_budget_usd)
    # Keep the lexical project-venv entrypoint.  Resolving this final symlink
    # selects uv's managed base interpreter, which does not carry Crucible's
    # editable project installation and makes the otherwise exact argv
    # unexecutable.  Plans are already checkout/path-bound and must be rebuilt
    # after relocation, so binding this source checkout's venv is deliberate.
    project_python = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    try:
        resolved_python = project_python.resolve(strict=True)
        metadata = resolved_python.stat()
    except OSError as exc:
        raise DeepSWEQualityCohortError(
            "project virtual-environment Python is unavailable"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or not os.access(project_python, os.X_OK):
        raise DeepSWEQualityCohortError(
            "project virtual-environment Python is not executable"
        )
    return (
        str(project_python),
        "-m",
        "crucible.workflows",
        "deepswe",
        "--spec",
        str(root),
        "--dataset",
        str(corpus.manifest_path),
        "--task-id",
        task.task_id,
        "--backend-config",
        str(arm.routing_path),
        "--budget-usd",
        budget,
        "--budget-usd-per-instance",
        budget,
        "--max-turns",
        str(execution.max_turns),
        "--planner-count",
        str(execution.planner_count),
        "--backend-limit-mode",
        execution.backend_limit_mode,
        "--backend-limit-deadline-seconds",
        str(execution.backend_limit_deadline_seconds),
        "--planner-tool-call-limit",
        str(execution.planner_tool_call_limit),
        "--planner-tool-result-bytes",
        str(execution.planner_tool_result_bytes),
        "--max-patch-bytes",
        str(execution.max_patch_bytes),
        "--implement",
        "--lifecycle-out",
        str(root / "lifecycle.jsonl"),
    )


def build_quality_cohort_plan(
    matrix: QualityCohortMatrix,
    *,
    dataset_path: Path | str,
    output_root: Path | str,
) -> QualityCohortPlan:
    """Expand the authenticated matrix without executing a cell."""

    if type(matrix) is not QualityCohortMatrix:
        raise DeepSWEQualityCohortError("matrix must be loader-produced")
    canonical = load_quality_cohort_matrix(matrix.source_path)
    if matrix != canonical:
        raise DeepSWEQualityCohortError("caller matrix differs from source bytes")
    corpus = validate_quality_corpus(canonical, dataset_path)
    root = _resolve_with_nonexistent_tail(output_root, field="proposed output root")
    corpus_root = corpus.root.resolve(strict=True)
    _require_outside_root(root, corpus_root, field="output root")

    cells: list[QualityCell] = []
    first_block_terminal: str | None = None
    block_index = 0
    for task in corpus.tasks:
        for replicate, assignment in enumerate(task.replicate_orders, start=1):
            block_index += 1
            block_id = (
                f"quality-pilot.b{block_index:03d}.{task.task_id}.r{replicate:03d}"
            )
            predecessor: str | None = None
            for order_index, arm_key in enumerate(assignment, start=1):
                arm = canonical.arms[arm_key]
                cell_id = f"{block_id}.o{order_index:02d}.{arm_key}"
                cell_root = (
                    root
                    / "quality-pilot"
                    / f"block-{block_index:03d}"
                    / task.task_id
                    / f"replicate-{replicate:03d}"
                    / f"arm-{arm_key.lower()}"
                )
                dependencies: tuple[str, ...]
                if predecessor is not None:
                    dependencies = (predecessor,)
                elif block_index > 1:
                    assert first_block_terminal is not None
                    dependencies = (first_block_terminal,)
                else:
                    dependencies = ()
                cells.append(
                    QualityCell(
                        cell_id=cell_id,
                        block_id=block_id,
                        block_index=block_index,
                        task_id=task.task_id,
                        replicate=replicate,
                        assignment=assignment,
                        order_index=order_index,
                        arm=arm_key,
                        backend=arm.backend,
                        model=arm.model,
                        depends_on=dependencies,
                        root=cell_root,
                        proposed_argv=_proposed_cell_argv(
                            matrix=canonical,
                            corpus=corpus,
                            task=task,
                            arm=arm,
                            root=cell_root,
                        ),
                    )
                )
                predecessor = cell_id
            if block_index == 1:
                first_block_terminal = predecessor

    if block_index != 30 or len(cells) != 90:
        raise DeepSWEQualityCohortError("quality-pilot expansion count drifted")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise DeepSWEQualityCohortError("expanded cell identifiers collide")
    if len({cell.root for cell in cells}) != len(cells):
        raise DeepSWEQualityCohortError("expanded cell roots collide")
    return QualityCohortPlan(
        matrix=canonical,
        corpus=corpus,
        output_root=root,
        cells=tuple(cells),
    )


def validate_frozen_plan(
    plan_path: Path | str,
    *,
    expected: QualityCohortPlan,
) -> None:
    """Require one frozen plan file to equal the rebuilt canonical bytes."""

    if type(expected) is not QualityCohortPlan:
        raise DeepSWEQualityCohortError("expected plan must be builder-produced")
    path = Path(os.path.abspath(Path(plan_path)))
    observed = _read_regular(path, maximum=_MAX_MANIFEST_BYTES, field="frozen plan")
    if observed != expected.to_json_bytes():
        raise DeepSWEQualityCohortError(
            "frozen plan bytes differ from the rebuilt provider-free plan"
        )


__all__ = [
    "ANALYSIS_SCHEMA",
    "ARMS",
    "COHORT_ID",
    "EXCLUDED_REPOS",
    "EXCLUDED_TASK_IDS",
    "LANGUAGES",
    "MATRIX_SCHEMA",
    "ORDERS",
    "PLAN_SCHEMA",
    "QUALITY_PILOT_TASK_IDS",
    "SELECTION_SEED",
    "AnalysisPolicy",
    "BoundTask",
    "CorpusBinding",
    "CorpusIdentity",
    "DeepSWEQualityCohortError",
    "ExecutionPolicy",
    "QualityArm",
    "QualityCell",
    "QualityCohortMatrix",
    "QualityCohortPlan",
    "TaskDefinition",
    "build_quality_cohort_plan",
    "freeze_quality_cohort_plan",
    "load_quality_cohort_matrix",
    "validate_frozen_plan",
    "validate_plan_output_path",
    "validate_quality_corpus",
]
