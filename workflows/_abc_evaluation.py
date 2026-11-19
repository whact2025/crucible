# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Frozen, fail-closed machinery for the E12 provider-lane comparison.

This module deliberately does not invoke a backend.  It prepares immutable
paired blocks, binds every execution identity needed by checkpoints, validates
offline result records, and produces repeated-measures statistics.  Live
workflow code may consume the prepared attempt manifests only after all three
arms pass preflight; route failure is an intention-to-treat result, never a
reason to select another backend.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from crucible.backends.config import (
    lint_routing_config,
    routing_feature_fingerprint,
    routing_fingerprint,
)
from crucible.common.benchmark_runtime import canonical_fingerprint, file_digest

from ._run import write_atomic_json


ARM_MANIFEST_SCHEMA = "crucible.e12.arms.v2"
COHORT_MANIFEST_SCHEMA = "crucible.e12.cohort.v2"
ATTEMPT_MANIFEST_SCHEMA = "crucible.e12.attempt.v2"
RESULT_SCHEMA = "crucible.e12.result.v1"
REPORT_SCHEMA = "crucible.e12.report.v1"

ArmId = Literal["A", "B", "C"]
Phase = Literal["synthetic", "pilot", "held_out"]
_ARMS: tuple[ArmId, ...] = ("A", "B", "C")
_ORDERS: tuple[str, ...] = ("ABC", "BCA", "CAB")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[^\s]+$")
_ENDPOINT_ENV_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_CODEX_CLI_ENDPOINT = "codex-cli://workspace"
_CODEX_CREDENTIAL_SOURCE = "chatgpt_session"
_CLAUDE_SESSION_ENDPOINT = "claude-cli://organization-session"
_CLAUDE_CREDENTIAL_SOURCE = "claude_session"
_CLAUDE_EXPECTED_ORG_ID = "76f2dd92-7d56-4550-afaf-abd363e3235c"
_CLAUDE_EXPECTED_ORG_NAME = "NVIDIA Corporation - Power Users"
_KIMI_ENDPOINT_TEMPLATE = "${KIMI_BASE_URL}"
_KIMI_API_KEY_TEMPLATE = "${MOONSHOT_API_KEY}"
_KIMI_WORKER_SOURCE = "project_locked"
_KIMI_ARCHITECT_PLANNER_MAX_RESPONSE_TOKENS = 16_384
_KIMI_DEEPSWE_IMPLEMENTER_TIMEOUT_SECONDS = 1_200
_HARD_SAFETY_METRICS = frozenset({
    "hidden_data_access",
    "capability_escalation",
    "shared_checkout_concurrent_writer",
    "scope_escape_surviving_rollback",
    "unresolved_mutation_journal",
    "candidate_verifier_hash_mismatch",
    "unaccounted_cost_or_missing_terminal_trace",
    "evidence_source_target_closure",
    "cleanup_or_recovery_failure",
})
_TERMINAL_STATUSES = frozenset({
    "resolved",
    "unresolved",
    "access_failed",
    "outage",
    "setup_failed",
    "timeout",
    "budget_exhausted",
    "cancelled",
    "transport_failed",
    "verification_failed",
    "infrastructure_failed",
})

# The lane identities are part of E12, not operator-selectable aliases.
_EXPECTED_LANES: Mapping[ArmId, tuple[str, str, str | None]] = {
    "A": ("codex", "gpt-5.6-sol", "max"),
    "B": (
        "claude",
        "claude-opus-5",
        "high",
    ),
    "C": ("kimi", "kimi-k3.1-pro", "max"),
}
_EXPECTED_QUALITY_SETTINGS: Mapping[ArmId, str] = {
    "A": "max",
    "B": "high",
    "C": "max",
}
_EXPECTED_ROLE_EFFORT_OVERRIDES: Mapping[
    tuple[str, ArmId],
    Mapping[str, str | None],
] = {
    # Keep the E12 lane's default at the provider-quality setting used by
    # implementation and Breaker roles while freezing DeepSWE's bounded
    # planning/review overlays.
    # No other product/arm may silently introduce role-specific effort drift.
    ("deepswe", "A"): {
        "architect_planner": "high",
        "architect_critic": "high",
        "architect_judge": "high",
    },
    ("deepswe", "B"): {
        "architect_planner": "medium",
        "architect_critic": "medium",
        "architect_judge": "medium",
    },
}


class E12ValidationError(ValueError):
    """An E12 manifest or result is unsafe, incomplete, or inconsistent."""


class E12PreflightError(E12ValidationError):
    """At least one exact arm is unavailable or not promotion-qualified."""

    def __init__(self, blockers: Mapping[str, Sequence[str]]) -> None:
        normalized = {
            arm: tuple(reasons)
            for arm, reasons in sorted(blockers.items())
            if reasons
        }
        self.blockers = normalized
        detail = "; ".join(
            f"{arm}: {', '.join(reasons)}"
            for arm, reasons in normalized.items()
        )
        super().__init__(f"E12 preflight blocked ({detail})")


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise E12ValidationError(f"{field} must be a mapping")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise E12ValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _digest(value: Any, *, field: str) -> str:
    text = _string(value, field=field)
    if _DIGEST_RE.fullmatch(text) is None:
        raise E12ValidationError(f"{field} must be sha256:<64 lowercase hex>")
    return text


def _revision(value: Any, *, field: str) -> str:
    text = _string(value, field=field)
    if _REVISION_RE.fullmatch(text) is None:
        raise E12ValidationError(f"{field} must contain no whitespace")
    return text


def _strict_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
    field: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise E12ValidationError(f"{field} is missing {sorted(missing)}")
    if unknown:
        raise E12ValidationError(f"{field} has unknown fields {sorted(unknown)}")


def _normalize_endpoint(value: str) -> str:
    endpoint = _string(value, field="endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme in {"http", "https"}:
        if (
            not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise E12ValidationError(
                "HTTP endpoint must have a host and contain no "
                "userinfo/query/fragment"
            )
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                "",
            )
        )
    # Provider-managed transports use stable URI-like identities such as
    # codex-cli://workspace.  They still must not contain credentials.
    if (
        any(marker in endpoint for marker in ("@", "?", "#"))
        or any(char.isspace() for char in endpoint)
    ):
        raise E12ValidationError("endpoint identity is malformed")
    return endpoint.rstrip("/")


def _endpoint_template(value: str, *, field: str) -> str:
    """Validate one secret-free HTTP endpoint literal or whole env reference."""

    template = _string(value, field=field)
    if _ENDPOINT_ENV_RE.fullmatch(template) is not None:
        return template
    if "${" in template or "$" in template:
        raise E12ValidationError(
            f"{field} must be a literal endpoint or one whole ${{ENV_VAR}}"
        )
    normalized = _normalize_endpoint(template)
    if urlsplit(normalized).scheme not in {"http", "https"}:
        raise E12ValidationError(f"{field} must be an HTTP(S) endpoint")
    return normalized


def _json_safe_dict(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise E12ValidationError(f"{field} is not canonical JSON") from exc
    if not isinstance(decoded, dict):
        raise E12ValidationError(f"{field} must encode a JSON object")
    return decoded


@dataclasses.dataclass(frozen=True)
class E12Lane:
    arm: ArmId
    backend: str
    model: str
    effort: str | None
    role_efforts: Mapping[str, str | None]
    quality_setting: str
    endpoint_template: str
    routing_path: Path
    roles: tuple[str, ...]
    promotion_gate: Literal["qualified", "blocked"]
    gate_ticket: str
    routing_digest: str
    routing_fingerprint: str
    feature_fingerprint: str

    def identity(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "backend": self.backend,
            "model": self.model,
            "effort": self.effort,
            "role_efforts": {
                role: self.role_efforts[role]
                for role in sorted(self.role_efforts)
            },
            "quality_setting": self.quality_setting,
            "endpoint_template": self.endpoint_template,
            "roles": list(self.roles),
            "promotion_gate": self.promotion_gate,
            "gate_ticket": self.gate_ticket,
            "routing_digest": self.routing_digest,
            "routing_fingerprint": self.routing_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
        }


def _resolve_lane_endpoint(lane: E12Lane) -> str:
    """Resolve only the endpoint named by a lane, never a credential."""

    if lane.backend == "codex":
        if lane.endpoint_template != _CODEX_CLI_ENDPOINT:
            raise E12ValidationError(
                f"Codex E12 endpoint must be exactly {_CODEX_CLI_ENDPOINT}"
            )
        return lane.endpoint_template
    if lane.backend == "claude":
        if lane.endpoint_template != _CLAUDE_SESSION_ENDPOINT:
            raise E12ValidationError(
                "Claude E12 endpoint must be exactly "
                f"{_CLAUDE_SESSION_ENDPOINT}"
            )
        return lane.endpoint_template
    match = _ENDPOINT_ENV_RE.fullmatch(lane.endpoint_template)
    if match is not None:
        variable = match.group(1)
        value = os.environ.get(variable)
        if value is None or not value.strip():
            raise E12ValidationError(
                f"endpoint template ${{{variable}}} is unset or empty"
            )
        endpoint = value
    else:
        endpoint = lane.endpoint_template
    normalized = _normalize_endpoint(endpoint)
    if urlsplit(normalized).scheme not in {"http", "https"}:
        raise E12ValidationError(
            f"arm {lane.arm} endpoint must resolve to HTTP(S)"
        )
    return normalized


@dataclasses.dataclass(frozen=True)
class E12ArmSet:
    workflow: Literal["swebench", "deepswe"]
    source_path: Path
    source_digest: str
    lanes: Mapping[ArmId, E12Lane]

    @property
    def digest(self) -> str:
        return canonical_fingerprint({
            "schema": ARM_MANIFEST_SCHEMA,
            "workflow": self.workflow,
            "source_digest": self.source_digest,
            "lanes": {
                arm: self.lanes[arm].identity()
                for arm in _ARMS
            },
        })


@dataclasses.dataclass(frozen=True)
class E12Qualification:
    available: bool
    promotable: bool
    requested_endpoint: str
    served_endpoint: str
    served_model: str
    requested_effort: str | None
    deployment_revision: str | None
    immutable_deployment_identity: bool
    adapter_version: str
    runtime_version: str
    pricing_digest: str
    feature_profile_digest: str
    routing_fingerprint: str
    conformance_digest: str
    capability_probe_digest: str
    prompt_profile_digest: str
    tool_profile_digest: str
    policy_profile_digest: str
    source_revisions: Mapping[str, str]
    blockers: tuple[str, ...] = ()
    cost_comparable: bool = False
    shadow_cost_contract_digest: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "available",
            "promotable",
            "immutable_deployment_identity",
            "cost_comparable",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise E12ValidationError(
                    f"qualification.{field_name} must be an exact boolean"
                )
        object.__setattr__(
            self,
            "requested_endpoint",
            _normalize_endpoint(self.requested_endpoint),
        )
        object.__setattr__(
            self,
            "served_endpoint",
            _normalize_endpoint(self.served_endpoint),
        )
        _string(self.served_model, field="served_model")
        if self.requested_effort is not None and (
            type(self.requested_effort) is not str
            or _string(self.requested_effort, field="requested_effort")
            != self.requested_effort
        ):
            raise E12ValidationError(
                "qualification.requested_effort must be null or an exact "
                "canonical string"
            )
        for name in (
            "pricing_digest",
            "feature_profile_digest",
            "routing_fingerprint",
            "conformance_digest",
            "capability_probe_digest",
            "prompt_profile_digest",
            "tool_profile_digest",
            "policy_profile_digest",
        ):
            _digest(getattr(self, name), field=name)
        _revision(self.adapter_version, field="adapter_version")
        _revision(self.runtime_version, field="runtime_version")
        if self.deployment_revision is not None:
            _revision(
                self.deployment_revision,
                field="deployment_revision",
            )
        revisions_raw = _mapping(
            self.source_revisions,
            field="qualification.source_revisions",
        )
        revisions = {
            _string(name, field="source revision name"): _revision(
                revision,
                field=f"source_revisions.{name}",
            )
            for name, revision in revisions_raw.items()
        }
        if not {"crucible", "assay"} <= set(revisions):
            raise E12ValidationError(
                "source_revisions must bind both crucible and assay"
            )
        object.__setattr__(self, "source_revisions", revisions)
        if (
            isinstance(self.blockers, (str, bytes))
            or not isinstance(self.blockers, Sequence)
        ):
            raise E12ValidationError(
                "qualification.blockers must be a sequence of strings"
            )
        object.__setattr__(self, "blockers", tuple(
            _string(item, field="blocker") for item in self.blockers
        ))
        if self.shadow_cost_contract_digest is not None:
            _digest(
                self.shadow_cost_contract_digest,
                field="shadow_cost_contract_digest",
            )

    def identity(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class E12CommonBinding:
    task_manifest_digest: str
    prompt_abi_digest: str
    semantic_prompt_digest: str
    schema_digest: str
    rubric_digest: str
    tool_schema_digest: str
    execution_policy_digest: str
    hidden_evaluation_policy_digest: str
    source_revisions: Mapping[str, str]
    pricing_policy_revision: str

    def __post_init__(self) -> None:
        for field in (
            "task_manifest_digest",
            "prompt_abi_digest",
            "semantic_prompt_digest",
            "schema_digest",
            "rubric_digest",
            "tool_schema_digest",
            "execution_policy_digest",
            "hidden_evaluation_policy_digest",
        ):
            _digest(getattr(self, field), field=field)
        revisions_raw = _mapping(
            self.source_revisions,
            field="common.source_revisions",
        )
        revisions = {
            _string(name, field="source revision name"): _revision(
                revision,
                field=f"source_revisions.{name}",
            )
            for name, revision in revisions_raw.items()
        }
        if not {"crucible", "assay"} <= set(revisions):
            raise E12ValidationError(
                "common source_revisions must bind crucible and assay"
            )
        object.__setattr__(self, "source_revisions", revisions)
        _revision(
            self.pricing_policy_revision,
            field="pricing_policy_revision",
        )

    def identity(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class E12Task:
    task_id: str
    task_digest: str
    image_digest: str
    base_commit: str
    workspace_fingerprint: str
    stratum: str
    replicates: int

    def __post_init__(self) -> None:
        _string(self.task_id, field="task_id")
        _digest(self.task_digest, field="task_digest")
        _digest(self.image_digest, field="image_digest")
        _revision(self.base_commit, field="base_commit")
        _digest(self.workspace_fingerprint, field="workspace_fingerprint")
        _string(self.stratum, field="stratum")
        if (
            isinstance(self.replicates, bool)
            or not isinstance(self.replicates, int)
            or self.replicates <= 0
        ):
            raise E12ValidationError("replicates must be a positive integer")

    def identity(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _validated_common(value: Any) -> E12CommonBinding:
    if type(value) is not E12CommonBinding:
        raise E12ValidationError(
            "common must be an exact E12CommonBinding"
        )
    try:
        return dataclasses.replace(value)
    except E12ValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise E12ValidationError(f"common binding is malformed: {exc}") from exc


def _validated_tasks(tasks: Sequence[E12Task]) -> tuple[E12Task, ...]:
    if isinstance(tasks, (str, bytes)) or not isinstance(tasks, Sequence):
        raise E12ValidationError("tasks must be a sequence")
    snapshots: list[E12Task] = []
    for index, task in enumerate(tasks):
        if type(task) is not E12Task:
            raise E12ValidationError(
                f"tasks[{index}] must be an exact E12Task"
            )
        try:
            snapshots.append(dataclasses.replace(task))
        except E12ValidationError:
            raise
        except (AttributeError, TypeError, ValueError) as exc:
            raise E12ValidationError(
                f"tasks[{index}] is malformed: {exc}"
            ) from exc
    return tuple(snapshots)


@dataclasses.dataclass(frozen=True)
class E12StateRoots:
    checkout: Path
    corpus: Path
    session: Path
    log: Path
    checkpoint: Path
    candidate: Path
    runtime: Path

    def identity(self) -> dict[str, str]:
        return {
            field.name: str(getattr(self, field.name).resolve())
            for field in dataclasses.fields(self)
        }

    def materialize(self) -> None:
        paths = [getattr(self, field.name) for field in dataclasses.fields(self)]
        normalized = [path.resolve() for path in paths]
        if len(normalized) != len(set(normalized)):
            raise E12ValidationError("state roots must be pairwise distinct")
        for path in paths:
            path.mkdir(parents=True, exist_ok=False, mode=0o700)


@dataclasses.dataclass(frozen=True)
class E12AttemptManifest:
    experiment: str
    cohort: str
    phase: Phase
    workflow: Literal["swebench", "deepswe"]
    task: E12Task
    replicate: int
    assignment: str
    order_index: int
    arm: ArmId
    lane: E12Lane
    qualification: E12Qualification
    common: E12CommonBinding
    state_roots: E12StateRoots
    arm_set_digest: str
    cohort_manifest_digest: str

    @property
    def execution_identity_digest(self) -> str:
        """Digest of every route/runtime/prompt/tool/policy execution input."""

        return canonical_fingerprint({
            "lane": self.lane.identity(),
            "qualification": self.qualification.identity(),
            "common": self.common.identity(),
            "arm_set_digest": self.arm_set_digest,
            "cohort_manifest_digest": self.cohort_manifest_digest,
        })

    def payload(self, *, include_digest: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": ATTEMPT_MANIFEST_SCHEMA,
            "experiment": self.experiment,
            "cohort": self.cohort,
            "phase": self.phase,
            "workflow": self.workflow,
            "task": self.task.identity(),
            "replicate": self.replicate,
            "assignment": self.assignment,
            "order_index": self.order_index,
            "arm": self.arm,
            "lane": self.lane.identity(),
            "qualification": self.qualification.identity(),
            "common": self.common.identity(),
            "state_roots": self.state_roots.identity(),
            "arm_set_digest": self.arm_set_digest,
            "cohort_manifest_digest": self.cohort_manifest_digest,
            "execution_identity_digest": self.execution_identity_digest,
        }
        if include_digest:
            payload["manifest_digest"] = canonical_fingerprint(payload)
        return payload

    @property
    def digest(self) -> str:
        return self.payload()["manifest_digest"]


@dataclasses.dataclass(frozen=True)
class E12FrozenCohort:
    manifest_path: Path
    manifest_digest: str
    attempts: tuple[E12AttemptManifest, ...]

    def by_key(self) -> dict[tuple[str, int, ArmId], E12AttemptManifest]:
        return {
            (attempt.task.task_id, attempt.replicate, attempt.arm): attempt
            for attempt in self.attempts
        }


def load_e12_arm_set(path: Path | str) -> E12ArmSet:
    """Resolve exactly A/B/C from immutable routing files without secrets."""

    import yaml  # type: ignore[import-untyped]

    source = Path(path).resolve()
    try:
        raw_bytes = source.read_bytes()
        doc = yaml.safe_load(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise E12ValidationError(f"invalid E12 arm manifest {source}: {exc}") from exc
    root = _mapping(doc, field="arm manifest")
    _strict_fields(
        root,
        required={"schema_version", "workflow", "roles", "arms"},
        field="arm manifest",
    )
    if root["schema_version"] != ARM_MANIFEST_SCHEMA:
        raise E12ValidationError("unsupported E12 arm manifest schema")
    workflow = root["workflow"]
    if workflow not in {"swebench", "deepswe"}:
        raise E12ValidationError("workflow must be swebench or deepswe")
    roles_raw = root["roles"]
    if (
        not isinstance(roles_raw, list)
        or not roles_raw
        or any(not isinstance(role, str) or not role for role in roles_raw)
    ):
        raise E12ValidationError("roles must be a non-empty string list")
    roles = tuple(roles_raw)
    if len(set(roles)) != len(roles):
        raise E12ValidationError("roles must not contain duplicates")

    arms_raw = _mapping(root["arms"], field="arms")
    if set(arms_raw) != set(_ARMS):
        raise E12ValidationError("E12 requires exactly arms A, B, and C")

    lanes: dict[ArmId, E12Lane] = {}
    for arm in _ARMS:
        arm_raw = _mapping(arms_raw[arm], field=f"arms.{arm}")
        _strict_fields(
            arm_raw,
            required={
                "backend",
                "model",
                "effort",
                "quality_setting",
                "routing",
                "promotion_gate",
                "gate_ticket",
            },
            optional={"role_efforts"},
            field=f"arms.{arm}",
        )
        backend = _string(arm_raw["backend"], field=f"arms.{arm}.backend")
        model = _string(arm_raw["model"], field=f"arms.{arm}.model")
        effort_raw = arm_raw["effort"]
        effort = (
            None
            if effort_raw is None
            else _string(effort_raw, field=f"arms.{arm}.effort")
        )
        if (backend, model, effort) != _EXPECTED_LANES[arm]:
            raise E12ValidationError(
                f"arm {arm} must be exact lane {_EXPECTED_LANES[arm]!r}"
            )
        role_efforts_raw = _mapping(
            arm_raw.get("role_efforts", {}),
            field=f"arms.{arm}.role_efforts",
        )
        role_efforts: dict[str, str | None] = {}
        for role, role_effort_raw in role_efforts_raw.items():
            if type(role) is not str or role not in roles:
                raise E12ValidationError(
                    f"arms.{arm}.role_efforts contains unknown role {role!r}"
                )
            role_efforts[role] = (
                None
                if role_effort_raw is None
                else _string(
                    role_effort_raw,
                    field=f"arms.{arm}.role_efforts.{role}",
                )
            )
        expected_role_efforts = dict(
            _EXPECTED_ROLE_EFFORT_OVERRIDES.get((workflow, arm), {})
        )
        if role_efforts != expected_role_efforts:
            raise E12ValidationError(
                f"arm {arm} role_efforts must be exact frozen overrides "
                f"{expected_role_efforts!r}"
            )
        gate = arm_raw["promotion_gate"]
        if gate not in {"qualified", "blocked"}:
            raise E12ValidationError(
                f"arms.{arm}.promotion_gate must be qualified or blocked"
            )
        route_rel = Path(
            _string(arm_raw["routing"], field=f"arms.{arm}.routing")
        )
        if route_rel.is_absolute() or ".." in route_rel.parts:
            raise E12ValidationError(
                f"arms.{arm}.routing must be a confined relative path"
            )
        route_path = (source.parent / route_rel).resolve()
        if source.parent not in route_path.parents:
            raise E12ValidationError(
                f"arms.{arm}.routing escapes the arm manifest directory"
            )
        try:
            routing = lint_routing_config(route_path)
        except (OSError, ValueError) as exc:
            raise E12ValidationError(
                f"arm {arm} routing file is invalid: {exc}"
            ) from exc
        endpoint_templates: set[str] = set()
        for role in roles:
            config = routing.for_role(role)
            observed = (config.backend, config.model, config.effort)
            expected_role_effort = role_efforts.get(role, effort)
            if observed != (backend, model, expected_role_effort):
                raise E12ValidationError(
                    f"arm {arm} role {role!r} drifts to {observed!r}"
                )
            if config.feature_admission != "enforce":
                raise E12ValidationError(
                    f"arm {arm} role {role!r} must enforce feature admission"
                )
            fallback_keys = {
                key for key in config.extra
                if "fallback" in key.lower()
            }
            if fallback_keys:
                raise E12ValidationError(
                    f"arm {arm} role {role!r} configures fallback fields "
                    f"{sorted(fallback_keys)}"
                )
            expected_extra: dict[str, object]
            if backend == "codex":
                if config.base_url is not None or config.api_key is not None:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must use the Codex CLI "
                        "workspace endpoint"
                    )
                expected_extra = {
                    "credential_source": _CODEX_CREDENTIAL_SOURCE,
                }
                if config.extra != expected_extra:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must configure exact "
                        f"Codex routing extra {expected_extra!r}"
                    )
                endpoint_templates.add(_CODEX_CLI_ENDPOINT)
            elif backend == "claude":
                if config.base_url is not None or config.api_key is not None:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must use the first-party "
                        "Claude organization session without base_url or "
                        "api_key"
                    )
                expected_extra = {
                    "credential_source": _CLAUDE_CREDENTIAL_SOURCE,
                    "expected_org_id": _CLAUDE_EXPECTED_ORG_ID,
                    "expected_org_name": _CLAUDE_EXPECTED_ORG_NAME,
                }
                if config.extra != expected_extra:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must configure exact "
                        f"Claude routing extra {expected_extra!r}"
                    )
                endpoint_templates.add(_CLAUDE_SESSION_ENDPOINT)
            elif backend == "kimi":
                if config.base_url != _KIMI_ENDPOINT_TEMPLATE:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must use exact direct "
                        f"Moonshot endpoint template {_KIMI_ENDPOINT_TEMPLATE}"
                    )
                if config.api_key != _KIMI_API_KEY_TEMPLATE:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must use exact direct "
                        f"Moonshot credential template {_KIMI_API_KEY_TEMPLATE}"
                    )
                expected_extra = {
                    "worker_source": _KIMI_WORKER_SOURCE,
                }
                if role == "architect_planner":
                    expected_extra["max_response_tokens"] = (
                        _KIMI_ARCHITECT_PLANNER_MAX_RESPONSE_TOKENS
                    )
                if workflow == "deepswe" and role == "implementer":
                    expected_extra["timeout_seconds"] = (
                        _KIMI_DEEPSWE_IMPLEMENTER_TIMEOUT_SECONDS
                    )
                if config.extra != expected_extra:
                    raise E12ValidationError(
                        f"arm {arm} role {role!r} must configure exact "
                        f"Kimi routing extra {expected_extra!r}"
                    )
                endpoint_templates.add(_KIMI_ENDPOINT_TEMPLATE)
            else:  # pragma: no cover - _EXPECTED_LANES is a closed set
                raise E12ValidationError(
                    f"arm {arm} uses unsupported backend {backend!r}"
                )
        if len(endpoint_templates) != 1:
            raise E12ValidationError(
                f"arm {arm} roles do not share one endpoint template"
            )
        endpoint_template = endpoint_templates.pop()
        route_identity = file_digest(route_path)
        if route_identity is None:
            raise E12ValidationError(f"arm {arm} routing file is missing")
        quality_setting = _string(
            arm_raw["quality_setting"],
            field=f"arms.{arm}.quality_setting",
        )
        if quality_setting != _EXPECTED_QUALITY_SETTINGS[arm]:
            raise E12ValidationError(
                f"arm {arm} quality_setting must be exact frozen value "
                f"{_EXPECTED_QUALITY_SETTINGS[arm]!r}"
            )
        lanes[arm] = E12Lane(
            arm=arm,
            backend=backend,
            model=model,
            effort=effort,
            role_efforts=role_efforts,
            quality_setting=quality_setting,
            endpoint_template=endpoint_template,
            routing_path=route_path,
            roles=roles,
            promotion_gate=gate,
            gate_ticket=_string(
                arm_raw["gate_ticket"],
                field=f"arms.{arm}.gate_ticket",
            ),
            routing_digest="sha256:" + str(route_identity["sha256"]),
            routing_fingerprint=routing_fingerprint(routing),
            feature_fingerprint=routing_feature_fingerprint(routing),
        )
    return E12ArmSet(
        workflow=workflow,
        source_path=source,
        source_digest="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
        lanes=lanes,
    )


def _canonical_arm_set(arm_set: E12ArmSet) -> E12ArmSet:
    """Reload and authenticate every caller-supplied arm-set field."""

    if type(arm_set) is not E12ArmSet:
        raise E12ValidationError("arm_set must be an exact E12ArmSet")
    if not isinstance(arm_set.source_path, Path):
        raise E12ValidationError("arm_set.source_path must be a Path")
    canonical = load_e12_arm_set(arm_set.source_path)
    mismatches: list[str] = []
    if arm_set.source_path != canonical.source_path:
        mismatches.append("source path")
    if arm_set.source_digest != canonical.source_digest:
        mismatches.append("source bytes/digest")
    if arm_set.workflow != canonical.workflow:
        mismatches.append("workflow")
    if type(arm_set.lanes) is not dict:
        raise E12ValidationError(
            "arm_set.lanes must be the loader-produced plain dict"
        )
    lane_keys = set(arm_set.lanes)
    if any(type(key) is not str for key in lane_keys):
        raise E12ValidationError(
            "arm_set.lanes keys must be exact strings"
        )
    if lane_keys != set(_ARMS):
        mismatches.append("lane inventory")
    else:
        for arm in _ARMS:
            lane = arm_set.lanes[arm]
            if type(lane) is not E12Lane:
                raise E12ValidationError(
                    f"arm_set.lanes[{arm!r}] must be an exact E12Lane"
                )
            expected = canonical.lanes[arm]
            for field in dataclasses.fields(E12Lane):
                if getattr(lane, field.name) != getattr(expected, field.name):
                    mismatches.append(f"arm {arm} {field.name}")
    if mismatches:
        raise E12ValidationError(
            "arm set differs from its canonical source: "
            + ", ".join(mismatches)
        )
    return canonical


def _validated_qualification(
    value: Any,
    *,
    arm: ArmId,
) -> E12Qualification:
    """Return an exact, revalidated snapshot of one qualification."""

    if type(value) is not E12Qualification:
        raise E12ValidationError(
            f"qualification {arm} must be an exact E12Qualification"
        )
    try:
        return dataclasses.replace(value)
    except E12ValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise E12ValidationError(
            f"qualification {arm} is malformed: {exc}"
        ) from exc


def _qualification_snapshots(
    qualifications: Mapping[str, E12Qualification],
) -> tuple[dict[ArmId, E12Qualification], set[str]]:
    if type(qualifications) is not dict:
        raise E12ValidationError(
            "qualifications must be a plain dict"
        )
    keys = set(qualifications)
    if any(type(key) is not str for key in keys):
        raise E12ValidationError(
            "qualification arm keys must be exact strings"
        )
    snapshots: dict[ArmId, E12Qualification] = {}
    for arm in _ARMS:
        if arm in keys:
            snapshots[arm] = _validated_qualification(
                qualifications[arm],
                arm=arm,
            )
    return snapshots, keys


def _lane_preflight_blockers(
    lane: E12Lane,
    qualification: E12Qualification,
) -> list[str]:
    blockers: list[str] = []
    if lane.promotion_gate != "qualified":
        blockers.append(f"{lane.gate_ticket} has not promoted this lane")
    try:
        resolved_endpoint = _resolve_lane_endpoint(lane)
    except E12ValidationError as exc:
        blockers.append(str(exc))
    else:
        if resolved_endpoint != qualification.requested_endpoint:
            blockers.append(
                "qualification requested endpoint differs from the "
                "resolved lane endpoint"
            )
    if not qualification.available:
        blockers.append("exact route is unavailable")
    if not qualification.promotable:
        blockers.append("qualification is diagnostic-only")
    if qualification.blockers:
        blockers.extend(qualification.blockers)
    if qualification.served_model != lane.model:
        blockers.append(
            f"served model {qualification.served_model!r} != "
            f"requested {lane.model!r}"
        )
    if qualification.requested_effort != lane.effort:
        blockers.append(
            f"qualified effort {qualification.requested_effort!r} != "
            f"requested {lane.effort!r}"
        )
    if qualification.requested_endpoint != qualification.served_endpoint:
        blockers.append("served endpoint differs from requested endpoint")
    if not qualification.immutable_deployment_identity:
        blockers.append("immutable deployment identity is unavailable")
    if qualification.deployment_revision is None:
        blockers.append("deployment revision is unavailable")
    if qualification.feature_profile_digest != lane.feature_fingerprint:
        blockers.append(
            "qualified feature profile differs from frozen routing profile"
        )
    if qualification.routing_fingerprint != lane.routing_fingerprint:
        blockers.append(
            "qualified route differs from frozen routing fingerprint"
        )
    if (
        lane.arm == "C"
        and qualification.cost_comparable
        and qualification.shadow_cost_contract_digest is None
    ):
        blockers.append(
            "Kimi comparable cost lacks an approved shadow-cost contract"
        )
    return blockers


def _preflight_canonical(
    arm_set: E12ArmSet,
    qualifications: Mapping[ArmId, E12Qualification],
    qualification_keys: set[str],
) -> None:
    blockers: dict[str, list[str]] = defaultdict(list)
    if qualification_keys != set(_ARMS):
        missing = set(_ARMS) - qualification_keys
        extra = qualification_keys - set(_ARMS)
        for arm in sorted(missing):
            blockers[arm].append("qualification evidence is missing")
        for unknown_arm in sorted(extra):
            blockers[unknown_arm].append("unknown arm")
    for arm in _ARMS:
        qualification = qualifications.get(arm)
        if qualification is None:
            lane = arm_set.lanes[arm]
            if lane.promotion_gate != "qualified":
                blockers[arm].append(
                    f"{lane.gate_ticket} has not promoted this lane"
                )
            continue
        lane_blockers = _lane_preflight_blockers(
            arm_set.lanes[arm],
            qualification,
        )
        if lane_blockers:
            blockers[arm].extend(lane_blockers)
    if blockers:
        raise E12PreflightError(blockers)


def preflight_e12(
    arm_set: E12ArmSet,
    qualifications: Mapping[str, E12Qualification],
) -> None:
    """Fail unless source-authenticated exact lanes are promotion-ready."""

    canonical = _canonical_arm_set(arm_set)
    snapshots, keys = _qualification_snapshots(qualifications)
    _preflight_canonical(canonical, snapshots, keys)


def counterbalanced_assignments(
    tasks: Sequence[E12Task],
    *,
    seed: str,
) -> dict[tuple[str, int], str]:
    """Return deterministic ABC/BCA/CAB orders for every paired block.

    Tasks are first placed in a seed-bound deterministic order.  Replicates
    then rotate, so every three-replicate pilot task sees all three orders and
    global order counts differ by at most one.
    """

    _string(seed, field="assignment seed")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise E12ValidationError("task ids must be unique")
    ordered = sorted(
        tasks,
        key=lambda task: hashlib.sha256(
            f"{seed}\0{task.task_id}".encode("utf-8")
        ).digest(),
    )
    assignments: dict[tuple[str, int], str] = {}
    for task_index, task in enumerate(ordered):
        for replicate in range(1, task.replicates + 1):
            assignments[(task.task_id, replicate)] = _ORDERS[
                (task_index + replicate - 1) % len(_ORDERS)
            ]
    counts = Counter(assignments.values())
    if counts and max(counts.values()) - min(counts.values()) > 1:
        raise AssertionError("counterbalanced assignment invariant failed")
    return assignments


def _validate_cohort_shape(
    workflow: str,
    phase: Phase,
    tasks: Sequence[E12Task],
) -> None:
    if not tasks:
        raise E12ValidationError("cohort must contain at least one task")
    if phase == "synthetic":
        return
    if phase == "pilot":
        minimum, maximum = (20, 30) if workflow == "swebench" else (10, 15)
        if not minimum <= len(tasks) <= maximum:
            raise E12ValidationError(
                f"{workflow} pilot requires {minimum}-{maximum} tasks"
            )
        if any(task.replicates != 3 for task in tasks):
            raise E12ValidationError("every pilot task requires three replicates")
        return
    if phase != "held_out":
        raise E12ValidationError(f"unknown phase {phase!r}")
    if len(tasks) < 100:
        raise E12ValidationError("held-out cohort requires at least 100 tasks")
    stability = [task for task in tasks if task.replicates == 3]
    invalid = [task for task in tasks if task.replicates not in {1, 3}]
    if invalid or len(stability) > 20:
        raise E12ValidationError(
            "held-out tasks require one replicate, with at most 20 "
            "predeclared three-replicate stability tasks"
        )


def _state_roots(
    root: Path,
    *,
    task_id: str,
    replicate: int,
    arm: ArmId,
) -> E12StateRoots:
    task_key = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    base = root / "attempts" / task_key / f"replicate-{replicate:03d}" / arm
    return E12StateRoots(
        checkout=base / "checkout",
        corpus=base / "corpus",
        session=base / "session",
        log=base / "log",
        checkpoint=base / "checkpoint",
        candidate=base / "candidate",
        runtime=base / "runtime",
    )


def freeze_e12_cohort(
    *,
    root: Path | str,
    arm_set: E12ArmSet,
    qualifications: Mapping[str, E12Qualification],
    common: E12CommonBinding,
    tasks: Sequence[E12Task],
    cohort: str,
    phase: Phase,
    assignment_seed: str,
    materialize_state_roots: bool = True,
) -> E12FrozenCohort:
    """Create one immutable cohort plus separate attempt/state manifests."""

    common = _validated_common(common)
    tasks = _validated_tasks(tasks)
    canonical_arm_set = _canonical_arm_set(arm_set)
    qualification_snapshots, qualification_keys = _qualification_snapshots(
        qualifications
    )
    _preflight_canonical(
        canonical_arm_set,
        qualification_snapshots,
        qualification_keys,
    )
    # All cohort identities below are built from the canonical reload and
    # revalidated qualification snapshots, never caller-owned replacements.
    arm_set = canonical_arm_set
    qualification_by_arm = qualification_snapshots
    cohort_name = _string(cohort, field="cohort")
    _validate_cohort_shape(arm_set.workflow, phase, tasks)
    if common.source_revisions != qualification_by_arm["A"].source_revisions:
        raise E12ValidationError(
            "common and Arm A source revisions differ"
        )
    for other_arm in _ARMS[1:]:
        if (
            qualification_by_arm[other_arm].source_revisions
            != common.source_revisions
        ):
            raise E12ValidationError(
                f"common and Arm {other_arm} source revisions differ"
            )
    assignments = counterbalanced_assignments(tasks, seed=assignment_seed)
    cohort_root = Path(root).resolve()
    cohort_payload = {
        "schema_version": COHORT_MANIFEST_SCHEMA,
        "experiment": "E12",
        "cohort": cohort_name,
        "phase": phase,
        "workflow": arm_set.workflow,
        "assignment_seed": assignment_seed,
        "arm_set_digest": arm_set.digest,
        "arm_manifest_source_digest": arm_set.source_digest,
        "arms": {
            arm: {
                "lane": arm_set.lanes[arm].identity(),
                "qualification": qualification_by_arm[arm].identity(),
            }
            for arm in _ARMS
        },
        "common": common.identity(),
        "tasks": [task.identity() for task in tasks],
        "assignments": [
            {
                "task_id": task_id,
                "replicate": replicate,
                "order": order,
            }
            for (task_id, replicate), order in sorted(assignments.items())
        ],
    }
    cohort_digest = canonical_fingerprint(cohort_payload)
    cohort_payload["manifest_digest"] = cohort_digest
    manifest_path = cohort_root / "cohort_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise E12ValidationError(
                "existing cohort manifest is unreadable"
            ) from exc
        if existing != cohort_payload:
            raise E12ValidationError(
                "cohort root is already bound to different frozen inputs"
            )
    else:
        write_atomic_json(manifest_path, cohort_payload)

    attempts: list[E12AttemptManifest] = []
    all_paths: set[Path] = set()
    tasks_by_id = {task.task_id: task for task in tasks}
    for (task_id, replicate), assignment in sorted(assignments.items()):
        task = tasks_by_id[task_id]
        for order_index, arm_text in enumerate(assignment):
            arm: ArmId = arm_text  # type: ignore[assignment]
            roots = _state_roots(
                cohort_root,
                task_id=task_id,
                replicate=replicate,
                arm=arm,
            )
            root_paths = {
                Path(path) for path in roots.identity().values()
            }
            if all_paths & root_paths:
                raise AssertionError("cross-arm state-root collision")
            all_paths.update(root_paths)
            if materialize_state_roots:
                roots.materialize()
            attempt = E12AttemptManifest(
                experiment="E12",
                cohort=cohort_name,
                phase=phase,
                workflow=arm_set.workflow,
                task=task,
                replicate=replicate,
                assignment=assignment,
                order_index=order_index,
                arm=arm,
                lane=arm_set.lanes[arm],
                qualification=qualification_by_arm[arm],
                common=common,
                state_roots=roots,
                arm_set_digest=arm_set.digest,
                cohort_manifest_digest=cohort_digest,
            )
            attempt_path = (
                roots.checkpoint.parent / "attempt_manifest.json"
            )
            if attempt_path.exists():
                raise E12ValidationError(
                    f"attempt manifest already exists: {attempt_path}"
                )
            write_atomic_json(attempt_path, attempt.payload())
            attempts.append(attempt)
    return E12FrozenCohort(
        manifest_path=manifest_path,
        manifest_digest=cohort_digest,
        attempts=tuple(attempts),
    )


E12AttemptKey = tuple[str, int, ArmId]


def _snapshot_e12_lane(value: Any, *, field: str) -> E12Lane:
    if type(value) is not E12Lane:
        raise E12ValidationError(f"{field} must be an exact E12Lane")
    if value.arm not in _ARMS:
        raise E12ValidationError(f"{field}.arm must be A, B, or C")
    for name in ("backend", "model", "quality_setting", "gate_ticket"):
        item = getattr(value, name)
        if type(item) is not str or _string(item, field=f"{field}.{name}") != item:
            raise E12ValidationError(
                f"{field}.{name} must be an exact canonical string"
            )
    if (
        value.effort is not None
        and (
            type(value.effort) is not str
            or _string(value.effort, field=f"{field}.effort") != value.effort
        )
    ):
        raise E12ValidationError(
            f"{field}.effort must be null or an exact canonical string"
        )
    if type(value.endpoint_template) is not str:
        raise E12ValidationError(
            f"{field}.endpoint_template must be an exact string"
        )
    _normalize_endpoint(value.endpoint_template)
    if not isinstance(value.routing_path, Path):
        raise E12ValidationError(f"{field}.routing_path must be a Path")
    if type(value.roles) is not tuple or any(
        type(role) is not str or not role for role in value.roles
    ):
        raise E12ValidationError(
            f"{field}.roles must be an exact tuple of strings"
        )
    if len(set(value.roles)) != len(value.roles):
        raise E12ValidationError(f"{field}.roles must not contain duplicates")
    if type(value.role_efforts) is not dict:
        raise E12ValidationError(
            f"{field}.role_efforts must be an exact mapping"
        )
    role_efforts: dict[str, str | None] = {}
    for role, role_effort in value.role_efforts.items():
        if type(role) is not str or role not in value.roles:
            raise E12ValidationError(
                f"{field}.role_efforts contains unknown role {role!r}"
            )
        if (
            role_effort is not None
            and (
                type(role_effort) is not str
                or _string(
                    role_effort,
                    field=f"{field}.role_efforts.{role}",
                )
                != role_effort
            )
        ):
            raise E12ValidationError(
                f"{field}.role_efforts.{role} must be null or an exact "
                "canonical string"
            )
        role_efforts[role] = role_effort
    if value.promotion_gate not in {"qualified", "blocked"}:
        raise E12ValidationError(
            f"{field}.promotion_gate must be qualified or blocked"
        )
    for name in (
        "routing_digest",
        "routing_fingerprint",
        "feature_fingerprint",
    ):
        _digest(getattr(value, name), field=f"{field}.{name}")
    return dataclasses.replace(
        value,
        roles=tuple(value.roles),
        role_efforts=role_efforts,
    )


def _snapshot_e12_state_roots(
    value: Any,
    *,
    field: str,
) -> E12StateRoots:
    if type(value) is not E12StateRoots:
        raise E12ValidationError(f"{field} must be an exact E12StateRoots")
    values: dict[str, Path] = {}
    for state_field in dataclasses.fields(E12StateRoots):
        path = getattr(value, state_field.name)
        if not isinstance(path, Path) or not path.is_absolute():
            raise E12ValidationError(
                f"{field}.{state_field.name} must be an absolute Path"
            )
        values[state_field.name] = Path(path)
    return E12StateRoots(**values)


def _snapshot_e12_attempt(
    value: Any,
    *,
    field: str,
) -> E12AttemptManifest:
    """Return a detached, revalidated snapshot of one attempt."""

    if type(value) is not E12AttemptManifest:
        raise E12ValidationError(
            f"{field} must be an exact E12AttemptManifest"
        )
    if value.experiment != "E12":
        raise E12ValidationError(f"{field}.experiment must be E12")
    if (
        type(value.cohort) is not str
        or _string(value.cohort, field=f"{field}.cohort") != value.cohort
    ):
        raise E12ValidationError(
            f"{field}.cohort must be an exact canonical string"
        )
    if value.phase not in {"synthetic", "pilot", "held_out"}:
        raise E12ValidationError(f"{field}.phase is invalid")
    if value.workflow not in {"swebench", "deepswe"}:
        raise E12ValidationError(f"{field}.workflow is invalid")
    if (
        type(value.replicate) is not int
        or value.replicate <= 0
        or type(value.order_index) is not int
        or value.order_index not in range(len(_ARMS))
    ):
        raise E12ValidationError(
            f"{field} replicate/order index is invalid"
        )
    if (
        type(value.assignment) is not str
        or value.assignment not in _ORDERS
        or value.arm not in _ARMS
        or value.assignment[value.order_index] != value.arm
    ):
        raise E12ValidationError(
            f"{field} assignment/arm identity is invalid"
        )
    _digest(value.arm_set_digest, field=f"{field}.arm_set_digest")
    _digest(
        value.cohort_manifest_digest,
        field=f"{field}.cohort_manifest_digest",
    )
    task = _validated_tasks((value.task,))[0]
    if value.replicate > task.replicates:
        raise E12ValidationError(
            f"{field}.replicate exceeds the task replicate count"
        )
    try:
        return dataclasses.replace(
            value,
            task=task,
            lane=_snapshot_e12_lane(value.lane, field=f"{field}.lane"),
            qualification=_validated_qualification(
                value.qualification,
                arm=value.arm,
            ),
            common=_validated_common(value.common),
            state_roots=_snapshot_e12_state_roots(
                value.state_roots,
                field=f"{field}.state_roots",
            ),
        )
    except E12ValidationError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise E12ValidationError(f"{field} is malformed: {exc}") from exc


def _cohort_manifest_tasks(value: Any) -> tuple[E12Task, ...]:
    if type(value) is not list:
        raise E12ValidationError(
            "frozen cohort manifest tasks must be an exact list"
        )
    tasks: list[E12Task] = []
    for index, item in enumerate(value):
        row = _mapping(item, field=f"cohort tasks[{index}]")
        _strict_fields(
            row,
            required={
                "task_id",
                "task_digest",
                "image_digest",
                "base_commit",
                "workspace_fingerprint",
                "stratum",
                "replicates",
            },
            field=f"cohort tasks[{index}]",
        )
        try:
            task = E12Task(
                task_id=row["task_id"],
                task_digest=row["task_digest"],
                image_digest=row["image_digest"],
                base_commit=row["base_commit"],
                workspace_fingerprint=row["workspace_fingerprint"],
                stratum=row["stratum"],
                replicates=row["replicates"],
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise E12ValidationError(
                f"cohort tasks[{index}] is malformed: {exc}"
            ) from exc
        if task.identity() != row:
            raise E12ValidationError(
                f"cohort tasks[{index}] is not canonical"
            )
        tasks.append(task)
    if len({task.task_id for task in tasks}) != len(tasks):
        raise E12ValidationError(
            "frozen cohort manifest contains duplicate task ids"
        )
    return tuple(tasks)


def _cohort_manifest_assignments(
    value: Any,
) -> dict[tuple[str, int], str]:
    if type(value) is not list:
        raise E12ValidationError(
            "frozen cohort manifest assignments must be an exact list"
        )
    assignments: dict[tuple[str, int], str] = {}
    for index, item in enumerate(value):
        row = _mapping(item, field=f"cohort assignments[{index}]")
        _strict_fields(
            row,
            required={"task_id", "replicate", "order"},
            field=f"cohort assignments[{index}]",
        )
        task_id = row["task_id"]
        replicate = row["replicate"]
        order = row["order"]
        if type(task_id) is not str or _string(
            task_id,
            field=f"cohort assignments[{index}].task_id",
        ) != task_id:
            raise E12ValidationError(
                f"cohort assignments[{index}].task_id is not canonical"
            )
        if type(replicate) is not int or replicate <= 0:
            raise E12ValidationError(
                f"cohort assignments[{index}].replicate is invalid"
            )
        if type(order) is not str or order not in _ORDERS:
            raise E12ValidationError(
                f"cohort assignments[{index}].order is invalid"
            )
        key = (task_id, replicate)
        if key in assignments:
            raise E12ValidationError(
                f"frozen cohort contains duplicate assignment {key!r}"
            )
        assignments[key] = order
    return assignments


def _authenticated_cohort_index(
    cohort: E12FrozenCohort,
) -> tuple[Path, str, dict[E12AttemptKey, E12AttemptManifest]]:
    """Authenticate the cohort manifest and snapshot its exact inventory."""

    if type(cohort) is not E12FrozenCohort:
        raise E12ValidationError(
            "cohort must be an exact E12FrozenCohort"
        )
    manifest_path = cohort.manifest_path
    if not isinstance(manifest_path, Path):
        raise E12ValidationError("cohort.manifest_path must be a Path")
    if (
        not manifest_path.is_absolute()
        or manifest_path.name != "cohort_manifest.json"
        or manifest_path != manifest_path.resolve()
    ):
        raise E12ValidationError(
            "cohort.manifest_path must be the canonical frozen path"
        )
    expected_digest = _digest(
        cohort.manifest_digest,
        field="cohort.manifest_digest",
    )
    attempts = cohort.attempts
    if type(attempts) is not tuple:
        raise E12ValidationError("cohort.attempts must be an exact tuple")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise E12ValidationError(
            "frozen cohort manifest is unreadable"
        ) from exc
    if type(manifest) is not dict:
        raise E12ValidationError("frozen cohort manifest must be an object")
    _strict_fields(
        manifest,
        required={
            "schema_version",
            "experiment",
            "cohort",
            "phase",
            "workflow",
            "assignment_seed",
            "arm_set_digest",
            "arm_manifest_source_digest",
            "arms",
            "common",
            "tasks",
            "assignments",
            "manifest_digest",
        },
        field="frozen cohort manifest",
    )
    unsigned_manifest = dict(manifest)
    stored_digest = _digest(
        unsigned_manifest.pop("manifest_digest"),
        field="cohort manifest digest",
    )
    if (
        stored_digest != expected_digest
        or canonical_fingerprint(unsigned_manifest) != stored_digest
    ):
        raise E12ValidationError(
            "frozen cohort manifest digest is unauthenticated"
        )
    if manifest["schema_version"] != COHORT_MANIFEST_SCHEMA:
        raise E12ValidationError("unsupported frozen cohort schema")
    if manifest["experiment"] != "E12":
        raise E12ValidationError("frozen cohort experiment must be E12")
    cohort_name = _string(manifest["cohort"], field="cohort manifest cohort")
    phase = manifest["phase"]
    if phase not in {"synthetic", "pilot", "held_out"}:
        raise E12ValidationError("frozen cohort phase is invalid")
    workflow = manifest["workflow"]
    if workflow not in {"swebench", "deepswe"}:
        raise E12ValidationError("frozen cohort workflow is invalid")
    assignment_seed = _string(
        manifest["assignment_seed"],
        field="cohort manifest assignment seed",
    )
    arm_set_digest = _digest(
        manifest["arm_set_digest"],
        field="cohort manifest arm-set digest",
    )
    _digest(
        manifest["arm_manifest_source_digest"],
        field="cohort manifest arm-source digest",
    )
    arms = _mapping(manifest["arms"], field="cohort manifest arms")
    if set(arms) != set(_ARMS):
        raise E12ValidationError(
            "frozen cohort manifest requires exactly arms A, B, and C"
        )
    arm_payloads: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for arm in _ARMS:
        arm_payload = _mapping(arms[arm], field=f"cohort arms.{arm}")
        _strict_fields(
            arm_payload,
            required={"lane", "qualification"},
            field=f"cohort arms.{arm}",
        )
        arm_payloads[arm] = (
            _mapping(arm_payload["lane"], field=f"cohort arms.{arm}.lane"),
            _mapping(
                arm_payload["qualification"],
                field=f"cohort arms.{arm}.qualification",
            ),
        )
    common_payload = _mapping(
        manifest["common"],
        field="cohort manifest common",
    )
    tasks = _cohort_manifest_tasks(manifest["tasks"])
    _validate_cohort_shape(workflow, phase, tasks)
    tasks_by_id = {task.task_id: task for task in tasks}
    assignments = _cohort_manifest_assignments(manifest["assignments"])
    expected_assignments = counterbalanced_assignments(
        tasks,
        seed=assignment_seed,
    )
    if assignments != expected_assignments:
        raise E12ValidationError(
            "frozen cohort assignments differ from task/seed declarations"
        )

    canonical_by_key: dict[E12AttemptKey, E12AttemptManifest] = {}
    for index, candidate in enumerate(attempts):
        snapshot = _snapshot_e12_attempt(
            candidate,
            field=f"cohort.attempts[{index}]",
        )
        key: E12AttemptKey = (
            snapshot.task.task_id,
            snapshot.replicate,
            snapshot.arm,
        )
        if key in canonical_by_key:
            raise E12ValidationError(
                f"frozen cohort contains duplicate attempt key {key!r}"
            )
        assignment = assignments.get((key[0], key[1]))
        task = tasks_by_id.get(key[0])
        expected_roots = _state_roots(
            manifest_path.parent,
            task_id=key[0],
            replicate=key[1],
            arm=key[2],
        )
        lane_payload, qualification_payload = arm_payloads[key[2]]
        mismatches: list[str] = []
        if snapshot.experiment != "E12":
            mismatches.append("experiment")
        if snapshot.cohort != cohort_name:
            mismatches.append("cohort")
        if snapshot.phase != phase:
            mismatches.append("phase")
        if snapshot.workflow != workflow:
            mismatches.append("workflow")
        if snapshot.arm_set_digest != arm_set_digest:
            mismatches.append("arm_set_digest")
        if snapshot.cohort_manifest_digest != stored_digest:
            mismatches.append("cohort_manifest_digest")
        if task is None or snapshot.task.identity() != task.identity():
            mismatches.append("task")
        if (
            assignment is None
            or snapshot.assignment != assignment
            or snapshot.order_index != assignment.index(snapshot.arm)
        ):
            mismatches.append("assignment")
        if _json_safe_dict(
            snapshot.lane.identity(),
            field=f"cohort.attempts[{index}].lane",
        ) != lane_payload:
            mismatches.append("lane")
        if _json_safe_dict(
            snapshot.qualification.identity(),
            field=f"cohort.attempts[{index}].qualification",
        ) != qualification_payload:
            mismatches.append("qualification")
        if _json_safe_dict(
            snapshot.common.identity(),
            field=f"cohort.attempts[{index}].common",
        ) != common_payload:
            mismatches.append("common")
        if snapshot.state_roots != expected_roots:
            mismatches.append("state_roots")
        if mismatches:
            raise E12ValidationError(
                "attempt differs from frozen cohort manifest: "
                + ", ".join(mismatches)
            )
        canonical_by_key[key] = snapshot

    expected_keys: set[E12AttemptKey] = {
        (task_id, replicate, arm)
        for task_id, replicate in assignments
        for arm in _ARMS
    }
    if set(canonical_by_key) != expected_keys:
        missing = sorted(expected_keys - set(canonical_by_key))
        extra = sorted(set(canonical_by_key) - expected_keys)
        raise E12ValidationError(
            "frozen cohort attempt inventory is inexact "
            f"(missing={missing!r}, extra={extra!r})"
        )
    return manifest_path.parent, stored_digest, canonical_by_key


def _authenticate_persisted_attempt(
    attempt: E12AttemptManifest,
    *,
    cohort_root: Path,
    cohort_manifest_digest: str,
) -> E12AttemptManifest:
    expected_roots = _state_roots(
        cohort_root,
        task_id=attempt.task.task_id,
        replicate=attempt.replicate,
        arm=attempt.arm,
    )
    attempt_path = expected_roots.checkpoint.parent / "attempt_manifest.json"
    try:
        stored_attempt = json.loads(
            attempt_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise E12ValidationError(
            f"frozen attempt manifest is unreadable: {attempt_path}"
        ) from exc
    canonical_payload = _json_safe_dict(
        attempt.payload(),
        field="canonical attempt payload",
    )
    if stored_attempt != canonical_payload:
        raise E12ValidationError(
            "canonical attempt differs from its frozen manifest"
        )
    if attempt.cohort_manifest_digest != cohort_manifest_digest:
        raise E12ValidationError(
            "canonical attempt has the wrong cohort manifest digest"
        )
    return attempt


def _authenticated_cohort_attempts(
    cohort: E12FrozenCohort,
) -> dict[E12AttemptKey, E12AttemptManifest]:
    """Authenticate all persisted attempts once and return detached snapshots."""

    cohort_root, stored_digest, canonical_by_key = (
        _authenticated_cohort_index(cohort)
    )
    return {
        key: _authenticate_persisted_attempt(
            attempt,
            cohort_root=cohort_root,
            cohort_manifest_digest=stored_digest,
        )
        for key, attempt in canonical_by_key.items()
    }


def _authenticated_cohort_attempt(
    cohort: E12FrozenCohort,
    attempt: E12AttemptManifest,
) -> E12AttemptManifest:
    """Authenticate a caller attempt against frozen in-memory/on-disk state."""

    cohort_root, stored_digest, canonical_by_key = (
        _authenticated_cohort_index(cohort)
    )
    supplied = _snapshot_e12_attempt(attempt, field="attempt")
    key: E12AttemptKey = (
        supplied.task.task_id,
        supplied.replicate,
        supplied.arm,
    )
    canonical = canonical_by_key.get(key)
    if canonical is None:
        raise E12ValidationError(
            "attempt is outside the exact frozen cohort"
        )
    field_mismatches = [
        field.name
        for field in dataclasses.fields(E12AttemptManifest)
        if getattr(supplied, field.name) != getattr(canonical, field.name)
    ]
    if field_mismatches or supplied.digest != canonical.digest:
        raise E12ValidationError(
            "attempt differs from the canonical frozen cohort: "
            + ", ".join(field_mismatches or ["manifest digest"])
        )
    return _authenticate_persisted_attempt(
        canonical,
        cohort_root=cohort_root,
        cohort_manifest_digest=stored_digest,
    )


def validate_e12_execution_boundary(
    attempt: E12AttemptManifest,
    *,
    cohort: E12FrozenCohort,
    arm_set: E12ArmSet,
) -> E12AttemptManifest:
    """Re-authenticate one attempt immediately before backend transport.

    E12's future live runner must call this at the last boundary before a
    request and execute from the returned snapshot.  That prevents source,
    routing, qualification, or endpoint-environment drift between cohort
    freeze and transport.
    """

    if type(attempt) is not E12AttemptManifest:
        raise E12ValidationError(
            "attempt must be an exact E12AttemptManifest"
        )
    canonical_attempt = _authenticated_cohort_attempt(cohort, attempt)
    canonical_arm_set = _canonical_arm_set(arm_set)
    if canonical_attempt.arm not in _ARMS:
        raise E12ValidationError("attempt arm must be A, B, or C")
    canonical_lane = canonical_arm_set.lanes[canonical_attempt.arm]
    if type(canonical_attempt.lane) is not E12Lane:
        raise E12ValidationError("attempt lane must be an exact E12Lane")
    lane_mismatches = [
        field.name
        for field in dataclasses.fields(E12Lane)
        if getattr(canonical_attempt.lane, field.name)
        != getattr(canonical_lane, field.name)
    ]
    if lane_mismatches:
        raise E12ValidationError(
            "attempt lane differs from its canonical source: "
            + ", ".join(lane_mismatches)
        )
    if canonical_attempt.workflow != canonical_arm_set.workflow:
        raise E12ValidationError(
            "attempt workflow differs from its canonical source"
        )
    if canonical_attempt.arm_set_digest != canonical_arm_set.digest:
        raise E12ValidationError(
            "attempt arm-set digest differs from its canonical source"
        )
    qualification = _validated_qualification(
        canonical_attempt.qualification,
        arm=canonical_attempt.arm,
    )
    blockers = _lane_preflight_blockers(canonical_lane, qualification)
    if blockers:
        raise E12PreflightError({canonical_attempt.arm: blockers})
    return canonical_attempt


@dataclasses.dataclass(frozen=True)
class E12Result:
    task_id: str
    replicate: int
    arm: ArmId
    assignment: str
    attempt_manifest_digest: str
    execution_identity_digest: str
    terminal_status: str
    terminal_event_count: int
    resolved: bool
    usage_complete: bool
    quota_complete: bool
    trace_complete: bool
    fallback_used: bool
    hard_safety_counts: Mapping[str, int]
    served_model: str
    served_endpoint: str
    deployment_revision: str | None
    wall_time_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    cost_comparable: bool
    diagnostic_only: bool

    def __post_init__(self) -> None:
        self._validate_scalars()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "E12Result":
        _strict_fields(
            value,
            required={
                "schema_version",
                "task_id",
                "replicate",
                "arm",
                "assignment",
                "attempt_manifest_digest",
                "execution_identity_digest",
                "terminal_status",
                "terminal_event_count",
                "resolved",
                "usage_complete",
                "quota_complete",
                "trace_complete",
                "fallback_used",
                "hard_safety_counts",
                "served_model",
                "served_endpoint",
                "deployment_revision",
                "wall_time_seconds",
                "input_tokens",
                "output_tokens",
                "cost_usd",
                "cost_comparable",
                "diagnostic_only",
            },
            field="result",
        )
        if value["schema_version"] != RESULT_SCHEMA:
            raise E12ValidationError("unsupported E12 result schema")
        arm = value["arm"]
        if arm not in _ARMS:
            raise E12ValidationError("result arm must be A, B, or C")
        result = cls(
            task_id=_string(value["task_id"], field="result.task_id"),
            replicate=value["replicate"],
            arm=arm,
            assignment=_string(
                value["assignment"],
                field="result.assignment",
            ),
            attempt_manifest_digest=_digest(
                value["attempt_manifest_digest"],
                field="result.attempt_manifest_digest",
            ),
            execution_identity_digest=_digest(
                value["execution_identity_digest"],
                field="result.execution_identity_digest",
            ),
            terminal_status=_string(
                value["terminal_status"],
                field="result.terminal_status",
            ),
            terminal_event_count=value["terminal_event_count"],
            resolved=value["resolved"],
            usage_complete=value["usage_complete"],
            quota_complete=value["quota_complete"],
            trace_complete=value["trace_complete"],
            fallback_used=value["fallback_used"],
            hard_safety_counts=_json_safe_dict(
                _mapping(
                    value["hard_safety_counts"],
                    field="result.hard_safety_counts",
                ),
                field="result.hard_safety_counts",
            ),
            served_model=_string(
                value["served_model"],
                field="result.served_model",
            ),
            served_endpoint=_normalize_endpoint(value["served_endpoint"]),
            deployment_revision=(
                None
                if value["deployment_revision"] is None
                else _revision(
                    value["deployment_revision"],
                    field="result.deployment_revision",
                )
            ),
            wall_time_seconds=value["wall_time_seconds"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            cost_usd=value["cost_usd"],
            cost_comparable=value["cost_comparable"],
            diagnostic_only=value["diagnostic_only"],
        )
        return result

    def _validate_scalars(self) -> None:
        if (
            isinstance(self.replicate, bool)
            or not isinstance(self.replicate, int)
            or self.replicate <= 0
        ):
            raise E12ValidationError("result replicate must be positive")
        if self.assignment not in _ORDERS:
            raise E12ValidationError("result assignment is not counterbalanced")
        if (
            isinstance(self.terminal_event_count, bool)
            or self.terminal_event_count != 1
        ):
            raise E12ValidationError("result must contain exactly one terminal")
        for field in (
            "resolved",
            "usage_complete",
            "quota_complete",
            "trace_complete",
            "fallback_used",
            "cost_comparable",
            "diagnostic_only",
        ):
            if not isinstance(getattr(self, field), bool):
                raise E12ValidationError(f"result.{field} must be boolean")
        if self.terminal_status not in _TERMINAL_STATUSES:
            raise E12ValidationError(
                f"unknown terminal status {self.terminal_status!r}"
            )
        if self.resolved != (self.terminal_status == "resolved"):
            raise E12ValidationError(
                "resolved flag and terminal_status=resolved must agree"
            )
        if not self.usage_complete and not self.quota_complete:
            raise E12ValidationError(
                "result requires complete usage or quota attribution"
            )
        if not self.trace_complete:
            raise E12ValidationError("result trace is incomplete")
        if self.fallback_used:
            raise E12ValidationError("fallback is forbidden in E12")
        if set(self.hard_safety_counts) != _HARD_SAFETY_METRICS:
            missing = sorted(
                _HARD_SAFETY_METRICS - set(self.hard_safety_counts)
            )
            extra = sorted(
                set(self.hard_safety_counts) - _HARD_SAFETY_METRICS
            )
            raise E12ValidationError(
                "hard safety inventory is incomplete or unknown "
                f"(missing={missing}, extra={extra})"
            )
        for name, count in self.hard_safety_counts.items():
            _string(name, field="hard safety metric")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise E12ValidationError(
                    "hard safety counts must be non-negative integers"
                )
            if count:
                raise E12ValidationError(
                    f"hard safety violation {name!r}={count}"
                )
        if (
            isinstance(self.wall_time_seconds, bool)
            or not isinstance(self.wall_time_seconds, (int, float))
            or not math.isfinite(self.wall_time_seconds)
            or self.wall_time_seconds < 0
        ):
            raise E12ValidationError("wall_time_seconds must be finite/non-negative")
        for field in ("input_tokens", "output_tokens"):
            value = getattr(self, field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise E12ValidationError(f"{field} must be non-negative or null")
        if self.usage_complete and (
            self.input_tokens is None or self.output_tokens is None
        ):
            raise E12ValidationError(
                "usage_complete requires input and output tokens"
            )
        if self.cost_usd is not None and (
            isinstance(self.cost_usd, bool)
            or not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise E12ValidationError("cost_usd must be finite/non-negative or null")
        if self.cost_comparable and self.cost_usd is None:
            raise E12ValidationError(
                "comparable cost requires an attributed USD value"
            )


def validate_e12_result(
    result: E12Result,
    attempt: E12AttemptManifest,
) -> None:
    """Validate one result and all response-time identity against its attempt."""

    if type(result) is not E12Result:
        raise E12ValidationError("result must be an exact E12Result")
    # Revalidate at every public admission boundary. The frozen dataclass can
    # still contain caller-owned Mapping values, so construction-time checks
    # alone are insufficient against later safety-inventory mutation.
    result._validate_scalars()

    expected = (
        attempt.task.task_id,
        attempt.replicate,
        attempt.arm,
        attempt.assignment,
        attempt.digest,
        attempt.execution_identity_digest,
    )
    observed = (
        result.task_id,
        result.replicate,
        result.arm,
        result.assignment,
        result.attempt_manifest_digest,
        result.execution_identity_digest,
    )
    if observed != expected:
        raise E12ValidationError(
            "result does not bind its exact task/replicate/arm/assignment manifest"
        )
    qualification = _validated_qualification(
        attempt.qualification,
        arm=attempt.arm,
    )
    drifts: list[str] = []
    if result.served_model != qualification.served_model:
        drifts.append("served model")
    if result.served_endpoint != qualification.served_endpoint:
        drifts.append("served endpoint")
    if result.deployment_revision != qualification.deployment_revision:
        drifts.append("deployment revision")
    if result.cost_comparable != qualification.cost_comparable:
        drifts.append("cost comparability policy")
    expected_diagnostic = (
        not qualification.immutable_deployment_identity
        or not qualification.promotable
    )
    if result.diagnostic_only != expected_diagnostic:
        drifts.append("promotion/diagnostic classification")
    if drifts:
        raise E12ValidationError(
            "response identity drift invalidates paired block: "
            + ", ".join(drifts)
        )


def ingest_e12_results(
    cohort: E12FrozenCohort,
    records: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = True,
) -> tuple[E12Result, ...]:
    """Validate records as a paired set; any arm drift rejects its whole block."""

    # Snapshot and authenticate the entire persisted inventory before reading
    # any result.  Incomplete admission must not let an unmentioned corrupt
    # attempt evade validation.
    expected = _authenticated_cohort_attempts(cohort)
    accepted: dict[tuple[str, int, ArmId], E12Result] = {}
    invalid_blocks: dict[tuple[str, int], list[str]] = defaultdict(list)
    for record in records:
        result = E12Result.from_mapping(record)
        key = (result.task_id, result.replicate, result.arm)
        if key in accepted:
            raise E12ValidationError(f"duplicate E12 result for {key!r}")
        attempt = expected.get(key)
        if attempt is None:
            raise E12ValidationError(f"result is outside frozen cohort: {key!r}")
        try:
            validate_e12_result(result, attempt)
        except E12ValidationError as exc:
            invalid_blocks[(result.task_id, result.replicate)].append(str(exc))
        else:
            accepted[key] = result
    if invalid_blocks:
        detail = "; ".join(
            f"{block!r}: {', '.join(errors)}"
            for block, errors in sorted(invalid_blocks.items())
        )
        raise E12ValidationError(f"paired block invalidation: {detail}")
    if require_complete and set(accepted) != set(expected):
        missing = sorted(set(expected) - set(accepted))
        raise E12ValidationError(
            f"E12 results are incomplete; missing {len(missing)} attempts"
        )
    return tuple(accepted[key] for key in sorted(accepted))


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise E12ValidationError("cannot summarize an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_bootstrap_interval(
    differences: Mapping[str, float],
    *,
    confidence: float,
    seed: str,
    samples: int,
) -> tuple[float, float]:
    tasks = sorted(differences)
    if not tasks:
        raise E12ValidationError("paired contrast has no tasks")
    if len(tasks) == 1:
        value = differences[tasks[0]]
        return value, value
    rng = random.Random(
        int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest(), "big")
    )
    estimates: list[float] = []
    for _ in range(samples):
        draw = [differences[rng.choice(tasks)] for _ in tasks]
        estimates.append(sum(draw) / len(draw))
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _paired_sign_flip_p(
    differences: Mapping[str, float],
    *,
    seed: str,
    samples: int = 20_000,
) -> float:
    values = list(differences.values())
    if not values or all(value == 0 for value in values):
        return 1.0
    observed = abs(sum(values) / len(values))
    extreme = 0
    if len(values) <= 16:
        total = 1 << len(values)
        for mask in range(total):
            estimate = sum(
                value if mask & (1 << index) else -value
                for index, value in enumerate(values)
            ) / len(values)
            extreme += abs(estimate) >= observed - 1e-15
        return extreme / total
    rng = random.Random(
        int.from_bytes(hashlib.sha256(seed.encode("utf-8")).digest(), "big")
    )
    for _ in range(samples):
        estimate = sum(
            value if rng.getrandbits(1) else -value
            for value in values
        ) / len(values)
        extreme += abs(estimate) >= observed - 1e-15
    return (extreme + 1) / (samples + 1)


def _holm_adjust(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def build_e12_report(
    cohort: E12FrozenCohort,
    results: Sequence[E12Result],
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Build ITT rates, paired clustered inference, and promotion diagnostics."""

    if bootstrap_samples < 1_000:
        raise E12ValidationError("bootstrap_samples must be at least 1000")
    expected = _authenticated_cohort_attempts(cohort)
    result_keys = [
        (result.task_id, result.replicate, result.arm)
        for result in results
    ]
    duplicate_keys = sorted(
        key
        for key, count in Counter(result_keys).items()
        if count > 1
    )
    if duplicate_keys:
        raise E12ValidationError(
            f"report contains duplicate result keys: {duplicate_keys!r}"
        )
    observed = dict(zip(result_keys, results, strict=True))
    if set(observed) != set(expected):
        raise E12ValidationError("report requires the complete frozen cohort")
    for key, result in observed.items():
        validate_e12_result(result, expected[key])

    by_task_arm: dict[tuple[str, ArmId], list[E12Result]] = defaultdict(list)
    by_arm: dict[ArmId, list[E12Result]] = defaultdict(list)
    validated_results = tuple(observed.values())
    for result in validated_results:
        by_task_arm[(result.task_id, result.arm)].append(result)
        by_arm[result.arm].append(result)
    task_ids = sorted({result.task_id for result in validated_results})
    task_rates: dict[ArmId, dict[str, float]] = {
        arm: {
            task_id: sum(
                1.0 if result.resolved else 0.0
                for result in by_task_arm[(task_id, arm)]
            ) / len(by_task_arm[(task_id, arm)])
            for task_id in task_ids
        }
        for arm in _ARMS
    }
    arm_summaries: dict[str, Any] = {}
    for arm in _ARMS:
        arm_results = by_arm[arm]
        comparable = all(result.cost_comparable for result in arm_results)
        resolved_count = sum(result.resolved for result in arm_results)
        costs = [
            float(result.cost_usd)
            for result in arm_results
            if result.cost_usd is not None
        ]
        arm_summaries[arm] = {
            "attempts": len(arm_results),
            "tasks": len(task_ids),
            "resolved_attempts": resolved_count,
            "itt_resolved_rate": resolved_count / len(arm_results),
            "task_mean_resolved_rate": (
                sum(task_rates[arm].values()) / len(task_ids)
            ),
            "full_task_p95_seconds": _percentile(
                [result.wall_time_seconds for result in arm_results],
                0.95,
            ),
            "total_input_tokens": (
                sum(result.input_tokens or 0 for result in arm_results)
                if all(result.usage_complete for result in arm_results)
                else None
            ),
            "total_output_tokens": (
                sum(result.output_tokens or 0 for result in arm_results)
                if all(result.usage_complete for result in arm_results)
                else None
            ),
            "cost_comparable": comparable,
            "total_comparable_cost_usd": sum(costs) if comparable else None,
            "comparable_cost_per_resolved": (
                sum(costs) / resolved_count
                if comparable and resolved_count
                else None
            ),
            "diagnostic_only": any(
                result.diagnostic_only for result in arm_results
            ),
        }

    raw_p: dict[str, float] = {}
    contrasts: dict[str, Any] = {}
    for arm in ("B", "C"):
        name = f"{arm}-A"
        differences = {
            task_id: task_rates[arm][task_id] - task_rates["A"][task_id]
            for task_id in task_ids
        }
        delta = sum(differences.values()) / len(differences)
        interval = _cluster_bootstrap_interval(
            differences,
            confidence=0.95,
            seed=f"{cohort.manifest_digest}:{name}:95",
            samples=bootstrap_samples,
        )
        simultaneous = _cluster_bootstrap_interval(
            differences,
            # Bonferroni simultaneous coverage for the two primary contrasts.
            confidence=0.975,
            seed=f"{cohort.manifest_digest}:{name}:simultaneous",
            samples=bootstrap_samples,
        )
        raw_p[name] = _paired_sign_flip_p(
            differences,
            seed=f"{cohort.manifest_digest}:{name}:sign-flip",
        )
        control = arm_summaries["A"]
        alternate = arm_summaries[arm]
        p95_gain = (
            1.0
            - alternate["full_task_p95_seconds"]
            / control["full_task_p95_seconds"]
            if control["full_task_p95_seconds"] > 0
            else None
        )
        control_cost = control["comparable_cost_per_resolved"]
        alternate_cost = alternate["comparable_cost_per_resolved"]
        cost_gain = (
            1.0 - alternate_cost / control_cost
            if (
                control_cost is not None
                and alternate_cost is not None
                and control_cost > 0
            )
            else None
        )
        contrasts[name] = {
            "paired_task_delta": delta,
            "cluster_bootstrap_95_interval": list(interval),
            "familywise_97_5_interval": list(simultaneous),
            "raw_p_value": raw_p[name],
            "p95_wall_time_reduction": p95_gain,
            "comparable_cost_per_resolved_reduction": cost_gain,
        }
    adjusted = _holm_adjust(raw_p)
    for name, value in adjusted.items():
        contrasts[name]["holm_adjusted_p_value"] = value

    held_out = (
        next(iter(expected.values())).phase == "held_out"
        and len(task_ids) >= 100
    )
    for arm in ("B", "C"):
        name = f"{arm}-A"
        contrast = contrasts[name]
        lower = contrast["familywise_97_5_interval"][0]
        quality = (
            contrast["paired_task_delta"] >= 0.05
            and lower > 0.0
            and contrast["holm_adjusted_p_value"] <= 0.05
        )
        noninferior = lower > -0.03
        efficient = (
            (
                contrast["comparable_cost_per_resolved_reduction"] is not None
                and contrast["comparable_cost_per_resolved_reduction"] >= 0.10
            )
            or (
                contrast["p95_wall_time_reduction"] is not None
                and contrast["p95_wall_time_reduction"] >= 0.15
            )
        )
        promotion_evidence = (
            held_out
            and not arm_summaries[arm]["diagnostic_only"]
            and (quality or (noninferior and efficient))
        )
        contrast["promotion"] = {
            "eligible_evidence": promotion_evidence,
            "quality_branch": quality,
            "noninferiority_branch": noninferior and efficient,
            "requires_held_out_100_tasks": not held_out,
        }

    payload = {
        "schema_version": REPORT_SCHEMA,
        "experiment": "E12",
        "cohort_manifest_digest": cohort.manifest_digest,
        "phase": next(iter(expected.values())).phase,
        "tasks": len(task_ids),
        "replicates_nested_within_task": True,
        "estimand": "intention_to_treat_hidden_resolved_pass_at_1",
        "arms": arm_summaries,
        "primary_contrasts": contrasts,
        "exploratory_contrast": "B-C",
        "bootstrap_samples": bootstrap_samples,
        "hard_safety_counts": {
            name: sum(
                result.hard_safety_counts.get(name, 0)
                for result in results
            )
            for name in sorted({
                key
                for result in results
                for key in result.hard_safety_counts
            })
        },
    }
    payload["report_digest"] = canonical_fingerprint(payload)
    return payload


__all__ = [
    "ARM_MANIFEST_SCHEMA",
    "ATTEMPT_MANIFEST_SCHEMA",
    "COHORT_MANIFEST_SCHEMA",
    "E12ArmSet",
    "E12AttemptManifest",
    "E12CommonBinding",
    "E12FrozenCohort",
    "E12Lane",
    "E12PreflightError",
    "E12Qualification",
    "E12Result",
    "E12StateRoots",
    "E12Task",
    "E12ValidationError",
    "REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "build_e12_report",
    "counterbalanced_assignments",
    "freeze_e12_cohort",
    "ingest_e12_results",
    "load_e12_arm_set",
    "preflight_e12",
    "validate_e12_execution_boundary",
    "validate_e12_result",
]
