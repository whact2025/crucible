# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provenance binding for executable TierDAG reuse.

An accepted plan is derived state.  Reusing it is safe only while the
authoritative source semantics and the machinery that generated/adjudicated
the plan are unchanged.  This module records that relationship without
teaching Architect any product behavior.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, TypeGuard

import assay.primitives.plan as assay_plan_module
import crucible.rubrics as rubric_module
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.types.artifacts import LifecycleEvent

from . import _review as review_module
from ._run import exec_ctx_now


_PLAN_CONTRACT_SCHEMA_V2 = "crucible.plan-contract.v2"
_PLAN_CONTRACT_SCHEMA = "crucible.plan-contract.v3"
# This identity replaces a recursive hash of this source file.  Bump it for a
# semantic change to component selection, digest construction, or comparison.
PLAN_CONTRACT_FINGERPRINT_ALGORITHM = (
    "crucible.plan-contract-fingerprint.v3"
)
_PLAN_CONTRACT_EVENT = "plan_contract_bound"
_PLAN_CONTRACT_REASON_PREFIX = "accepted plan contract: "
_PLAN_TERMINALS = frozenset({
    "planned",
    "plan_rejected",
    "plan_inconclusive",
})


@dataclass(frozen=True, slots=True)
class PlanContractState:
    """Whether the current executable DAG has a reusable provenance bind."""

    status: Literal["absent", "fresh", "stale"]
    expected_fingerprint: str
    actual_fingerprint: str | None = None
    reason: str | None = None
    changed_components: tuple[str, ...] = ()


class PlanContractFingerprint(str):
    """String-compatible fingerprint carrying its v3 component digests.

    The string behavior preserves the public API used by existing workflow
    integrations.  The attached immutable mapping lets bind/state helpers
    persist and compare scoped components without a second computation or a
    process-global cache.
    """

    def __new__(
        cls,
        value: str,
        component_digests: Mapping[str, str],
    ) -> PlanContractFingerprint:
        instance = super().__new__(cls, value)
        instance._component_digests = MappingProxyType(  # type: ignore[attr-defined]
            dict(component_digests)
        )
        return instance

    @property
    def component_digests(self) -> Mapping[str, str]:
        return self._component_digests  # type: ignore[attr-defined]


def _file_digest(path: str | Path | None, *, label: str) -> str:
    if path is None:
        raise ValueError(f"{label} source path is unavailable")
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{label} source is unavailable: {exc}") from exc
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_digest(value: Any, *, label: str) -> str:
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise ValueError(f"{label} source identity is unavailable") from exc
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _source_file(value: Any, *, label: str) -> str:
    try:
        source_file = inspect.getsourcefile(value)
    except TypeError:
        source_file = None
    if source_file is None:
        module = inspect.getmodule(value)
        source_file = getattr(module, "__file__", None)
    if not isinstance(source_file, str) or not source_file:
        raise ValueError(
            f"{label} source identity is unavailable: module source path"
        )
    return source_file


def _callable_identity(
    callback: Callable[[Any], Any] | None,
) -> dict[str, Any]:
    if callback is None:
        return {
            "kind": "none",
            "module": None,
            "qualname": None,
            "module_digest": None,
            "source_digest": None,
            "class_digest": None,
            "call_digest": None,
        }

    if inspect.ismethod(callback):
        source_target = callback.__func__
        module = getattr(source_target, "__module__", None)
        qualname = getattr(source_target, "__qualname__", None)
        source_file = _source_file(source_target, label="candidate transform")
        return {
            "kind": "bound_method",
            "module": module,
            "qualname": qualname,
            "module_digest": _file_digest(
                source_file, label="candidate transform module",
            ),
            "source_digest": _source_digest(
                source_target, label="candidate transform",
            ),
            "class_digest": None,
            "call_digest": None,
        }

    if inspect.isfunction(callback):
        source_file = _source_file(callback, label="candidate transform")
        return {
            "kind": "function",
            "module": getattr(callback, "__module__", None),
            "qualname": getattr(callback, "__qualname__", None),
            "module_digest": _file_digest(
                source_file, label="candidate transform module",
            ),
            "source_digest": _source_digest(
                callback, label="candidate transform",
            ),
            "class_digest": None,
            "call_digest": None,
        }

    if not callable(callback):
        raise TypeError("candidate transform must be callable or None")
    callback_class = type(callback)
    call_method = getattr(callback_class, "__call__", None)
    if call_method is None:
        raise ValueError("candidate transform class has no source-visible __call__")
    source_file = _source_file(
        callback_class, label="candidate transform class",
    )
    return {
        "kind": "callable_object",
        "module": getattr(callback_class, "__module__", None),
        "qualname": getattr(callback_class, "__qualname__", None),
        "module_digest": _file_digest(
            source_file, label="candidate transform module",
        ),
        "source_digest": None,
        "class_digest": _source_digest(
            callback_class, label="candidate transform class",
        ),
        "call_digest": _source_digest(
            call_method, label="candidate transform __call__",
        ),
    }


def _workspace_head_identity(spec: Any, cwd: str | Path | None) -> dict | None:
    """Return the clean authority identity without hashing mutable dirt.

    The workspace may contain implementation changes from prior tiers, so the
    working tree itself is deliberately not part of plan reuse.  The checked
    out commit is authoritative and must exactly match the Spec's declared
    base before a plan can be bound or reused.
    """

    if cwd is None:
        return None
    from ._mutation_journal import (  # noqa: PLC0415
        MutationJournalError,
        _git_argv,
        _reject_git_redirect_env,
        _sanitized_git_env,
    )

    try:
        _reject_git_redirect_env()
    except MutationJournalError as exc:
        raise ValueError(str(exc)) from exc
    git_env = _sanitized_git_env()
    try:
        workspace = Path(cwd).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"planning workspace is unavailable: {exc}") from exc
    result = subprocess.run(
        _git_argv("rev-parse", "--verify", "HEAD^{commit}"),
        cwd=str(workspace),
        env=git_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise ValueError(
            "planning workspace HEAD is unavailable"
            + (f": {detail}" if detail else "")
        )
    head = result.stdout.strip().lower()
    expected = str(spec.project_context.base_commit).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,64}", expected):
        raise ValueError(
            f"Spec base_commit {expected or '<empty>'} is not a Git commit id"
        )
    declared = subprocess.run(
        _git_argv(
            "rev-parse", "--verify", f"{expected}^{{commit}}",
        ),
        cwd=str(workspace),
        env=git_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    declared_head = declared.stdout.strip().lower()
    if declared.returncode != 0 or not declared_head:
        raise ValueError(
            f"Spec base_commit {expected} is unavailable in planning workspace"
        )
    if not head or head != declared_head:
        raise ValueError(
            f"planning workspace HEAD {head or '<empty>'} does not match "
            f"Spec base_commit {expected} ({declared_head})"
        )
    return {"head": head}


def _component_digest(value: Any) -> str:
    """Return one canonical component digest."""

    return hash_record(value)


def _selected_rubric_component_digests(
    rubric: Any,
    *,
    label: str,
) -> dict[str, str]:
    name = getattr(rubric, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"selected {label} rubric has no stable name")
    meta = rubric_module.contract(name)
    return {
        f"review.{label}.rubric": _component_digest(rubric),
        f"review.{label}.contract": _component_digest(
            rubric_module.plan_review_contract_semantics(meta)
        ),
    }


def plan_contract_component_digests(
    spec: Any,
    *,
    critic_rubric: Any,
    judge_rubric: Any,
    candidate_transform: Callable[[Any], Any] | None,
    cwd: str | Path | None = None,
) -> dict[str, str]:
    """Return scoped v3 digests for the effective planning contract.

    Backend/model selection and turn budgets are deliberately excluded: they
    affect how a candidate is produced, not whether an already accepted DAG
    still expresses the same source contract.  The pinned project context,
    root semantics, planner implementation, reviewer contracts, and candidate
    transform are validity inputs and therefore invalidate reuse on change.

    Reviewer identity is limited to the two selected, fully loaded Rubrics and
    the effective semantic projections of their selected manifest contracts.
    Shared rubric inventory and non-planning review implementation are bound
    by explicit plan-review ABI revisions instead of whole-file hashes.
    """

    workflow_plan_path = Path(__file__).with_name("plan.py")
    components = {
        "authority.project_context": _component_digest(spec.project_context),
        "authority.intent": _component_digest(spec.intent),
        "authority.constraints": _component_digest([
            hash_record(item) for item in spec.constraints
        ]),
        "authority.done_criteria": _component_digest([
            hash_record(item) for item in spec.done_criteria
        ]),
        "authority.plan_policy": _component_digest(spec.plan_policy),
        "authority.workspace": _component_digest(
            _workspace_head_identity(spec, cwd)
        ),
        "planner.assay_module": _file_digest(
            inspect.getsourcefile(assay_plan_module),
            label="Assay planner",
        ),
        "planner.crucible_adapter": _file_digest(
            workflow_plan_path, label="Crucible plan adapter",
        ),
        **{
            f"planner.{name}": _file_digest(
                workflow_plan_path.with_name(f"_{name}.py"),
                label=f"Crucible {name} acceptance contract",
            )
            for name in ("plan_competition", "plan_selection", "plan_source_snapshot")
        },
        "planner.source_evidence": _file_digest(
            workflow_plan_path.parent.parent / "common" / "read_only_sources.py",
            label="Crucible source evidence capture",
        ),
        "planner.source_reference_rule": _file_digest(
            workflow_plan_path.parent.parent / "rules" / "read_only_references.py",
            label="Crucible source reference rule",
        ),
        "planner.candidate_transform": _component_digest(
            _callable_identity(candidate_transform)
        ),
        "review.runtime_abi": _component_digest({
            "rubric": rubric_module.PLAN_REVIEW_RUBRIC_RUNTIME_ABI,
            "adapter": review_module.PLAN_REVIEW_RUNTIME_ABI,
        }),
        "fingerprint.algorithm": _component_digest(
            PLAN_CONTRACT_FINGERPRINT_ALGORITHM
        ),
    }
    components.update(_selected_rubric_component_digests(
        critic_rubric, label="critic",
    ))
    components.update(_selected_rubric_component_digests(
        judge_rubric, label="judge",
    ))
    components.update(_selected_rubric_component_digests(
        rubric_module.load("plan_selection"), label="selection",
    ))
    return dict(sorted(components.items()))


def plan_contract_fingerprint(
    spec: Any,
    *,
    critic_rubric: Any,
    judge_rubric: Any,
    candidate_transform: Callable[[Any], Any] | None,
    cwd: str | Path | None = None,
    diagnostic_cost_policy: str | None = None,
) -> str:
    """Hash the scoped authority and planning/review contract for one DAG."""

    components = plan_contract_component_digests(
        spec,
        critic_rubric=critic_rubric,
        judge_rubric=judge_rubric,
        candidate_transform=candidate_transform,
        cwd=cwd,
    )
    if diagnostic_cost_policy is not None:
        from crucible.backends.cost_bounds import DIAGNOSTIC_COST_POLICY

        if diagnostic_cost_policy != DIAGNOSTIC_COST_POLICY:
            raise ValueError("unknown diagnostic plan cost policy")
        # A diagnostic plan must not become a reusable exact-accounting
        # contract merely because the source/rubric inputs are unchanged.
        components["diagnostic_cost_policy"] = str(hash_record(diagnostic_cost_policy))
    fingerprint = hash_record({
        "schema_version": _PLAN_CONTRACT_SCHEMA,
        "component_digests": components,
    })
    return PlanContractFingerprint(fingerprint, components)


def bind_plan_contract(
    corpus: Corpus,
    staged_spec: Any,
    *,
    fingerprint: str,
    component_digests: Mapping[str, str] | None = None,
) -> Any:
    """Idempotently bind an accepted DAG to its exact plan terminal.

    The Assay plan primitive already advanced the staged Spec exactly once.
    Use ``replace`` so this event participates in that same atomic publication
    rather than manufacturing an uncommitted intermediate version.
    """

    if staged_spec.tier_dag is None:
        raise ValueError("cannot bind plan provenance without a tier_dag")

    terminal_index, planned_ref, planned_event = _latest_plan_terminal(
        corpus, staged_spec,
    )
    if planned_event.kind != "planned":
        raise ValueError(
            "cannot bind plan provenance: latest plan terminal is "
            f"{planned_event.kind!r}"
        )
    if component_digests is None and isinstance(
        fingerprint, PlanContractFingerprint,
    ):
        component_digests = fingerprint.component_digests
    if component_digests is None:
        raise ValueError(
            "v3 plan provenance requires scoped component digests; use "
            "plan_contract_fingerprint() or pass component_digests explicitly"
        )
    normalized_components = _validated_component_digests(component_digests)
    computed_fingerprint = hash_record({
        "schema_version": _PLAN_CONTRACT_SCHEMA,
        "component_digests": normalized_components,
    })
    if str(fingerprint) != computed_fingerprint:
        raise ValueError(
            "plan provenance fingerprint does not match its component digests"
        )
    payload = {
        "component_digests": normalized_components,
        "fingerprint": str(fingerprint),
        "planned_event_ref": str(planned_ref),
        "schema_version": _PLAN_CONTRACT_SCHEMA,
        "tier_dag_digest": hash_record(staged_spec.tier_dag),
    }

    # Validation and later workflow records may advance the Spec without
    # changing its accepted DAG. Rebinding that same authority must not append
    # duplicate lifecycle evidence on every Architect invocation.
    for ref in reversed(staged_spec.lifecycle_event_refs[terminal_index + 1:]):
        event = _load_lifecycle_event(corpus, ref)
        if event.kind != _PLAN_CONTRACT_EVENT:
            continue
        if (
            _bound_payload(event) == payload
            and event.affected_versions == planned_event.affected_versions
        ):
            return staged_spec
        raise ValueError(
            "cannot replace plan provenance for the same accepted plan "
            "terminal; explicit plan reacceptance is required"
        )

    reason = _PLAN_CONTRACT_REASON_PREFIX + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    event = LifecycleEvent(
        kind=_PLAN_CONTRACT_EVENT,
        reason=reason,
        affected_versions=planned_event.affected_versions,
        execution=exec_ctx_now("plan_contract"),
    )
    event_ref = corpus.put_record(event, kind="lifecycle_events")
    return replace(
        staged_spec,
        lifecycle_event_refs=(
            *staged_spec.lifecycle_event_refs,
            event_ref,
        ),
    )


def _is_sha256(value: object) -> TypeGuard[str]:
    return bool(
        isinstance(value, str)
        and len(value) == len("sha256:") + 64
        and value.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in value[7:])
    )


def _validated_component_digests(
    value: Mapping[str, str] | object,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("plan provenance component_digests must be a mapping")
    normalized: dict[str, str] = {}
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or re.fullmatch(r"[a-z][a-z0-9_.-]*", key) is None
            or not _is_sha256(digest)
        ):
            raise ValueError(
                "plan provenance component_digests contains an invalid entry"
            )
        normalized[key] = digest
    return dict(sorted(normalized.items()))


def validate_plan_digest_format(digest: object) -> str:
    """Return one canonical plan digest or reject malformed authority."""

    if not _is_sha256(digest):
        raise ValueError(
            "plan digest must be lowercase sha256:<64 hex chars>"
        )
    return digest


def validate_reaccept_plan_digest(spec: Any, digest: object) -> str:
    """Validate explicit authority to reaccept one retained TierDAG.

    Reacceptance is intentionally value-bound rather than a boolean override:
    the operator must name the exact canonical DAG digest they inspected.
    This helper is side-effect free so callers can reject malformed or stale
    authority before opening backends or staging lifecycle records.
    """

    digest = validate_plan_digest_format(digest)
    if spec.tier_dag is None:
        raise ValueError(
            "cannot reaccept a plan because the Spec has no tier_dag"
        )
    actual = hash_record(spec.tier_dag)
    if digest != actual:
        raise ValueError(
            "reaccept plan digest does not match the retained tier_dag: "
            f"expected {actual}, got {digest}"
        )
    return digest


def stage_explicit_plan_reacceptance(
    corpus: Corpus,
    spec: Any,
    *,
    tier_dag_digest: object,
) -> Any:
    """Stage a new ``planned`` terminal for a digest-confirmed DAG.

    The returned next-version Spec is not published. Architect validates it,
    attaches a current provenance bind, and commits the whole draft only when
    deterministic validation passes. A later rejected/inconclusive plan
    terminal therefore cannot be bypassed accidentally, while a failed
    reacceptance attempt leaves the visible Spec lineage unchanged.
    """

    digest = validate_reaccept_plan_digest(spec, tier_dag_digest)
    event = LifecycleEvent(
        kind="planned",
        reason=(
            "plan explicitly reaccepted after exact tier_dag digest "
            f"confirmation: {digest}"
        ),
        affected_versions=(spec.version + 1,),
        execution=exec_ctx_now("plan_reaccept"),
    )
    event_ref = corpus.put_record(event, kind="lifecycle_events")
    return spec.with_appended_refs(lifecycle_event_refs=(event_ref,))


def _bound_payload(event: LifecycleEvent) -> dict[str, Any] | None:
    if event.kind != _PLAN_CONTRACT_EVENT:
        return None
    if not event.reason.startswith(_PLAN_CONTRACT_REASON_PREFIX):
        return None
    try:
        payload = json.loads(
            event.reason[len(_PLAN_CONTRACT_REASON_PREFIX):]
        )
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    common_keys = {
        "fingerprint",
        "planned_event_ref",
        "schema_version",
        "tier_dag_digest",
    }
    schema_version = payload.get("schema_version")
    if schema_version == _PLAN_CONTRACT_SCHEMA_V2:
        if set(payload) != common_keys:
            return None
    elif schema_version == _PLAN_CONTRACT_SCHEMA:
        if set(payload) != common_keys | {"component_digests"}:
            return None
        try:
            components = _validated_component_digests(
                payload.get("component_digests")
            )
        except ValueError:
            return None
        expected = hash_record({
            "schema_version": _PLAN_CONTRACT_SCHEMA,
            "component_digests": components,
        })
        if payload.get("fingerprint") != expected:
            return None
        payload["component_digests"] = components
    else:
        return None
    for key in ("fingerprint", "planned_event_ref", "tier_dag_digest"):
        value = payload.get(key)
        if not _is_sha256(value):
            return None
    return payload


def _load_lifecycle_event(corpus: Corpus, ref: Any) -> LifecycleEvent:
    try:
        event = corpus.get_record(ref, as_type=LifecycleEvent)
    except Exception as exc:  # noqa: BLE001 - fail closed with context
        raise ValueError(
            f"plan lifecycle record {ref!s} is unreadable"
        ) from exc
    if not isinstance(event, LifecycleEvent):
        raise ValueError(f"plan lifecycle record {ref!s} has the wrong type")
    return event


def _latest_plan_terminal(
    corpus: Corpus,
    spec: Any,
) -> tuple[int, Any, LifecycleEvent]:
    for index in range(len(spec.lifecycle_event_refs) - 1, -1, -1):
        ref = spec.lifecycle_event_refs[index]
        event = _load_lifecycle_event(corpus, ref)
        if event.kind in _PLAN_TERMINALS:
            return index, ref, event
    raise ValueError("tier_dag has no plan lifecycle terminal")


def accepted_plan_contract_state(
    corpus: Corpus,
    spec: Any,
    *,
    expected_fingerprint: str,
    expected_component_digests: Mapping[str, str] | None = None,
) -> PlanContractState:
    """Classify the latest plan terminal and its adjacent provenance bind."""

    if expected_component_digests is None and isinstance(
        expected_fingerprint, PlanContractFingerprint,
    ):
        expected_component_digests = expected_fingerprint.component_digests
    if expected_component_digests is not None:
        expected_component_digests = _validated_component_digests(
            expected_component_digests
        )
    expected_fingerprint = str(expected_fingerprint)
    if spec.tier_dag is None:
        return PlanContractState("absent", expected_fingerprint)

    terminal: tuple[int, Any, LifecycleEvent] | None = None
    corrupt_reason: str | None = None
    for index in range(len(spec.lifecycle_event_refs) - 1, -1, -1):
        ref = spec.lifecycle_event_refs[index]
        try:
            event = _load_lifecycle_event(corpus, ref)
        except ValueError as exc:
            corrupt_reason = str(exc)
            continue
        if event.kind in _PLAN_TERMINALS:
            terminal = (index, ref, event)
            break

    if terminal is None:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            reason=corrupt_reason or "tier_dag has no accepted plan lifecycle event",
        )
    terminal_index, planned_ref, planned_event = terminal
    if planned_event.kind != "planned":
        return PlanContractState("absent", expected_fingerprint)
    if corrupt_reason is not None:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            reason="plan lifecycle lineage contains an unreadable record",
        )

    binding_event: LifecycleEvent | None = None
    binding_payload: dict[str, Any] | None = None
    for ref in reversed(spec.lifecycle_event_refs[terminal_index + 1:]):
        try:
            event = _load_lifecycle_event(corpus, ref)
        except ValueError:
            return PlanContractState(
                "stale",
                expected_fingerprint,
                reason="plan lifecycle lineage contains an unreadable record",
            )
        if event.kind == _PLAN_CONTRACT_EVENT:
            binding_event = event
            binding_payload = _bound_payload(event)
            break

    if binding_event is None:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            reason="accepted plan predates provenance binding",
        )
    if binding_payload is None:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            reason="plan provenance event is malformed or unsupported",
        )
    actual = binding_payload["fingerprint"]
    if binding_payload["planned_event_ref"] != str(planned_ref):
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason="plan provenance event targets a different plan terminal",
        )
    if binding_event.affected_versions != planned_event.affected_versions:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason="plan provenance event is not bound to the accepted version",
        )
    if binding_payload["tier_dag_digest"] != hash_record(spec.tier_dag):
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason="accepted tier_dag differs from its provenance binding",
        )
    if binding_payload["schema_version"] == _PLAN_CONTRACT_SCHEMA_V2:
        mismatch = " and does not match the current contract" if (
            actual != expected_fingerprint
        ) else ""
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason=(
                "accepted plan uses legacy v2 provenance"
                f"{mismatch}; explicit reacceptance under v3 is required"
            ),
        )
    actual_components = binding_payload["component_digests"]
    changed_components: tuple[str, ...] = ()
    if expected_component_digests is not None:
        changed_components = tuple(sorted(
            key
            for key in set(actual_components) | set(expected_component_digests)
            if actual_components.get(key) != expected_component_digests.get(key)
        ))
    if actual != expected_fingerprint:
        component_detail = (
            "; changed components: " + ", ".join(changed_components)
            if changed_components
            else ""
        )
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason=(
                "root semantics or the planning/review contract changed"
                + component_detail
            ),
            changed_components=changed_components,
        )
    if changed_components:
        return PlanContractState(
            "stale",
            expected_fingerprint,
            actual_fingerprint=actual,
            reason=(
                "plan provenance component binding differs despite a matching "
                "aggregate fingerprint; changed components: "
                + ", ".join(changed_components)
            ),
            changed_components=changed_components,
        )
    return PlanContractState(
        "fresh",
        expected_fingerprint,
        actual_fingerprint=actual,
    )
