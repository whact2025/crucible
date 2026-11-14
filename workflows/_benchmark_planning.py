# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bind benchmark plans to the exact planning recipe that produced them.

Generation policy is benchmark execution identity, separate from the reusable
goal's plan contract. A changed benchmark recipe requires a new accepted plan;
an old DAG cannot acquire current benchmark provenance by being reused.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import metadata
from pathlib import Path
from typing import Any

import assay
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.types.artifacts import LifecycleEvent
from assay.types.spec import Spec

from ._plan_provenance import _latest_plan_terminal, validate_plan_digest_format
from ._run import exec_ctx_now


BENCHMARK_PLANNING_BUDGET_SHARE = 0.40
_VERSION = "crucible.benchmark-planning.v1"
_EVENT = "benchmark_planning_policy"
_RECIPE_PATHS = (
    "common/cost.py",
    "common/policies.py",
    "common/policy_loader.py",
    "common/read_only_sources.py",
    "tools/_builder.py",
    "tools/_context.py",
    "tools/_dispatch_budget.py",
    "tools/_paths.py",
    "tools/source/__init__.py",
    "tools/source/handlers.py",
    "workflows/architect.py",
    "workflows/plan.py",
    "workflows/validate.py",
    "workflows/_analysis_handoff.py",
    "workflows/_atomic_persistence.py",
    "workflows/_backends.py",
    "workflows/_benchmark_planning.py",
    "workflows/_development_budget.py",
    "workflows/_diagnostic_cost.py",
    "workflows/_mutation_journal.py",
    "workflows/_outcome.py",
    "workflows/_plan_competition.py",
    "workflows/_plan_provenance.py",
    "workflows/_plan_selection.py",
    "workflows/_plan_source_snapshot.py",
    "workflows/_plan_validation.py",
    "workflows/_planner_session.py",
    "workflows/_readonly_scheduler.py",
    "workflows/_review.py",
    "workflows/_run.py",
    "workflows/_run_policy.py",
)


def _digest_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _assay_recipe_identity(digest_file: Callable[[Path], Any]) -> dict[str, Any]:
    """Bind the loaded package, whether editable, installed, or relocated."""

    source = inspect.getsourcefile(assay)
    if source is None:
        raise ValueError("imported Assay package source is unavailable")
    try:
        package_file = Path(source).resolve(strict=True)
    except OSError as exc:
        raise ValueError("imported Assay package source is unavailable") from exc
    if package_file.name != "__init__.py" or not package_file.is_file():
        raise ValueError("imported Assay package requires Python package source")
    root = package_file.parent
    modules = tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))
    # Missing or redirected source for an executing dependency must not turn
    # into a partial inventory. Resolve imported modules, not a sibling repo.
    for module_name in (
        "assay.canonical.serialize", "assay.canonical.registry",
        "assay.corpus.store", "assay.primitives.plan",
        "assay.primitives.review", "assay.primitives.validate",
        "assay.types.artifacts", "assay.types.dag", "assay.types.findings",
        "assay.types.policy", "assay.types.spec",
    ):
        module_source = inspect.getsourcefile(importlib.import_module(module_name))
        if module_source is None:
            raise ValueError(f"imported Assay source is unavailable: {module_name}")
        module_path = Path(module_source).resolve(strict=True)
        if not module_path.is_relative_to(root) or module_path not in modules:
            raise ValueError(f"imported Assay source is outside its package: {module_name}")
    try:
        distribution_version = metadata.version("assay")
    except metadata.PackageNotFoundError:
        # A source checkout need not have distribution metadata. The complete
        # loaded source tree remains mandatory and authoritative in that case.
        distribution_version = None
    return {
        "distribution": {"name": "assay", "version": distribution_version},
        "modules": {
            path.relative_to(root).as_posix(): digest_file(path) for path in modules
        },
    }


def planning_recipe_identity(
    crucible_root: Path | str,
    *,
    digest_file: Callable[[Path], Any],
) -> dict[str, Any]:
    """Inventory planning authority with the caller's checkpoint hash format."""

    root = Path(crucible_root)
    rules = tuple(sorted((root / "rules").glob("*.py")))
    if not rules:
        raise FileNotFoundError("planning validation rule source recipe is empty")
    return {
        "modules": {
            relative: digest_file(root / relative) for relative in _RECIPE_PATHS
        },
        "rules": {
            path.relative_to(root).as_posix(): digest_file(path) for path in rules
        },
        "assay": _assay_recipe_identity(digest_file),
    }


def _validate_policy(planner_count: int, share: float | None) -> None:
    if type(planner_count) is not int or not 1 <= planner_count <= 8:
        raise ValueError("planner_count must be an integer from 1 through 8")
    if share is not None and (
        type(share) not in (int, float)
        or not math.isfinite(share)
        or not 0 < share <= 1
    ):
        raise ValueError("planning_budget_share must be finite and in (0, 1]")


def benchmark_planning_identity(
    planner_count: int,
    *,
    planning_budget_share: float | None = BENCHMARK_PLANNING_BUDGET_SHARE,
) -> dict[str, Any]:
    """Return current semantics without the encompassing execution fingerprint.

    Checkpoints can include this identity in their execution fingerprint without
    creating a recursive fingerprint of themselves.
    """

    _validate_policy(planner_count, planning_budget_share)
    return {
        "version": _VERSION,
        "planner_count": planner_count,
        "mode": "single_track" if planner_count == 1 else "competition",
        "planning_budget_share": (
            float(planning_budget_share) if planning_budget_share is not None else None
        ),
        "planning_recipe": planning_recipe_identity(
            Path(__file__).resolve().parents[1], digest_file=_digest_file,
        ),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkPlanningPolicy:
    """One immutable execution's generation policy and source identity."""

    planner_count: int = 3
    planning_budget_share: float | None = BENCHMARK_PLANNING_BUDGET_SHARE
    execution_fingerprint: str | None = None
    _identity_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        identity = benchmark_planning_identity(
            self.planner_count, planning_budget_share=self.planning_budget_share,
        )
        if self.execution_fingerprint is not None:
            validate_plan_digest_format(self.execution_fingerprint)
        identity["execution_fingerprint"] = self.execution_fingerprint
        object.__setattr__(
            self, "_identity_json",
            json.dumps(identity, sort_keys=True, separators=(",", ":")),
        )

    def identity(self) -> dict[str, Any]:
        """Return a fresh copy of the recipe frozen before execution starts."""

        return json.loads(self._identity_json)


def _accepted_terminal(corpus: Corpus, spec: Spec) -> tuple[int, Any, LifecycleEvent]:
    if spec.tier_dag is None:
        raise ValueError("benchmark planning policy requires an accepted tier_dag")
    index, ref, event = _latest_plan_terminal(corpus, spec)
    if (
        event.kind != "planned"
        or len(event.affected_versions) != 1
        or type(event.affected_versions[0]) is not int
        or not 1 <= event.affected_versions[0] <= spec.version
    ):
        raise ValueError("benchmark planning policy requires a valid latest planned terminal")
    return index, ref, event


def _payload(spec: Spec, planned_ref: Any, policy: BenchmarkPlanningPolicy) -> dict[str, Any]:
    return {
        "schema_version": _VERSION,
        "planned_event_ref": str(planned_ref),
        "tier_dag_digest": str(hash_record(spec.tier_dag)),
        "policy": policy.identity(),
    }


def _bindings(corpus: Corpus, spec: Spec, terminal_index: int) -> list[LifecycleEvent]:
    events = []
    for ref in spec.lifecycle_event_refs[terminal_index + 1:]:
        event = corpus.get_record(ref, as_type=LifecycleEvent)
        if not isinstance(event, LifecycleEvent):
            raise ValueError("benchmark planning lifecycle record has the wrong type")
        if event.kind == _EVENT:
            events.append(event)
    return events


def benchmark_plan_matches_policy(
    corpus: Corpus, spec: Spec, policy: BenchmarkPlanningPolicy,
) -> bool:
    """Admit only an exact policy stamp on the current accepted plan terminal."""

    try:
        if planning_recipe_identity(
            Path(__file__).resolve().parents[1], digest_file=_digest_file,
        ) != policy.identity()["planning_recipe"]:
            return False
        index, planned_ref, planned = _accepted_terminal(corpus, spec)
        bindings = _bindings(corpus, spec, index)
        expected = _payload(spec, planned_ref, policy)
        return bool(bindings) and all(
            event.affected_versions == planned.affected_versions
            and json.loads(event.reason) == expected
            for event in bindings
        )
    except Exception:  # noqa: BLE001 - corrupt or missing authority fails closed
        return False


def _competition_proof(
    corpus: Corpus,
    spec: Spec,
    *,
    terminal_index: int,
    planned: LifecycleEvent,
    policy: BenchmarkPlanningPolicy,
    source_spec: Spec,
) -> None:
    events = []
    for ref in spec.lifecycle_event_refs[terminal_index + 1:]:
        event = corpus.get_record(ref, as_type=LifecycleEvent)
        if isinstance(event, LifecycleEvent) and event.kind == "plan_competition":
            events.append(event)
    if len(events) != 1 or events[0].affected_versions != planned.affected_versions:
        raise ValueError("accepted benchmark competition requires current competition evidence")
    try:
        binding = json.loads(events[0].reason)
        report = json.loads(corpus.get_artifact(binding["report_ref"]))
        if (
            report["schema_version"] != "crucible.plan-competition.v1"
            or type(report["planner_count"]) is not int
            or report["planner_count"] != policy.planner_count
            or report["status"] != "accepted"
            or report["accounting_complete"] is not True
            or binding["source_spec"] != str(hash_record(source_spec))
            or report["source_identity"]["spec"] != binding["source_spec"]
        ):
            raise ValueError("competition report does not establish this planning policy")
        modules = policy.identity()["planning_recipe"]["modules"]
        for name in (
            "_plan_competition.py", "_plan_selection.py", "_plan_source_snapshot.py",
        ):
            if report["implementation_sources"][name] != modules["workflows/" + name]:
                raise ValueError("competition report uses another planning implementation")
        candidates = report["candidates"]
        if (
            len(candidates) != policy.planner_count
            or len({row["id"] for row in candidates}) != policy.planner_count
        ):
            raise ValueError("competition report does not contain every independent candidate")
        comparison = report["comparison"]
        if comparison["mode"] == "select":
            candidates = [
                row for row in candidates
                if row["id"] == report["selected_id"] == comparison["selected_id"]
            ]
            if len(candidates) != 1:
                raise ValueError("competition report has no unique selected candidate")
            winner = candidates[0]
            if winner["digest"] != str(hash_record(spec.tier_dag)):
                raise ValueError("competition selected another DAG")
        elif comparison["mode"] == "synthesize":
            winner = report["synthesis"]
        else:
            raise ValueError("competition did not select or synthesize a plan")
        if winner["accepted"] is not True or winner["status"] != "accepted":
            raise ValueError("competition winner did not pass review and validation")
        # The serialized final candidate is authority, including synthesized
        # winners whose report intentionally carries no separate DAG digest.
        from .plan import _decode_review_candidate  # noqa: PLC0415

        final_dag = _decode_review_candidate(corpus.get_artifact(winner["candidate_ref"]))
        if hash_record(final_dag) != hash_record(spec.tier_dag):
            raise ValueError("competition final candidate differs from the accepted DAG")
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        raise ValueError("accepted benchmark competition evidence is malformed") from exc


def bind_benchmark_planning_policy(
    corpus: Corpus, staged_spec: Spec, policy: BenchmarkPlanningPolicy,
) -> Spec:
    """Stage one stamp after fresh acceptance; the caller owns publication."""

    index, planned_ref, planned = _accepted_terminal(corpus, staged_spec)
    current_recipe = planning_recipe_identity(
        Path(__file__).resolve().parents[1], digest_file=_digest_file,
    )
    if current_recipe != policy.identity()["planning_recipe"]:
        raise ValueError("benchmark planning source recipe changed during execution")
    bindings = _bindings(corpus, staged_spec, index)
    if bindings:
        if benchmark_plan_matches_policy(corpus, staged_spec, policy):
            return staged_spec
        raise ValueError("cannot replace benchmark planning policy without a fresh accepted plan")
    base = corpus.latest_spec()
    if (
        base is None
        or staged_spec.version != base.version + 1
        or staged_spec.parent != hash_record(base)
        or planned.affected_versions != (staged_spec.version,)
        or staged_spec.lifecycle_event_refs[:len(base.lifecycle_event_refs)]
        != base.lifecycle_event_refs
        or planned_ref in base.lifecycle_event_refs
    ):
        raise ValueError("benchmark planning policy can bind only a fresh staged accepted plan")
    if policy.planner_count > 1:
        _competition_proof(
            corpus, staged_spec, terminal_index=index, planned=planned,
            policy=policy, source_spec=base,
        )
    elif any(
        corpus.get_record(ref, as_type=LifecycleEvent).kind == "plan_competition"
        for ref in staged_spec.lifecycle_event_refs[index + 1:]
    ):
        raise ValueError("cannot bind a competition outcome as single-track planning")
    event = LifecycleEvent(
        kind=_EVENT,
        reason=json.dumps(_payload(staged_spec, planned_ref, policy), sort_keys=True),
        affected_versions=planned.affected_versions,
        execution=exec_ctx_now("benchmark_planning_policy"),
    )
    ref = corpus.put_record(event, kind="lifecycle_events")
    return replace(
        staged_spec, lifecycle_event_refs=(*staged_spec.lifecycle_event_refs, ref),
    )


__all__ = [
    "BENCHMARK_PLANNING_BUDGET_SHARE", "BenchmarkPlanningPolicy",
    "benchmark_planning_identity", "planning_recipe_identity",
    "benchmark_plan_matches_policy", "bind_benchmark_planning_policy",
]
