# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Final plan bindings after an accepted mutating breach."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ContentHash,
    Evidence,
    ExecutionCompleteness,
    Finding,
    Implementation,
    InvocationResult,
    LifecycleEvent,
    ProjectContext,
    ReviewCoverage,
    Spec,
    Usage,
    UsagePricingEvidence,
    Verdict,
)
from assay.corpus.runs import RunResult
from assay.corpus.store import Corpus
from assay.primitives.implement import capture_workspace_delta
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy, ReviewPolicy
from crucible.tools._context import ToolContext
from crucible.tools.profiles import build_autonomous_implementer_profile
from crucible.workflows import _partition
from crucible.workflows._run import exec_ctx_now, open_run_with_capture
from crucible.workflows.tier import (
    _DEFAULT_TIER_RESPONSE_RETRY_BUDGET,
    _single_tier_execution_binding,
    _tier_tool_authority_digests,
)


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub",
    requested_model="stub-model",
    events=(
        BackendExecutionEvent(
            sequence=0,
            kind="terminal",
            name="end_turn",
        ),
    ),
)


_DEEPSWE_DEV_VERIFY_RUNNER = (
    "crucible.products.deepswe.runners.deepswe_dev_verify"
)
_SWEBENCH_DEV_VERIFY_RUNNER = (
    "crucible.products.swebench.runners.swebench_pytest"
)


def _contract_aware_build_verify_stub(callback):
    """Give deterministic gate fakes their exact synchronous boundary."""

    def settled(*args, **kwargs):
        outcome = callback(*args, **kwargs)
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_final_binding_gate_stub",
            succeeded=outcome.classification.startswith("success"),
        )
        return outcome

    return settled


def _settle_fake_breach_writer_epoch(argv) -> None:
    """Give the synchronous accepted-Breach fake its parent-family proof."""

    from crucible.workflows import _build_verify, _mutation_journal

    parent_id = argv[
        argv.index("--parent-mutation-transaction-id") + 1
    ]
    parent = _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
        parent_id
    )
    _build_verify._settle_no_process_writer_epoch(
        _mutation_journal.live_writer_epoch_family(parent),
        phase="test_final_binding_breach_child",
        succeeded=True,
    )


def test_aggregate_gate_preserves_only_unambiguous_canonical_slice_routing():
    first = Slice(
        id="python_slice",
        directive=ImplementationDirective(
            slice_id="python_slice",
            intent="qualify Python dispatch",
            files_in_scope=("release.py",),
            creates_paths=(),
            verify_mode="build_only",
            blast_radius={"max_files": 1, "languages": ["python"]},
        ),
    )
    single = Tier(tier_id="t4", slices=(first,), depends_on=())

    slice_id, directive = _single_tier_execution_binding(
        single,
        fallback_slice_id="t4__final",
    )

    assert slice_id == first.id
    assert directive is not None
    assert directive.slice_id == first.id
    assert directive.blast_radius == first.directive.blast_radius
    assert directive.verify_mode == "tests"

    bench_slice = Slice(
        id="bench_slice",
        directive=ImplementationDirective(
            slice_id="bench_slice",
            intent="retain benchmark dispatch",
            files_in_scope=("bench.rs",),
            creates_paths=(),
            verify_mode="bench",
            blast_radius={"max_files": 1, "languages": ["rust"]},
        ),
    )
    bench_id, bench_directive = _single_tier_execution_binding(
        Tier(tier_id="bench", slices=(bench_slice,), depends_on=()),
        fallback_slice_id="bench__final",
    )
    assert bench_id == "bench_slice"
    assert bench_directive is bench_slice.directive
    assert bench_directive.verify_mode == "bench"

    second = Slice(
        id="rust_slice",
        directive=ImplementationDirective(
            slice_id="rust_slice",
            intent="qualify Rust runtime",
            files_in_scope=("runtime.rs",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1, "languages": ["rust"]},
        ),
    )
    aggregate = Tier(tier_id="t4", slices=(first, second), depends_on=())

    assert _single_tier_execution_binding(
        aggregate,
        fallback_slice_id="t4__final",
    ) == ("t4__final", None)

def _tool_authority_for(repo: Path, tier: Tier, slice_id: str) -> str:
    context = ToolContext(workspace_root=repo, repo_root=repo)
    profile = build_autonomous_implementer_profile(context)
    return _tier_tool_authority_digests(
        context,
        tier,
        profile,
    )[slice_id]


class _UnusedBackend:
    name = "unused"
    model = "unused"

    def attach_toolset(self, toolset, *, context_budget=None):  # noqa: ARG002
        return None

    def close(self):
        return None


class _WritePlanBackend(_UnusedBackend):
    def __init__(self, repo: Path):
        self.repo = repo

    def invoke(self, _invocation):
        (self.repo / "a.py").write_text(
            "value = 'failed phase-one seed'\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="implemented a seed for deterministic verification",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _ReviewRetryBackend(_UnusedBackend):
    def __init__(self, repo: Path):
        self.repo = repo
        self.review_calls = 0
        self.implement_calls = 0

    def _write_review_retry(self) -> None:
        (self.repo / "a.py").write_text(
            "value = 'successful review retry'\n",
            encoding="utf-8",
        )

    def invoke(self, invocation):
        if "assay.review.v2" in (getattr(invocation, "system", "") or ""):
            self.review_calls += 1
            revise = self.review_calls == 1
            message = {
                "schema_version": "assay.review.v2",
                "decision": "revise" if revise else "pass",
                "reasoning": (
                    "The first reviewed postimage is incomplete."
                    if revise else "The retry completes the tier."
                ),
                "coverage": {
                    "complete": True,
                    "sources_reviewed": ["artifact"],
                    "omissions": [],
                },
                "findings": ([{
                    "id": "retry_required",
                    "criterion_id": "slice_delivery",
                    "severity": "High",
                    "confidence": 0.95,
                    "disposition": "confirmed",
                    "category": "partial_delivery",
                    "targets": [],
                    "evidence": [{
                        "source_id": "artifact",
                        "kind": "diff",
                        "location": "a.py",
                        "excerpt": "defender final",
                    }],
                    "proposal": {
                        "action": "Complete the reviewed implementation.",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }] if revise else []),
            }
        else:
            self.implement_calls += 1
            self._write_review_retry()
            message = "implemented review feedback"
        return InvocationResult(
            final_message=(
                json.dumps(message) if isinstance(message, dict) else message
            ),
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _AtomicReviewRetryBackend(_ReviewRetryBackend):
    """Publish the retry through the same inode-replacing shape as write_source."""

    def _write_review_retry(self) -> None:
        target = self.repo / "a.py"
        stage = self.repo / ".a.py.review-retry-stage"
        stage.write_text(
            "value = 'successful review retry'\n",
            encoding="utf-8",
        )
        stage.chmod(target.stat().st_mode)
        os.replace(stage, target)


class _FallbackReviewBackend(_ReviewRetryBackend):
    """Reject Defender and retry candidates, then accept restored preimage."""

    def invoke(self, invocation):
        if "assay.review.v2" in (getattr(invocation, "system", "") or ""):
            self.review_calls += 1
            revise = self.review_calls < 3
            message = {
                "schema_version": "assay.review.v2",
                "decision": "revise" if revise else "pass",
                "reasoning": (
                    "The amended candidate is rejected."
                    if revise else "The exact restored candidate is acceptable."
                ),
                "coverage": {
                    "complete": True,
                    "sources_reviewed": ["artifact"],
                    "omissions": [],
                },
                "findings": ([{
                    "id": "amendment_rejected",
                    "criterion_id": "slice_delivery",
                    "severity": "High",
                    "confidence": 0.99,
                    "disposition": "confirmed",
                    "category": "partial_delivery",
                    "targets": [],
                    "evidence": [{
                        "source_id": "artifact",
                        "kind": "diff",
                        "location": "a.py",
                        "excerpt": "a.py",
                    }],
                    "proposal": {
                        "action": "Use the original plan candidate.",
                        "replacement": None,
                    },
                    "attack_spec": None,
                }] if revise else []),
            }
            return InvocationResult(
                final_message=json.dumps(message),
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        self.implement_calls += 1
        (self.repo / "a.py").write_text(
            "value = 'rejected review amendment'\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="implemented rejected review amendment",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _AlwaysRejectReviewBackend(_FallbackReviewBackend):
    def invoke(self, invocation):
        if "assay.review.v2" not in (getattr(invocation, "system", "") or ""):
            return super().invoke(invocation)
        self.review_calls += 1
        message = {
            "schema_version": "assay.review.v2",
            "decision": "revise",
            "reasoning": "Every candidate remains incomplete.",
            "coverage": {
                "complete": True,
                "sources_reviewed": ["artifact"],
                "omissions": [],
            },
            "findings": [{
                "id": "still_rejected",
                "criterion_id": "slice_delivery",
                "severity": "High",
                "confidence": 0.99,
                "disposition": "confirmed",
                "category": "partial_delivery",
                "targets": [],
                "evidence": [{
                    "source_id": "artifact",
                    "kind": "diff",
                    "location": "a.py",
                    "excerpt": "a.py",
                }],
                "proposal": {
                    "action": "Do not accept this candidate.",
                    "replacement": None,
                },
                "attack_spec": None,
            }],
        }
        return InvocationResult(
            final_message=json.dumps(message),
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _FallbackInfrastructureFailureBackend(_FallbackReviewBackend):
    """Semantic rejection first; malformed responses on fallback review."""

    def invoke(self, invocation):
        is_review = "assay.review.v2" in (
            getattr(invocation, "system", "") or ""
        )
        if is_review and self.review_calls >= 2:
            self.review_calls += 1
            return InvocationResult(
                final_message="{}",
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        return super().invoke(invocation)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _setup(
    tmp_path: Path,
    *,
    verify_mode: str = "skip",
    runners: dict[str, str] | None = None,
    plan_owns_breach_test: bool = False,
    creates_only: bool = False,
    intent: str | None = None,
    include_successor: bool = False,
    seed_initial: bool = True,
    seed_initial_delete: bool = False,
    extra_creates_paths: tuple[str, ...] = (),
) -> tuple[Path, Corpus, Tier, Slice]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tier-binding@example.invalid")
    _git(repo, "config", "user.name", "Tier Binding")
    (repo / "a.py").write_text("value = 'base'\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")

    slice_ = Slice(
        id="plan_slice",
        directive=ImplementationDirective(
            slice_id="plan_slice",
            intent=(
                intent
                if intent is not None
                else ("create new.py" if creates_only else "update a.py")
            ),
            files_in_scope=(() if creates_only else ("a.py",)),
            creates_paths=tuple(dict.fromkeys((
                *(("new.py",) if creates_only else ()),
                *(("tests/test_breach.py",) if plan_owns_breach_test else ()),
                *extra_creates_paths,
            ))),
            verify_mode=verify_mode,
            blast_radius={
                "max_files": max(1, 1 + len(extra_creates_paths)),
                "languages": ["python"],
            },
        ),
    )
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    successor = Tier(
        tier_id="t1",
        slices=(Slice(
            id="successor_slice",
            directive=ImplementationDirective(
                slice_id="successor_slice",
                intent="consume the accepted predecessor",
                files_in_scope=("successor.py",),
                creates_paths=(),
                verify_mode="tests",
                blast_radius={"max_files": 1, "languages": ["python"]},
            ),
        ),),
        depends_on=("t0",),
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier-final-binding",
            repo_url="https://example.invalid/tier-final-binding.git",
            base_commit=base,
            branch=None,
            submodule_path=None,
            runners=(
                runners
                if runners is not None
                else {"verify": "crucible.common.runners.polyglot"}
            ),
        ),
        execution=None,
        intent="bind accepted breach postimage",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(
            tiers=(tier, successor) if include_successor else (tier,)
        ),
    )
    spec_dir = tmp_path / "corpus" / "spec"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)

    if seed_initial:
        # Seed the normal plan-bound implementation that tier will resume.
        plan_path = repo / ("new.py" if creates_only else "a.py")
        if seed_initial_delete:
            plan_path.unlink()
        else:
            plan_path.write_text("value = 'initial implementation'\n")
        diff, files, captured_base = capture_workspace_delta(
            cwd=str(repo),
            path_scope=tuple(dict.fromkeys((
                *slice_.directive.files_in_scope,
                *slice_.directive.creates_paths,
            ))),
        )
        initial = Implementation(
            slice_id=slice_.id,
            base_commit=captured_base,
            result_commit=None,
            diff_ref=corpus.put_artifact(
                diff, kind="diffs", name_hint="initial.diff",
            ),
            transcript_ref=corpus.put_artifact(
                b"[]", kind="transcripts", name_hint="initial.json",
            ),
            files_modified=files,
            classification="success",
            execution=exec_ctx_now("test"),
            scope_escapes=(),
            artifacts={},
        )
        initial = _partition.attach_resume_identity(
            corpus,
            spec,
            tier,
            slice_,
            initial,
            tool_authority_digest=_tool_authority_for(
                repo,
                tier,
                slice_.id,
            ),
        )
        corpus.commit(implementations=(initial,))
    return repo, corpus, tier, slice_


def _install_accepted_breach(
    monkeypatch,
    repo: Path,
    corpus: Corpus,
    *,
    final_value: str = "defender final",
    status: str = "proceed",
    remove_plan_path: str | None = None,
    candidate_mutation=None,
    candidate_scope: tuple[str, ...] | None = None,
    admitted_attack_ids: tuple[str, ...] = (),
    argv_sink: list[tuple[str, ...]] | None = None,
) -> None:
    def _accepted_breach(
        argv,
        *,
        fresh_lineage: bool,
        development_budget_ledger,
    ):
        assert fresh_lineage is False
        assert development_budget_ledger is None
        if argv_sink is not None:
            argv_sink.append(tuple(argv))
        with open_run_with_capture(corpus, "breach") as handle:
            if candidate_mutation is not None:
                candidate_mutation()
            elif remove_plan_path is not None:
                (repo / remove_plan_path).unlink()
            else:
                (repo / "a.py").write_text(f"value = '{final_value}'\n")
            breach_test = repo / "tests" / "test_breach.py"
            breach_test.parent.mkdir(parents=True, exist_ok=True)
            breach_test.write_text(
                "def test_defender_regression():\n"
                "    assert 'adversarial test survives final gate'\n",
                encoding="utf-8",
            )
            _settle_fake_breach_writer_epoch(argv)
            plan_scope = (
                candidate_scope
                if candidate_scope is not None
                else (remove_plan_path or "a.py",)
            )
            diff, files, base = capture_workspace_delta(
                cwd=str(repo), path_scope=plan_scope,
            )
            non_plan = Implementation(
                slice_id="breach_defender_r0",
                base_commit=base,
                result_commit=None,
                diff_ref=corpus.put_artifact(
                    diff, kind="diffs", name_hint="breach-final.diff",
                ),
                transcript_ref=corpus.put_artifact(
                    b"[]", kind="transcripts", name_hint="breach-final.json",
                ),
                files_modified=files,
                classification="success",
                execution=exec_ctx_now("test_breach"),
                scope_escapes=(),
                artifacts={},
            )
            committed = corpus.commit(implementations=(non_plan,))
            owned_implementation_ref = (
                committed.refs.implementation_refs[0]
            )
            handle.write_summary({
                "status": status,
                "spent_usd": 0.0,
                "costs": {"total_usd": 0.0},
                "implementation_refs": [str(owned_implementation_ref)],
                "rounds": ([{
                    "attack_ledger": [
                        {
                            "attack_id": attack_id,
                            "post_fix": {"status": "passed"},
                        }
                        for attack_id in admitted_attack_ids
                    ],
                }] if admitted_attack_ids else []),
                "test_file": str(breach_test),
            })
            handle.finalize(RunResult(status=status, cost_usd=0.0))
        return 0

    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        _accepted_breach,
    )


def _install_successful_final_gate(monkeypatch, *, on_gate=None):
    from crucible.workflows import tier as tier_workflow
    from crucible.common.runners.declared_verification import (
        accepted_verification_reference_payload,
        surface_from_directives,
    )
    from crucible.products.deepswe.runners import (
        deepswe_dev_verify as dev_runner,
    )
    from crucible.products.deepswe.runners.dev_verify_outcome import (
        parse_test_outcome,
    )
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    calls = []

    def _successful_final_gate(*args, **kwargs):  # noqa: ARG001
        calls.append(kwargs)
        if on_gate is not None:
            on_gate(kwargs)
        directives = tuple(kwargs.get("verification_directives") or ())
        if not directives:
            return BuildVerifyOutcome("success", build=None, verify=None)
        surface = surface_from_directives(directives)
        paths = tuple(dict.fromkeys(
            path
            for directive in directives
            for path in (
                *directive.files_in_scope,
                *directive.creates_paths,
            )
            if dev_runner._is_declared_native_test_path(path, "python")
        ))
        entries = []
        for path in paths:
            command = dev_runner._declared_test_probe_command(
                "candidate",
                path,
                language="python",
                parser_kind="pytest",
                metadata={},
                parameters={},
            )
            parsed = parse_test_outcome(
                language="python",
                command=command,
                actual_returncode=0,
                stdout="1 passed in 0.01s",
                stderr="",
                parser_hint="pytest",
            )
            entries.append((
                path,
                hashlib.sha256(command.encode()).hexdigest(),
                parsed.to_payload(),
            ))
        reference = accepted_verification_reference_payload(
            predecessor_surface=surface,
            language="python",
            parser_kind="pytest",
            outcomes=entries,
        )
        values = {
            "dev_verify_declared_test_files_required": float(len(entries)),
            "dev_verify_declared_test_files_executed": float(len(entries)),
            "dev_verify_declared_test_files_passed": float(len(entries)),
            "dev_verify_declared_test_files_failed": 0.0,
        }
        result = RunnerResult(
            role="dev_verify",
            runner=_DEEPSWE_DEV_VERIFY_RUNNER,
            returncode=0,
            values=values,
            raw_output=json.dumps({
                "values": values,
                "declared_verification_reference": reference,
            }).encode(),
            stderr=b"",
            duration_s=0.0,
        )
        return BuildVerifyOutcome(
            "success", build=None, verify=None, dev_verify=result
        )

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_successful_final_gate),
    )
    return calls


def _run_tier(
    repo: Path,
    corpus: Corpus,
    tier: Tier,
    *,
    skip_build: bool,
    skip_review: bool = True,
    review_policy: ReviewPolicy | None = None,
    breach_policy: BreachPolicy | None = None,
    extra_parameters: dict | None = None,
    build_retries: int = 2,
) -> int:
    from crucible.workflows.tier import run_tier

    return run_tier(
        spec=corpus.spec_dir,
        tier_id=tier.tier_id,
        cwd=str(repo),
        skip_build_verify=skip_build,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=skip_review,
        breach_policy=(
            breach_policy or BreachPolicy(enabled=True, max_rounds=1)
        ),
        review_policy=review_policy or ReviewPolicy(rubrics=()),
        extra_parameters=extra_parameters,
        build_retries=build_retries,
    )


def test_swebench_breach_binds_exact_tier_authority(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.products.swebench.spec_synthesis import (
        SWEBENCH_BREACH_POLICY,
    )
    from crucible.workflows.breach import breach_authority_digest

    repo, corpus, tier, slice_ = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    captured_argv: list[tuple[str, ...]] = []
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        argv_sink=captured_argv,
    )
    _install_successful_final_gate(monkeypatch)

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        breach_policy=SWEBENCH_BREACH_POLICY,
    )

    assert rc == 0
    assert len(captured_argv) == 1
    argv = captured_argv[0]
    authority_path = Path(argv[argv.index("--authority-file") + 1])
    authority_digest = argv[argv.index("--authority-digest") + 1]
    encoded = authority_path.read_bytes()
    assert breach_authority_digest(encoded) == authority_digest
    payload = json.loads(encoded)
    assert payload["schema_version"] == "crucible.breach-authority.v2"
    assert payload["tier_id"] == tier.tier_id
    assert payload["issue_text"] == "bind accepted breach postimage"
    assert payload["slices"] == [{
        "slice_id": slice_.id,
        "directive": {
            "slice_id": slice_.directive.slice_id,
            "intent": slice_.directive.intent,
            "files_in_scope": list(slice_.directive.files_in_scope),
            "creates_paths": list(slice_.directive.creates_paths),
            "verify_mode": slice_.directive.verify_mode,
                "blast_radius": {
                    "max_files": 1,
                    "languages": ["python"],
                },
                "mutation_regions": [],
                "requirement_refs": list(
                    slice_.directive.requirement_refs
                ),
            },
        }]


def test_deepswe_generic_breaker_binds_exact_tier_authority(
    tmp_path: Path,
    monkeypatch,
):
    """Authority is a policy capability, not a SWE-specific rubric check."""
    from crucible.products.deepswe.spec_synthesis import (
        DEEPSWE_BREACH_POLICY,
    )
    from crucible.workflows.breach import breach_authority_digest

    repo, corpus, tier, _slice = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    captured_argv: list[tuple[str, ...]] = []
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        argv_sink=captured_argv,
    )
    _install_successful_final_gate(monkeypatch)

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        breach_policy=DEEPSWE_BREACH_POLICY,
    )

    assert rc == 0
    argv = captured_argv[0]
    assert argv[argv.index("--breaker-rubric") + 1] == "impl_breaker"
    authority_path = Path(argv[argv.index("--authority-file") + 1])
    authority_digest = argv[argv.index("--authority-digest") + 1]
    encoded = authority_path.read_bytes()
    assert breach_authority_digest(encoded) == authority_digest
    payload = json.loads(encoded)
    assert payload["schema_version"] == "crucible.breach-authority.v2"
    assert payload["tier_id"] == tier.tier_id
    assert payload["issue_text"] == "bind accepted breach postimage"


def test_explicit_legacy_breach_opt_out_omits_authority_packet(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, _slice = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    captured_argv: list[tuple[str, ...]] = []
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        argv_sink=captured_argv,
    )
    _install_successful_final_gate(monkeypatch)

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        breach_policy=BreachPolicy(
            enabled=True,
            breaker_rubric="impl_breaker",
            judge_rubric="breach_judge",
            require_attack_authority=False,
        ),
    )

    assert rc == 0
    argv = captured_argv[0]
    assert "--authority-file" not in argv
    assert "--authority-digest" not in argv


def test_breach_disabled_container_tier_runs_cumulative_final_gate(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER},
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    calls = _install_successful_final_gate(monkeypatch)

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    )

    assert rc == 0
    assert [call["slice_id"] for call in calls] == [slice_.id]
    assert calls[0]["gate"] == "dev_verify"
    assert calls[0]["slice_directive"].slice_id == slice_.id
    assert calls[0]["slice_directive"].verify_mode == "tests"
    assert calls[0]["verification_directives"] == (slice_.directive,)

    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_cumulative_verification"] == {
        "classification": "success",
        "directive_count": 1,
        "scope_escapes": [],
        "integrity_error": None,
        "status": "passed",
    }


def test_swebench_container_without_checkpoint_capability_lands_cleanly(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _build_verify
    from crucible.workflows import tier as tier_workflow

    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _SWEBENCH_DEV_VERIFY_RUNNER},
        include_successor=True,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    calls = []

    def swebench_gate(*_args, **kwargs):
        calls.append(kwargs)
        values = {"tests_passed": 1.0, "tests_failed": 0.0}
        result = _build_verify.RunnerResult(
            role="dev_verify",
            runner=_SWEBENCH_DEV_VERIFY_RUNNER,
            returncode=0,
            values=values,
            raw_output=json.dumps({
                "values": values,
                "raw_output": "1 passed",
            }).encode(),
            stderr=b"",
            duration_s=0.0,
        )
        return _build_verify.BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=result,
        )

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(swebench_gate),
    )

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "swebench_metadata": {
                "container_id": "authoring-container",
                "repo": "example/project",
                "version": "1.0",
            },
        },
    )

    assert rc == 0
    assert [call["slice_id"] for call in calls] == [slice_.id]
    assert calls[0]["gate"] == "dev_verify"
    assert calls[0]["slice_directive"].slice_id == slice_.id
    assert calls[0]["verification_directives"] == ()
    assert calls[0]["accepted_verification_reference"] is None
    latest = corpus.latest_spec()
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=latest,
    )
    assert not pending
    assert all(
        _build_verify.ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT
        not in implementation.artifacts
        for implementation in landed.values()
    )
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_cumulative_verification"]["directive_count"] == 0
    assert summary["status"] == "complete"


def test_deepswe_checkpoint_capability_requires_complete_runner_reference(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _build_verify
    from crucible.workflows import tier as tier_workflow

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER},
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )

    def incomplete_gate(*_args, **_kwargs):
        values = {
            "dev_verify_declared_test_files_required": 0.0,
            "dev_verify_declared_test_files_executed": 0.0,
            "dev_verify_declared_test_files_passed": 0.0,
            "dev_verify_declared_test_files_failed": 0.0,
        }
        result = _build_verify.RunnerResult(
            role="dev_verify",
            runner=_DEEPSWE_DEV_VERIFY_RUNNER,
            returncode=0,
            values=values,
            raw_output=json.dumps({"values": values}).encode(),
            stderr=b"",
            duration_s=0.0,
        )
        return _build_verify.BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
            dev_verify=result,
        )

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(incomplete_gate),
    )

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    )

    assert rc == 1
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["error_kind"] == (
        "accepted_verification_checkpoint_invalid"
    )
    assert "accepted verification reference" in summary["error_message"]


def test_clean_container_tier_publishes_resume_stable_predecessor_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _build_verify

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER},
        include_successor=True,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_successful_final_gate(monkeypatch)
    kwargs = {
        "skip_build": False,
        "breach_policy": BreachPolicy(enabled=False),
        "extra_parameters": {
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    }

    assert _run_tier(repo, corpus, tier, **kwargs) == 0
    after_first = corpus.latest_spec()
    reference = _build_verify.accepted_predecessor_verification_reference(
        corpus,
        after_first,
        current_tier_id="t1",
    )

    assert reference is not None
    assert reference["predecessor_surface"]["slices"][0][
        "slice_id"
    ] == "plan_slice"
    assert reference["outcomes"] == []

    implementation_count = len(after_first.implementation_refs)

    # A clean resume reuses the content-addressed checkpoint rather than
    # growing an unbounded chain of otherwise identical reference bindings.
    assert _run_tier(repo, corpus, tier, **kwargs) == 0
    assert len(corpus.latest_spec().implementation_refs) == implementation_count


def test_explicit_skipped_container_gate_lands_without_test_checkpoint(
    tmp_path: Path,
    monkeypatch,
):
    """An operator skip is not silently promoted to accepted test proof."""

    from crucible.workflows import _build_verify

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={},
        include_successor=True,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    ) == 0

    latest = corpus.latest_spec()
    landed, pending = _partition.partition_for_resume(
        corpus, tier, spec=latest,
    )
    assert not pending
    assert all(
        _build_verify.ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT
        not in implementation.artifacts
        for implementation in landed.values()
    )
    with pytest.raises(ValueError, match="lifecycle checkpoint"):
        _build_verify.accepted_predecessor_verification_reference(
            corpus,
            latest,
            current_tier_id="t1",
        )

    # The same explicit bypass remains usable across multiple tiers even
    # though no test authority is published. Switching a later tier back to
    # verified mode would hit the missing-checkpoint rejection above.
    successor_tier = next(
        candidate
        for candidate in latest.tier_dag.tiers
        if candidate.tier_id == "t1"
    )
    successor_slice = successor_tier.slices[0]
    (repo / "successor.py").write_text(
        "value = 'explicitly unverified'\n", encoding="utf-8",
    )
    diff, files, base = capture_workspace_delta(
        cwd=str(repo), path_scope=("successor.py",),
    )
    successor = Implementation(
        slice_id=successor_slice.id,
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=files,
        classification="success",
        execution=exec_ctx_now("explicit_skip"),
        scope_escapes=(),
        artifacts={},
    )
    successor = _partition.attach_resume_identity(
        corpus,
        latest,
        successor_tier,
        successor_slice,
        successor,
        tool_authority_digest=_tool_authority_for(
            repo,
            successor_tier,
            successor_slice.id,
        ),
    )
    corpus.commit(implementations=(successor,))
    assert _run_tier(
        repo,
        corpus,
        successor_tier,
        skip_build=True,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    ) == 0


def test_explicit_skipped_cumulative_gate_settles_implementation_epoch(
    tmp_path: Path,
    monkeypatch,
):
    """A skipped final runner still closes its new checkpoint family."""

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER},
        seed_initial=False,
    )
    backend = _WritePlanBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    ) == 0

    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_cumulative_verification"] == {
        "classification": "success_skipped_by_policy",
        "directive_count": 1,
        "scope_escapes": [],
        "integrity_error": None,
        "status": "passed",
    }


def test_ordinary_restart_rejects_old_later_tier_before_backend(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows.tier import run_tier

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER},
        include_successor=True,
    )
    spec = corpus.latest_spec()
    successor_tier = next(
        candidate
        for candidate in spec.tier_dag.tiers
        if candidate.tier_id == "t1"
    )
    successor_slice = successor_tier.slices[0]
    (repo / "successor.py").write_text("old attempt\n", encoding="utf-8")
    diff, files, base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("successor.py",),
    )
    successor = Implementation(
        slice_id=successor_slice.id,
        base_commit=base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=files,
        classification="success",
        execution=exec_ctx_now("old_attempt"),
        scope_escapes=(),
        artifacts={},
    )
    successor = _partition.attach_resume_identity(
        corpus,
        spec,
        successor_tier,
        successor_slice,
        successor,
        tool_authority_digest=_tool_authority_for(
            repo,
            successor_tier,
            successor_slice.id,
        ),
    )
    corpus.commit(implementations=(successor,))

    def unexpected_backend(*_args, **_kwargs):
        raise AssertionError("ordinary restart must reject before backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        unexpected_backend,
    )

    common = dict(
        spec=corpus.spec_dir,
        tier_id=tier.tier_id,
        cwd=str(repo),
        restart=True,
        persistent_workspace=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
        review_policy=ReviewPolicy(rubrics=()),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    )
    assert run_tier(**common) == 2

    # Only the programmatic DeepSWE fresh-workspace capability may ignore the
    # old later-tier lineage. Reaching the locked body proves preflight accepts
    # it; the restart-lineage checkpoint regression in test_build_verify then
    # proves the next tier selects the newly replayed t0 checkpoint.
    entered = []

    def fresh_locked(*_args, **kwargs):
        entered.append(kwargs["accepted_predecessor_reference"])
        return 0

    monkeypatch.setattr(
        "crucible.workflows.tier._run_tier_locked_body",
        fresh_locked,
    )
    assert run_tier(**common, fresh_lineage=True) == 0
    assert entered == [None]


def test_container_tier_without_dev_verify_fails_before_backend_or_landing(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="tests",
        runners={},
    )

    def unexpected_backend(*_args, **_kwargs):
        raise AssertionError("missing runner must fail before backend acquisition")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        unexpected_backend,
    )

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        breach_policy=BreachPolicy(enabled=False),
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "authoring-container",
                "language": "python",
            },
        },
    )

    assert rc == 2
    latest = corpus.latest_spec()
    events = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert not any(event.kind == "landed" for event in events)


def test_accepted_breach_publishes_superseding_plan_bound_postimage(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)

    observed_gate_tests = []

    def _observe_gate_test(call) -> None:
        breach_test = repo / "tests" / "test_breach.py"
        observed_gate_tests.append((call["slice_id"], breach_test.exists()))
        if breach_test.exists():
            assert b"adversarial test survives final gate" in breach_test.read_bytes()
        else:
            assert not breach_test.exists()

    final_gate_calls = _install_successful_final_gate(
        monkeypatch,
        on_gate=_observe_gate_test,
    )

    rc = _run_tier(repo, corpus, tier, skip_build=True)

    assert rc == 0
    assert [call["slice_id"] for call in final_gate_calls] == [
        slice_.id, slice_.id,
    ]
    assert all(
        call["slice_directive"].slice_id == slice_.id
        for call in final_gate_calls
    )
    assert all(
        call["verification_directives"] == ()
        for call in final_gate_calls
    )
    assert observed_gate_tests == [
        (slice_.id, True),
        (slice_.id, False),
    ]
    assert not (repo / "tests" / "test_breach.py").exists()
    latest = corpus.latest_spec()
    landed, pending = _partition.partition_for_resume(
        corpus, tier, spec=latest,
    )
    assert pending == ()
    final = landed[slice_.id]
    assert final.classification == "success"
    assert "tier_final_workspace_binding" in final.artifacts
    final_binding = json.loads(corpus.get_artifact(
        final.artifacts["tier_final_workspace_binding"],
    ))
    assert final_binding["verification"] == "success"
    assert final.execution_completeness == _COMPLETE_EXECUTION
    assert final.pricing_evidence == UsagePricingEvidence(
        segments=(),
        complete=True,
    )
    assert final.cost_usd == 0.0
    assert final.input_tokens == 0
    assert final.output_tokens == 0
    assert final.cache_creation_tokens == 0
    assert final.cache_read_tokens == 0
    final_diff = corpus.get_artifact(final.diff_ref).decode("utf-8")
    assert "defender final" in final_diff
    assert "initial implementation" not in final_diff

    # Exercise reconcile's isolated-index postimage reconstruction, not only
    # resume partition selection. The bound diff must reconstruct the exact
    # Defender bytes and match the live workspace under mode+content checks.
    from crucible.workflows.reconcile import (
        _implementation_postimages,
        _workspace_matches_postimage,
    )

    postimage_result = _implementation_postimages(
        repo,
        final.base_commit,
        corpus.get_artifact(final.diff_ref),
        {"a.py"},
    )
    assert postimage_result is not None
    postimages, touched_paths, rename_pairs = postimage_result
    assert touched_paths == frozenset({"a.py"})
    assert rename_pairs == frozenset()
    assert postimages["a.py"].content == b"value = 'defender final'\n"
    assert _workspace_matches_postimage(repo, "a.py", postimages["a.py"])

    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_final_bindings"] == [{
        "slice_id": slice_.id,
        "classification": "success",
        "verification": "success",
        "diff_ref": str(final.diff_ref),
        "files_modified": ["a.py"],
        "attached": True,
    }]


@pytest.mark.parametrize(
    "transition",
    ("delete", "rename", "executable", "symlink", "byte_identical_replace"),
)
def test_accepted_breach_retains_exact_reviewed_path_transition(
    tmp_path: Path,
    monkeypatch,
    transition: str,
):
    extra_creates_paths = ("renamed.py",) if transition == "rename" else ()
    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        extra_creates_paths=extra_creates_paths,
    )
    baseline_mode = stat.S_IMODE((repo / "a.py").stat().st_mode)
    accepted_identity: list[tuple[int, int]] = []

    def mutate_candidate() -> None:
        target = repo / "a.py"
        if transition == "delete":
            target.unlink()
        elif transition == "rename":
            os.replace(target, repo / "renamed.py")
        elif transition == "executable":
            target.chmod(baseline_mode | 0o111)
        elif transition == "symlink":
            target.unlink()
            target.symlink_to("renamed.py")
        else:
            stage = repo / ".a.py.byte-identical-stage"
            stage.write_bytes(target.read_bytes())
            stage.chmod(target.stat().st_mode)
            os.replace(stage, target)
            current = target.stat()
            accepted_identity.append((current.st_dev, current.st_ino))

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    candidate_scope = (
        ("a.py", "renamed.py")
        if transition == "rename"
        else ("a.py",)
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        candidate_mutation=mutate_candidate,
        candidate_scope=candidate_scope,
    )
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id,
        slice_.id,
    ]
    assert not (repo / "tests" / "test_breach.py").exists()
    if transition == "delete":
        assert not (repo / "a.py").exists()
    elif transition == "rename":
        assert not (repo / "a.py").exists()
        assert (repo / "renamed.py").read_text() == (
            "value = 'initial implementation'\n"
        )
    elif transition == "executable":
        assert stat.S_IMODE((repo / "a.py").stat().st_mode) == (
            baseline_mode | 0o111
        )
    elif transition == "symlink":
        assert (repo / "a.py").is_symlink()
        assert os.readlink(repo / "a.py") == "renamed.py"
    else:
        current = (repo / "a.py").stat()
        assert (current.st_dev, current.st_ino) == accepted_identity[0]
        assert (repo / "a.py").read_text() == (
            "value = 'initial implementation'\n"
        )

    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    if transition == "byte_identical_replace":
        tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
        summary = json.loads((tier_run / "summary.json").read_text())
        assert summary["phase_5_final_bindings"] == [{
            "slice_id": slice_.id,
            "classification": "unchanged_plan_postimage",
            "attached": False,
        }]
    else:
        assert "tier_final_workspace_binding" in landed[slice_.id].artifacts


def test_accepted_breach_retains_preexisting_plan_deletion(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        seed_initial_delete=True,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        candidate_mutation=lambda: None,
        candidate_scope=("a.py",),
    )
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id,
        slice_.id,
    ]
    assert not (repo / "a.py").exists()
    assert not (repo / "tests" / "test_breach.py").exists()
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    assert "tier_final_workspace_binding" not in landed[slice_.id].artifacts
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_final_bindings"] == [{
        "slice_id": slice_.id,
        "classification": "unchanged_plan_postimage",
        "attached": False,
    }]


def test_shipping_preflight_rolls_writer_epoch_before_normal_gate(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _build_verify

    repo, corpus, tier, slice_ = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)
    original_settle = _build_verify._settle_no_process_writer_epoch
    epoch_phases: list[tuple[str, tuple[str, ...], bool]] = []

    def record_settlement(members, *, phase: str, succeeded: bool):
        epoch_phases.append((
            phase,
            tuple(member.slice_id for member in members or ()),
            succeeded,
        ))
        return original_settle(
            members,
            phase=phase,
            succeeded=succeeded,
        )

    monkeypatch.setattr(
        _build_verify,
        "_settle_no_process_writer_epoch",
        record_settlement,
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id,
        slice_.id,
    ]
    shipping_preflights = [
        (members, succeeded)
        for phase, members, succeeded in epoch_phases
        if phase == "shipping_gate_preflight"
    ]
    assert len(shipping_preflights) == 1
    members, succeeded = shipping_preflights[0]
    assert succeeded is True
    assert "t0__shipping_gate" in members


@pytest.mark.timeout(300)
def test_deferred_failed_seed_lands_one_exact_family_settlement(
    tmp_path: Path,
    monkeypatch,
):
    """Tier acceptance settles both mutation owners before returning success."""

    from assay.canonical import hash_record
    from crucible.workflows import _mutation_journal
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        verify_mode="tests",
        seed_initial=False,
    )
    backend = _WritePlanBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    final_gate_calls = _install_successful_final_gate(monkeypatch)
    successful_gate = tier_workflow._build_verify.build_and_verify
    phase_one_calls: list[str] = []

    def fail_phase_one_then_pass(*args, **kwargs):
        if not phase_one_calls:
            assert kwargs["slice_id"] == slice_.id
            phase_one_calls.append(kwargs["slice_id"])
            return BuildVerifyOutcome(
                "verify_failed",
                build=RunnerResult(
                    role="build",
                    runner="tests.family_build",
                    returncode=0,
                    values={},
                    raw_output=b"build passed",
                    stderr=b"",
                    duration_s=0.01,
                ),
                verify=RunnerResult(
                    role="verify",
                    runner="tests.family_verify",
                    returncode=1,
                    values={"tests_failed": 1.0},
                    raw_output=b"one failing test",
                    stderr=b"verification failed",
                    duration_s=0.01,
                ),
            )
        return successful_gate(*args, **kwargs)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(fail_phase_one_then_pass),
    )

    rc = _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        extra_parameters={"swebench_metadata": {}},
        build_retries=0,
    )

    assert rc == 0
    assert phase_one_calls == [slice_.id]
    assert [call["slice_id"] for call in final_gate_calls] == [
        slice_.id,
        slice_.id,
    ]
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'defender final'\n"
    )

    latest = corpus.latest_spec()
    assert latest is not None
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=latest,
    )
    assert pending == ()
    final = landed[slice_.id]
    final_ref = str(hash_record(final))
    lifecycle_matches = []
    for ref in latest.lifecycle_event_refs:
        event = corpus.get_record(ref, as_type=LifecycleEvent)
        if (
            _mutation_journal.mutation_lifecycle_reason_base(event.reason)
            == "tier t0: complete"
        ):
            lifecycle_matches.append((str(ref), event))
    assert len(lifecycle_matches) == 1
    lifecycle_ref, lifecycle = lifecycle_matches[0]
    _settlement_ref, payload, operations = (
        _mutation_journal._validate_mutation_lifecycle_event(
            corpus,
            latest,
            lifecycle_ref,
            lifecycle,
        )
    )
    assert payload["implementation_refs"] == [final_ref]
    assert len(operations) == 2
    by_slice = {
        str(operation["slice_id"]): operation
        for operation in operations.values()
    }
    assert set(by_slice) == {slice_.id, "t0__post_breach"}
    assert by_slice[slice_.id]["implementation_refs"] == []
    assert by_slice["t0__post_breach"]["implementation_refs"] == [
        final_ref
    ]
    deferred_transaction = str(by_slice[slice_.id]["transaction_id"])
    post_breach_transaction = str(
        by_slice["t0__post_breach"]["transaction_id"]
    )
    assert payload["family_parents"] == {
        deferred_transaction: None,
        post_breach_transaction: deferred_transaction,
    }
    assert _mutation_journal.discover_mutation_journals(
        corpus,
        workspace=repo,
    ) == ()

    before_reap = (repo / "a.py").read_bytes()
    report = reap_orphaned_dirt(corpus, repo)
    assert report.failures == ()
    assert report.mutation_journals_recovered == ()
    assert report.mutation_journals_preserved == ()
    assert (repo / "a.py").read_bytes() == before_reap


def test_build_only_slice_runs_verify_for_final_post_breach_binding(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import breach as breach_workflow

    runner = tmp_path / "final_gate_runner.py"
    marker = tmp_path / "final_gate_roles.txt"
    runner.write_text(
        "import json, os, sys\n"
        "cfg = json.loads(sys.stdin.read())\n"
        "role = (cfg.get('parameters') or {}).get('role')\n"
        "with open(os.environ['FINAL_GATE_MARKER'], 'a') as stream:\n"
        "    stream.write(str(role) + '\\n')\n"
        "sys.stdout.write(json.dumps({'values': {'returncode': 0.0}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FINAL_GATE_MARKER", str(marker))
    pythonpath = str(tmp_path)
    if os.environ.get("PYTHONPATH"):
        pythonpath += os.pathsep + os.environ["PYTHONPATH"]
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    trusted_runner_has_capability = (
        breach_workflow.trusted_runner_has_capability
    )
    monkeypatch.setattr(
        breach_workflow,
        "trusted_runner_has_capability",
        lambda candidate, capability: (
            candidate == "final_gate_runner"
            or trusted_runner_has_capability(candidate, capability)
        ),
    )

    repo, corpus, tier, _slice = _setup(
        tmp_path,
        verify_mode="build_only",
        runners={
            "build": "final_gate_runner",
            "verify": "final_gate_runner",
        },
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert marker.read_text(encoding="utf-8").splitlines() == [
        "build", "verify", "build", "verify",
    ]


def test_failed_final_gate_does_not_supersede_prior_plan_success(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: BuildVerifyOutcome(
                "verify_failed", build=None, verify=None,
            )
        ),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert (repo / "a.py").read_text() == "value = 'initial implementation'\n"
    assert not (repo / "tests" / "test_breach.py").exists()
    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs


def test_failed_shipping_gate_does_not_publish_adversarial_postimage(
    tmp_path: Path,
    monkeypatch,
):
    """The restored workspace must pass after the adversarial test is gone."""

    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = []

    def _gate(*args, **kwargs):  # noqa: ARG001
        gate_calls.append(kwargs["slice_id"])
        classification = "success" if len(gate_calls) == 1 else "verify_failed"
        return BuildVerifyOutcome(classification, build=None, verify=None)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert gate_calls == [slice_.id, slice_.id]
    assert (repo / "a.py").read_text() == "value = 'initial implementation'\n"
    assert not (repo / "tests" / "test_breach.py").exists()
    final_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs


@pytest.mark.parametrize("candidate_kind", ("tracked_same_inode", "new_file"))
def test_post_cleanup_candidate_substitution_is_rejected_before_shipping(
    tmp_path: Path,
    monkeypatch,
    candidate_kind: str,
):
    from crucible.workflows import _mutation_journal

    creates_only = candidate_kind == "new_file"
    repo, corpus, tier, slice_ = _setup(tmp_path, creates_only=creates_only)
    target = repo / ("new.py" if creates_only else "a.py")
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    initial_bytes = target.read_bytes()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    if creates_only:
        _install_accepted_breach(
            monkeypatch,
            repo,
            corpus,
            candidate_mutation=lambda: target.write_text(
                "value = 'reviewed new candidate'\n",
                encoding="utf-8",
            ),
            candidate_scope=("new.py",),
        )
    else:
        _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)
    original_restore = _mutation_journal.WorkspaceMutationJournal.restore

    def restore_then_substitute(self, *args, **kwargs):
        restored = original_restore(self, *args, **kwargs)
        if (
            restored
            and self.slice_id == "t0__breach_test"
            and kwargs.get("full") is False
        ):
            before = target.stat()
            target.write_text(
                "value = 'substituted after cleanup'\n",
                encoding="utf-8",
            )
            after = target.stat()
            assert (after.st_dev, after.st_ino) == (
                before.st_dev,
                before.st_ino,
            )
        return restored

    monkeypatch.setattr(
        _mutation_journal.WorkspaceMutationJournal,
        "restore",
        restore_then_substitute,
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert [call["slice_id"] for call in gate_calls] == [slice_.id]
    assert target.read_bytes() == initial_bytes
    assert not (repo / "tests" / "test_breach.py").exists()
    final_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["error_kind"] == (
        "breach_test_cleanup_changed_candidate"
    )


def test_shipping_journal_registration_substitution_is_rejected_before_gate(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _mutation_journal

    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    initial_bytes = (repo / "a.py").read_bytes()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)
    original_begin = _mutation_journal.WorkspaceMutationJournal.begin

    def begin_then_substitute(cls, **kwargs):  # noqa: ARG001
        journal = original_begin(**kwargs)
        if kwargs["slice_id"] == "t0__shipping_gate":
            target = repo / "a.py"
            before = target.stat()
            target.write_text(
                "value = 'substituted during shipping setup'\n",
                encoding="utf-8",
            )
            after = target.stat()
            assert (after.st_dev, after.st_ino) == (
                before.st_dev,
                before.st_ino,
            )
        return journal

    monkeypatch.setattr(
        _mutation_journal.WorkspaceMutationJournal,
        "begin",
        classmethod(begin_then_substitute),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert [call["slice_id"] for call in gate_calls] == [slice_.id]
    assert (repo / "a.py").read_bytes() == initial_bytes
    assert not (repo / "tests" / "test_breach.py").exists()
    final_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["error_kind"] == (
        "breach_test_cleanup_changed_candidate"
    )


def test_post_reproof_xattr_substitution_is_rejected_before_shipping_gate(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import _mutation_journal

    repo, corpus, tier, slice_ = _setup(tmp_path)
    target = repo / "a.py"
    try:
        os.setxattr(target, "user.crucible_probe", b"probe")
        os.removexattr(target, "user.crucible_probe")
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            pytest.skip("filesystem does not support user xattrs")
        raise
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    initial_bytes = target.read_bytes()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)
    journal_type = _mutation_journal.WorkspaceMutationJournal
    original_reprove = (
        journal_type.reprove_consumed_active_parent_postimage_handoff
    )

    def reprove_then_substitute(self, handoff):
        original_reprove(self, handoff)
        os.setxattr(target, "user.crucible_probe", b"post-reproof")

    monkeypatch.setattr(
        journal_type,
        "reprove_consumed_active_parent_postimage_handoff",
        reprove_then_substitute,
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert [call["slice_id"] for call in gate_calls] == [slice_.id]
    assert target.read_bytes() == initial_bytes
    assert "user.crucible_probe" not in os.listxattr(target)
    assert not (repo / "tests" / "test_breach.py").exists()
    final_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["error_kind"] == (
        "breach_test_cleanup_changed_candidate"
    )


@pytest.mark.parametrize("metadata_drift", ("xattr", "nonexec_mode"))
def test_post_first_reproof_new_path_metadata_drift_is_rejected_before_shipping(
    tmp_path: Path,
    monkeypatch,
    metadata_drift: str,
):
    from crucible.workflows import _mutation_journal

    repo, corpus, tier, slice_ = _setup(tmp_path, creates_only=True)
    target = repo / "new.py"
    if metadata_drift == "xattr":
        try:
            os.setxattr(target, "user.crucible_probe", b"probe")
            os.removexattr(target, "user.crucible_probe")
        except OSError as exc:
            if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip("filesystem does not support user xattrs")
            raise
    initial_bytes = target.read_bytes()
    initial_mode = stat.S_IMODE(target.stat().st_mode)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        candidate_mutation=lambda: target.write_text(
            "value = 'reviewed new candidate'\n",
            encoding="utf-8",
        ),
        candidate_scope=("new.py",),
    )
    gate_calls = _install_successful_final_gate(monkeypatch)
    journal_type = _mutation_journal.WorkspaceMutationJournal
    original_reprove = (
        journal_type.reprove_consumed_active_parent_postimage_handoff
    )
    reproof_attempts = 0

    def first_reproof_then_drift(self, handoff):
        nonlocal reproof_attempts
        reproof_attempts += 1
        original_reprove(self, handoff)
        if reproof_attempts != 1:
            return
        if metadata_drift == "xattr":
            os.setxattr(target, "user.crucible_probe", b"post-first-proof")
        else:
            changed_mode = 0o600 if initial_mode != 0o600 else 0o644
            target.chmod(changed_mode)

    monkeypatch.setattr(
        journal_type,
        "reprove_consumed_active_parent_postimage_handoff",
        first_reproof_then_drift,
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert reproof_attempts == 2
    assert [call["slice_id"] for call in gate_calls] == [slice_.id]
    assert target.read_bytes() == initial_bytes
    assert stat.S_IMODE(target.stat().st_mode) == initial_mode
    if metadata_drift == "xattr":
        assert "user.crucible_probe" not in os.listxattr(target)
    assert not (repo / "tests" / "test_breach.py").exists()
    final_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["error_kind"] == (
        "breach_test_cleanup_changed_candidate"
    )


def test_empty_scoped_shipping_postimage_is_rejected_and_rolled_back(
    tmp_path: Path,
    monkeypatch,
):
    """A final binding cannot turn a promised edit into resumable no-op."""

    repo, corpus, tier, slice_ = _setup(tmp_path, creates_only=True)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        remove_plan_path="new.py",
    )
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id, slice_.id,
    ]
    assert (repo / "new.py").read_text() == "value = 'initial implementation'\n"
    assert not (repo / "tests" / "test_breach.py").exists()

    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["status"] == "final_binding_failed"
    assert summary["phase_5_final_bindings"] == []


def test_review_retry_success_is_not_published_when_final_gate_later_fails(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    backend = _ReviewRetryBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: BuildVerifyOutcome(
                "verify_failed", build=None, verify=None,
            )
        ),
    )

    with (repo / "a.py").open("rb") as pinned_preimage:
        pinned_identity = (
            os.fstat(pinned_preimage.fileno()).st_dev,
            os.fstat(pinned_preimage.fileno()).st_ino,
        )
        assert (
            _run_tier(
                repo,
                corpus,
                tier,
                skip_build=True,
                skip_review=False,
                review_policy=ReviewPolicy(
                    rubrics=("tier_completeness",),
                    retry_budget=1,
                ),
            )
            == 1
        )
        live = (repo / "a.py").stat()
        assert (live.st_dev, live.st_ino) != pinned_identity
        assert backend.review_calls == 2
        assert backend.implement_calls == 1
        assert (repo / "a.py").read_text() == (
            "value = 'initial implementation'\n"
        )
        assert not (repo / "tests" / "test_breach.py").exists()

    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs


def test_review_retry_success_is_bound_only_by_final_atomic_acceptance(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_identity = (repo / "a.py").stat().st_ino
    backend = _AtomicReviewRetryBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    _install_successful_final_gate(monkeypatch)

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 0
    assert backend.review_calls == 2
    assert backend.implement_calls == 1
    assert (repo / "a.py").stat().st_ino != initial_identity
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    final = landed[slice_.id]
    assert "tier_final_workspace_binding" in final.artifacts
    assert b"successful review retry" in corpus.get_artifact(final.diff_ref)


def test_rejected_post_breach_retry_can_land_exact_reviewed_preimage(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    backend = _FallbackReviewBackend(repo)
    from crucible.workflows import tier as tier_mod

    strict_review = tier_mod.review
    review_chunk_chars = []

    def recording_review(*args, **kwargs):
        review_chunk_chars.append(kwargs.get("chunk_chars"))
        return strict_review(*args, **kwargs)

    monkeypatch.setattr(tier_mod, "review", recording_review)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(
        monkeypatch,
        on_gate=lambda call: (
            (repo / "a.py").read_text(encoding="utf-8"),
            (repo / "tests" / "test_breach.py").exists(),
        ),
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 0

    assert backend.review_calls == 3
    assert review_chunk_chars == [80_000] * 3
    assert backend.implement_calls == 1
    assert [call["slice_id"] for call in gate_calls] == [slice_.id]
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    assert not (repo / "tests" / "test_breach.py").exists()
    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    fallback = summary["phase_4_pre_breach_fallback"]
    assert fallback["status"] == "landed"
    assert fallback["digest_verified"] is True
    assert fallback["recorded_sha256"] == fallback["restored_sha256"]
    fallback_reviews = [
        item for item in summary["phase_4_reviews"]
        if item.get("candidate") == "pre_breach_fallback"
    ]
    assert len(fallback_reviews) == 1
    assert fallback_reviews[0]["passed"] is True
    assert not list(run.glob("**/manifest.json"))


def test_rejected_pre_breach_fallback_review_remains_fail_closed(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, _slice = _setup(tmp_path)
    backend = _AlwaysRejectReviewBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 3

    assert backend.review_calls == 3
    assert backend.implement_calls == 1
    assert gate_calls == []
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    assert not (repo / "tests" / "test_breach.py").exists()
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "review_failed"
    fallback = summary["phase_4_pre_breach_fallback"]
    assert fallback["status"] == "review_failed"
    assert fallback["digest_verified"] is True


def test_pre_breach_fallback_is_ineligible_with_admitted_attack(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, _slice = _setup(tmp_path)
    backend = _FallbackReviewBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        admitted_attack_ids=("defender_regression",),
    )
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 3

    assert backend.review_calls == 2
    assert backend.implement_calls == 1
    assert gate_calls == []
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    assert not (repo / "tests" / "test_breach.py").exists()
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "review_failed"
    assert "phase_4_pre_breach_fallback" not in summary


def test_pre_breach_fallback_digest_mismatch_is_terminal(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, tier, _slice = _setup(tmp_path)
    backend = _FallbackReviewBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        tier_workflow,
        "_diff_sha256",
        lambda _diff: next(digests),
    )
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 1

    assert backend.review_calls == 2
    assert backend.implement_calls == 1
    assert gate_calls == []
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "final_binding_failed"
    fallback = summary["phase_4_pre_breach_fallback"]
    assert fallback["status"] == "digest_mismatch"
    assert fallback["digest_verified"] is False


def test_pre_breach_fallback_review_infrastructure_failure_is_terminal(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, _slice = _setup(tmp_path)
    backend = _FallbackInfrastructureFailureBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    gate_calls = _install_successful_final_gate(monkeypatch)

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 3

    # Two semantic reviews plus the initial malformed fallback response and
    # the configured default number of strict-response recovery calls.
    assert backend.review_calls == (
        3 + _DEFAULT_TIER_RESPONSE_RETRY_BUDGET
    )
    assert backend.implement_calls == 1
    assert gate_calls == []
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    fallback = summary["phase_4_pre_breach_fallback"]
    assert fallback["status"] == "review_failed"
    assert fallback["infrastructure_failure"] is True
    # A lost fallback reviewer keeps the tier status of the review it
    # replaced, and the summary does not name a review without a verdict.
    assert summary["status"] == "review_failed"
    assert "phase_4_review_inconclusive" not in summary


def test_post_breach_review_amendment_reproves_admitted_attacks(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    repo, corpus, tier, slice_ = _setup(tmp_path)
    backend = _ReviewRetryBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        admitted_attack_ids=("defender_regression",),
    )
    calls: list[tuple[str, dict | None]] = []

    def _gate(*args, **kwargs):  # noqa: ARG001
        parameters = kwargs.get("extra_parameters")
        calls.append((kwargs["slice_id"], parameters))
        if parameters and parameters.get("breach_observations") is True:
            assert parameters["breach_observations"] is True
            assert parameters["breach_test_file"] == "tests/test_breach.py"
            assert parameters["breach_attack_ids"] == [
                "defender_regression"
            ]
            result = RunnerResult(
                role="verify",
                runner="test.runner",
                returncode=0,
                values={"returncode": 0.0},
                raw_output=json.dumps({
                    "test_observations": [{
                        "language": "python",
                        "id": (
                            "tests/test_breach.py::"
                            "test_defender_regression"
                        ),
                        "status": "passed",
                        "line": (
                            "tests/test_breach.py::"
                            "test_defender_regression"
                        ),
                    }],
                }).encode(),
                stderr=b"",
                duration_s=0.0,
            )
            return BuildVerifyOutcome(
                "success", build=None, verify=result
            )
        assert not parameters or "breach_observations" not in parameters
        return BuildVerifyOutcome("success", build=None, verify=None)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 0
    assert [slice_id for slice_id, _parameters in calls] == [
        slice_.id, slice_.id,
    ]
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    reproof = summary["phase_5_breach_obligation_reproof"]
    assert reproof["required"] is True
    assert reproof["passed"] is True
    assert reproof["attack_ids"] == ["defender_regression"]
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    assert b"successful review retry" in corpus.get_artifact(
        landed[slice_.id].diff_ref
    )


def test_conflict_reconciliation_at_prebreach_digest_still_reproves_attacks(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    clause = "Remove PublicSymbol from the public export."
    repo, corpus, tier, _slice = _setup(
        tmp_path,
        intent=f"Change conversion behavior. {clause} Preserve behavior.",
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        admitted_attack_ids=("defender_regression",),
    )

    class Backend(_UnusedBackend):
        name = "conflict-reproof"
        model = "stub"

        def __init__(self):
            self.calls = 0

        def invoke(self, invocation):  # noqa: ARG002
            self.calls += 1
            (repo / "a.py").write_text(
                (
                    "value = 'retry removed PublicSymbol'\n"
                    if self.calls == 1
                    else "value = 'initial implementation'\n"
                ),
                encoding="utf-8",
            )
            return InvocationResult(
                final_message="implemented",
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )

    backend = Backend()
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    rubric_calls = {"diff_scope": 0, "tier_completeness": 0}

    def _verdict(rubric, artifact, decision, findings):
        artifact_bytes = (
            artifact if isinstance(artifact, bytes)
            else corpus.get_artifact(ContentHash(str(artifact)))
        )
        return Verdict(
            rubric=rubric,
            decision=decision,
            findings=tuple(findings),
            artifact_ref=corpus.put_artifact(
                artifact_bytes,
                kind="analysis_inputs",
                name_hint=f"{rubric}.json",
            ),
            artifact_commit=None,
            reasoning="deterministic conflict test",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name="stub",
            transcript_ref=None,
            cost_usd=0.0,
            execution=None,
        )

    def _review(artifact, rubric, **kwargs):  # noqa: ARG001
        if rubric.name == "review_conflict_authority":
            packet = json.loads(artifact.decode("utf-8"))["packet"]
            gate_excerpt = "ImportError: PublicSymbol is required"
            authority_findings = (
                Finding(
                    id="review_conflict_000",
                    severity="Low",
                    category="actionable",
                    evidence=(Evidence(
                        source_id="artifact",
                        kind="review_finding",
                        location="finding:review_conflict_000",
                        excerpt="defender final",
                    ),),
                    suggested_action=None,
                    source="llm",
                    criterion_id="review_conflict_authority",
                    confidence=0.99,
                    disposition="addressed",
                ),
                Finding(
                    id="review_conflict_001",
                    severity="Low",
                    category="verification_conflict",
                    evidence=(
                        Evidence(
                            source_id="artifact",
                            kind="review_finding",
                            location="finding:review_conflict_001",
                            excerpt=clause,
                        ),
                        Evidence(
                            source_id="artifact",
                            kind="slice_directive",
                            location="slice:plan_slice",
                            excerpt=clause,
                        ),
                        Evidence(
                            source_id="artifact",
                            kind="gate_diagnostic",
                            location="gate:0",
                            excerpt=gate_excerpt,
                        ),
                    ),
                    suggested_action=None,
                    source="llm",
                    criterion_id="review_conflict_authority",
                    confidence=0.99,
                    disposition="dismissed",
                ),
            )
            assert packet["gate_rejections"][0]["diagnostic"][
                "stdout"
            ] == gate_excerpt
            return _verdict(
                rubric.name, artifact, "pass", authority_findings
            )
        rubric_calls[rubric.name] += 1
        if rubric_calls[rubric.name] > 1:
            return _verdict(rubric.name, artifact, "pass", ())
        if rubric.name == "diff_scope":
            finding = Finding(
                id="defender-drift",
                severity="Medium",
                category="incidental_change",
                evidence=(Evidence(
                    source_id="artifact",
                    kind="diff_change",
                    location="a.py",
                    excerpt="defender final",
                ),),
                suggested_action="restore the plan candidate",
                source="llm",
                criterion_id="incidental_changes",
                confidence=0.99,
                disposition="confirmed",
            )
        else:
            finding = Finding(
                id="remove-public-symbol",
                severity="Medium",
                category="partial_delivery",
                evidence=(Evidence(
                    source_id="artifact",
                    kind="slice_directive",
                    location="plan_slice",
                    excerpt=clause,
                ),),
                suggested_action="remove PublicSymbol",
                source="llm",
                criterion_id="slice_delivery",
                confidence=0.99,
                disposition="confirmed",
            )
        return _verdict(rubric.name, artifact, "revise", (finding,))

    monkeypatch.setattr(tier_workflow, "review", _review)
    gate_calls = []

    def _gate(*args, **kwargs):  # noqa: ARG001
        gate_calls.append(kwargs)
        if len(gate_calls) == 1:
            failed = RunnerResult(
                role="verify",
                runner="test.runner",
                returncode=1,
                values={},
                raw_output=b"ImportError: PublicSymbol is required",
                stderr=b"",
                duration_s=0.0,
            )
            return BuildVerifyOutcome(
                "verify_failed", build=None, verify=failed
            )
        if len(gate_calls) == 3:
            assert (repo / "tests" / "test_breach.py").is_file()
        return BuildVerifyOutcome("success", build=None, verify=None)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=False,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("diff_scope", "tier_completeness"),
            retry_budget=1,
        ),
    ) == 1

    assert backend.calls == 2
    assert [call["slice_id"] for call in gate_calls] == [
        "plan_slice", "plan_slice", "plan_slice",
    ]
    final_parameters = gate_calls[-1]["extra_parameters"]
    assert final_parameters["breach_test_file"] == "tests/test_breach.py"
    assert final_parameters["breach_attack_ids"] == [
        "defender_regression"
    ]
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    conflict = summary["phase_4_review_conflict_authority"]
    assert conflict["reconciliation_sha256"] == (
        summary["phase_3_pre_breach_candidate"]["sha256"]
    )
    reproof = summary["phase_5_breach_obligation_reproof"]
    assert reproof["required"] is True
    assert reproof["passed"] is False
    assert summary["status"] == "final_binding_failed"
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )


def test_failed_post_review_breach_reproof_is_terminal_and_restores(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    repo, corpus, tier, _slice = _setup(tmp_path)
    backend = _ReviewRetryBackend(repo)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        admitted_attack_ids=("defender_regression",),
    )

    def _gate(*args, **kwargs):  # noqa: ARG001
        result = RunnerResult(
            role="verify",
            runner="test.runner",
            returncode=0,
            values={"returncode": 0.0},
            raw_output=json.dumps({
                "test_observations": [{
                    "language": "python",
                    "id": "tests/test_breach.py::test_defender_regression",
                    "status": "failed",
                    "line": "tests/test_breach.py::test_defender_regression",
                }],
            }).encode(),
            stderr=b"",
            duration_s=0.0,
        )
        return BuildVerifyOutcome("success", build=None, verify=result)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(_gate),
    )

    assert _run_tier(
        repo,
        corpus,
        tier,
        skip_build=True,
        skip_review=False,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    ) == 1
    assert (repo / "a.py").read_text(encoding="utf-8") == (
        "value = 'initial implementation'\n"
    )
    assert not (repo / "tests" / "test_breach.py").exists()
    run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "final_binding_failed"
    reproof = summary["phase_5_breach_obligation_reproof"]
    assert reproof["required"] is True
    assert reproof["passed"] is False
    assert reproof["status"] == "conflict"


def test_unchanged_plan_postimage_is_still_gated_but_not_superseded(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        final_value="initial implementation",
    )

    gate_calls = _install_successful_final_gate(
        monkeypatch,
        on_gate=lambda call: (
            (repo / "tests" / "test_breach.py").read_bytes()
            if (repo / "tests" / "test_breach.py").exists()
            else None
        ),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id, slice_.id,
    ]
    assert all(
        call["slice_directive"].slice_id == slice_.id for call in gate_calls
    )
    assert not (repo / "tests" / "test_breach.py").exists()
    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    tier_run = sorted(corpus.spec_dir.joinpath("runs").glob("*tier*"))[-1]
    summary = json.loads((tier_run / "summary.json").read_text())
    assert summary["phase_5_final_bindings"] == [{
        "slice_id": slice_.id,
        "classification": "unchanged_plan_postimage",
        "attached": False,
    }]


def test_no_attacks_status_after_mutation_still_runs_final_gate_and_binds(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(
        monkeypatch,
        repo,
        corpus,
        final_value="mutation before no attacks",
        status="no_attacks",
    )
    gate_calls = _install_successful_final_gate(
        monkeypatch,
        on_gate=lambda call: (
            (repo / "tests" / "test_breach.py").read_bytes()
            if (repo / "tests" / "test_breach.py").exists()
            else None
        ),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [call["slice_id"] for call in gate_calls] == [
        slice_.id, slice_.id,
    ]
    assert not (repo / "tests" / "test_breach.py").exists()
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    final_diff = corpus.get_artifact(landed[slice_.id].diff_ref)
    assert b"mutation before no attacks" in final_diff


def test_preexisting_breach_test_is_restored_exactly_after_final_gate(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    breach_test = repo / "tests" / "test_breach.py"
    breach_test.parent.mkdir(parents=True)
    original = b"# local user test\ndef test_local():\n    assert True\n"
    breach_test.write_bytes(original)
    breach_test.chmod(0o640)
    original_mode = breach_test.stat().st_mode
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)

    observed = []

    def _assert_correct_test_reaches_each_gate(call) -> None:
        observed.append((call["slice_id"], breach_test.read_bytes()))
        if len(observed) == 1:
            assert b"adversarial test survives final gate" in observed[-1][1]
        else:
            assert observed[-1][1] == original

    _install_successful_final_gate(
        monkeypatch,
        on_gate=_assert_correct_test_reaches_each_gate,
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    assert [slice_id for slice_id, _contents in observed] == [
        slice_.id, slice_.id,
    ]
    assert observed == [
        (
            slice_.id,
            b"def test_defender_regression():\n"
            b"    assert 'adversarial test survives final gate'\n",
        ),
        (slice_.id, original),
    ]
    assert breach_test.read_bytes() == original
    assert breach_test.stat().st_mode == original_mode


def test_plan_owned_breach_test_remains_and_is_bound_after_final_gate(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(
        tmp_path,
        plan_owns_breach_test=True,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    _install_successful_final_gate(
        monkeypatch,
        on_gate=lambda call: (repo / "tests/test_breach.py").read_bytes(),
    )

    assert _run_tier(repo, corpus, tier, skip_build=True) == 0
    breach_test = repo / "tests/test_breach.py"
    assert breach_test.is_file()
    landed, pending = _partition.partition_for_resume(
        corpus,
        tier,
        spec=corpus.latest_spec(),
    )
    assert pending == ()
    final = landed[slice_.id]
    assert "tests/test_breach.py" in final.files_modified
    assert b"test_defender_regression" in corpus.get_artifact(final.diff_ref)


def test_concurrent_spec_change_rejects_atomic_binding_and_restores_workspace(
    tmp_path: Path,
    monkeypatch,
):
    repo, corpus, tier, slice_ = _setup(tmp_path)
    initial_plan_refs = tuple(
        ref for ref in corpus.latest_spec().implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: _UnusedBackend(),
    )
    _install_accepted_breach(monkeypatch, repo, corpus)
    _install_successful_final_gate(monkeypatch)

    original_commit = Corpus.commit
    race_injected = False

    def _racing_commit(self, **kwargs):
        nonlocal race_injected
        if (
            not race_injected
            and kwargs.get("transform") is not None
            and kwargs.get("lifecycle_event_factory") is not None
        ):
            race_injected = True
            original_commit(
                self,
                lifecycle_event_factory=lambda affected_version: (
                    LifecycleEvent(
                        kind="abandoned",
                        reason="concurrent writer",
                        affected_versions=(affected_version,),
                        execution=exec_ctx_now("concurrent_test"),
                    )
                ),
            )
        return original_commit(self, **kwargs)

    monkeypatch.setattr(Corpus, "commit", _racing_commit)

    assert _run_tier(repo, corpus, tier, skip_build=True) == 1
    assert race_injected
    assert (repo / "a.py").read_text() == "value = 'initial implementation'\n"
    assert not (repo / "tests" / "test_breach.py").exists()

    latest = corpus.latest_spec()
    final_plan_refs = tuple(
        ref for ref in latest.implementation_refs
        if corpus.get_record(ref, as_type=Implementation).slice_id == slice_.id
    )
    assert final_plan_refs == initial_plan_refs
    lifecycle_reasons = {
        corpus.get_record(ref, as_type=LifecycleEvent).reason
        for ref in latest.lifecycle_event_refs
    }
    assert "concurrent writer" in lifecycle_reasons
    assert "tier t0: complete" not in lifecycle_reasons
