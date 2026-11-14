# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hostile WRK-01 and NAT-01 qualification-boundary tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    ExternalCancellation,
    Implementation,
    ImplementationDirective,
    Invocation,
    InvocationResult,
    LifecycleEvent,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.canonical import hash_record
from assay.corpus.store import Corpus

from crucible.backends.features import feature_profile_for_route
from crucible.tools import ToolMutationIndeterminate
from crucible.tools.write import build_write_tools
from crucible.workflows._mutation_journal import (
    MutationJournalCleanupError,
    MutationJournalError,
    WorkspaceMutationJournal,
    _unregister_live,
    workspace_mutation_lock,
)
from crucible.workflows._run_policy import (
    WriterLanePolicy,
)
from crucible.workflows._writer_scheduler import (
    IntegrationGateResult,
    WriterCleanupError,
    WriterLaneRequest,
    WriterSchedulerError,
    run_isolated_writer_fanout,
)
from crucible.workflows import tier as tier_workflow
from crucible.workflows import _backends as backend_pool_module
from crucible.workflows import _mutation_journal as mutation_journal_module
from crucible.workflows import _worktree as worktree_module
from crucible.workflows import _writer_scheduler as writer_scheduler_module
from crucible.workflows._backends import BackendCleanupError, _BackendPool
from crucible.workflows.tier import TierWriterFrontier


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "writer@example.invalid")
    _git(repo, "config", "user.name", "Writer Test")
    (repo / "a.txt").write_text("base-a\n", encoding="utf-8")
    (repo / "b.txt").write_text("base-b\n", encoding="utf-8")
    _git(repo, "add", "a.txt", "b.txt")
    _git(repo, "commit", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _invocation(*, budget: float = 2.0) -> Invocation:
    return Invocation(
        system="author",
        user="edit",
        tools=None,
        max_turns=8,
        budget_usd=budget,
    )


def _complete_result(
    lane_id: str,
    *,
    cost: float = 0.1,
    cancellation: bool = False,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    pricing_evidence: UsagePricingEvidence | None = None,
) -> InvocationResult:
    events = (
        BackendExecutionEvent(sequence=0, kind="request", name="invoke"),
        BackendExecutionEvent(sequence=1, kind="usage", name="usage"),
        BackendExecutionEvent(
            sequence=2,
            kind="terminal",
            name="cancelled" if cancellation else "complete",
        ),
    )
    return InvocationResult(
        final_message=f"writer:{lane_id}",
        transcript=(),
        usage=Usage(
            input_tokens=10,
            output_tokens=5,
            cost_usd=cost,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
        stop_reason="cancelled" if cancellation else "end_turn",
        error=(
            ExternalCancellation(
                cause="scheduler",
                message="cancelled",
            )
            if cancellation
            else None
        ),
        completeness=ExecutionCompleteness(
            token_usage_complete=True,
            cost_complete=True,
            terminal_complete=True,
            trace_complete=True,
        ),
        execution_trace=BackendExecutionTrace(
            backend="synthetic",
            requested_model="writer-test",
            events=events,
        ),
        pricing_evidence=pricing_evidence,
    )


class _WriterToolSet:
    read_only_names = {"read_source"}
    write_names = {"write_source"}

    def __init__(self, context) -> None:
        self.context = context

    def to_anthropic_tools(self) -> list[dict]:
        return [
            {
                "name": "read_source",
                "description": "read",
                "input_schema": {"type": "object"},
            },
            {
                "name": "write_source",
                "description": "write",
                "input_schema": {"type": "object"},
            },
        ]

    def dispatch(self, name: str, arguments: dict) -> str:
        assert name == "write_source"
        path = self.context.repo_root / str(arguments["path"])
        path.write_text(str(arguments["content"]), encoding="utf-8")
        return "ok"


@dataclass
class _Concurrency:
    active: int = 0
    max_active: int = 0
    started: int = 0
    closed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    two_started: threading.Event = field(default_factory=threading.Event)

    def start(self) -> None:
        with self.lock:
            self.active += 1
            self.started += 1
            self.max_active = max(self.max_active, self.active)
            if self.started >= 2:
                self.two_started.set()

    def finish(self) -> None:
        with self.lock:
            self.active -= 1

    def close(self) -> None:
        with self.lock:
            self.closed += 1


class _WriterBackend:
    name = "synthetic"
    model = "writer-test"

    def __init__(
        self,
        request: WriterLaneRequest,
        toolset: _WriterToolSet,
        tracker: _Concurrency,
        *,
        writes: dict[str, tuple[str, str]],
        delay: float = 0.0,
        scope_escape: bool = False,
        block: bool = False,
        accounting_complete: bool = True,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        pricing_evidence: UsagePricingEvidence | None = None,
    ) -> None:
        self.request = request
        self.toolset = toolset
        self.tracker = tracker
        self.writes = writes
        self.delay = delay
        self.scope_escape = scope_escape
        self.block = block
        self.accounting_complete = accounting_complete
        self.cache_creation_tokens = cache_creation_tokens
        self.cache_read_tokens = cache_read_tokens
        self.pricing_evidence = pricing_evidence
        self.closed = threading.Event()
        self._close_lock = threading.Lock()
        self._close_counted = False

    def invoke(self, invocation: Invocation) -> InvocationResult:
        self.tracker.start()
        try:
            if self.block:
                assert self.closed.wait(5)
                return _complete_result(self.request.lane_id, cancellation=True)
            path, content = self.writes[self.request.lane_id]
            self.toolset.dispatch(
                "write_source",
                {"path": path, "content": content},
            )
            if self.scope_escape:
                escape = self.toolset.context.repo_root / "escape.txt"
                escape.write_text("escape\n", encoding="utf-8")
            if self.delay:
                time.sleep(self.delay)
            result = _complete_result(
                self.request.lane_id,
                cache_creation_tokens=self.cache_creation_tokens,
                cache_read_tokens=self.cache_read_tokens,
                pricing_evidence=self.pricing_evidence,
            )
            if not self.accounting_complete:
                result = replace(
                    result,
                    completeness=replace(
                        result.completeness,
                        token_usage_complete=False,
                        cost_complete=False,
                    ),
                )
            return result
        finally:
            self.tracker.finish()

    def close(self) -> None:
        with self._close_lock:
            if not self._close_counted:
                self._close_counted = True
                self.tracker.close()
        self.closed.set()


def _policy(
    *,
    lanes: int = 2,
    max_concurrency: int = 2,
    require_disjoint_scopes: bool = True,
) -> WriterLanePolicy:
    return WriterLanePolicy(
        mode="isolated_writer",
        eligible_roles=("implementer",),
        max_lanes=lanes,
        max_concurrency=max_concurrency,
        max_total_cost_share=0.5,
        max_lane_cost_share=0.25,
        require_disjoint_scopes=require_disjoint_scopes,
    )


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("native_delegation", "read_only"),
        ("project_instruction_policy", "inherit"),
        ("approval_policy", "allow"),
        ("cancellation_policy", "abandon"),
        ("integration_order", "completion"),
        ("base_policy", "dirty_allowed"),
        ("landing_mode", "apply"),
    ],
)
def test_writer_policy_closed_literals_fail_closed(
    field_name,
    value,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        WriterLanePolicy(**{field_name: value})




def _requests(
    scopes: tuple[tuple[str, ...], ...],
) -> tuple[WriterLaneRequest, ...]:
    return tuple(
        WriterLaneRequest(
            lane_id=f"lane-{index}",
            role=f"writer-{index}",
            route_role="implementer",
            scope=scope,
            invocation=_invocation(),
        )
        for index, scope in enumerate(scopes)
    )


def _gate(
    worktree: Path,
    scope: tuple[str, ...],
    requests: tuple[WriterLaneRequest, ...],
) -> IntegrationGateResult:
    payload = {
        "scope": scope,
        "lanes": tuple(request.lane_id for request in requests),
        "a": (worktree / "a.txt").read_text(encoding="utf-8"),
        "b": (worktree / "b.txt").read_text(encoding="utf-8"),
    }
    return IntegrationGateResult(
        success=True,
        evidence_digest=hash_record(payload),
    )


def _state_root(tmp_path: Path, name: str = "writer") -> Path:
    return tmp_path / "spec" / "runs" / "run-1" / name


def _assert_writer_ownership_settled(repo: Path) -> None:
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = repo / common
    common = common.resolve()
    for name in (
        "crucible-mutation-journals",
        "crucible-worktree-leases",
    ):
        owner_root = common / name
        assert not owner_root.exists() or not any(owner_root.iterdir())
    worktrees = _git(repo, "worktree", "list", "--porcelain")
    assert sum(line.startswith("worktree ") for line in worktrees.splitlines()) == 1


def test_writer_lanes_are_bounded_isolated_and_integrate_in_order(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    requests = _requests((("a.txt",), ("b.txt",)))
    tracker = _Concurrency()
    toolsets: list[_WriterToolSet] = []
    backends: list[_WriterBackend] = []
    contexts: list[object] = []
    worktrees: list[Path] = []
    lane_journal_bindings: dict[str, tuple[str, Path]] = {}
    lane_journal_lock = threading.Lock()

    def toolset_factory(request, context):
        journal = mutation_journal_module.resolve_live_mutation_journal(
            context.mutation_journal_dir,
            context.mutation_journal_transaction_id,
        )
        assert journal.slice_id == f"writer-lane::{request.lane_id}"
        assert journal.journal_dir == context.mutation_journal_dir
        assert journal.transaction_id == (
            context.mutation_journal_transaction_id
        )
        with lane_journal_lock:
            lane_journal_bindings[request.lane_id] = (
                journal.transaction_id,
                journal.journal_dir,
            )
        contexts.append(context)
        toolset = _WriterToolSet(context)
        toolsets.append(toolset)
        return toolset

    def backend_factory(request, toolset, debug_dir, worktree):
        assert debug_dir.name == request.lane_id
        worktrees.append(worktree)
        backend = _WriterBackend(
            request,
            toolset,
            tracker,
            writes={
                "lane-0": ("a.txt", "writer-a\n"),
                "lane-1": ("b.txt", "writer-b\n"),
            },
            delay=0.05,
        )
        backends.append(backend)
        return backend

    result = run_isolated_writer_fanout(
        requests,
        policy=_policy(),
        repo_root=repo,
        base_commit=base,
        backend_factory=backend_factory,
        toolset_factory=toolset_factory,
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert result.successful
    assert result.status == "succeeded"
    assert result.usage == Usage(input_tokens=20, output_tokens=10, cost_usd=0.2)
    legacy_summary = result.to_summary()
    assert legacy_summary["schema_version"] == ("crucible.writer-fanout-result.v1")
    assert "cache_creation_tokens" not in legacy_summary
    assert "pricing_evidence" not in legacy_summary["children"][0]
    assert result.integrated_files == ("a.txt", "b.txt")
    assert b"+writer-a" in result.integrated_patch
    assert b"+writer-b" in result.integrated_patch
    assert tracker.max_active == 2
    assert tracker.closed == 2
    assert len({id(item) for item in contexts}) == 2
    assert len({id(item) for item in toolsets}) == 2
    assert len({id(item) for item in backends}) == 2
    assert set(lane_journal_bindings) == {"lane-0", "lane-1"}
    assert len({str(path) for path in worktrees}) == 2
    assert all(not path.exists() for path in worktrees)
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "base-b\n"
    assert _git(repo, "status", "--porcelain") == ""
    subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo,
        input=result.integrated_patch,
        check=True,
    )


def test_writer_lane_retains_indeterminate_mutation_cause_without_retry(
    tmp_path: Path,
) -> None:
    from assay.types.invocation import TransportError

    repo, base = _repo(tmp_path)
    calls = 0
    gate_calls = 0

    class _IndeterminateWriter:
        name = "synthetic"
        model = "indeterminate-writer"

        def __init__(self, request, toolset) -> None:
            self.request = request
            self.toolset = toolset

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            nonlocal calls
            calls += 1
            self.toolset.dispatch(
                "write_source",
                {"path": "a.txt", "content": "indeterminate\n"},
            )
            return replace(
                _complete_result(self.request.lane_id),
                stop_reason="tool_mutation_indeterminate",
                error=TransportError(
                    message="delete settlement is unknown",
                    retriable=False,
                ),
            )

        def close(self) -> None:
            return None

    def integration_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return _gate(*args, **kwargs)

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=replace(
            _policy(lanes=1, max_concurrency=1),
            fail_fast=False,
        ),
        repo_root=repo,
        base_commit=base,
        backend_factory=(
            lambda request, toolset, _debug, _worktree: _IndeterminateWriter(
                request,
                toolset,
            )
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "indeterminate-writer"),
        integration_gate=integration_gate,
    )

    assert calls == 1
    assert gate_calls == 0
    assert result.status == "failed"
    assert result.integrated_patch == b""
    assert result.children[0].status == "failed"
    assert result.children[0].error_kind == "tool_mutation_indeterminate"
    summary = result.to_summary()
    assert summary["children"][0]["error_kind"] == (
        "tool_mutation_indeterminate"
    )
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""


class _RaisedIndeterminateWriter:
    name = "synthetic"
    model = "raised-indeterminate-writer"

    def __init__(
        self,
        request: WriterLaneRequest,
        toolset: _WriterToolSet,
        *,
        close_settles: bool,
    ) -> None:
        self.request = request
        self.toolset = toolset
        self.close_settles = close_settles
        self.invoke_calls = 0
        self.close_calls = 0

    def invoke(self, _invocation: Invocation) -> InvocationResult:
        self.invoke_calls += 1
        self.toolset.dispatch(
            "write_source",
            {"path": "a.txt", "content": "indeterminate\n"},
        )
        raise ToolMutationIndeterminate(
            "writer mutation publication settlement is unknown"
        )

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("writer provider descendants still live")


def test_writer_raised_indeterminate_settles_once_without_paid_retry(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    gate_calls = 0
    backend: _RaisedIndeterminateWriter | None = None

    def backend_factory(request, toolset, _debug, _worktree):
        nonlocal backend
        backend = _RaisedIndeterminateWriter(
            request,
            toolset,
            close_settles=True,
        )
        return backend

    def integration_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return _gate(*args, **kwargs)

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=replace(
            _policy(lanes=1, max_concurrency=1),
            fail_fast=False,
        ),
        repo_root=repo,
        base_commit=base,
        backend_factory=backend_factory,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "raised-indeterminate-writer"),
        integration_gate=integration_gate,
    )

    assert backend is not None
    assert backend.invoke_calls == 1
    assert backend.close_calls == 1
    assert gate_calls == 0
    assert result.status == "failed"
    child = result.children[0]
    assert child.error_kind == "tool_mutation_indeterminate"
    assert child.token_usage_complete is False
    assert child.cost_complete is False
    assert child.pricing_evidence is None
    assert child.backend_cleanup_complete is True
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_raised_indeterminate_retains_direct_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    gate_calls = 0
    capture_calls = 0
    backend: _RaisedIndeterminateWriter | None = None
    real_capture = writer_scheduler_module.capture_workspace_delta

    def backend_factory(request, toolset, _debug, _worktree):
        nonlocal backend
        backend = _RaisedIndeterminateWriter(
            request,
            toolset,
            close_settles=False,
        )
        return backend

    def integration_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return _gate(*args, **kwargs)

    def observed_capture(*args, **kwargs):
        nonlocal capture_calls
        capture_calls += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(
        writer_scheduler_module,
        "capture_workspace_delta",
        observed_capture,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=replace(
                _policy(lanes=1, max_concurrency=1),
                fail_fast=False,
            ),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "unsettled-indeterminate-writer"),
            integration_gate=integration_gate,
        )

    assert backend is not None
    assert backend.invoke_calls == 1
    assert backend.close_calls == 1
    assert gate_calls == 0
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, WriterCleanupError)
    assert cleanup_owner.primary_error is caught.value
    assert cleanup_owner.partial_result is not None
    child = cleanup_owner.partial_result.children[0]
    retained_worktree = Path(child.worktree)
    assert child.error_kind == "tool_mutation_indeterminate"
    assert child.token_usage_complete is False
    assert child.cost_complete is False
    assert child.pricing_evidence is None
    assert capture_calls == 0
    assert cleanup_owner.retained_backend_ids == ("lane-0",)
    assert cleanup_owner.retained_journal_ids
    assert cleanup_owner.retained_epoch_id is not None
    assert retained_worktree.is_dir()
    assert (retained_worktree / "a.txt").read_text(encoding="utf-8") == (
        "indeterminate\n"
    )
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""

    backend.close_settles = True
    settled = cleanup_owner.retry_cleanup()
    assert settled is not None
    assert settled.children[0].error_kind == "tool_mutation_indeterminate"
    assert backend.close_calls == 2
    assert capture_calls == 0
    assert cleanup_owner.retained_backend_ids == ()
    assert cleanup_owner.retained_journal_ids == ()
    assert cleanup_owner.retained_epoch_id is None
    assert not retained_worktree.exists()
    _assert_writer_ownership_settled(repo)


def test_writer_returned_indeterminate_malformed_usage_is_unknown_lower_bound(
    tmp_path: Path,
) -> None:
    from assay.types.invocation import TransportError

    repo, base = _repo(tmp_path)
    gate_calls = 0

    class _MalformedUsageIndeterminateWriter:
        name = "synthetic"
        model = "malformed-usage-indeterminate-writer"

        def __init__(self, request, toolset) -> None:
            self.request = request
            self.toolset = toolset

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            self.toolset.dispatch(
                "write_source",
                {"path": "a.txt", "content": "indeterminate\n"},
            )
            complete = _complete_result(self.request.lane_id)
            terminal = replace(
                complete,
                stop_reason="tool_mutation_indeterminate",
                error=TransportError(
                    message="mutation settlement and usage are unknown",
                    retriable=False,
                ),
            )
            object.__setattr__(terminal.usage, "cost_usd", float("nan"))
            return terminal

        def close(self) -> None:
            return None

    def integration_gate(*args, **kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return _gate(*args, **kwargs)

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=replace(
            _policy(lanes=1, max_concurrency=1),
            fail_fast=False,
        ),
        repo_root=repo,
        base_commit=base,
        backend_factory=(
            lambda request, toolset, _debug, _worktree: (
                _MalformedUsageIndeterminateWriter(request, toolset)
            )
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "malformed-indeterminate-writer"),
        integration_gate=integration_gate,
    )

    assert gate_calls == 0
    child = result.children[0]
    assert child.error_kind == "tool_mutation_indeterminate"
    assert child.cost_usd == 0.0
    assert child.input_tokens == 0
    assert child.output_tokens == 0
    assert child.cache_creation_tokens == 0
    assert child.cache_read_tokens == 0
    assert child.token_usage_complete is False
    assert child.cost_complete is False
    assert child.pricing_evidence is None
    assert "unknown lower bound" in (child.error_message or "")
    assert result.evidence_complete is False
    _assert_writer_ownership_settled(repo)


def test_writer_scheduler_preserves_cache_usage_and_lane_pricing_evidence(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(
                input_tokens=10,
                output_tokens=5,
                cache_creation_tokens=2,
                cache_read_tokens=5,
                cost_usd=0.1,
            ),
        ),
        complete=True,
    )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "priced writer\n")},
            cache_creation_tokens=2,
            cache_read_tokens=5,
            pricing_evidence=evidence,
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "priced-writer"),
        integration_gate=_gate,
    )

    assert result.successful
    assert result.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
        cache_creation_tokens=2,
        cache_read_tokens=5,
    )
    assert result.children[0].pricing_evidence is evidence
    summary = result.to_summary()
    assert summary["schema_version"] == "crucible.writer-fanout-result.v2"
    assert summary["cache_creation_tokens"] == 2
    assert summary["cache_read_tokens"] == 5
    assert summary["children"][0]["pricing_evidence"] == {
        "segments": [
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_tokens": 2,
                "cache_read_tokens": 5,
                "cost_usd": 0.1,
            },
        ],
        "complete": True,
    }


def test_materialized_worktree_allows_bounded_direct_tool_dispatch(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)

    class _DirectDispatchBackend:
        name = "synthetic"
        model = "direct-dispatch"

        def __init__(self, request, toolset) -> None:
            self.request = request
            self.toolset = toolset

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            completed = threading.Event()
            response: list[str] = []

            def dispatch() -> None:
                response.append(
                    self.toolset.dispatch(
                        "write_source",
                        {
                            "file_path": "a.txt",
                            "content": "direct-dispatch\n",
                            "expected_sha256": hashlib.sha256(b"base-a\n").hexdigest(),
                        },
                    )
                )
                completed.set()

            worker = threading.Thread(
                target=dispatch,
                daemon=True,
                name="direct-tool-dispatch-probe",
            )
            worker.start()
            if not completed.wait(2):
                raise RuntimeError(
                    "direct ToolSet dispatch deadlocked on materialized "
                    "worktree ownership"
                )
            worker.join(timeout=1)
            assert response and not response[0].startswith("Error")
            return _complete_result(self.request.lane_id)

        def close(self) -> None:
            return None

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=(
            lambda request, toolset, _debug, _worktree: _DirectDispatchBackend(
                request, toolset
            )
        ),
        toolset_factory=lambda _request, context: build_write_tools(context).filter_to(
            {"write_source"}
        ),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
    )

    assert result.successful
    assert b"+direct-dispatch" in result.integrated_patch
    assert _git(repo, "status", "--porcelain") == ""


def test_writer_transient_close_failure_retries_and_withholds_patch(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()

    class _FailOnceWriter(_WriterBackend):
        def __init__(
            self,
            request: WriterLaneRequest,
            toolset: _WriterToolSet,
        ) -> None:
            super().__init__(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            )
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("transient writer close failure")
            super().close()

    backends: list[_FailOnceWriter] = []

    def _backend_factory(request, toolset, _debug_dir, _worktree):
        backend = _FailOnceWriter(request, toolset)
        backends.append(backend)
        return backend

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=_backend_factory,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert len(backends) == 1
    assert backends[0].close_calls == 2
    assert backends[0].closed.is_set()
    assert result.status == "cleanup_failed"
    assert result.cancellation_drained
    assert result.integrated_patch == b""
    assert result.children[0].status == "cleanup_failed"
    assert not result.children[0].backend_cleanup_complete
    assert result.children[0].error_kind == "RuntimeError"
    assert _git(repo, "status", "--porcelain") == ""


def test_writer_provider_generation_closes_before_source_frontier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    events: list[str] = []
    generation: object | None = None
    real_capture = writer_scheduler_module.capture_workspace_delta

    class ProviderGeneration:
        closed = False

        def __enter__(self):
            events.append("provider_enter")

            def backend_factory(request, toolset, _debug, _worktree):
                events.append("backend_factory")
                return _WriterBackend(
                    request,
                    toolset,
                    tracker,
                    writes={"lane-0": ("a.txt", "provider-bound\n")},
                )

            return backend_factory

        def __exit__(self, *_exc_info):
            events.append("provider_close")
            self.closed = True
            return False

    def provider_factory(members):
        nonlocal generation
        assert len(members) == 2
        assert members[0].parent_transaction_id is None
        assert members[1].parent_transaction_id == members[0].transaction_id
        generation = ProviderGeneration()
        return generation

    def capture_after_provider_close(*args, **kwargs):
        assert generation is not None
        assert generation.closed
        events.append("source_capture")
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(
        writer_scheduler_module,
        "capture_workspace_delta",
        capture_after_provider_close,
    )
    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda *_args: pytest.fail(
            "unbound backend factory must not run"
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "provider-generation-order"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
        provider_generation_factory=provider_factory,
    )

    assert result.successful
    assert events.index("provider_enter") < events.index("backend_factory")
    assert events.index("backend_factory") < events.index("provider_close")
    assert events.index("provider_close") < events.index("source_capture")
    _assert_writer_ownership_settled(repo)


def test_writer_provider_close_failure_retains_family_until_caller_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    restore_calls: list[str] = []
    real_restore = WorkspaceMutationJournal.restore

    class ProviderGeneration:
        closed = False
        retry_calls = 0

        def __enter__(self):
            return lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "provider-retry\n")},
            )

        def retry_cleanup(self) -> None:
            self.retry_calls += 1
            if self.retry_calls == 1:
                raise BackendCleanupError(
                    (("provider pool", RuntimeError("still live")),),
                    retry_cleanup=self.retry_cleanup,
                )
            self.closed = True

        def __exit__(self, *_exc_info):
            raise BackendCleanupError(
                (("provider pool", RuntimeError("first close failed")),),
                retry_cleanup=self.retry_cleanup,
            )

    generation = ProviderGeneration()

    def restore_only_after_provider_close(journal, *, full=None, _defer_terminal_cleanup=False):
        assert generation.closed
        restore_calls.append(journal.slice_id)
        return real_restore(journal, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        restore_only_after_provider_close,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "unbound backend factory must not run"
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "provider-close-retry"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            provider_generation_factory=lambda _members: generation,
        )

    cleanup_error = caught.value
    assert cleanup_error.provider_generation_retained
    assert cleanup_error.retained_epoch_id is not None
    assert restore_calls == []
    partial = cleanup_error.retry_cleanup()

    assert partial is not None
    assert partial.cancellation_drained
    assert generation.closed
    assert generation.retry_calls == 2
    assert restore_calls
    assert not cleanup_error.provider_generation_retained
    _assert_writer_ownership_settled(repo)


def test_writer_provider_enter_cleanup_failure_has_typed_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    restore_calls: list[str] = []
    real_restore = WorkspaceMutationJournal.restore

    class ProviderGeneration:
        closed = False
        retry_calls = 0

        def retry_cleanup(self) -> None:
            self.retry_calls += 1
            if self.retry_calls < 3:
                raise BackendCleanupError(
                    (("provider setup", RuntimeError("setup owner live")),),
                    retry_cleanup=self.retry_cleanup,
                )
            self.closed = True

        def __enter__(self):
            raise BackendCleanupError(
                (("provider setup", RuntimeError("enter failed")),),
                retry_cleanup=self.retry_cleanup,
            )

        def __exit__(self, *_exc_info):
            pytest.fail("failed provider entry must use its retry owner")

    generation = ProviderGeneration()

    def restore_only_after_provider_close(journal, *, full=None, _defer_terminal_cleanup=False):
        assert generation.closed
        restore_calls.append(journal.slice_id)
        return real_restore(journal, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        restore_only_after_provider_close,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "backend factory must not run after provider enter failure"
            ),
            toolset_factory=lambda *_args: pytest.fail(
                "toolset factory must not run after provider enter failure"
            ),
            state_root=_state_root(tmp_path, "provider-enter-retry"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            provider_generation_factory=lambda _members: generation,
        )

    cleanup_error = caught.value
    assert cleanup_error.provider_generation_retained
    assert cleanup_error.retained_epoch_id is not None
    assert restore_calls == []
    assert cleanup_error.retry_cleanup() is None
    assert generation.closed
    assert generation.retry_calls == 3
    assert restore_calls
    _assert_writer_ownership_settled(repo)


def test_writer_provider_factory_cleanup_failure_has_typed_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    restore_calls: list[str] = []
    real_restore = WorkspaceMutationJournal.restore

    class ProviderFactoryOwner:
        closed = False
        retry_calls = 0

        def retry_cleanup(self) -> None:
            self.retry_calls += 1
            if self.retry_calls < 3:
                raise BackendCleanupError(
                    (("provider factory", RuntimeError("factory owner live")),),
                    retry_cleanup=self.retry_cleanup,
                )
            self.closed = True

        def create(self, _members):
            raise BackendCleanupError(
                (("provider factory", RuntimeError("factory failed")),),
                retry_cleanup=self.retry_cleanup,
            )

    owner = ProviderFactoryOwner()

    def restore_only_after_provider_close(journal, *, full=None, _defer_terminal_cleanup=False):
        assert owner.closed
        restore_calls.append(journal.slice_id)
        return real_restore(journal, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        restore_only_after_provider_close,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "backend factory must not run after provider factory failure"
            ),
            toolset_factory=lambda *_args: pytest.fail(
                "toolset factory must not run after provider factory failure"
            ),
            state_root=_state_root(tmp_path, "provider-factory-retry"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            provider_generation_factory=owner.create,
        )

    cleanup_error = caught.value
    assert cleanup_error.provider_generation_retained
    assert cleanup_error.retained_epoch_id is not None
    assert restore_calls == []
    assert cleanup_error.retry_cleanup() is None
    assert owner.closed
    assert owner.retry_calls == 3
    assert restore_calls
    _assert_writer_ownership_settled(repo)


@pytest.mark.parametrize("failure_stage", ("factory", "protocol", "enter"))
def test_writer_untyped_provider_setup_failure_retains_epoch_without_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    repo, base = _repo(tmp_path)
    resource_closed = False
    events: list[str] = []
    restore_calls: list[str] = []
    real_restore = WorkspaceMutationJournal.restore

    class ProviderGeneration:
        def __enter__(self):
            events.append("enter_published_writer")
            raise RuntimeError("untyped enter failure after writer publication")

        def __exit__(self, *_exc_info):
            events.append("unsafe_exit")
            pytest.fail("failed __enter__ must not imply safe __exit__ cleanup")

    def provider_factory(_members):
        if failure_stage == "factory":
            events.append("factory_published_writer")
            raise RuntimeError("untyped factory failure after writer publication")
        if failure_stage == "protocol":
            events.append("factory_returned_invalid_owner")
            return object()
        return ProviderGeneration()

    def restore_only_after_explicit_cleanup(journal, *, full=None, _defer_terminal_cleanup=False):
        assert resource_closed
        restore_calls.append(journal.slice_id)
        return real_restore(journal, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        restore_only_after_explicit_cleanup,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "backend factory must not run after provider setup failure"
            ),
            toolset_factory=lambda *_args: pytest.fail(
                "toolset factory must not run after provider setup failure"
            ),
            state_root=_state_root(
                tmp_path,
                f"provider-untyped-{failure_stage}",
            ),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            provider_generation_factory=provider_factory,
        )

    cleanup_error = caught.value
    assert cleanup_error.provider_generation_retained
    assert cleanup_error.retained_epoch_id is not None
    assert restore_calls == []
    assert "unsafe_exit" not in events
    with pytest.raises(WriterCleanupError) as still_caught:
        cleanup_error.retry_cleanup()
    still_live = still_caught.value
    assert still_live.provider_generation_retained
    assert still_live.retained_epoch_id is not None
    assert restore_calls == []
    assert "unsafe_exit" not in events

    # Test-only out-of-band recovery models an operator-specific cleanup
    # procedure becoming available after the fail-closed result. Production
    # code deliberately exposes no way to invent this missing proof.
    managed_generation = still_live._retained_provider_generation
    assert managed_generation is not None
    if failure_stage == "protocol":
        assert managed_generation._context is not None

    def explicit_cleanup() -> None:
        nonlocal resource_closed
        resource_closed = True

    with managed_generation._lock:
        managed_generation._retry_cleanup = explicit_cleanup
    assert still_live.retry_cleanup() is None
    assert resource_closed
    assert restore_calls
    _assert_writer_ownership_settled(repo)


def test_writer_untyped_provider_exit_failure_retains_epoch_without_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    resource_closed = False
    exit_calls = 0
    restore_calls: list[str] = []
    captures = 0
    real_restore = WorkspaceMutationJournal.restore
    real_capture = writer_scheduler_module.capture_workspace_delta

    class ProviderGeneration:
        def __enter__(self):
            return lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "provider-exit-failure\n")},
            )

        def __exit__(self, *_exc_info):
            nonlocal exit_calls
            exit_calls += 1
            raise RuntimeError("untyped exit failure with provider still live")

    def restore_only_after_explicit_cleanup(journal, *, full=None, _defer_terminal_cleanup=False):
        assert resource_closed
        restore_calls.append(journal.slice_id)
        return real_restore(journal, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def capture_only_after_provider_settlement(*args, **kwargs):
        nonlocal captures
        assert resource_closed
        captures += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        restore_only_after_explicit_cleanup,
    )
    monkeypatch.setattr(
        writer_scheduler_module,
        "capture_workspace_delta",
        capture_only_after_provider_settlement,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "unbound backend factory must not run"
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "provider-untyped-exit"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            provider_generation_factory=lambda _members: ProviderGeneration(),
        )

    cleanup_error = caught.value
    assert cleanup_error.provider_generation_retained
    assert cleanup_error.retained_epoch_id is not None
    assert exit_calls == 1
    assert captures == 0
    assert restore_calls == []
    with pytest.raises(WriterCleanupError) as still_caught:
        cleanup_error.retry_cleanup()
    still_live = still_caught.value
    assert still_live.provider_generation_retained
    assert still_live.retained_epoch_id is not None
    assert exit_calls == 1
    assert captures == 0
    assert restore_calls == []

    managed_generation = still_live._retained_provider_generation
    assert managed_generation is not None

    def explicit_cleanup() -> None:
        nonlocal resource_closed
        resource_closed = True

    with managed_generation._lock:
        managed_generation._retry_cleanup = explicit_cleanup
    partial = still_live.retry_cleanup()
    assert partial is not None
    assert partial.cancellation_drained
    assert resource_closed
    assert captures == 0
    assert restore_calls
    _assert_writer_ownership_settled(repo)


def test_writer_provider_generation_settles_on_precancelled_fanout(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    cancel = threading.Event()
    cancel.set()
    events: list[str] = []

    class ProviderGeneration:
        def __enter__(self):
            events.append("enter")
            return lambda *_args: pytest.fail(
                "pre-cancelled lane must not construct a backend"
            )

        def __exit__(self, *_exc_info):
            events.append("close")
            return False

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda *_args: pytest.fail(
            "unbound backend factory must not run"
        ),
        toolset_factory=lambda *_args: pytest.fail(
            "pre-cancelled lane must not construct a toolset"
        ),
        state_root=_state_root(tmp_path, "provider-cancel"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
        cancel_event=cancel,
        provider_generation_factory=lambda _members: ProviderGeneration(),
    )

    assert result.status == "cancelled"
    assert result.cancellation_drained
    assert events == ["enter", "close"]
    _assert_writer_ownership_settled(repo)


def test_writer_provider_generation_closes_after_inflight_cancellation_drains_backend(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    cancel = threading.Event()
    tracker = _Concurrency()
    events: list[str] = []

    class ProviderGeneration:
        def __enter__(self):
            events.append("enter")

            def backend_factory(request, toolset, _debug, _worktree):
                return _WriterBackend(
                    request,
                    toolset,
                    tracker,
                    writes={},
                    block=True,
                )

            return backend_factory

        def __exit__(self, *_exc_info):
            # Provider settlement must follow both backend closure and worker
            # join, including the external-cancellation path.
            assert tracker.active == 0
            assert tracker.closed == 2
            events.append("close")
            return False

    trigger = threading.Thread(
        target=lambda: (tracker.two_started.wait(30), cancel.set()),
        daemon=True,
    )
    trigger.start()
    try:
        result = run_isolated_writer_fanout(
            _requests((("a.txt",), ("b.txt",))),
            policy=_policy(lanes=2, max_concurrency=2),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail(
                "unbound backend factory must not run"
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "provider-inflight-cancel"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            cancel_event=cancel,
            provider_generation_factory=lambda _members: ProviderGeneration(),
        )
    finally:
        cancel.set()
        trigger.join(timeout=5)

    assert result.status == "cancelled"
    assert result.all_joined
    assert result.cancellation_drained
    assert tracker.active == 0
    assert events == ["enter", "close"]
    _assert_writer_ownership_settled(repo)


def test_writer_backend_setup_cleanup_is_drained_by_provider_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    cleanup_calls = 0
    provider_closed = False
    captures = 0
    real_capture = writer_scheduler_module.capture_workspace_delta

    class ProviderGeneration:
        pending = False

        def retry_setup_cleanup(self) -> None:
            nonlocal cleanup_calls, provider_closed
            cleanup_calls += 1
            if cleanup_calls == 1:
                raise BackendCleanupError(
                    (("backend setup", RuntimeError("setup cleanup live")),),
                    retry_cleanup=self.retry_setup_cleanup,
                )
            self.pending = False
            provider_closed = True

        def __enter__(self):
            def fail_backend_setup(*_args):
                self.pending = True
                raise BackendCleanupError(
                    (("backend setup", RuntimeError("attach failed")),),
                    retry_cleanup=self.retry_setup_cleanup,
                )

            return fail_backend_setup

        def __exit__(self, *_exc_info):
            nonlocal provider_closed
            if self.pending:
                self.retry_setup_cleanup()
            provider_closed = True
            return False

    def capture_after_cleanup(*args, **kwargs):
        nonlocal captures
        assert provider_closed
        captures += 1
        return real_capture(*args, **kwargs)

    monkeypatch.setattr(
        writer_scheduler_module,
        "capture_workspace_delta",
        capture_after_cleanup,
    )
    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda *_args: pytest.fail(
            "unbound backend factory must not run"
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "provider-setup-cleanup"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
        provider_generation_factory=lambda _members: ProviderGeneration(),
    )

    assert result.status == "cancelled"
    assert result.cancellation_drained
    assert result.children[0].error_kind == "BackendCleanupError"
    assert cleanup_calls == 2
    assert provider_closed
    assert captures == 1
    _assert_writer_ownership_settled(repo)


def test_writer_transient_worktree_cleanup_settles_without_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    close_attempts = 0

    def fail_first_worktree_close(self):
        nonlocal close_attempts
        close_attempts += 1
        if close_attempts == 1:
            raise RuntimeError("transient worktree cleanup failure")
        return real_close_locked(self)

    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "_close_locked",
        fail_first_worktree_close,
    )
    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "transient-worktree-cleanup"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert close_attempts == 2
    assert result.status == "cleanup_failed"
    assert result.integrated_patch == b""
    assert result.children[0].status == "cleanup_failed"
    assert not result.children[0].worktree_cleanup_complete
    assert result.children[0].error_kind == "WriterSchedulerError"
    assert "recovered failures" in (result.children[0].error_message or "")
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_cleanup_error_requires_live_retry_authority() -> None:
    with pytest.raises(ValueError, match="live retryable cleanup authority"):
        WriterCleanupError(
            primary_error=RuntimeError("primary"),
            failures=(("historical cleanup", RuntimeError("settled")),),
        )


def test_writer_backend_without_close_is_never_dispatched_or_restored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    real_restore = WorkspaceMutationJournal.restore
    restore_calls = 0
    invoked = False
    backends: list[object] = []

    class _MissingCloseWriter:
        name = "missing-close"
        model = "missing-close"

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            nonlocal invoked
            invoked = True
            return _complete_result("lane-0")

    def backend_factory(*_args):
        backend = _MissingCloseWriter()
        backends.append(backend)
        return backend

    def observe_restore(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal restore_calls
        restore_calls += 1
        assert callable(getattr(backends[0], "close", None))
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_restore,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "missing-close"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    assert not invoked
    assert restore_calls == 0
    assert cleanup_error.retained_backend_ids == ("lane-0",)
    assert cleanup_error.retained_journal_ids
    assert any(resolver.owned for _label, resolver in cleanup_error.retained_resolvers)

    setattr(backends[0], "close", lambda: None)
    settled = cleanup_error.retry_cleanup()

    assert settled is not None
    assert settled.cancellation_drained
    assert restore_calls
    assert cleanup_error.retained_backend_ids == ()
    assert cleanup_error.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    _assert_writer_ownership_settled(repo)


def test_writer_backend_uses_prebound_close_after_descriptor_revocation(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    invoked = False
    closed = False

    class _RevokesCloseWriter:
        name = "revokes-close"
        model = "revokes-close"

        def __init__(self, toolset: _WriterToolSet) -> None:
            self.toolset = toolset

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            nonlocal invoked
            invoked = True
            type(self).close = None
            self.toolset.dispatch(
                "write_source",
                {"path": "a.txt", "content": "changed\n"},
            )
            return _complete_result("lane-0")

        def close(self) -> None:
            nonlocal closed
            closed = True

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=(
            lambda _request, toolset, _debug, _worktree: _RevokesCloseWriter(toolset)
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "revoked-close"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert invoked
    assert closed
    assert result.successful
    _assert_writer_ownership_settled(repo)


def test_writer_factory_failure_without_cleanup_owner_retains_armed_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    real_restore = WorkspaceMutationJournal.restore
    restore_calls: list[str] = []

    def observe_restore(self, *, full=None, _defer_terminal_cleanup=False):
        restore_calls.append(self.slice_id)
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def unsafe_toolset_factory(_request, context):
        (context.repo_root / "a.txt").write_text(
            "factory-entered\n",
            encoding="utf-8",
        )
        raise RuntimeError("factory failed without returning a cleanup owner")

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_restore,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda *_args: pytest.fail("backend factory must not run"),
            toolset_factory=unsafe_toolset_factory,
            state_root=_state_root(tmp_path, "unowned-factory"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    assert cleanup_error.retained_epoch_id is not None
    assert restore_calls == []
    assert cleanup_error.partial_result is not None
    assert not cleanup_error.partial_result.cancellation_drained
    child_worktree = Path(cleanup_error.partial_result.children[0].worktree)
    assert (child_worktree / "a.txt").read_text(encoding="utf-8") == (
        "factory-entered\n"
    )
    assert _git(repo, "status", "--porcelain") == ""

    with pytest.raises(WriterCleanupError) as retried:
        cleanup_error.retry_cleanup()

    assert retried.value.retained_epoch_id == cleanup_error.retained_epoch_id
    assert restore_calls == []
    assert child_worktree.exists()


def test_writer_late_mutator_is_retained_until_close_and_epoch_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    release_writer = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()
    backend_instances: list[object] = []
    restore_observations: list[str] = []
    real_restore = WorkspaceMutationJournal.restore

    class _LateWriter:
        name = "late-writer"
        model = "late-writer"

        def __init__(self, toolset: _WriterToolSet) -> None:
            self.toolset = toolset
            self.thread: threading.Thread | None = None

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            def write_late() -> None:
                writer_started.set()
                assert release_writer.wait(10)
                self.toolset.dispatch(
                    "write_source",
                    {"path": "a.txt", "content": "late-write\n"},
                )
                writer_finished.set()

            self.thread = threading.Thread(
                target=write_late,
                daemon=True,
                name="writer-epoch-late-mutator",
            )
            self.thread.start()
            assert writer_started.wait(5)
            return _complete_result("lane-0")

        def close(self) -> None:
            thread = self.thread
            if thread is not None and thread.is_alive():
                raise RuntimeError("late writer is still live")

    def backend_factory(_request, toolset, _debug, _worktree):
        backend = _LateWriter(toolset)
        backend_instances.append(backend)
        return backend

    def observe_restore(self, *, full=None, _defer_terminal_cleanup=False):
        if self.slice_id.startswith("writer-lane::"):
            cwd = Path(str(self._manifest["cwd"]))
            restore_observations.append((cwd / "a.txt").read_text(encoding="utf-8"))
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_restore,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "late-mutator"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    assert cleanup_error.retained_epoch_id is not None
    assert cleanup_error.retained_backend_ids == ("lane-0",)
    assert cleanup_error.partial_result is not None
    assert not cleanup_error.partial_result.cancellation_drained
    assert restore_observations == []
    assert len(backend_instances) == 1

    release_writer.set()
    backend = backend_instances[0]
    assert isinstance(backend, _LateWriter)
    assert backend.thread is not None
    backend.thread.join(timeout=5)
    assert writer_finished.is_set()

    settled = cleanup_error.retry_cleanup()

    assert settled is not None
    assert settled.cancellation_drained
    assert restore_observations
    assert all(
        observation == "late-write\n"
        for observation in restore_observations
    )
    assert cleanup_error.retained_epoch_id is None
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_worktree_cleanup_error_requires_live_retry_authority(
    tmp_path: Path,
) -> None:
    resolver = worktree_module.WorktreeResolver(tmp_path)
    with pytest.raises(ValueError, match="live cleanup authority"):
        worktree_module.WorktreeCleanupError(
            owner=resolver,
            primary_error=RuntimeError("primary"),
            cleanup_errors=(RuntimeError("settled"),),
        )


def test_writer_persistent_backend_cleanup_has_caller_retry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    restore_close_counts: list[int] = []

    class _NeedsCallerRetryWriter(_WriterBackend):
        close_calls = 0

        def __init__(
            self,
            request: WriterLaneRequest,
            toolset: _WriterToolSet,
        ) -> None:
            super().__init__(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            )

        def close(self) -> None:
            type(self).close_calls += 1
            if type(self).close_calls <= 5:
                raise RuntimeError("persistent writer close failure")
            super().close()

    def observe_restore_after_backend_close(self, *, full=None, _defer_terminal_cleanup=False):
        restore_close_counts.append(_NeedsCallerRetryWriter.close_calls)
        assert _NeedsCallerRetryWriter.close_calls >= 6
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_restore_after_backend_close,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=(
                lambda request, toolset, _debug_dir, _worktree: _NeedsCallerRetryWriter(
                    request, toolset
                )
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert _NeedsCallerRetryWriter.close_calls == 4
    assert cleanup_error.retained_backend_ids == ("lane-0",)
    assert not partial.cancellation_drained
    assert partial.status == "cleanup_failed"
    assert partial.integrated_patch == b""
    assert partial.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    assert partial.children[0].status == "cleanup_failed"
    assert partial.children[0].error_kind == "BackendCleanupError"
    assert restore_close_counts == []
    assert cleanup_error.retained_journal_ids
    assert any(resolver.owned for _label, resolver in cleanup_error.retained_resolvers)
    assert _git(repo, "status", "--porcelain") == ""

    with pytest.raises(WriterCleanupError) as retried:
        cleanup_error.retry_cleanup()

    replacement = retried.value
    assert _NeedsCallerRetryWriter.close_calls == 5
    assert restore_close_counts == []
    assert replacement.retained_backend_ids == ("lane-0",)
    assert replacement.retained_journal_ids
    assert any(resolver.owned for _label, resolver in replacement.retained_resolvers)

    settled = replacement.retry_cleanup()
    assert settled is not None
    assert _NeedsCallerRetryWriter.close_calls == 6
    assert restore_close_counts
    assert all(close_calls >= 6 for close_calls in restore_close_counts)
    assert cleanup_error.retained_backend_ids == ()
    assert cleanup_error.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert settled.cancellation_drained
    assert not settled.successful
    assert settled.usage == partial.usage


def test_writer_outer_unwind_retains_backend_for_caller_retry(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()

    class _FatalLaneEscape(BaseException):
        pass

    class _NeedsOuterRetryWriter(_WriterBackend):
        close_calls = 0

        def __init__(
            self,
            request: WriterLaneRequest,
            toolset: _WriterToolSet,
        ) -> None:
            super().__init__(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            )

        def invoke(self, _invocation: Invocation) -> InvocationResult:
            raise _FatalLaneEscape("fatal worker escape")

        def close(self) -> None:
            type(self).close_calls += 1
            if type(self).close_calls <= 6:
                raise RuntimeError("persistent outer-unwind close failure")
            super().close()

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=(
                lambda request, toolset, _debug_dir, _worktree: _NeedsOuterRetryWriter(
                    request, toolset
                )
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert partial.usage == Usage()
    assert partial.all_joined
    assert not partial.cancellation_drained
    assert not partial.successful
    child = partial.children[0]
    assert child.status == "cleanup_failed"
    assert child.cost_usd == 0.0
    assert child.input_tokens == 0
    assert child.output_tokens == 0
    assert not child.token_usage_complete
    assert not child.cost_complete
    assert not child.terminal_complete
    assert not child.trace_complete
    assert not child.backend_cleanup_complete
    assert cleanup_error.retained_backend_ids == ("lane-0",)
    assert _NeedsOuterRetryWriter.close_calls == 6
    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert settled.cancellation_drained
    assert not settled.successful
    assert settled.usage == partial.usage
    assert _NeedsOuterRetryWriter.close_calls == 7
    assert cleanup_error.retained_backend_ids == ()
    assert tracker.closed == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_fatal_sibling_preserves_completed_cost_and_retry_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    release_fatal = threading.Event()
    real_wait = writer_scheduler_module.wait
    real_restore = WorkspaceMutationJournal.restore
    fatal_lane_restore_calls = 0

    class _FatalSiblingEscape(BaseException):
        pass

    class _FatalSiblingWriter(_WriterBackend):
        bad_close_calls = 0

        def invoke(self, invocation: Invocation) -> InvocationResult:
            if self.request.lane_id == "lane-0":
                return super().invoke(invocation)
            self.tracker.start()
            try:
                assert release_fatal.wait(5)
                raise _FatalSiblingEscape("fatal sibling after completed lane")
            finally:
                self.tracker.finish()

        def close(self) -> None:
            if self.request.lane_id == "lane-1":
                type(self).bad_close_calls += 1
                if type(self).bad_close_calls <= 6:
                    raise RuntimeError("persistent fatal-sibling close failure")
            super().close()

    def release_after_first_completion(
        futures,
        timeout=None,
        return_when=None,
    ):
        if return_when is None:
            completed = real_wait(futures, timeout=timeout)
        else:
            completed = real_wait(
                futures,
                timeout=timeout,
                return_when=return_when,
            )
        done, _pending = completed
        if (
            return_when == writer_scheduler_module.FIRST_COMPLETED
            and done
            and not release_fatal.is_set()
        ):
            assert len(done) == 1
            release_fatal.set()
        return completed

    def observe_fatal_lane_restore(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal fatal_lane_restore_calls
        if self.slice_id == "writer-lane::lane-1":
            fatal_lane_restore_calls += 1
            assert _FatalSiblingWriter.bad_close_calls >= 7
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        writer_scheduler_module,
        "wait",
        release_after_first_completion,
    )
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_fatal_lane_restore,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",), ("b.txt",))),
            policy=_policy(),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: (
                _FatalSiblingWriter(
                    request,
                    toolset,
                    tracker,
                    writes={
                        "lane-0": ("a.txt", "completed\n"),
                        "lane-1": ("b.txt", "never-written\n"),
                    },
                )
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "fatal-sibling"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert tuple(child.lane_id for child in partial.children) == (
        "lane-0",
        "lane-1",
    )
    completed, fatal = partial.children
    assert completed.result is not None
    assert completed.cost_usd == pytest.approx(0.1)
    assert completed.input_tokens == 10
    assert completed.output_tokens == 5
    assert completed.token_usage_complete
    assert completed.cost_complete
    assert completed.terminal_complete
    assert completed.trace_complete
    assert fatal.result is None
    assert fatal.status == "cleanup_failed"
    assert fatal.error_kind == "_FatalSiblingEscape"
    assert fatal.cost_usd == 0.0
    assert fatal.input_tokens == 0
    assert fatal.output_tokens == 0
    assert not fatal.token_usage_complete
    assert not fatal.cost_complete
    assert not fatal.terminal_complete
    assert not fatal.trace_complete
    assert not fatal.backend_cleanup_complete
    assert not fatal.journal_cleanup_complete
    assert not fatal.worktree_cleanup_complete
    assert partial.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    assert partial.all_joined
    assert not partial.cancellation_drained
    assert cleanup_error.retained_backend_ids == ("lane-1",)
    assert cleanup_error.retained_journal_ids == (
        "lane 'lane-1' mutation journal",
        "lane 'lane-0' mutation journal",
        "parent mutation journal",
    )
    assert {
        label for label, resolver in cleanup_error.retained_resolvers if resolver.owned
    } == {"lane-0", "lane-1"}
    assert fatal_lane_restore_calls == 0
    error: BaseException | None = cleanup_error
    while error is not None:
        assert not isinstance(error, KeyError)
        error = getattr(error, "primary_error", None)

    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert settled.cancellation_drained
    assert not settled.successful
    assert settled.usage == partial.usage
    assert _FatalSiblingWriter.bad_close_calls == 7
    assert fatal_lane_restore_calls >= 1
    assert cleanup_error.retained_backend_ids == ()
    assert cleanup_error.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_fatal_with_settled_cleanup_reraises_original(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()

    class _SettledFatalEscape(BaseException):
        pass

    class _SettledFatalWriter(_WriterBackend):
        def invoke(self, _invocation: Invocation) -> InvocationResult:
            raise _SettledFatalEscape("fatal escape with settled cleanup")

    with pytest.raises(
        _SettledFatalEscape,
        match="fatal escape with settled cleanup",
    ):
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: (
                _SettledFatalWriter(
                    request,
                    toolset,
                    tracker,
                    writes={"lane-0": ("a.txt", "never-written\n")},
                )
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "settled-fatal"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    assert tracker.closed == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_fatal_unwind_does_not_retain_settled_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    lane_restore_calls = 0

    class _FatalEscape(BaseException):
        pass

    class _FatalWriter(_WriterBackend):
        def invoke(self, _invocation: Invocation) -> InvocationResult:
            raise _FatalEscape("fatal escape before settled journal cleanup")

    def settle_lane_but_report_false(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal lane_restore_calls
        restored = real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)
        if (
            self.slice_id == "writer-lane::lane-0"
            and not _defer_terminal_cleanup
            and restored is True
            and not self.cleanup_pending
        ):
            lane_restore_calls += 1
            if lane_restore_calls == 1:
                return False
        return restored

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        settle_lane_but_report_false,
    )
    with pytest.raises(
        _FatalEscape,
        match="fatal escape before settled journal cleanup",
    ):
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _FatalWriter(
                request,
                toolset,
                tracker,
                writes={},
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "fatal-settled-journal"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    assert lane_restore_calls == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_journal_fail_once_is_recoverable_from_caught_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    lane_restore_failures = 0

    def fail_lane_restore_once(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal lane_restore_failures
        if (
            self.slice_id == "writer-lane::lane-0"
            and lane_restore_failures == 0
        ):
            lane_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_lane_restore_once,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "journal-fail-once"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert partial.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.1,
    )
    assert "lane 'lane-0' mutation journal" in (cleanup_error.retained_journal_ids)
    assert any(
        label == "lane-0" and resolver.owned
        for label, resolver in cleanup_error.retained_resolvers
    )
    assert cleanup_error._epoch_settlement_factory is None

    real_retry_cleanup = WorkspaceMutationJournal.retry_cleanup
    retry_started = threading.Event()
    journal_retry_entered = threading.Event()
    retry_results = []
    retry_errors: list[BaseException] = []

    def observe_journal_retry(self):
        journal_retry_entered.set()
        return real_retry_cleanup(self)

    def retry_in_thread() -> None:
        retry_started.set()
        try:
            retry_results.append(cleanup_error.retry_cleanup())
        except BaseException as error:
            retry_errors.append(error)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "retry_cleanup",
        observe_journal_retry,
    )
    retry_thread = threading.Thread(target=retry_in_thread)
    retry_thread_started = False
    try:
        with workspace_mutation_lock(repo):
            retry_thread.start()
            retry_thread_started = True
            assert retry_started.wait(1)
            assert not journal_retry_entered.wait(0.2)
            assert retry_thread.is_alive()
    finally:
        if retry_thread_started:
            # Full coverage makes exact journal restoration substantially
            # slower.  Always release the held lock and join the retry owner
            # before an assertion can unwind into the next test.
            retry_thread.join(60)

    assert not retry_thread.is_alive()
    assert journal_retry_entered.is_set()
    assert retry_errors == []
    assert len(retry_results) == 1
    settled = retry_results[0]
    assert settled is not None
    assert settled.usage == partial.usage
    assert cleanup_error.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert lane_restore_failures == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_settled_journal_false_report_does_not_retain_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    lane_restore_calls = 0

    def settle_lane_but_report_false(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal lane_restore_calls
        restored = real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)
        if (
            self.slice_id == "writer-lane::lane-0"
            and restored is True
            and not self.cleanup_pending
        ):
            lane_restore_calls += 1
            if lane_restore_calls == 1:
                return False
        return restored

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        settle_lane_but_report_false,
    )
    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "journal-settled-false"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert lane_restore_calls == 1
    assert result.status == "cleanup_failed"
    assert result.children[0].status == "cleanup_failed"
    assert not result.children[0].journal_cleanup_complete
    assert result.children[0].worktree_cleanup_complete
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_retry_settled_journal_false_report_has_no_stale_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore

    def retain_lane_once(self, *, full=None, _defer_terminal_cleanup=False):
        if self.slice_id == "writer-lane::lane-0":
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        retain_lane_once,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "journal-retry-settled-false"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        real_restore,
    )
    real_retry_cleanup = WorkspaceMutationJournal.retry_cleanup

    def settle_lane_but_report_false(self):
        settled = real_retry_cleanup(self)
        if self.slice_id == "writer-lane::lane-0":
            assert settled is True
            assert not self.cleanup_pending
            return False
        return settled

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "retry_cleanup",
        settle_lane_but_report_false,
    )
    with pytest.raises(
        WriterSchedulerError,
        match="recovered failures after all authority settled",
    ) as retried:
        caught.value.retry_cleanup()

    assert not isinstance(retried.value, WriterCleanupError)
    assert caught.value.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in caught.value.retained_resolvers
    )
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_persistent_journal_failure_returns_replacement_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    real_resolver_close = worktree_module.WorktreeResolver.close
    lane_restore_failures = 0
    resolver_close_calls = 0

    def fail_lane_restore_twice(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal lane_restore_failures
        if (
            self.slice_id == "writer-lane::lane-0"
            and lane_restore_failures < 2
        ):
            lane_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def count_resolver_close(self):
        nonlocal resolver_close_calls
        resolver_close_calls += 1
        return real_resolver_close(self)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_lane_restore_twice,
    )
    monkeypatch.setattr(
        worktree_module.WorktreeResolver,
        "close",
        count_resolver_close,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "journal-persistent"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    partial = caught.value.partial_result
    assert partial is not None
    assert partial.usage.cost_usd == pytest.approx(0.1)
    with pytest.raises(WriterCleanupError) as retried:
        caught.value.retry_cleanup()

    replacement = retried.value
    assert resolver_close_calls == 0
    assert replacement.partial_result is not None
    assert replacement.partial_result.usage == partial.usage
    assert replacement.retained_journal_ids == (
        "lane 'lane-0' mutation journal",
        "parent mutation journal",
    )
    assert any(
        label == "lane-0" and resolver.owned
        for label, resolver in replacement.retained_resolvers
    )

    settled = replacement.retry_cleanup()
    assert settled is not None
    assert settled.usage == partial.usage
    assert replacement.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in replacement.retained_resolvers
    )
    assert lane_restore_failures == 2
    assert resolver_close_calls == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_later_journal_failure_defers_earlier_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    later_restore_failures = 0

    def fail_later_restore_once(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal later_restore_failures
        if (
            self.slice_id == "writer-lane::lane-1"
            and later_restore_failures == 0
        ):
            later_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_later_restore_once,
    )
    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",), ("b.txt",))),
            policy=_policy(),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={
                    "lane-0": ("a.txt", "first\n"),
                    "lane-1": ("b.txt", "later\n"),
                },
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "ordered-journal-retry"),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert partial.usage == Usage(
        input_tokens=20,
        output_tokens=10,
        cost_usd=0.2,
    )
    assert cleanup_error.retained_journal_ids == (
        "lane 'lane-1' mutation journal",
        "lane 'lane-0' mutation journal",
        "parent mutation journal",
    )
    assert {
        label for label, resolver in cleanup_error.retained_resolvers if resolver.owned
    } == {"lane-0", "lane-1"}

    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert settled.usage == partial.usage
    assert later_restore_failures == 1
    assert cleanup_error.retained_journal_ids == ()
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_terminal_journal_marker_failure_retains_retry_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_unlink = mutation_journal_module.os.unlink
    marker_unlink_failures = 0
    backend_invoked = False

    class _MarkerFailureWriter(_WriterBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            nonlocal backend_invoked
            backend_invoked = True
            return super().invoke(invocation)

    def fail_one_registry_marker_unlink(path, *args, **kwargs):
        nonlocal marker_unlink_failures
        if (
            backend_invoked
            and str(path).startswith(".crucible-cleanup-")
            and marker_unlink_failures == 0
        ):
            marker_unlink_failures += 1
            raise OSError("injected registry marker unlink failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            mutation_journal_module.os,
            "unlink",
            fail_one_registry_marker_unlink,
        )
        with pytest.raises(WriterCleanupError) as caught:
            run_isolated_writer_fanout(
                _requests((("a.txt",),)),
                policy=_policy(lanes=1, max_concurrency=1),
                repo_root=repo,
                base_commit=base,
                backend_factory=lambda request, toolset, _debug, _worktree: (
                    _MarkerFailureWriter(
                        request,
                        toolset,
                        tracker,
                        writes={"lane-0": ("a.txt", "changed\n")},
                    )
                ),
                toolset_factory=lambda _request, context: _WriterToolSet(context),
                state_root=_state_root(tmp_path, "marker-unlink-failure"),
                integration_gate=_gate,
                parent_budget_usd=4.0,
            )

    cleanup_error = caught.value
    assert marker_unlink_failures == 1
    assert cleanup_error.partial_result is not None
    assert cleanup_error.retained_journal_ids == (
        "lane 'lane-0' mutation journal",
        "parent mutation journal",
    )
    lane_journal = cleanup_error._retained_journals[0][1]
    assert not lane_journal.active
    assert lane_journal.cleanup_pending
    assert any(
        label == "lane-0" and resolver.owned
        for label, resolver in cleanup_error.retained_resolvers
    )

    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert cleanup_error.retained_journal_ids == ()
    assert not lane_journal.cleanup_pending
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_terminal_journal_symlink_loop_conservatively_retains_owner(
    tmp_path: Path,
) -> None:
    loop = tmp_path / "registry-loop"
    loop.symlink_to(loop.name)
    transaction_id = "a" * 32
    journal = WorkspaceMutationJournal(
        tmp_path / "journal",
        {
            "state": "rolled_back",
            "transaction_id": transaction_id,
            "registry_directory": str(loop),
            "registry_marker": str(loop / f"{transaction_id}.json"),
        },
    )

    assert journal.cleanup_pending


@pytest.mark.parametrize(
    (
        "target_slice",
        "scopes",
        "expected_journal_ids",
        "expected_resolver_ids",
        "expected_backend_constructions",
        "expect_partial_result",
    ),
    (
        (
            "writer-fanout-parent",
            (("a.txt",),),
            ("parent mutation journal",),
            {"lane-0"},
            0,
            False,
        ),
        (
            "writer-lane::lane-1",
            (("a.txt",), ("b.txt",)),
            (
                "lane 'lane-1' mutation journal",
                "lane 'lane-0' mutation journal",
                "parent mutation journal",
            ),
            {"lane-0", "lane-1"},
            0,
            False,
        ),
        (
            "writer-serial-integration",
            (("a.txt",),),
            ("integration mutation journal",),
            {"integration"},
            1,
            True,
        ),
    ),
)
def test_writer_adopts_journal_owner_when_begin_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_slice: str,
    scopes: tuple[tuple[str, ...], ...],
    expected_journal_ids: tuple[str, ...],
    expected_resolver_ids: set[str],
    expected_backend_constructions: int,
    expect_partial_result: bool,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_register = mutation_journal_module._register_live_journal
    real_restore = WorkspaceMutationJournal.restore
    register_failures = 0
    target_restore_failures = 0
    backend_constructions = 0

    def fail_target_registration_once(journal):
        nonlocal register_failures
        if journal.slice_id == target_slice and register_failures == 0:
            register_failures += 1
            raise RuntimeError("injected post-publication setup failure")
        return real_register(journal)

    def fail_target_restore_twice(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal target_restore_failures
        if self.slice_id == target_slice and target_restore_failures < 2:
            target_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def backend_factory(request, toolset, _debug_dir, _worktree):
        nonlocal backend_constructions
        backend_constructions += 1
        return _WriterBackend(
            request,
            toolset,
            tracker,
            writes={
                "lane-0": ("a.txt", "changed-a\n"),
                "lane-1": ("b.txt", "changed-b\n"),
            },
        )

    monkeypatch.setattr(
        mutation_journal_module,
        "_register_live_journal",
        fail_target_registration_once,
    )
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_target_restore_twice,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests(scopes),
            policy=_policy(
                lanes=len(scopes),
                max_concurrency=len(scopes),
            ),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, target_slice.replace(":", "-")),
            integration_gate=_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    setup_error: MutationJournalCleanupError | None = None
    error: BaseException | None = cleanup_error
    while error is not None:
        if isinstance(error, MutationJournalCleanupError):
            setup_error = error
            break
        error = getattr(error, "primary_error", None)
    assert setup_error is not None
    assert setup_error.owner.slice_id == target_slice
    assert setup_error.owner.cleanup_pending
    assert register_failures == 1
    assert target_restore_failures == 2
    assert backend_constructions == expected_backend_constructions
    assert (cleanup_error.partial_result is not None) is expect_partial_result
    assert cleanup_error.retained_journal_ids == expected_journal_ids
    assert {
        label for label, resolver in cleanup_error.retained_resolvers if resolver.owned
    } == expected_resolver_ids
    retained_target = next(
        journal
        for _label, journal in cleanup_error._retained_journals
        if journal.slice_id == target_slice
    )
    assert retained_target is setup_error.owner

    settled = cleanup_error.retry_cleanup()
    assert (settled is not None) is expect_partial_result
    assert target_restore_failures == 2
    assert cleanup_error.retained_journal_ids == ()
    assert not setup_error.owner.cleanup_pending
    assert all(
        not resolver.owned for _label, resolver in cleanup_error.retained_resolvers
    )
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_mutation_journal_cleanup_error_retry_reacquires_workspace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _base = _repo(tmp_path)
    with workspace_mutation_lock(repo):
        journal = WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=_state_root(tmp_path, "direct-cleanup-owner"),
            slice_id="direct-cleanup-owner",
            scope=(),
            sequence=0,
            spec=None,
        )
    cleanup_error = MutationJournalCleanupError(
        owner=journal,
        primary_error=RuntimeError("injected setup failure"),
    )
    real_retry_cleanup = WorkspaceMutationJournal.retry_cleanup
    retry_started = threading.Event()
    journal_retry_entered = threading.Event()
    retry_errors: list[BaseException] = []

    def observe_journal_retry(self):
        journal_retry_entered.set()
        return real_retry_cleanup(self)

    def retry_in_thread() -> None:
        retry_started.set()
        try:
            cleanup_error.retry_cleanup()
        except BaseException as error:
            retry_errors.append(error)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "retry_cleanup",
        observe_journal_retry,
    )
    retry_thread = threading.Thread(target=retry_in_thread)
    retry_thread_started = False
    try:
        with workspace_mutation_lock(repo):
            retry_thread.start()
            retry_thread_started = True
            assert retry_started.wait(1)
            assert not journal_retry_entered.wait(0.2)
            assert retry_thread.is_alive()
    finally:
        if retry_thread_started:
            # Full coverage makes exact journal restoration substantially
            # slower.  Always release the held lock and join the retry owner
            # before an assertion can unwind into the next test.
            retry_thread.join(60)

    assert not retry_thread.is_alive()
    assert journal_retry_entered.is_set()
    assert retry_errors == []
    assert not journal.cleanup_pending
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_mutation_journal_cleanup_error_rejects_settled_owner(
    tmp_path: Path,
) -> None:
    settled = WorkspaceMutationJournal(
        tmp_path / "settled-journal",
        {
            "state": "rolled_back",
            "transaction_id": "a" * 32,
        },
    )

    assert not settled.cleanup_pending
    with pytest.raises(
        ValueError,
        match="requires live cleanup authority",
    ):
        MutationJournalCleanupError(
            owner=settled,
            primary_error=RuntimeError("stale owner"),
        )


@pytest.mark.parametrize("cleanup_mode", ("return_false", "raise"))
def test_journal_begin_reraises_primary_when_cleanup_settles_transiently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    repo, _base = _repo(tmp_path)
    real_register = mutation_journal_module._register_live_journal
    real_restore = WorkspaceMutationJournal.restore
    register_failures = 0
    restore_calls = 0

    def fail_registration_once(journal):
        nonlocal register_failures
        if journal.slice_id == "transiently-settled-begin" and register_failures == 0:
            register_failures += 1
            raise RuntimeError("injected primary journal setup failure")
        return real_register(journal)

    def settle_then_report_failure(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal restore_calls
        restored = real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)
        if (
            self.slice_id == "transiently-settled-begin"
            and restored is True
            and not self.cleanup_pending
        ):
            restore_calls += 1
            if cleanup_mode == "raise":
                raise OSError("injected cleanup error after settlement")
            return False
        return restored

    monkeypatch.setattr(
        mutation_journal_module,
        "_register_live_journal",
        fail_registration_once,
    )
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        settle_then_report_failure,
    )

    with workspace_mutation_lock(repo):
        with pytest.raises(
            RuntimeError,
            match="injected primary journal setup failure",
        ) as caught:
            WorkspaceMutationJournal.begin(
                cwd=repo,
                journal_root=_state_root(
                    tmp_path,
                    f"transiently-settled-begin-{cleanup_mode}",
                ),
                slice_id="transiently-settled-begin",
                scope=(),
                sequence=0,
                spec=None,
            )

    assert type(caught.value) is RuntimeError
    assert register_failures == 1
    assert restore_calls == 1
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


@pytest.mark.parametrize("cleanup_mode", ("return_false", "raise"))
def test_journal_cleanup_retry_uses_plain_error_after_transient_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_mode: str,
) -> None:
    repo, _base = _repo(tmp_path)
    with workspace_mutation_lock(repo):
        journal = WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=_state_root(
                tmp_path,
                f"transiently-settled-retry-{cleanup_mode}",
            ),
            slice_id="transiently-settled-retry",
            scope=(),
            sequence=0,
            spec=None,
        )
    cleanup_error = MutationJournalCleanupError(
        owner=journal,
        primary_error=RuntimeError("injected setup failure"),
    )
    real_retry_cleanup = WorkspaceMutationJournal.retry_cleanup

    def settle_then_report_failure(self):
        settled = real_retry_cleanup(self)
        if self is journal:
            assert settled
            if cleanup_mode == "raise":
                raise OSError("injected cleanup error after settlement")
            return False
        return settled

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "retry_cleanup",
        settle_then_report_failure,
    )

    with pytest.raises(MutationJournalError) as caught:
        cleanup_error.retry_cleanup()

    assert type(caught.value) is MutationJournalError
    assert "after all authority settled" in str(caught.value)
    assert not journal.cleanup_pending
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_journal_begin_sweeper_preserves_typed_owner_when_marker_is_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _base = _repo(tmp_path)
    real_register = mutation_journal_module._register_live_journal
    real_restore = WorkspaceMutationJournal.restore
    real_is_file = Path.is_file
    register_failures = 0
    restore_calls = 0

    def fail_direct_registration_once(journal):
        nonlocal register_failures
        if journal.slice_id == "hidden-marker-owner" and register_failures == 0:
            register_failures += 1
            raise RuntimeError("injected setup failure with hidden marker")
        return real_register(journal)

    def fail_first_direct_restore(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal restore_calls
        if self.slice_id == "hidden-marker-owner":
            restore_calls += 1
            if restore_calls == 1:
                return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def hide_registry_marker(path):
        if path.parent.name == "crucible-mutation-journals" and path.name.endswith(
            ".json"
        ):
            return False
        return real_is_file(path)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            mutation_journal_module,
            "_register_live_journal",
            fail_direct_registration_once,
        )
        patcher.setattr(
            WorkspaceMutationJournal,
            "restore",
            fail_first_direct_restore,
        )
        patcher.setattr(Path, "is_file", hide_registry_marker)
        with workspace_mutation_lock(repo):
            with pytest.raises(MutationJournalCleanupError) as caught:
                WorkspaceMutationJournal.begin(
                    cwd=repo,
                    journal_root=_state_root(tmp_path, "hidden-marker-owner"),
                    slice_id="hidden-marker-owner",
                    scope=(),
                    sequence=0,
                    spec=None,
                )

        cleanup_error = caught.value
        assert cleanup_error.owner.cleanup_pending
        assert cleanup_error.owner.journal_dir.is_dir()
        assert (cleanup_error.owner.journal_dir / "manifest.json").is_file()

    cleanup_error.retry_cleanup()
    assert register_failures == 1
    assert restore_calls == 1
    assert not cleanup_error.owner.cleanup_pending
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


@pytest.mark.parametrize("fail_unwind_listing", (False, True))
def test_journal_begin_sweeper_preserves_descendant_owner_and_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_unwind_listing: bool,
) -> None:
    repo, _base = _repo(tmp_path)
    journal_root = tmp_path / "synthetic-journals"
    journal_root.mkdir()
    candidate = journal_root / "new-parent"
    owner_dir = candidate / "nested_mutations" / "retained-child"
    primary_error = RuntimeError("synthetic retained descendant")
    owner: WorkspaceMutationJournal | None = None
    real_iterdir = Path.iterdir
    root_listing_calls = 0

    def fail_with_descendant_owner(cls, **_kwargs):
        nonlocal owner
        owner_dir.mkdir(parents=True)
        owner = WorkspaceMutationJournal(
            owner_dir,
            {"state": "invoking"},
        )
        raise MutationJournalCleanupError(
            owner=owner,
            primary_error=primary_error,
        )

    def optionally_fail_unwind_listing(path):
        nonlocal root_listing_calls
        if path == journal_root:
            root_listing_calls += 1
            if fail_unwind_listing and root_listing_calls == 2:
                raise OSError("injected unwind journal-root listing failure")
        return real_iterdir(path)

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "_begin_unchecked",
        classmethod(fail_with_descendant_owner),
    )
    monkeypatch.setattr(Path, "iterdir", optionally_fail_unwind_listing)

    with pytest.raises(MutationJournalCleanupError) as caught:
        WorkspaceMutationJournal.begin(
            cwd=repo,
            journal_root=journal_root,
            slice_id="synthetic-descendant",
            scope=(),
            sequence=0,
            spec=None,
        )

    assert caught.value.primary_error is primary_error
    assert caught.value.owner is owner
    assert candidate.is_dir()
    assert owner_dir.is_dir()
    assert root_listing_calls == 2


def test_writer_settled_begin_cleanup_reraises_original_setup_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    real_register = mutation_journal_module._register_live_journal
    real_restore = WorkspaceMutationJournal.restore
    register_failures = 0
    parent_restore_failures = 0
    backend_constructed = False

    def fail_parent_registration_once(journal):
        nonlocal register_failures
        if journal.slice_id == "writer-fanout-parent" and register_failures == 0:
            register_failures += 1
            raise RuntimeError("injected settled journal setup failure")
        return real_register(journal)

    def fail_first_parent_restore(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal parent_restore_failures
        if self.slice_id == "writer-fanout-parent" and parent_restore_failures == 0:
            parent_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def unexpected_backend(*_args):
        nonlocal backend_constructed
        backend_constructed = True
        raise AssertionError("backend must not run after setup failure")

    monkeypatch.setattr(
        mutation_journal_module,
        "_register_live_journal",
        fail_parent_registration_once,
    )
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_first_parent_restore,
    )

    with pytest.raises(
        RuntimeError,
        match="injected settled journal setup failure",
    ) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=unexpected_backend,
            toolset_factory=unexpected_backend,
            state_root=_state_root(tmp_path, "settled-journal-setup"),
            integration_gate=_gate,
        )

    assert type(caught.value) is RuntimeError
    assert register_failures == 1
    assert parent_restore_failures == 1
    assert not backend_constructed
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_settled_integration_begin_cleanup_reports_primary_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_register = mutation_journal_module._register_live_journal
    real_restore = WorkspaceMutationJournal.restore
    register_failures = 0
    integration_restore_failures = 0

    def fail_integration_registration_once(journal):
        nonlocal register_failures
        if journal.slice_id == "writer-serial-integration" and register_failures == 0:
            register_failures += 1
            raise RuntimeError("injected settled integration setup failure")
        return real_register(journal)

    def fail_first_integration_restore(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal integration_restore_failures
        if (
            self.slice_id == "writer-serial-integration"
            and integration_restore_failures == 0
        ):
            integration_restore_failures += 1
            return False
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    monkeypatch.setattr(
        mutation_journal_module,
        "_register_live_journal",
        fail_integration_registration_once,
    )
    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        fail_first_integration_restore,
    )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "settled-integration-setup"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert register_failures == 1
    assert integration_restore_failures == 1
    assert result.status == "integration_failed"
    assert result.error_kind == "RuntimeError"
    assert "injected settled integration setup failure" in (result.error_message or "")
    assert result.integration_cleanup_complete
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_writer_settled_integration_journal_false_report_removes_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    integration_restore_calls = 0

    def settle_integration_but_report_false(self, *, full=None, _defer_terminal_cleanup=False):
        nonlocal integration_restore_calls
        restored = real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)
        if (
            self.slice_id == "writer-serial-integration"
            and restored is True
            and not self.cleanup_pending
        ):
            integration_restore_calls += 1
            if integration_restore_calls == 1:
                return False
        return restored

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        settle_integration_but_report_false,
    )
    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "integration-settled-false"),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert integration_restore_calls == 1
    assert result.status == "cleanup_failed"
    assert not result.integration_cleanup_complete
    assert result.integrated_patch == b""
    assert result.error_kind == "WriterSchedulerError"
    assert "recovered failures" in (result.error_message or "")
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_dirty_base_rejected_before_backend_construction(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    (repo / "dirty.txt").write_text("operator\n", encoding="utf-8")
    constructed = False

    def backend_factory(*_args):
        nonlocal constructed
        constructed = True
        raise AssertionError

    with pytest.raises(WriterSchedulerError, match="exact clean base"):
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path),
            integration_gate=_gate,
        )
    assert not constructed
    assert (repo / "dirty.txt").read_text(encoding="utf-8") == "operator\n"


def test_scope_escape_fails_and_every_worktree_is_removed(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    observed_worktrees: list[Path] = []

    def backend_factory(request, toolset, _debug_dir, worktree):
        observed_worktrees.append(worktree)
        return _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
            scope_escape=True,
        )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=backend_factory,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
    )

    assert result.status == "failed"
    assert result.integrated_patch == b""
    assert result.children[0].error_kind == "MutationJournalError"
    assert "scope escape" in (result.children[0].error_message or "")
    assert result.children[0].journal_cleanup_complete
    assert result.children[0].worktree_cleanup_complete
    assert all(not path.exists() for path in observed_worktrees)
    assert _git(repo, "status", "--porcelain") == ""


def test_hostile_commit_ref_index_and_objects_are_rejected_and_restored(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    observed_worktrees: list[Path] = []

    class CommitBackend(_WriterBackend):
        def invoke(self, invocation: Invocation) -> InvocationResult:
            worktree = self.toolset.context.repo_root
            self.toolset.dispatch(
                "write_source",
                {"path": "a.txt", "content": "hostile\n"},
            )
            _git(worktree, "add", "a.txt")
            _git(worktree, "commit", "-m", "model-authored commit")
            return _complete_result(self.request.lane_id)

    def backend_factory(request, toolset, _debug, worktree):
        observed_worktrees.append(worktree)
        return CommitBackend(
            request,
            toolset,
            tracker,
            writes={},
        )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=backend_factory,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
    )

    assert result.status == "failed"
    assert result.integrated_patch == b""
    assert result.children[0].error_kind == "MutationJournalError"
    assert "changed Git HEAD" in (result.children[0].error_message or "")
    assert result.children[0].journal_cleanup_complete
    assert all(not path.exists() for path in observed_worktrees)
    assert _git(repo, "rev-parse", "HEAD") == base
    assert _git(repo, "status", "--porcelain") == ""


def test_serial_integration_conflict_withholds_patch(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    requests = _requests((("a.txt",), ("a.txt",)))

    result = run_isolated_writer_fanout(
        requests,
        policy=_policy(require_disjoint_scopes=False),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={
                "lane-0": ("a.txt", "first\n"),
                "lane-1": ("a.txt", "second\n"),
            },
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert result.status == "integration_failed"
    assert result.integrated_patch == b""
    assert result.error_kind == "WriterIntegrationConflict"
    assert "lane-1" in (result.error_message or "")
    assert all(child.successful for child in result.children)
    assert result.integration_cleanup_complete
    assert result.base_unchanged
    assert _git(repo, "status", "--porcelain") == ""


def test_lane_setup_post_add_failure_retains_retryable_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    real_write_atomic_json = worktree_module.write_atomic_json
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    retained_worktree: Path | None = None
    retained_lease: Path | None = None

    def fail_created_lease(path, payload):
        nonlocal retained_worktree, retained_lease
        if payload.get("state") == "created":
            retained_worktree = Path(payload["worktree"])
            retained_lease = Path(path)
            raise OSError("injected lane post-add lease publication failure")
        real_write_atomic_json(path, payload)

    def fail_remove(self):
        if retained_worktree is not None and self.worktree_path == retained_worktree:
            raise RuntimeError("injected lane worktree remove failure")
        real_close_locked(self)

    def unexpected_factory(*_args):
        raise AssertionError("backend factory must not run after setup failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            worktree_module,
            "write_atomic_json",
            fail_created_lease,
        )
        patcher.setattr(
            worktree_module.WorktreeResolver,
            "_close_locked",
            fail_remove,
        )
        with pytest.raises(WriterCleanupError) as caught:
            run_isolated_writer_fanout(
                _requests((("a.txt",),)),
                policy=_policy(lanes=1, max_concurrency=1),
                repo_root=repo,
                base_commit=base,
                backend_factory=unexpected_factory,
                toolset_factory=unexpected_factory,
                state_root=_state_root(tmp_path),
                integration_gate=_gate,
            )

    cleanup_error = caught.value
    assert retained_worktree is not None and retained_worktree.is_dir()
    assert retained_lease is not None and retained_lease.is_file()
    assert cleanup_error.retained_resolvers
    _, retained_resolver = cleanup_error.retained_resolvers[0]
    assert retained_resolver.worktree_path == retained_worktree
    assert retained_resolver.lease_path == retained_lease
    assert "post-add lease publication failure" in str(cleanup_error)
    assert "worktree remove failure" in str(cleanup_error)

    cleanup_error.retry_cleanup()
    assert not retained_resolver.owned
    assert not retained_worktree.exists()
    assert not retained_lease.exists()
    assert _git(repo, "status", "--porcelain") == ""


def test_lane_post_return_validation_failure_retries_owned_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    real_status = writer_scheduler_module._status
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    rejected_worktree: Path | None = None

    def reject_returned_worktree(cwd):
        nonlocal rejected_worktree
        candidate = Path(cwd).resolve()
        if candidate != repo:
            rejected_worktree = candidate
            return b"? injected-dirty-worktree"
        return real_status(cwd)

    def fail_rejected_remove(self):
        if rejected_worktree is not None and self.worktree_path == rejected_worktree:
            raise RuntimeError("injected validation unwind remove failure")
        real_close_locked(self)

    def unexpected_factory(*_args):
        raise AssertionError("backend factory must not run after validation failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(
            writer_scheduler_module,
            "_status",
            reject_returned_worktree,
        )
        patcher.setattr(
            worktree_module.WorktreeResolver,
            "_close_locked",
            fail_rejected_remove,
        )
        with pytest.raises(WriterCleanupError) as caught:
            run_isolated_writer_fanout(
                _requests((("a.txt",),)),
                policy=_policy(lanes=1, max_concurrency=1),
                repo_root=repo,
                base_commit=base,
                backend_factory=unexpected_factory,
                toolset_factory=unexpected_factory,
                state_root=_state_root(tmp_path),
                integration_gate=_gate,
            )

    cleanup_error = caught.value
    assert isinstance(cleanup_error.primary_error, WriterSchedulerError)
    assert "did not start at the exact clean base" in str(cleanup_error.primary_error)
    assert cleanup_error.retained_resolvers
    _, retained_resolver = cleanup_error.retained_resolvers[0]
    retained_worktree = retained_resolver.worktree_path
    retained_lease = retained_resolver.lease_path
    assert retained_worktree == rejected_worktree
    assert retained_worktree is not None and retained_worktree.is_dir()
    assert retained_lease is not None and retained_lease.is_file()
    assert "validation unwind remove failure" in str(cleanup_error)

    cleanup_error.retry_cleanup()
    assert not retained_resolver.owned
    assert not retained_worktree.exists()
    assert not retained_lease.exists()
    assert _git(repo, "status", "--porcelain") == ""


def test_lane_cleanup_after_outcome_has_caller_owned_retry_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_close_locked = worktree_module.WorktreeResolver._close_locked

    def fail_lane_remove(self):
        if self.worktree_path is not None:
            raise RuntimeError("injected post-outcome worktree remove failure")
        real_close_locked(self)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            worktree_module.WorktreeResolver,
            "_close_locked",
            fail_lane_remove,
        )
        with pytest.raises(WriterCleanupError) as caught:
            run_isolated_writer_fanout(
                _requests((("a.txt",),)),
                policy=_policy(lanes=1, max_concurrency=1),
                repo_root=repo,
                base_commit=base,
                backend_factory=(
                    lambda request, toolset, _debug, _worktree: _WriterBackend(
                        request,
                        toolset,
                        tracker,
                        writes={"lane-0": ("a.txt", "changed\n")},
                    )
                ),
                toolset_factory=(lambda _request, context: _WriterToolSet(context)),
                state_root=_state_root(tmp_path),
                integration_gate=_gate,
            )

    cleanup_error = caught.value
    partial = cleanup_error.partial_result
    assert partial is not None
    assert partial.status == "cleanup_failed"
    assert partial.usage.cost_usd == 0.1
    assert partial.children[0].status == "cleanup_failed"
    assert not partial.children[0].worktree_cleanup_complete
    assert cleanup_error.retained_resolvers
    label, retained_resolver = cleanup_error.retained_resolvers[0]
    assert label == "lane-0"
    retained_worktree = retained_resolver.worktree_path
    retained_lease = retained_resolver.lease_path
    assert retained_worktree is not None and retained_worktree.is_dir()
    assert retained_lease is not None and retained_lease.is_file()
    assert "post-outcome worktree remove failure" in str(cleanup_error)

    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert not retained_resolver.owned
    assert not retained_worktree.exists()
    assert not retained_lease.exists()
    assert settled.usage == partial.usage
    assert not settled.successful
    assert _git(repo, "status", "--porcelain") == ""


def test_worktree_close_retries_lease_after_git_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    resolver = worktree_module.WorktreeResolver(repo, base_ref=base)
    worktree = resolver.ensure_worktree()
    lease = resolver.lease_path
    assert lease is not None
    real_unlink = Path.unlink
    unlink_attempts = 0

    def fail_lease_unlink_once(path, *args, **kwargs):
        nonlocal unlink_attempts
        if path == lease:
            unlink_attempts += 1
            if unlink_attempts == 1:
                raise OSError("injected lease unlink failure")
        return real_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(Path, "unlink", fail_lease_unlink_once)
            with pytest.raises(
                worktree_module.WorktreeCleanupError,
                match="lease unlink failure",
            ) as caught:
                resolver.close()

            assert resolver.worktree_path is None
            assert resolver.lease_path == lease
            assert resolver.owned
            assert not worktree.exists()
            assert lease.is_file()

            caught.value.retry_cleanup()
            assert unlink_attempts == 2
            assert not resolver.owned
            assert not lease.exists()
    finally:
        if resolver.owned:
            resolver.close()


def test_worktree_add_failure_retains_lease_until_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    resolver = worktree_module.WorktreeResolver(repo, base_ref=base)
    real_run = worktree_module.subprocess.run
    real_unlink = Path.unlink
    lease_unlink_attempts = 0

    def fail_worktree_add(argv, *args, **kwargs):
        if "worktree" in argv and "add" in argv:
            raise subprocess.CalledProcessError(
                128,
                argv,
                stderr=b"injected git worktree add failure",
            )
        return real_run(argv, *args, **kwargs)

    def fail_lease_unlink_once(path, *args, **kwargs):
        nonlocal lease_unlink_attempts
        if path.parent.name == "crucible-worktree-leases" and path.name.endswith(
            ".json"
        ):
            lease_unlink_attempts += 1
            if lease_unlink_attempts == 1:
                raise OSError("injected add-failure lease unlink failure")
        return real_unlink(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                worktree_module.subprocess,
                "run",
                fail_worktree_add,
            )
            patcher.setattr(Path, "unlink", fail_lease_unlink_once)
            with pytest.raises(
                worktree_module.WorktreeCleanupError,
            ) as caught:
                resolver.ensure_worktree()

            cleanup_error = caught.value
            lease = resolver.lease_path
            assert cleanup_error.owner is resolver
            assert "git worktree add failure" in str(cleanup_error.primary_error)
            assert "lease unlink failure" in str(cleanup_error)
            assert resolver.worktree_path is None
            assert resolver.intent_path is None
            assert lease is not None and lease.is_file()
            intended_worktree = Path(
                json.loads(lease.read_text(encoding="utf-8"))["worktree"]
            )
            assert not intended_worktree.exists()
            assert resolver.owned

            cleanup_error.retry_cleanup()
            assert lease_unlink_attempts == 2
            assert not resolver.owned
            assert not lease.exists()
    finally:
        if resolver.owned:
            resolver.close()


def test_integration_post_add_lease_publish_and_remove_failure_retains_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_write_atomic_json = worktree_module.write_atomic_json
    real_close_locked = worktree_module.WorktreeResolver._close_locked
    created_publications = 0

    def fail_integration_created_lease(path, payload):
        nonlocal created_publications
        if payload.get("state") == "created":
            created_publications += 1
            if created_publications == 2:
                raise OSError("injected post-add lease publication failure")
        real_write_atomic_json(path, payload)

    def fail_integration_remove(self):
        if created_publications >= 2 and self.worktree_path is not None:
            raise RuntimeError("injected integration worktree remove failure")
        real_close_locked(self)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            worktree_module,
            "write_atomic_json",
            fail_integration_created_lease,
        )
        patcher.setattr(
            worktree_module.WorktreeResolver,
            "_close_locked",
            fail_integration_remove,
        )
        with pytest.raises(WriterCleanupError) as caught:
            run_isolated_writer_fanout(
                _requests((("a.txt",),)),
                policy=_policy(lanes=1, max_concurrency=1),
                repo_root=repo,
                base_commit=base,
                backend_factory=lambda request, toolset, _debug, _worktree: (
                    _WriterBackend(
                        request,
                        toolset,
                        tracker,
                        writes={"lane-0": ("a.txt", "changed\n")},
                    )
                ),
                toolset_factory=(lambda _request, context: _WriterToolSet(context)),
                state_root=_state_root(tmp_path),
                integration_gate=_gate,
            )

    cleanup_error = caught.value
    result = cleanup_error.partial_result
    assert result is not None
    assert created_publications == 2
    assert result.status == "cleanup_failed"
    assert not result.integration_cleanup_complete
    assert result.integrated_patch == b""
    assert result.error_kind == "WorktreeCleanupError"
    assert "remove failure" in (result.error_message or "")
    assert cleanup_error.retained_resolvers
    label, retained_resolver = cleanup_error.retained_resolvers[0]
    assert label == "integration"
    retained_worktree = retained_resolver.worktree_path
    retained_lease = retained_resolver.lease_path
    assert retained_worktree is not None and retained_worktree.is_dir()
    assert retained_lease is not None and retained_lease.is_file()
    assert str(retained_lease) in str(cleanup_error)
    assert _git(repo, "status", "--porcelain") == ""

    settled = cleanup_error.retry_cleanup()
    assert settled is not None
    assert not retained_resolver.owned
    assert not retained_worktree.exists()
    assert not retained_lease.exists()
    assert settled.usage == result.usage


def test_union_gate_scope_escape_is_rejected_and_integration_is_removed(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()

    def escaping_gate(worktree, _scope, _requests):
        (worktree / "escape.txt").write_text("gate escape\n", encoding="utf-8")
        return IntegrationGateResult(
            success=True,
            evidence_digest=hash_record({"gate": "escaped"}),
        )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=escaping_gate,
    )

    assert result.status == "integration_failed"
    assert result.integrated_patch == b""
    assert result.error_kind == "MutationJournalError"
    assert "integration lane scope escape" in (result.error_message or "")
    assert result.integration_cleanup_complete
    assert _git(repo, "status", "--porcelain") == ""


def test_union_gate_in_scope_mutation_is_rejected_and_integration_is_removed(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()

    def mutating_gate(worktree, _scope, _requests):
        (worktree / "a.txt").write_text(
            "gate-replaced-candidate\n",
            encoding="utf-8",
        )
        return IntegrationGateResult(
            success=True,
            evidence_digest=hash_record({"gate": "mutated-in-scope"}),
        )

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "changed\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=mutating_gate,
    )

    assert result.status == "integration_failed"
    assert result.integrated_patch == b""
    assert result.error_kind == "MutationJournalError"
    assert "gate mutated candidate source" in (result.error_message or "")
    assert result.integration_cleanup_complete
    assert _git(repo, "status", "--porcelain") == ""


def test_union_gate_exception_retains_armed_integration_epoch_without_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    real_restore = WorkspaceMutationJournal.restore
    restored_slices: list[str] = []

    def observe_restore(self, *, full=None, _defer_terminal_cleanup=False):
        restored_slices.append(self.slice_id)
        return real_restore(self, full=full, _defer_terminal_cleanup=_defer_terminal_cleanup)

    def exploding_gate(worktree, _scope, _requests):
        (worktree / "a.txt").write_text(
            "gate-raised-after-write\n",
            encoding="utf-8",
        )
        raise RuntimeError("integration gate lost settlement authority")

    monkeypatch.setattr(
        WorkspaceMutationJournal,
        "restore",
        observe_restore,
    )

    with pytest.raises(WriterCleanupError) as caught:
        run_isolated_writer_fanout(
            _requests((("a.txt",),)),
            policy=_policy(lanes=1, max_concurrency=1),
            repo_root=repo,
            base_commit=base,
            backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
                request,
                toolset,
                tracker,
                writes={"lane-0": ("a.txt", "changed\n")},
            ),
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path, "gate-exception"),
            integration_gate=exploding_gate,
            parent_budget_usd=4.0,
        )

    cleanup_error = caught.value
    assert cleanup_error.retained_epoch_id is not None
    assert "writer-serial-integration" not in restored_slices
    assert cleanup_error.partial_result is not None
    assert cleanup_error.partial_result.integrated_patch == b""
    integration_resolver = next(
        resolver
        for label, resolver in cleanup_error.retained_resolvers
        if label == "integration"
    )
    assert (integration_resolver.worktree_path / "a.txt").read_text(
        encoding="utf-8",
    ) == "gate-raised-after-write\n"
    assert _git(repo, "status", "--porcelain") == ""

    with pytest.raises(WriterCleanupError) as retried:
        cleanup_error.retry_cleanup()

    assert retried.value.retained_epoch_id == cleanup_error.retained_epoch_id
    assert "writer-serial-integration" not in restored_slices
    assert integration_resolver.owned


def test_shared_backend_instance_is_rejected(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)

    class SharedBackend:
        name = "shared"
        model = "shared"

        def invoke(self, invocation):
            time.sleep(0.05)
            return _complete_result("shared")

        def close(self):
            return None

    shared = SharedBackend()

    result = run_isolated_writer_fanout(
        _requests((("a.txt",), ("b.txt",))),
        policy=_policy(),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda *_args: shared,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path),
        integration_gate=_gate,
        parent_budget_usd=4.0,
    )

    assert result.status in {"failed", "cancelled"}
    assert result.integrated_patch == b""
    assert any(child.error_kind == "WriterIsolationError" for child in result.children)
    assert _git(repo, "status", "--porcelain") == ""


def test_parent_cancellation_closes_drains_and_joins(tmp_path: Path) -> None:
    repo, base = _repo(tmp_path)
    tracker = _Concurrency()
    cancel = threading.Event()

    def backend_factory(request, toolset, _debug, _worktree):
        return _WriterBackend(
            request,
            toolset,
            tracker,
            writes={},
            block=True,
        )

    trigger = threading.Thread(
        target=lambda: (tracker.two_started.wait(30), cancel.set()),
        daemon=True,
    )
    trigger.start()
    try:
        result = run_isolated_writer_fanout(
            _requests((("a.txt",), ("b.txt",))),
            policy=_policy(),
            repo_root=repo,
            base_commit=base,
            backend_factory=backend_factory,
            toolset_factory=lambda _request, context: _WriterToolSet(context),
            state_root=_state_root(tmp_path),
            integration_gate=_gate,
            parent_budget_usd=4.0,
            cancel_event=cancel,
        )
    finally:
        cancel.set()
        trigger.join(timeout=5)

    assert result.status == "cancelled"
    assert result.all_joined
    assert result.cancellation_drained
    assert result.integrated_patch == b""
    assert tracker.active == 0
    assert tracker.closed == 2
    assert _git(repo, "status", "--porcelain") == ""


def test_crash_left_journal_is_recovered_before_new_writer_baseline(
    tmp_path: Path,
) -> None:
    repo, base = _repo(tmp_path)
    crash_root = tmp_path / "spec" / "runs" / "crash" / "mutations"
    crash_root.mkdir(parents=True)
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=crash_root,
        slice_id="crash-left",
        scope=("a.txt",),
        sequence=1,
        spec=None,
    )
    (repo / "a.txt").write_text("crashed writer\n", encoding="utf-8")
    _unregister_live(journal.transaction_id)
    tracker = _Concurrency()

    result = run_isolated_writer_fanout(
        _requests((("a.txt",),)),
        policy=_policy(lanes=1, max_concurrency=1),
        repo_root=repo,
        base_commit=base,
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "new writer\n")},
        ),
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        state_root=_state_root(tmp_path, "after-crash"),
        integration_gate=_gate,
    )

    assert result.successful
    assert not journal.journal_dir.exists()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""


@pytest.mark.parametrize(
    "publication_fault",
    (None, "before", "after", "cleanup"),
    ids=(
        "normal",
        "raise-before-publish",
        "publish-then-raise",
        "cleanup-retry",
    ),
)
@pytest.mark.timeout(600)
def test_tier_programmatic_frontier_reproves_lands_and_binds_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_fault: str | None,
) -> None:
    repo, base = _repo(tmp_path)
    slice_ = Slice(
        id="lane-0",
        directive=ImplementationDirective(
            slice_id="lane-0",
            intent="replace a",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1, "languages": ["text"]},
        ),
    )
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="writer-frontier",
            repo_url="https://example.invalid/writer-frontier.git",
            base_commit=base,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise WRK-01 owning workflow seam",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(tier,)),
    )
    spec_dir = tmp_path / "corpus" / "spec"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)
    publication_faults: list[str] = []
    if publication_fault in {"before", "after"}:
        original_commit = Corpus.commit

        def publish_then_raise(bound_corpus, *args, **kwargs):
            implementations = tuple(kwargs.get("implementations", ()))
            is_frontier_publication = (
                Path(bound_corpus.spec_dir) == spec_dir
                and kwargs.get("lifecycle_event_factory") is not None
                and [item.slice_id for item in implementations] == ["lane-0"]
            )
            if is_frontier_publication and publication_fault == "before":
                publication_faults.append("before")
                raise OSError("writer-frontier commit rejected before publish")
            result = original_commit(bound_corpus, *args, **kwargs)
            if is_frontier_publication and publication_fault == "after":
                publication_faults.append("after")
                raise OSError("lost writer-frontier commit acknowledgement")
            return result

        monkeypatch.setattr(Corpus, "commit", publish_then_raise)
    cleanup_faults: list[str] = []
    if publication_fault == "cleanup":
        original_remove_registry_marker = (
            WorkspaceMutationJournal._remove_registry_marker
        )

        def fail_first_landing_cleanup(bound_journal, *args, **kwargs):
            if (
                bound_journal.slice_id == "t0__writer_frontier_landing"
                and bound_journal.state == "committed"
                and not cleanup_faults
            ):
                cleanup_faults.append("first")
                raise OSError("lost writer-frontier cleanup acknowledgement")
            return original_remove_registry_marker(
                bound_journal,
                *args,
                **kwargs,
            )

        monkeypatch.setattr(
            WorkspaceMutationJournal,
            "_remove_registry_marker",
            fail_first_landing_cleanup,
        )
    orphan = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(spec_dir / "runs" / "pre-frontier" / "phase1_mutations"),
        slice_id="pre-frontier-orphan",
        scope=("a.txt",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.txt").write_text("crashed pre-frontier bytes\n", encoding="utf-8")
    _unregister_live(orphan.transaction_id)
    tracker = _Concurrency()
    verifier_calls: list[Path] = []

    def final_verify(_spec, **kwargs):
        owning_checkout = Path(kwargs["cwd"])
        verifier_calls.append(owning_checkout)
        assert owning_checkout == repo
        writer_epoch_members = kwargs["writer_epoch_members"]
        assert len(writer_epoch_members) == 2
        landing_authority, verification_guard = writer_epoch_members
        assert landing_authority.active
        assert verification_guard.active
        assert verification_guard.parent_transaction_id == (
            landing_authority.transaction_id
        )
        assert writer_epoch_members == (
            tier_workflow._mutation_journal.live_writer_epoch_family(
                landing_authority
            )
        )
        assert (owning_checkout / "a.txt").read_text(
            encoding="utf-8"
        ) == "frontier landed\n"
        return tier_workflow._build_verify.BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
        )

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        final_verify,
    )
    pricing_evidence = UsagePricingEvidence(
        segments=(
            UsagePricingSegment(
                input_tokens=10,
                output_tokens=5,
                cache_creation_tokens=2,
                cache_read_tokens=5,
                cost_usd=0.1,
            ),
        ),
        complete=True,
    )
    bounded_toolsets = []

    def frontier_backend_factory(request, toolset, _debug, _worktree):
        bounded_toolsets.append(toolset)
        return _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "frontier landed\n")},
            cache_creation_tokens=2,
            cache_read_tokens=5,
            pricing_evidence=pricing_evidence,
        )

    frontier = TierWriterFrontier(
        policy=_policy(lanes=1, max_concurrency=1),
        requests=_requests((("a.txt",),)),
        integration_gate=_gate,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        backend_factory=frontier_backend_factory,
    )

    rc = tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=2.0,
        skip_smoke_check=True,
        skip_smoke_bench=True,
        skip_review=True,
        writer_frontier=frontier,
        implementer_read_call_limit=7,
    )

    assert len(bounded_toolsets) == 1
    assert isinstance(
        bounded_toolsets[0],
        tier_workflow.ProgressBoundedMutatingToolSet,
    )
    assert bounded_toolsets[0].snapshot().max_consecutive_read_calls == 7
    if publication_fault == "before":
        assert rc == 1
        assert publication_faults == ["before"]
        assert not orphan.journal_dir.exists()
        assert verifier_calls == [repo]
        assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
        assert _git(repo, "status", "--porcelain") == ""
        latest = corpus.latest_spec()
        assert latest is not None
        assert latest.implementation_refs == ()
        lifecycle = [
            corpus.get_record(ref, as_type=LifecycleEvent)
            for ref in latest.lifecycle_event_refs
        ]
        assert [event.kind for event in lifecycle] == ["abandoned"]
        run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
        assert summary["error_kind"] == "OSError"
        assert not tuple(
            (run_dir / "writer_frontier_landing_mutations").glob("*/manifest.json")
        )
        return

    assert rc == 0
    assert not orphan.journal_dir.exists()
    assert verifier_calls == [repo]
    assert (repo / "a.txt").read_text(encoding="utf-8") == "frontier landed\n"
    assert _git(repo, "rev-parse", "HEAD") == base
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.version == 1
    assert len(latest.implementation_refs) == 1
    implementation = corpus.get_record(
        latest.implementation_refs[0],
        as_type=Implementation,
    )
    assert implementation.slice_id == "lane-0"
    assert implementation.classification == "success"
    assert implementation.base_commit == base
    assert corpus.get_artifact(implementation.diff_ref)
    lifecycle = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert lifecycle[-1].kind == "landed"
    assert [event.kind for event in lifecycle] == ["landed"]
    run_dirs = sorted((spec_dir / "runs").glob("*_tier_*"))
    assert run_dirs
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert summary["reaper"]["mutation_journals_recovered"] == ["pre-frontier-orphan"]
    assert summary["writer_frontier"]["status"] == "succeeded"
    assert summary["writer_frontier"]["schema_version"] == (
        "crucible.writer-fanout-result.v2"
    )
    if publication_fault == "after":
        assert publication_faults == ["after"]
        reconciliation = summary["writer_frontier"]["publication_reconciliation"]
        assert reconciliation["status"] == "attached"
        assert reconciliation["disposition"] == "preserved"
        assert reconciliation["lifecycle_ref"] == str(latest.lifecycle_event_refs[-1])
    else:
        assert publication_faults == []
        assert "publication_reconciliation" not in summary["writer_frontier"]
    assert cleanup_faults == (["first"] if publication_fault == "cleanup" else [])
    assert summary["costs"]["total_cache_creation_tokens"] == 2
    assert summary["costs"]["total_cache_read_tokens"] == 5
    assert summary["costs"]["pricing_evidence"]["complete"] is True
    assert summary["costs"]["pricing_evidence_by_call"] == [
        {
            "role": "implementer",
            "backend": "synthetic",
            "model": "writer-test",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_tokens": 2,
                "cache_read_tokens": 5,
                "cost_usd": 0.1,
            },
            "pricing_evidence": {
                "segments": [
                    {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_creation_tokens": 2,
                        "cache_read_tokens": 5,
                        "cost_usd": 0.1,
                    },
                ],
                "complete": True,
            },
        },
    ]
    assert summary["phase_5_cumulative_verification"]["status"] == "passed"


def _setup_indeterminate_writer_frontier_spec(
    tmp_path: Path,
) -> tuple[Path, str, Path]:
    repo, base = _repo(tmp_path)
    slice_ = Slice(
        id="lane-0",
        directive=ImplementationDirective(
            slice_id="lane-0",
            intent="replace a",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1, "languages": ["text"]},
        ),
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="writer-frontier-indeterminate",
            repo_url="https://example.invalid/writer-frontier.git",
            base_commit=base,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="retain exact writer-frontier mutation uncertainty",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),)
        ),
    )
    spec_dir = tmp_path / "corpus" / "spec"
    spec_dir.mkdir(parents=True)
    Corpus(spec_dir).put_spec(spec)
    return repo, base, spec_dir


def test_tier_writer_frontier_preserves_settled_indeterminate_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _base, spec_dir = _setup_indeterminate_writer_frontier_spec(
        tmp_path
    )
    backend: _RaisedIndeterminateWriter | None = None
    verifier_calls = 0

    def backend_factory(request, toolset, _debug, _worktree):
        nonlocal backend
        backend = _RaisedIndeterminateWriter(
            request,
            toolset,
            close_settles=True,
        )
        return backend

    def unexpected_verify(*_args, **_kwargs):
        nonlocal verifier_calls
        verifier_calls += 1
        raise AssertionError("indeterminate writer reached final verification")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        unexpected_verify,
    )
    frontier = TierWriterFrontier(
        policy=replace(
            _policy(lanes=1, max_concurrency=1),
            fail_fast=False,
        ),
        requests=_requests((("a.txt",),)),
        integration_gate=_gate,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        backend_factory=backend_factory,
    )

    rc = tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=2.0,
        skip_smoke_check=True,
        skip_smoke_bench=True,
        skip_review=True,
        writer_frontier=frontier,
    )

    assert rc == 1
    assert backend is not None
    assert backend.invoke_calls == 1
    assert backend.close_calls == 1
    assert verifier_calls == 0
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["writer_frontier"]["children"][0]["error_kind"] == (
        "tool_mutation_indeterminate"
    )
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_tier_writer_frontier_rethrows_indeterminate_with_cleanup_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, _base, spec_dir = _setup_indeterminate_writer_frontier_spec(
        tmp_path
    )
    backend: _RaisedIndeterminateWriter | None = None

    def backend_factory(request, toolset, _debug, _worktree):
        nonlocal backend
        backend = _RaisedIndeterminateWriter(
            request,
            toolset,
            close_settles=False,
        )
        return backend

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        lambda *_args, **_kwargs: pytest.fail(
            "unsettled indeterminate writer reached verification"
        ),
    )
    frontier = TierWriterFrontier(
        policy=replace(
            _policy(lanes=1, max_concurrency=1),
            fail_fast=False,
        ),
        requests=_requests((("a.txt",),)),
        integration_gate=_gate,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        backend_factory=backend_factory,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=spec_dir,
            tier_id="t0",
            cwd=str(repo),
            budget_usd=2.0,
            skip_smoke_check=True,
            skip_smoke_bench=True,
            skip_review=True,
            writer_frontier=frontier,
        )

    assert backend is not None
    assert backend.invoke_calls == 1
    assert backend.close_calls == 1
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, WriterCleanupError)
    assert cleanup_owner.primary_error is caught.value
    assert cleanup_owner.partial_result is not None
    assert cleanup_owner.partial_result.children[0].error_kind == (
        "tool_mutation_indeterminate"
    )
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["provider_cleanup_pending"] is True
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }

    backend.close_settles = True
    settled = cleanup_owner.retry_cleanup()
    assert settled is not None
    assert backend.close_calls == 2
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""
    _assert_writer_ownership_settled(repo)


def test_tier_frontier_incomplete_lane_accounting_never_integrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    slice_ = Slice(
        id="lane-0",
        directive=ImplementationDirective(
            slice_id="lane-0",
            intent="replace a",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1, "languages": ["text"]},
        ),
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="writer-frontier-incomplete-accounting",
            repo_url="https://example.invalid/writer-frontier.git",
            base_commit=base,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="reject a writer lane with incomplete accounting",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),)),
    )
    spec_dir = tmp_path / "corpus" / "spec"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)
    tracker = _Concurrency()
    verifier_calls: list[Path] = []

    def unexpected_verify(_spec, **kwargs):
        verifier_calls.append(Path(kwargs["cwd"]))
        raise AssertionError("incomplete writer accounting reached verification")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        unexpected_verify,
    )
    frontier = TierWriterFrontier(
        policy=_policy(lanes=1, max_concurrency=1),
        requests=_requests((("a.txt",),)),
        integration_gate=_gate,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "must not land\n")},
            accounting_complete=False,
        ),
    )

    rc = tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=2.0,
        skip_smoke_check=True,
        skip_smoke_bench=True,
        skip_review=True,
        writer_frontier=frontier,
    )

    assert rc == 1
    assert verifier_calls == []
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert _git(repo, "status", "--porcelain") == ""
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.implementation_refs == ()
    run_dir = sorted((spec_dir / "runs").glob("*_tier_*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == tier_workflow.STATUS_INCOMPLETE_ACCOUNTING
    assert summary["writer_frontier"]["status"] != "succeeded"
    assert summary["costs"]["total_usd"] == pytest.approx(0.1)
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }


@pytest.mark.timeout(600)
def test_tier_frontier_final_verifier_mutation_restores_and_withholds_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _repo(tmp_path)
    slice_ = Slice(
        id="lane-0",
        directive=ImplementationDirective(
            slice_id="lane-0",
            intent="replace a",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1, "languages": ["text"]},
        ),
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="writer-frontier-rollback",
            repo_url="https://example.invalid/writer-frontier.git",
            base_commit=base,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="prove rejected frontier landing recovery",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),)),
    )
    spec_dir = tmp_path / "corpus" / "spec"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)
    tracker = _Concurrency()

    def mutating_verifier(_spec, **kwargs):
        owning_checkout = Path(kwargs["cwd"])
        assert (owning_checkout / "a.txt").read_text(encoding="utf-8") == "candidate\n"
        (owning_checkout / "escape.txt").write_text(
            "verifier mutation\n",
            encoding="utf-8",
        )
        return tier_workflow._build_verify.BuildVerifyOutcome(
            "success",
            build=None,
            verify=None,
        )

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        mutating_verifier,
    )
    frontier = TierWriterFrontier(
        policy=_policy(lanes=1, max_concurrency=1),
        requests=_requests((("a.txt",),)),
        integration_gate=_gate,
        toolset_factory=lambda _request, context: _WriterToolSet(context),
        backend_factory=lambda request, toolset, _debug, _worktree: _WriterBackend(
            request,
            toolset,
            tracker,
            writes={"lane-0": ("a.txt", "candidate\n")},
        ),
    )

    rc = tier_workflow.run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        budget_usd=2.0,
        skip_smoke_check=True,
        skip_smoke_bench=True,
        skip_review=True,
        writer_frontier=frontier,
    )

    assert rc == 1
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert not (repo / "escape.txt").exists()
    assert _git(repo, "status", "--porcelain") == ""
    latest = corpus.latest_spec()
    assert latest is not None
    assert latest.implementation_refs == ()
    lifecycle = [
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    ]
    assert lifecycle[-1].kind == "abandoned"
    run_dirs = sorted((spec_dir / "runs").glob("*_tier_*"))
    summary = json.loads((run_dirs[-1] / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert "scope escapes" in summary["error_message"]


def test_backend_pool_writer_lease_is_uncached_confined_and_native_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[str, dict, object]] = []

    class Backend:
        name = "synthetic"

        def __init__(self) -> None:
            self.attachments: list[tuple[object, int | None]] = []
            self.closed = False

        def attach_toolset(self, toolset, *, context_budget=None):
            self.attachments.append((toolset, context_budget))

        def close(self):
            self.closed = True

    def resolve_for_role(role, **kwargs):
        backend = Backend()
        constructed.append((role, kwargs, backend))
        return backend

    monkeypatch.setattr(
        backend_pool_module,
        "resolve_for_role",
        resolve_for_role,
    )
    pool = _BackendPool(object(), log_dir=tmp_path, cwd=str(tmp_path))
    first_toolset = _WriterToolSet(object())
    second_toolset = _WriterToolSet(object())
    first = pool.new_exclusive_writer(
        "implementer",
        toolset=first_toolset,
        log_dir=tmp_path / "lane-a",
        cwd=str(tmp_path),
        context_budget=101,
    )
    second = pool.new_exclusive_writer(
        "implementer",
        toolset=second_toolset,
        log_dir=tmp_path / "lane-b",
        cwd=str(tmp_path),
        context_budget=202,
    )

    assert first is not second
    assert len(constructed) == 2
    for _role, kwargs, _backend in constructed:
        assert kwargs["capability"] == "workspace_write"
        assert kwargs["native_shell"] is False
        requirements = kwargs["feature_requirements"]
        assert any(
            requirement.field == "native_delegation"
            and requirement.accepted == frozenset({"none"})
            for requirement in requirements
        )
    assert first.attachments == [(first_toolset, 101)]
    assert second.attachments == [(second_toolset, 202)]
    pool.close()
    assert not first.closed
    assert not second.closed
    first.close()
    second.close()
