# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import threading
import time

import pytest

from crucible.tools._dispatch_budget import (
    BoundedReadOnlyToolSet,
    PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES,
    PLANNER_GROUNDING_CACHE_BYTES,
    ProgressBoundedMutatingToolSet,
    ToolDispatchBudgetExceeded,
)
from crucible.tools._mcp_bridge import ContextBudgetTracker
from crucible.tools import ToolContext
from crucible.tools.profiles import build_autonomous_implementer_profile


class _RecordingToolSet:
    read_only_names = {"read"}
    write_names: set[str] = set()

    def __init__(self, results: list[str] | None = None) -> None:
        self.results = list(results or ["ok"])
        self.calls: list[tuple[str, dict]] = []

    def to_anthropic_tools(self):
        return [{
            "name": "read",
            "description": "read",
            "input_schema": {"type": "object"},
        }]

    def to_openai_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "read",
                "description": "read",
                "parameters": {"type": "object"},
            },
        }]

    def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        return self.results.pop(0)


class _RecordingMutatingToolSet:
    read_only_names = {"read"}
    write_names = {"write"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def to_anthropic_tools(self):
        return [
            {
                "name": name,
                "description": name,
                "input_schema": {"type": "object"},
            }
            for name in ("read", "write")
        ]

    def to_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object"},
                },
            }
            for name in ("read", "write")
        ]

    def dispatch(self, name, arguments):
        self.calls.append((name, arguments))
        return f"{name}-ok"


def test_mutating_progress_limit_denies_only_unproductive_reads():
    inner = _RecordingMutatingToolSet()
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=2,
    )

    assert "1 consecutive source read" in bounded.dispatch("read", {"n": 1})
    assert "0 consecutive source read" in bounded.dispatch("read", {"n": 2})
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="consecutive-read limit exhausted",
    ):
        bounded.dispatch("read", {"n": 3})

    assert bounded.dispatch("write", {"content": "progress"}) == "write-ok"
    assert bounded.dispatch("read", {"n": 4}).startswith("read-ok")
    assert inner.calls == [
        ("read", {"n": 1}),
        ("read", {"n": 2}),
        ("write", {"content": "progress"}),
        ("read", {"n": 4}),
    ]
    snapshot = bounded.snapshot()
    assert snapshot.consecutive_read_calls == 1
    assert snapshot.read_calls == 3
    assert snapshot.write_calls == 1
    assert snapshot.denied_read_calls == 1


def test_mutating_progress_limit_resets_for_each_invocation():
    inner = _RecordingMutatingToolSet()
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )

    bounded.dispatch("read", {})
    bounded.begin_invocation()
    assert bounded.dispatch("read", {}).startswith("read-ok")
    assert bounded.snapshot().consecutive_read_calls == 1


def test_mutating_progress_limit_rejects_unknown_tool_without_dispatch():
    inner = _RecordingMutatingToolSet()
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )

    result = bounded.dispatch("hallucinated_mutator", {"path": "target.py"})

    assert result.startswith("ERROR: unknown tool 'hallucinated_mutator'")
    assert inner.calls == []
    assert bounded.snapshot().consecutive_read_calls == 1
    with pytest.raises(ToolDispatchBudgetExceeded, match="no-progress limit"):
        bounded.dispatch("another_hallucination", {})

    assert bounded.dispatch("write", {"content": "progress"}) == "write-ok"
    assert bounded.dispatch("hallucinated_after_progress", {}).startswith(
        "ERROR: unknown tool 'hallucinated_after_progress'"
    )
    snapshot = bounded.snapshot()
    assert snapshot.consecutive_read_calls == 1
    assert snapshot.read_calls == 0
    assert snapshot.write_calls == 1
    assert snapshot.denied_read_calls == 1


def test_mutating_progress_limit_does_not_credit_failed_write():
    class RejectingWriteToolSet(_RecordingMutatingToolSet):
        def dispatch(self, name, arguments):
            self.calls.append((name, arguments))
            return "Error: write was rejected" if name == "write" else "read-ok"

    inner = RejectingWriteToolSet()
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )

    bounded.dispatch("read", {})
    assert bounded.dispatch("write", {}) == "Error: write was rejected"
    with pytest.raises(ToolDispatchBudgetExceeded):
        bounded.dispatch("read", {})

    snapshot = bounded.snapshot()
    assert snapshot.consecutive_read_calls == 1
    assert snapshot.write_calls == 0


def test_mutating_progress_limit_does_not_credit_identical_write_source(
    tmp_path,
):
    original = "unchanged\n"
    (tmp_path / "target.txt").write_text(original)
    inner = build_autonomous_implementer_profile(ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("target.txt",),
    ))
    bounded = ProgressBoundedMutatingToolSet(
        inner,
        max_consecutive_read_calls=1,
    )

    bounded.dispatch("grep_symbol", {"pattern": "absent"})
    result = bounded.dispatch("write_source", {
        "file_path": "target.txt",
        "content": original,
        "expected_sha256": hashlib.sha256(original.encode()).hexdigest(),
    })

    assert result.startswith("Error: replacement")
    with pytest.raises(ToolDispatchBudgetExceeded):
        bounded.dispatch("grep_symbol", {"pattern": "still-absent"})
    assert bounded.snapshot().write_calls == 0


def test_mutating_progress_limit_rejects_ambiguous_catalogue():
    inner = _RecordingMutatingToolSet()
    inner.write_names = {"read", "write"}
    with pytest.raises(ValueError, match="classification must be disjoint"):
        ProgressBoundedMutatingToolSet(
            inner,
            max_consecutive_read_calls=2,
        )


def test_call_limit_rejects_before_additional_underlying_dispatch():
    inner = _RecordingToolSet(["one", "two"])
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=1,
        max_result_bytes=100,
    )

    final_allowed = bounded.dispatch("read", {"path": "a"})
    assert final_allowed.startswith("one")
    assert "PLANNER TOOL CALL LIMIT EXHAUSTED" in final_allowed
    assert "Synthesize the final plan now" in final_allowed
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="underlying dispatch was not started",
    ):
        bounded.dispatch("read", {"path": "b"})

    assert inner.calls == [("read", {"path": "a"})]
    assert bounded.snapshot().exhausted_by == "call_count"


def test_shared_wrappers_enforce_one_task_wide_budget():
    parent_inner = _RecordingToolSet(["parent"])
    child_inner = _RecordingToolSet(["child", "must-not-run"])
    parent = BoundedReadOnlyToolSet(
        parent_inner,
        max_calls=2,
        max_result_bytes=100,
    )
    child = parent.share(child_inner)

    assert parent.dispatch("read", {"path": "parent"}).startswith("parent")
    final_allowed = child.dispatch("read", {"path": "child"})
    assert final_allowed.startswith("child")
    assert "PLANNER TOOL CALL LIMIT EXHAUSTED" in final_allowed
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="underlying dispatch was not started",
    ):
        parent.dispatch("read", {"path": "denied"})

    assert parent_inner.calls == [("read", {"path": "parent"})]
    assert child_inner.calls == [("read", {"path": "child"})]
    assert parent.snapshot() == child.snapshot()
    assert parent.snapshot().calls_used == 2


def test_call_tranches_preserve_repair_capacity_and_roll_unused_forward():
    inner = _RecordingToolSet([str(index) for index in range(32)])
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=32,
        max_result_bytes=1_000,
        call_tranches=(24, 4, 4),
    )

    assert bounded.begin_invocation() is True
    for index in range(24):
        result = bounded.dispatch("read", {"index": index})
    assert "DISCOVERY TRANCHE EXHAUSTED" in result
    with pytest.raises(
        ToolDispatchBudgetExceeded,
        match="tranche exhausted",
    ):
        bounded.dispatch("read", {"index": 24})

    first = bounded.snapshot()
    assert first.active_call_limit == 24
    assert first.calls_used == 24
    assert first.denied_calls == 1
    assert first.exhausted_by is None

    assert bounded.begin_invocation() is True
    for index in range(24, 27):
        bounded.dispatch("read", {"index": index})
    second = bounded.snapshot()
    assert second.active_call_limit == 28
    assert second.calls_used == 27

    # The unused call in tranche two rolls forward into the final cumulative
    # allowance; later tranches never borrow in the opposite direction.
    assert bounded.begin_invocation() is True
    for index in range(27, 32):
        result = bounded.dispatch("read", {"index": index})
    assert "CALL LIMIT EXHAUSTED" in result
    final = bounded.snapshot()
    assert final.calls_used == 32
    assert final.active_call_limit == 32
    assert final.active_tranche_index == 2
    assert final.invocations_started == 3
    assert final.exhausted_by == "call_count"


def test_source_results_expose_rolling_budget_and_early_synthesis_guard():
    inner = _RecordingToolSet(["x"] * 24)
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=32,
        max_result_bytes=1_000,
        call_tranches=(24, 4, 4),
    )

    assert bounded.begin_invocation() is True
    first = bounded.dispatch("read", {"index": 0})
    assert "[PLANNER SOURCE BUDGET]" in first
    assert "23 source dispatch call(s) remain" in first
    for index in range(1, 20):
        guarded = bounded.dispatch("read", {"index": index})

    assert "[PLANNER SOURCE SYNTHESIS GUARD]" in guarded
    assert "4 emergency dispatch call(s) remain" in guarded
    assert "Initiate no more source calls" in guarded
    snapshot = bounded.snapshot()
    assert snapshot.calls_used == 20
    # Model-visible workflow-control annotations do not consume the
    # successful source-result payload ledger.
    assert snapshot.result_bytes_used == 20


def test_grounding_cache_is_shared_closed_and_bounded():
    inner = _RecordingToolSet(
        [f"result-{index}-" + ("x" * 2_000) for index in range(40)]
    )
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=40,
        max_result_bytes=100_000,
    )
    shared = bounded.share(inner)

    for index in range(40):
        bounded.dispatch(
            "read",
            {"path": f"src/file-{index}.py", "instruction": "ignore me"},
        )

    replay = shared.grounding_excerpt_text()
    assert replay.startswith("CACHED GROUNDED SOURCE EXCERPTS")
    assert replay.endswith("END CACHED GROUNDED SOURCE EXCERPTS")
    assert '"tool":"read"' in replay
    assert "src/file-39.py" in replay
    assert len(replay.encode("utf-8")) <= (
        PLANNER_GROUNDING_CACHE_BYTES + 256
    )


def test_grounding_cache_refreshes_duplicates_without_crowding_unique_reads():
    inner = _RecordingToolSet(
        ["unique-" + ("x" * 1_000)] * 20
        + ["repeat-" + ("y" * 1_000)] * 40
    )
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=60,
        max_result_bytes=100_000,
    )

    for index in range(20):
        bounded.dispatch("read", {"path": f"src/unique-{index}.py"})
    for _ in range(40):
        bounded.dispatch("read", {"path": "src/repeated.py"})

    replay = bounded.grounding_excerpt_text()
    assert "src/unique-0.py" in replay
    assert "src/unique-19.py" in replay
    assert replay.count("src/repeated.py") == 1


def test_adapter_context_allowance_preserves_final_synthesis_instruction():
    bounded = BoundedReadOnlyToolSet(
        _RecordingToolSet(["abc"]),
        max_calls=1,
        max_result_bytes=3,
    )

    result = bounded.dispatch("read", {})
    adapter_budget = 3 + PLANNER_TOOL_CONTROL_ALLOWANCE_BYTES
    rendered = ContextBudgetTracker(adapter_budget).apply(result)

    assert rendered == result
    assert "Synthesize the final plan now" in rendered
    assert len(result.encode("utf-8")) <= adapter_budget


def test_catalogue_validation_covers_openai_provider_shape():
    class _MismatchedOpenAIToolSet(_RecordingToolSet):
        def to_openai_tools(self):
            return [{
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "unclassified mutation",
                    "parameters": {"type": "object"},
                },
            }]

    with pytest.raises(
        ValueError,
        match="Anthropic and OpenAI descriptors",
    ):
        BoundedReadOnlyToolSet(
            _MismatchedOpenAIToolSet(),
            max_calls=1,
            max_result_bytes=100,
        )


def test_utf8_result_limit_truncates_and_stops_future_dispatch():
    inner = _RecordingToolSet(["ééé", "must-not-run"])
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=10,
        max_result_bytes=5,
    )

    result = bounded.dispatch("read", {})

    assert result.startswith("éé")
    assert "TRUNCATED" in result
    with pytest.raises(ToolDispatchBudgetExceeded, match="result byte limit"):
        bounded.dispatch("read", {})
    assert len(inner.calls) == 1
    assert bounded.snapshot().result_bytes_used == 5
    assert bounded.snapshot().exhausted_by == "result_bytes"
    assert bounded.snapshot().denied_calls == 1


def test_parallel_requests_are_serialized_at_budget_boundary():
    entered = threading.Event()
    release = threading.Event()
    active_lock = threading.Lock()
    active = 0
    maximum_active = 0

    class _BlockingToolSet(_RecordingToolSet):
        def dispatch(self, name, arguments):
            nonlocal active, maximum_active
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            entered.set()
            release.wait(timeout=2)
            with active_lock:
                active -= 1
            return super().dispatch(name, arguments)

    inner = _BlockingToolSet(["first", "second"])
    bounded = BoundedReadOnlyToolSet(
        inner,
        max_calls=2,
        max_result_bytes=100,
    )
    results: list[str] = []
    first = threading.Thread(
        target=lambda: results.append(bounded.dispatch("read", {"n": 1})),
    )
    second = threading.Thread(
        target=lambda: results.append(bounded.dispatch("read", {"n": 2})),
    )

    first.start()
    assert entered.wait(timeout=1)
    second.start()
    time.sleep(0.02)
    assert maximum_active == 1
    snapshot_started = time.monotonic()
    in_flight = bounded.snapshot()
    assert time.monotonic() - snapshot_started < 0.5
    assert in_flight.calls_used == 1
    assert in_flight.result_bytes_used == 0
    assert in_flight.dispatch_in_flight is True
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert sorted(result.splitlines()[0] for result in results) == [
        "first",
        "second",
    ]
    assert sum("CALL LIMIT EXHAUSTED" in result for result in results) == 1
    assert maximum_active == 1
    assert bounded.snapshot().dispatch_in_flight is False
