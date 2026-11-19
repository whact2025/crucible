# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused recovery for empty/max-turn implementer attempts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Implementation,
    InvocationResult,
    Message,
    ProjectContext,
    Spec,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from assay.corpus.store import Corpus
from assay.types.dag import (
    ImplementationDirective,
    MutationRegion,
    Slice,
    Tier,
    TierDAG,
)
from crucible.common.policies import BreachPolicy


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_ACCOUNTING_DEBT_EXECUTION = ExecutionCompleteness(
    token_usage_complete=False,
    cost_complete=False,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub",
    requested_model="stub-model",
    events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
)


def _successful_gate_outcome():
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    def runner(role: str) -> RunnerResult:
        return RunnerResult(
            role=role,
            runner=f"tests.fake_{role}",
            returncode=0,
            values={"tests_passed": 1.0, "tests_failed": 0.0},
            raw_output=b'{"values":{"tests_passed":1}}',
            stderr=b"",
            duration_s=0.01,
        )

    return BuildVerifyOutcome(
        "success",
        build=runner("build"),
        verify=runner("verify"),
    )


def _contract_aware_build_verify_stub(callback):
    """Give deterministic gate fakes the same writer-epoch boundary as a runner."""

    def settled(*args, **kwargs):
        outcome = callback(*args, **kwargs)
        from crucible.workflows import _build_verify

        _build_verify._settle_no_process_writer_epoch(
            kwargs.get("writer_epoch_members"),
            phase="test_stub_gate",
            succeeded=True,
        )
        return outcome

    return settled


def _failed_gate_outcome(classification: str):
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )

    if classification == "runner_missing":
        return BuildVerifyOutcome(classification, build=None, verify=None)
    assert classification == "verify_failed"
    passed = _successful_gate_outcome()
    return BuildVerifyOutcome(
        classification,
        build=passed.build,
        verify=RunnerResult(
            role="verify",
            runner="tests.fake_verify",
            returncode=1,
            values={"tests_passed": 0.0, "tests_failed": 1.0},
            raw_output=b'{"values":{"tests_failed":1}}',
            stderr=b"verification failed",
            duration_s=0.01,
        ),
    )


class _MaxTurnsThenWriteBackend:
    name = "stub-empty-recovery"
    model = "stub-model"

    def __init__(
        self,
        repo: Path,
        *,
        target_path: str = "target.py",
        replacement: str = "value = 2\n",
    ):
        self.repo = repo
        self.target_path = target_path
        self.replacement = replacement
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        if len(self.invocations) == 1:
            return InvocationResult(
                final_message="still investigating",
                transcript=(),
                usage=Usage(input_tokens=11, output_tokens=3, cost_usd=0.0),
                stop_reason="max_turns",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        (self.repo / self.target_path).write_text(
            self.replacement,
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=7, output_tokens=2, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _AlwaysEmptyBackend:
    name = "stub-always-empty"
    model = "stub-model"

    def __init__(self):
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _EmptyThenBlockedBackend(_AlwaysEmptyBackend):
    name = "stub-empty-then-blocked"

    def invoke(self, invocation):
        self.invocations.append(invocation)
        message = (
            "DONE no changes"
            if len(self.invocations) == 1
            else "BLOCKED: no safe edit is available"
        )
        return InvocationResult(
            final_message=message,
            transcript=(Message(role="assistant", content=message),),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _ImplementerBlockedBackend:
    name = "stub-implementer-blocked"
    model = "stub-model"

    def __init__(
        self,
        *,
        repo: Path | None = None,
        mutate: bool = False,
        scope_escape: bool = False,
        messages: tuple[Message, ...] | None = None,
        completeness: ExecutionCompleteness = _COMPLETE_EXECUTION,
    ):
        self.repo = repo
        self.mutate = mutate
        self.scope_escape = scope_escape
        self.invocations = []
        blocked = (
            "BLOCKED — official provider bindings are unresolved; "
            "the directive forbids mock substitution."
        )
        self.messages = messages or (
            Message(role="assistant", content=blocked),
        )
        self.completeness = completeness

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        if self.mutate:
            assert self.repo is not None
            self.repo.joinpath("target.py").write_text(
                "value = partial\n",
                encoding="utf-8",
            )
        if self.scope_escape:
            assert self.repo is not None
            self.repo.joinpath("rogue.py").write_text(
                "escaped = True\n",
                encoding="utf-8",
            )
        message = self.messages[-1].content
        return InvocationResult(
            final_message=message,
            transcript=self.messages,
            usage=Usage(input_tokens=13, output_tokens=4, cost_usd=0.25),
            stop_reason="end_turn",
            completeness=self.completeness,
            execution_trace=_COMPLETE_TRACE,
        )


class _ScopeEscapeThenBlockedBackend:
    """Reach scope recovery once, then revoke all continuation authority."""

    name = "stub-scope-escape-then-blocked"
    model = "stub-model"

    def __init__(self):
        self.repo = None
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        if len(self.invocations) == 1:
            self.repo.joinpath("rogue.py").write_text(
                "escaped = True\n",
                encoding="utf-8",
            )
            message = "DONE initial attempt"
        else:
            self.repo.joinpath("target.py").write_text(
                "value = blocked_retry\n",
                encoding="utf-8",
            )
            message = "BLOCKED: scope-safe repair is unavailable"
        return InvocationResult(
            final_message=message,
            transcript=(Message(role="assistant", content=message),),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _WriteThenBlockedBackend:
    name = "stub-write-then-blocked"
    model = "stub-model"

    def __init__(self):
        self.repo = None
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        if len(self.invocations) == 1:
            self.repo.joinpath("target.py").write_text(
                "value = candidate\n",
                encoding="utf-8",
            )
            return InvocationResult(
                final_message="DONE initial candidate",
                transcript=(
                    Message(role="assistant", content="DONE initial candidate"),
                ),
                usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        message = "BLOCKED: the requested repair has no valid implementation"
        return InvocationResult(
            final_message=message,
            transcript=(Message(role="assistant", content=message),),
            usage=Usage(input_tokens=7, output_tokens=2, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _WriteTargetBackend:
    name = "stub-write-target"
    model = "stub-model"

    def __init__(self):
        self.repo = None
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        self.repo.joinpath("target.py").write_text(
            "value = 2\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _CreateNewBackend(_WriteTargetBackend):
    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        self.repo.joinpath("new.py").write_text(
            "created = True\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _CreateGeneratedBackend(_WriteTargetBackend):
    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        generated = self.repo / "generated"
        generated.mkdir(exist_ok=True)
        generated.joinpath("partial.py").write_text(
            "created = True\n",
            encoding="utf-8",
        )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=5, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _MissingDeclaredCreatesBackend:
    name = "stub-declared-create-recovery"
    model = "stub-model"

    def __init__(
        self,
        repo: Path,
        *,
        complete_on_retry: bool,
        initial_stop_reason: str = "max_turns",
    ):
        self.repo = repo
        self.complete_on_retry = complete_on_retry
        self.initial_stop_reason = initial_stop_reason
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        attempt = len(self.invocations)
        if attempt == 1:
            self.repo.joinpath("target.py").write_text(
                "value = partial\n",
                encoding="utf-8",
            )
        if attempt == 2 and self.complete_on_retry:
            self.repo.joinpath("generated_one.py").write_text(
                "ONE = 1\n",
                encoding="utf-8",
            )
        return InvocationResult(
            final_message=("still working" if attempt == 1 else "DONE"),
            transcript=(),
            usage=Usage(input_tokens=9, output_tokens=2, cost_usd=0.0),
            stop_reason=(
                self.initial_stop_reason if attempt == 1 else "end_turn"
            ),
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _IncompleteTerminalTwiceBackend:
    name = "stub-incomplete-terminal-twice"
    model = "stub-model"

    def __init__(
        self,
        *,
        terminal_kind: str,
        recovery_terminal_kind: str | None = None,
        accounting_complete: bool = True,
        blocked_report: bool = False,
        blocked_on_attempt: int | None = None,
    ):
        self.terminal_kind = terminal_kind
        self.recovery_terminal_kind = recovery_terminal_kind
        self.accounting_complete = accounting_complete
        self.blocked_report = blocked_report
        self.blocked_on_attempt = blocked_on_attempt
        self.repo = None
        self.invocations = []
        self.pre_call_states = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        from assay.types.invocation import TransportError

        self.invocations.append(invocation)
        assert self.repo is not None
        self.pre_call_states.append((
            self.repo.joinpath("target.py").read_text(encoding="utf-8"),
            self.repo.joinpath("failed_only.py").exists(),
        ))
        attempt = len(self.invocations)
        self.repo.joinpath("target.py").write_text(
            f"value = failed_{attempt}\n",
            encoding="utf-8",
        )
        self.repo.joinpath("failed_only.py").write_text(
            f"FAILED_{attempt} = True\n",
            encoding="utf-8",
        )
        terminal_kind = (
            self.recovery_terminal_kind
            if attempt > 1 and self.recovery_terminal_kind is not None
            else self.terminal_kind
        )
        blocked = (
            self.blocked_report
            or attempt == self.blocked_on_attempt
        )
        final_message = (
            "BLOCKED: provider terminal prevents safe completion"
            if blocked
            else "still working"
        )
        stop_reason = {
            "end_turn": "end_turn",
            "max_turns": "max_turns",
            "tool_mutation_indeterminate": "tool_mutation_indeterminate",
        }.get(terminal_kind, "error")
        return InvocationResult(
            final_message=final_message,
            transcript=(
                (Message(role="assistant", content=final_message),)
                if blocked else ()
            ),
            usage=Usage(input_tokens=5, output_tokens=2, cost_usd=0.0),
            stop_reason=stop_reason,
            error=(
                TransportError(message="worker timed out")
                if terminal_kind
                in {"transport", "tool_mutation_indeterminate"}
                else (
                    SimpleNamespace(
                        kind="timeout",
                        message="worker wall deadline expired",
                        retriable=False,
                    )
                    if terminal_kind == "timeout"
                    else None
                )
            ),
            completeness=(
                _COMPLETE_EXECUTION
                if self.accounting_complete
                else _ACCOUNTING_DEBT_EXECUTION
            ),
            execution_trace=_COMPLETE_TRACE,
        )


class _RaisedMutationIndeterminateBackend:
    name = "stub-raised-mutation-indeterminate"
    model = "stub-model"

    def __init__(self, *, close_settles: bool) -> None:
        self.repo: Path | None = None
        self.close_settles = close_settles
        self.invocations = []
        self.close_calls = 0
        self.terminal_error = None

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def invoke(self, invocation):
        from crucible.tools import ToolMutationIndeterminate

        self.invocations.append(invocation)
        assert self.repo is not None
        self.repo.joinpath("target.py").write_text(
            "value = indeterminate\n",
            encoding="utf-8",
        )
        self.repo.joinpath("failed_only.py").write_text(
            "FAILED = True\n",
            encoding="utf-8",
        )
        self.terminal_error = ToolMutationIndeterminate(
            "delete publication settlement is unknown"
        )
        raise self.terminal_error

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("provider tool worker remains live")


class _ReturnedMutationIndeterminateCloseFailureBackend(
    _IncompleteTerminalTwiceBackend
):
    def __init__(self) -> None:
        super().__init__(terminal_kind="tool_mutation_indeterminate")
        self.close_calls = 0
        self.close_settles = False

    def close(self) -> None:
        self.close_calls += 1
        if not self.close_settles:
            raise RuntimeError("returned-terminal provider remains live")


class _NonAuthorizingTerminalBackend:
    name = "stub-nonauthorizing-terminal"
    model = "stub-model"

    def __init__(self, *, error_kind: str):
        self.error_kind = error_kind
        self.repo = None
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        self.repo.joinpath("target.py").write_text(
            f"value = {self.error_kind}\n",
            encoding="utf-8",
        )
        self.repo.joinpath("failed_only.py").write_text(
            "FAILED_ONLY = True\n",
            encoding="utf-8",
        )
        evidence_incomplete = self.error_kind == "incomplete_evidence"
        return InvocationResult(
            final_message="terminal after partial write",
            transcript=(),
            usage=Usage(input_tokens=5, output_tokens=2, cost_usd=0.0),
            stop_reason="error",
            error=SimpleNamespace(
                kind=self.error_kind,
                message=f"{self.error_kind} terminal",
                retriable=False,
            ),
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=not evidence_incomplete,
                trace_complete=not evidence_incomplete,
            ),
            execution_trace=_COMPLETE_TRACE,
        )


class _ScopedDeclaredCreateBackend:
    """Exercise declared-create recovery through the real scoped tools."""

    name = "stub-scoped-declared-create-recovery"
    model = "stub-model"

    def __init__(self, *, fail_first_recovery: bool):
        self.fail_first_recovery = fail_first_recovery
        self.invocations = []
        self.dispatch_results: list[str] = []
        self.toolset = None
        self.attach_history = []
        self.recovery_toolsets = []
        self.recovery_calls = 0

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        self.toolset = toolset
        self.attach_history.append(toolset)

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        attempt = len(self.invocations)
        assert self.toolset is not None
        if attempt == 1:
            result = self.toolset.dispatch("apply_patch", {
                "file_path": "target.py",
                "before": "value = 1\n",
                "after": "value = partial\n",
            })
        else:
            self.recovery_calls += 1
            self.recovery_toolsets.append(self.toolset)
            if self.fail_first_recovery and self.recovery_calls == 1:
                # The exact-create surface does not expose an arbitrary
                # patcher, so this live-like production edit is impossible.
                result = self.toolset.dispatch("apply_patch", {
                    "file_path": "component.py",
                    "before": "class Component:\n",
                    "after": "class Component:\n    pass\n",
                })
            else:
                # Simulate a model still trying to choose a production path.
                # The recovery writer accepts content only and defensively
                # ignores this undeclared extra argument, so its closure must
                # materialize the currently bound creates_path instead.
                attempted_path = (
                    "component.py"
                    if self.recovery_calls % 2 else "provider.py"
                )
                result = self.toolset.dispatch("write_source", {
                    "file_path": attempted_path,
                    "content": f"CREATED_{self.recovery_calls} = True\n",
                })
        self.dispatch_results.append(result)
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=3, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _DirectRecoveryEscapeThenCreateBackend:
    """Bypass model tools to exercise the nested checkpoint authority."""

    name = "stub-direct-recovery-escape"
    model = "stub-model"

    def __init__(self):
        self.repo = None
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        attempt = len(self.invocations)
        assert self.repo is not None
        if attempt == 1:
            self.repo.joinpath("target.py").write_text(
                "value = partial\n",
                encoding="utf-8",
            )
        elif attempt == 2:
            self.repo.joinpath("target.py").write_text(
                "value = unrelated\n",
                encoding="utf-8",
            )
        else:
            self.repo.joinpath("generated_one.py").write_text(
                "ONE = 1\n",
                encoding="utf-8",
            )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=Usage(input_tokens=3, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
        )


class _RaisingRecoveryBackend:
    name = "stub-raising-recovery"
    model = "stub-model"

    def __init__(self):
        self.repo = None
        self.invocations = []
        self.attach_history = []
        self.toolset = None

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        self.toolset = toolset
        self.attach_history.append(toolset)

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        if len(self.invocations) == 1:
            self.repo.joinpath("target.py").write_text(
                "value = partial\n",
                encoding="utf-8",
            )
            return InvocationResult(
                final_message="DONE",
                transcript=(),
                usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
                stop_reason="end_turn",
                completeness=_COMPLETE_EXECUTION,
                execution_trace=_COMPLETE_TRACE,
            )
        raise RuntimeError("provider exploded during exact create")


class _DeclaredTestMutationBackend:
    name = "stub-declared-test-mutation"
    model = "stub-model"

    def __init__(
        self,
        *,
        production_path: str = "target.py",
        initial_test_path: str | None = None,
        recovery_test_path: str | None = None,
        scope_escape_path: str | None = None,
        complete_pricing: bool = False,
        cost_usd: float = 0.0,
    ):
        self.repo = None
        self.production_path = production_path
        self.initial_test_path = initial_test_path
        self.recovery_test_path = recovery_test_path
        self.scope_escape_path = scope_escape_path
        self.complete_pricing = complete_pricing
        self.cost_usd = cost_usd
        self.invocations = []

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation):
        self.invocations.append(invocation)
        assert self.repo is not None
        if len(self.invocations) == 1:
            self.repo.joinpath(self.production_path).write_text(
                "value = production\n",
                encoding="utf-8",
            )
            if self.scope_escape_path is not None:
                self.repo.joinpath(self.scope_escape_path).write_text(
                    "value = escaped\n",
                    encoding="utf-8",
                )
            test_path = self.initial_test_path
        else:
            test_path = self.recovery_test_path
        if test_path is not None:
            self.repo.joinpath(test_path).write_text(
                f"def test_local_{len(self.invocations)}():\n    assert True\n",
                encoding="utf-8",
            )
        usage = Usage(
            input_tokens=4,
            output_tokens=1,
            cost_usd=self.cost_usd,
        )
        return InvocationResult(
            final_message="DONE",
            transcript=(),
            usage=usage,
            stop_reason="end_turn",
            completeness=_COMPLETE_EXECUTION,
            execution_trace=_COMPLETE_TRACE,
            pricing_evidence=(
                UsagePricingEvidence(
                    segments=(
                        UsagePricingSegment(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_creation_tokens=0,
                            cache_read_tokens=0,
                            cost_usd=usage.cost_usd,
                        ),
                    ),
                    complete=True,
                )
                if self.complete_pricing
                else None
            ),
        )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "t"], cwd=path, check=True,
    )
    (path / "target.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "target.py"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "init"], cwd=path, check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _spec(
    base_commit: str,
    *,
    slice_id: str = "s0",
    no_surface: bool = False,
    verify_mode: str | None = None,
    creates_paths: tuple[str, ...] = (),
    files_in_scope: tuple[str, ...] | None = None,
    mutation_regions: tuple[MutationRegion, ...] = (),
) -> Spec:
    declared_files = (
        ()
        if no_surface
        else files_in_scope if files_in_scope is not None else ("target.py",)
    )
    slice_ = Slice(
        id=slice_id,
        directive=ImplementationDirective(
            slice_id=slice_id,
            intent="change target value",
            files_in_scope=declared_files,
            creates_paths=creates_paths,
            verify_mode=(
                verify_mode
                if verify_mode is not None
                else "skip" if no_surface else "tests"
            ),
            blast_radius={
                "max_files": max(
                    1,
                    len(declared_files) + len(creates_paths),
                )
            },
            mutation_regions=mutation_regions,
        ),
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_empty_retry_test",
            repo_url="https://example.invalid/repo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise empty implementation recovery",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=(slice_,), depends_on=()),),
        ),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _two_slice_spec(
    base_commit: str,
    *,
    first_files_in_scope: tuple[str, ...] | None = None,
) -> Spec:
    first_spec = _spec(
        base_commit,
        files_in_scope=first_files_in_scope,
    )
    assert first_spec.tier_dag is not None
    first_slice = first_spec.tier_dag.tiers[0].slices[0]
    sibling_slice = Slice(
        id="s1",
        directive=ImplementationDirective(
            slice_id="s1",
            intent="change sibling value",
            files_in_scope=("sibling.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    return replace(
        first_spec,
        tier_dag=TierDAG(
            tiers=(
                Tier(
                    tier_id="t0",
                    slices=(first_slice, sibling_slice),
                    depends_on=(),
                ),
            ),
        ),
    )


def _mixed_spec(base_commit: str, *, verification_first: bool) -> Spec:
    mutating = Slice(
        id="mutate",
        directive=ImplementationDirective(
            slice_id="mutate",
            intent="change target value",
            files_in_scope=("target.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    verification = Slice(
        id="verify_only",
        directive=ImplementationDirective(
            slice_id="verify_only",
            intent="verify the composed tier candidate",
            files_in_scope=(),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    slices = (
        (verification, mutating)
        if verification_first
        else (mutating, verification)
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_mixed_verification_test",
            repo_url="https://example.invalid/repo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise mixed mutating and verification-only slices",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _persistent_dependency_spec(
    base_commit: str,
    *,
    successor_creates: bool = False,
    successor_recursive: bool = False,
) -> Spec:
    assert not (successor_creates and successor_recursive)
    dependency = Slice(
        id="dependency_mutation",
        directive=ImplementationDirective(
            slice_id="dependency_mutation",
            intent="materialize the dependency postimage",
            files_in_scope=("target.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    verification = Slice(
        id="dependent_verification",
        directive=ImplementationDirective(
            slice_id="dependent_verification",
            intent="verify the live accepted dependency postimage",
            files_in_scope=("generated/",) if successor_recursive else (),
            creates_paths=("new.py",) if successor_creates else (),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        ),
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_persistent_dependency_test",
            repo_url="https://example.invalid/repo.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="prove persistent dependency workspace authority",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(dependency,), depends_on=()),
            Tier(
                tier_id="t1",
                slices=(verification,),
                depends_on=("t0",),
            ),
        )),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )


def _prepare_accepted_persistent_dependency(
    tmp_path,
    monkeypatch,
    *,
    successor_creates: bool = False,
    successor_recursive: bool = False,
):
    from crucible.workflows import tier as tier_workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    Corpus(corpus_dir).put_spec(_persistent_dependency_spec(
        base,
        successor_creates=successor_creates,
        successor_recursive=successor_recursive,
    ))
    backend = _WriteTargetBackend()
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    assert tier_workflow.run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
    ) == 0
    return repo, Corpus(corpus_dir), backend


def _run_persistent_successor(
    repo,
    corpus,
    monkeypatch,
    *,
    successor_backend=None,
):
    from crucible.workflows import tier as tier_workflow

    gate_slices: list[str] = []

    def passing_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    def reject_backend(*args, **kwargs):
        pytest.fail("persistent dependency proof must precede backend lookup")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(passing_gate),
    )
    if successor_backend is None:
        resolver = reject_backend
    else:
        successor_backend.repo = repo
        resolver = lambda *args, **kwargs: successor_backend
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        resolver,
    )
    rc = tier_workflow.run_tier(
        spec=corpus.spec_dir,
        tier_id="t1",
        cwd=str(repo),
        max_turns=24,
        persistent_workspace=True,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
    )
    summaries = []
    for run_dir in (corpus.spec_dir / "runs").iterdir():
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("tier_id") == "t1":
            summaries.append((run_dir.stat().st_mtime_ns, summary))
    assert summaries
    return rc, max(summaries, key=lambda item: item[0])[1], gate_slices


def _install_passing_completeness_review(
    monkeypatch,
    backend,
    reviewed_artifacts: list[str],
) -> None:
    from assay import ReviewCoverage, Verdict
    from crucible.workflows import tier as tier_workflow

    def passing_review(artifact, rubric, *, corpus, **kwargs):
        rendered = bytes(artifact)
        text = rendered.decode("utf-8")
        reviewed_artifacts.append(text)
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=corpus.put_artifact(
                rendered,
                kind="reviewed_artifacts",
            ),
            artifact_commit=None,
            reasoning="Exact composed verification evidence is sufficient.",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(text),
                artifact_chars_reviewed=len(text),
            ),
            backend_name=backend.name,
            transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
            cost_usd=0.0,
            execution=None,
        )

    monkeypatch.setattr(tier_workflow, "review", passing_review)


def _run_mixed(
    tmp_path: Path,
    monkeypatch,
    backend,
    *,
    verification_first: bool,
    abbreviated_base: bool = False,
    skip_review: bool = False,
):
    from crucible.common.policies import ReviewPolicy
    from crucible.workflows.tier import run_tier

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    Corpus(corpus_dir).put_spec(
        _mixed_spec(
            base[:8] if abbreviated_base else base,
            verification_first=verification_first,
        )
    )
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )
    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=skip_review,
        review_policy=ReviewPolicy(rubrics=("tier_completeness",)),
    )
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads(run_dir.joinpath("summary.json").read_text())
    return rc, summary, repo, Corpus(corpus_dir)


def _run(
    tmp_path: Path,
    monkeypatch,
    backend,
    *,
    slice_id="s0",
    no_surface=False,
    inherited_dirty=False,
    verify_mode=None,
    creates_paths=(),
    files_in_scope=None,
    committed_files=None,
    inherited_mutations=None,
    skip_build_verify=True,
    skip_smoke_check=True,
    skip_mutation_scope=True,
    skip_review=True,
    backend_resolver=None,
    mutation_regions=(),
    max_turns=24,
    **kwargs,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    for relative, content in (committed_files or {}).items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    if committed_files:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "add focused fixtures"],
            cwd=repo,
            check=True,
        )
        base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if inherited_dirty:
        (repo / "target.py").write_text(
            "value = inherited\n", encoding="utf-8",
        )
    for relative, content in (inherited_mutations or {}).items():
        (repo / relative).write_text(content, encoding="utf-8")
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    Corpus(corpus_dir).put_spec(_spec(
        base,
        slice_id=slice_id,
        no_surface=no_surface,
        verify_mode=verify_mode,
        creates_paths=creates_paths,
        files_in_scope=files_in_scope,
        mutation_regions=mutation_regions,
    ))
    if hasattr(backend, "repo"):
        backend.repo = repo

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        backend_resolver or (lambda role, *args, **kw: backend),
    )
    from crucible.workflows.tier import run_tier

    run_kwargs = dict(kwargs)
    if max_turns is not None:
        run_kwargs["max_turns"] = max_turns
    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_build_verify=skip_build_verify,
        skip_smoke_check=skip_smoke_check,
        skip_mutation_scope=skip_mutation_scope,
        skip_smoke_bench=True,
        skip_review=skip_review,
        **run_kwargs,
    )
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return rc, summary, repo


def test_declared_create_probe_rejects_directory_but_accepts_symlinks(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    repo.joinpath("generated_regular.py").write_text("OK = True\n")
    repo.joinpath("generated_directory.py").mkdir()
    repo.joinpath("generated_link.py").symlink_to("missing-target.py")
    repo.joinpath("generated_alias.py").symlink_to("generated_regular.py")
    directive = ImplementationDirective(
        slice_id="s0",
        intent="create outputs",
        files_in_scope=(),
        creates_paths=(
            "repo/generated_regular.py",
            "repo/generated_directory.py",
            "repo/generated_link.py",
            "repo/generated_alias.py",
            "repo/still_missing.py",
        ),
        verify_mode="tests",
        blast_radius={"max_files": 5},
    )
    from crucible.workflows.tier import _missing_declared_create_paths

    assert _missing_declared_create_paths(
        cwd=str(repo),
        directive=directive,
    ) == (
        "generated_directory.py",
        "still_missing.py",
    )


def test_max_turn_empty_attempt_gets_one_focused_retry(tmp_path, monkeypatch):
    backend = _MaxTurnsThenWriteBackend(tmp_path)

    rc, summary, repo = _run(
        tmp_path, monkeypatch, backend, inherited_dirty=True,
    )

    assert rc == 0
    assert (repo / "target.py").read_text(encoding="utf-8") == "value = 2\n"
    assert len(backend.invocations) == 2
    assert backend.invocations[0].max_turns == 24
    assert backend.invocations[1].max_turns == 8
    assert "EMPTY-DIFF RECOVERY" in backend.invocations[1].user
    assert summary["phase_1_slices"][0]["classification"] == "success"
    assert summary["costs"]["total_input_tokens"] == 18
    assert summary["costs"]["total_output_tokens"] == 5


def test_smoke_pass_through_settles_unused_gate_epoch_after_recovery(
    tmp_path,
    monkeypatch,
):
    backend = _MaxTurnsThenWriteBackend(
        tmp_path,
        target_path="target.cpp",
        replacement="int value = 2;\n",
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        committed_files={"target.cpp": "int value = 1;\n"},
        files_in_scope=("target.cpp",),
        skip_build_verify=True,
        skip_smoke_check=False,
    )

    assert rc == 0
    assert len(backend.invocations) == 2
    assert repo.joinpath("target.cpp").read_text(encoding="utf-8") == (
        "int value = 2;\n"
    )
    assert summary["phase_1_slices"][0]["classification"] == "success"


def test_pre_runner_mutation_scope_rejection_does_not_create_gate_epoch(
    tmp_path,
    monkeypatch,
):
    """A cheap rejection must not append an ownerless journal generation."""

    from crucible.workflows import tier as tier_workflow

    def unexpected_gate(*args, **kwargs):
        pytest.fail("mutation-scope rejection must precede build dispatch")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        unexpected_gate,
    )
    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        _WriteTargetBackend(),
        committed_files={
            "target.py": (
                "# OPTIM-START\n"
                "inside = 1\n"
                "# OPTIM-END\n"
                "outside = 1\n"
            ),
        },
        mutation_regions=(
            MutationRegion(
                file="target.py",
                start_marker="OPTIM-START",
                end_marker="OPTIM-END",
            ),
        ),
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=False,
    )

    assert rc == 1
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_mutation_scope"
    )
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "# OPTIM-START\n"
        "inside = 1\n"
        "# OPTIM-END\n"
        "outside = 1\n"
    )


@pytest.mark.parametrize("initial_stop_reason", ["max_turns", "end_turn"])
def test_missing_declared_creates_get_one_focused_completion(
    tmp_path,
    monkeypatch,
    initial_stop_reason,
):
    backend = _MissingDeclaredCreatesBackend(
        tmp_path,
        complete_on_retry=True,
        initial_stop_reason=initial_stop_reason,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        creates_paths=("repo/generated_one.py",),
    )

    assert rc == 0
    assert len(backend.invocations) == 2
    assert backend.invocations[0].max_turns == 24
    if initial_stop_reason == "max_turns":
        assert backend.invocations[1].max_turns == 24
        assert "INCOMPLETE-TERMINAL RECOVERY" in backend.invocations[1].user
        assert "Do not assume any prior model edit survived" in (
            backend.invocations[1].user
        )
        assert "- target.py" in backend.invocations[1].user
        assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
            "value = 1\n"
        )
        recovery = summary["phase_1_slices"][0][
            "incomplete_terminal_recovery"
        ]
        assert recovery == {
            "classification": "hard_fail_max_turns",
            "restored_paths": ["target.py"],
            "restored_to_pre_call_snapshot": True,
            "retry_attempted": True,
        }
    else:
        assert backend.invocations[1].max_turns == 8
        assert "EXACT DECLARED-CREATE RECOVERY" in (
            backend.invocations[1].user
        )
        assert "prior partial implementation is ALREADY APPLIED" in (
            backend.invocations[1].user
        )
        assert "- generated_one.py" in backend.invocations[1].user
        assert "ONE write obligation" in backend.invocations[1].user
        assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
            "value = partial\n"
        )
        assert summary["phase_1_slices"][0][
            "incomplete_terminal_recovery"
        ] is None
    assert repo.joinpath("generated_one.py").read_text(encoding="utf-8") == (
        "ONE = 1\n"
    )
    assert summary["phase_1_slices"][0]["classification"] == "success"
    assert summary["phase_1_slices"][0]["missing_create_paths"] == []


def test_incomplete_terminal_rollback_preserves_inherited_dirt_and_settles(
    tmp_path,
    monkeypatch,
):
    backend = _MissingDeclaredCreatesBackend(
        tmp_path,
        complete_on_retry=True,
        initial_stop_reason="max_turns",
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        creates_paths=("repo/generated_one.py",),
    )

    assert rc == 0
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert "INCOMPLETE-TERMINAL RECOVERY" in backend.invocations[1].user
    # The failed call's "partial" bytes cannot be inherited by the successful
    # retry, while the exact dirty bytes present before that call survive.
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert repo.joinpath("generated_one.py").read_text(encoding="utf-8") == (
        "ONE = 1\n"
    )
    assert summary["phase_1_slices"][0][
        "incomplete_terminal_recovery"
    ] == {
        "classification": "hard_fail_max_turns",
        "restored_paths": ["target.py"],
        "restored_to_pre_call_snapshot": True,
        "retry_attempted": True,
    }

    corpus = Corpus(tmp_path / "corpus")
    from crucible.workflows.tier import attest_mutation_journal_settlement

    settlement = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="incomplete-terminal-recovery-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []
    run_dir = next(
        path
        for path in (corpus.spec_dir / "runs").iterdir()
        if "tier" in path.name
    )
    assert list((run_dir / "phase1_mutations").iterdir()) == []


@pytest.mark.parametrize(
    ("terminal_kind", "expected_classification"),
    [
        ("max_turns", "hard_fail_max_turns"),
        ("transport", "hard_fail_transport"),
    ],
)
def test_repeated_incomplete_terminal_mutations_fail_closed_and_settle(
    tmp_path,
    monkeypatch,
    terminal_kind,
    expected_classification,
):
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind=terminal_kind,
    )
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert "INCOMPLETE-TERMINAL RECOVERY" in backend.invocations[1].user
    assert backend.pre_call_states == [
        ("value = inherited\n", False),
        ("value = inherited\n", False),
    ]
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()
    assert breach_called == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == expected_classification
    recovery = phase["incomplete_terminal_recovery"]
    assert recovery["classification"] == expected_classification
    assert set(recovery["restored_paths"]) == {
        "target.py",
        "failed_only.py",
    }
    assert recovery["restored_to_pre_call_snapshot"] is True
    assert recovery["retry_attempted"] is True

    corpus = Corpus(tmp_path / "corpus")
    from crucible.workflows.tier import attest_mutation_journal_settlement

    settlement = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id=f"repeated-{terminal_kind}-recovery-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []
    run_dir = next(
        path
        for path in (corpus.spec_dir / "runs").iterdir()
        if "tier" in path.name
    )
    assert list((run_dir / "phase1_mutations").iterdir()) == []


def test_indeterminate_mutation_restores_once_without_transport_retry(
    tmp_path,
    monkeypatch,
) -> None:
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="tool_mutation_indeterminate",
    )
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert backend.pre_call_states == [("value = inherited\n", False)]
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()
    assert breach_called == []
    assert summary["transport_failure_slices"] == 0
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert phase["incomplete_terminal_recovery"] == {
        "classification": "hard_fail_tool_mutation_indeterminate",
        "restored_paths": ["target.py", "failed_only.py"],
        "restored_to_pre_call_snapshot": True,
        "retry_attempted": False,
    }

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    assert spec is not None
    from assay import Implementation

    implementation = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert implementation.stop_reason == "tool_mutation_indeterminate"
    run_dir = next(
        path
        for path in (corpus.spec_dir / "runs").iterdir()
        if "tier" in path.name
    )
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert outcome["terminal_class"] == "permanent_failure"


def test_raised_indeterminate_mutation_synthesizes_and_restores_once(
    tmp_path,
    monkeypatch,
) -> None:
    backend = _RaisedMutationIndeterminateBackend(close_settles=True)

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert backend.close_calls == 1
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert not repo.joinpath("failed_only.py").exists()
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["accounting_debt"] is True


def test_raised_indeterminate_mutation_retains_unsettled_tier_owner(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(
        base,
        files_in_scope=("target.py", "failed_only.py"),
    ))
    backend = _RaisedMutationIndeterminateBackend(close_settles=False)
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    capture_calls = 0
    real_prepare = MutationTransaction.prepare_git_capture

    def observed_prepare(self) -> None:
        nonlocal capture_calls
        capture_calls += 1
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_prepare,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "synthesize_tool_mutation_indeterminate_implementation",
        lambda **_kwargs: pytest.fail("unsettled provider must not synthesize"),
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=corpus_dir,
            tier_id="t0",
            cwd=str(repo),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )

    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert caught.value is backend.terminal_error
    assert cleanup_owner.primary_error is caught.value
    assert len(backend.invocations) == 1
    assert backend.close_calls == 1
    assert capture_calls == 0
    assert repo.joinpath("target.py").read_text() == (
        "value = indeterminate\n"
    )
    assert repo.joinpath("failed_only.py").exists()
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True
    assert summary["cleanup_pending"] is True
    partial = summary["partial_summary"]
    assert partial["status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert partial["costs"]["by_role"]["implementer"]["calls"] == 1
    assert outcome["detail_status"] == "incomplete_accounting"
    assert outcome["terminal_class"] == "inconclusive"
    assert cleanup_owner.owner.cleanup_complete is False

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert backend.close_calls == 2
    assert cleanup_owner.owner.cleanup_complete is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_returned_indeterminate_close_failure_retains_known_usage_and_owner(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(
        base,
        files_in_scope=("target.py", "failed_only.py"),
    ))
    backend = _ReturnedMutationIndeterminateCloseFailureBackend()
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    capture_calls = 0
    real_prepare = MutationTransaction.prepare_git_capture

    def observed_prepare(self) -> None:
        nonlocal capture_calls
        capture_calls += 1
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        observed_prepare,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=corpus_dir,
            tier_id="t0",
            cwd=str(repo),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )

    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is caught.value
    assert len(backend.invocations) == 1
    assert backend.close_calls == 1
    assert capture_calls == 0
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["provider_cleanup_pending"] is True
    assert summary["mutation_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True
    partial = summary["partial_summary"]
    assert partial["status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert partial["accounting_evidence"] == "complete"
    assert partial["costs"]["by_role"]["implementer"]["calls"] == 1
    assert partial["costs"]["accounting_debt"] is False
    assert cleanup_owner.owner.cleanup_complete is False

    backend.close_settles = True
    cleanup_owner.retry_cleanup()
    assert backend.close_calls == 2
    assert cleanup_owner.owner.cleanup_complete is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_returned_indeterminate_rejects_foreign_cleanup_owner(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(
        base,
        files_in_scope=("target.py", "failed_only.py"),
    ))
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="tool_mutation_indeterminate",
    )
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
        ProviderSettlementExitStack,
    )

    real_run_after = ProviderSettlementExitStack.run_after_provider_quiescence
    foreign_stack = ProviderSettlementExitStack()
    foreign_owner = None
    injected = False

    def inject_foreign_owner(self, callback, /, *args, **kwargs):
        nonlocal foreign_owner, injected
        terminal_error = self._terminal_precedence
        if terminal_error is not None and not injected:
            injected = True
            foreign_owner = ProviderSettlementCleanupError(
                foreign_stack,
                primary_error=terminal_error,
                cleanup_error=RuntimeError("foreign cleanup owner"),
            )
            raise terminal_error from foreign_owner
        return real_run_after(self, callback, *args, **kwargs)

    monkeypatch.setattr(
        ProviderSettlementExitStack,
        "run_after_provider_quiescence",
        inject_foreign_owner,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=corpus_dir,
            tier_id="t0",
            cwd=str(repo),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )

    assert injected is True
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.__cause__.__cause__ is foreign_owner
    assert len(backend.invocations) == 1
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("failed_only.py").exists()
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary.get("provider_cleanup_pending") is not True


def test_returned_indeterminate_postinvoke_capture_failure_is_secondary(
    tmp_path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        MutationTransaction,
    )

    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="tool_mutation_indeterminate",
    )
    real_prepare = MutationTransaction.prepare_git_capture
    prepare_failed = False

    def fail_first_prepare(self) -> None:
        nonlocal prepare_failed
        if not prepare_failed:
            prepare_failed = True
            raise MutationJournalError(
                "injected returned-terminal postinvoke capture failure"
            )
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        fail_first_prepare,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert prepare_failed is True
    assert len(backend.invocations) == 1
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "MutationJournalError"
    assert summary["accounting_evidence"] == "complete"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_returned_indeterminate_accounting_failure_forces_incomplete_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="tool_mutation_indeterminate",
    )
    monkeypatch.setattr(
        "crucible.workflows.tier._record_completed_call",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected terminal accounting storage failure")
        ),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "RuntimeError"
    assert summary["accounting_error_kind"] == "RuntimeError"
    assert summary["accounting_evidence"] == "incomplete"
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_raised_indeterminate_prepare_failure_preserves_exact_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        MutationTransaction,
    )

    backend = _RaisedMutationIndeterminateBackend(close_settles=True)
    real_prepare = MutationTransaction.prepare_git_capture
    prepare_calls = 0

    def fail_prepare(self) -> None:
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls == 1:
            raise MutationJournalError("injected TMI prepare failure")
        real_prepare(self)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        fail_prepare,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert backend.close_calls == 1
    assert prepare_calls >= 2
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "MutationJournalError"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["accounting_evidence"] == "incomplete"
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_returned_indeterminate_validation_failure_preserves_exact_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        MutationTransaction,
    )

    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="tool_mutation_indeterminate",
    )

    def fail_validation(self, _candidate):
        raise MutationJournalError("injected TMI validation failure")

    monkeypatch.setattr(
        MutationTransaction,
        "validate_implementation",
        fail_validation,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "hard_fail_tool_mutation_indeterminate"
    assert summary["secondary_error_kind"] == "MutationJournalError"
    assert summary["accounting_evidence"] == "complete"
    assert summary["costs"]["accounting_debt"] is False
    assert repo.joinpath("target.py").read_text() == "value = inherited\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_raised_indeterminate_rollback_failure_requires_explicit_retry(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(
        base,
        files_in_scope=("target.py", "failed_only.py"),
    ))
    backend = _RaisedMutationIndeterminateBackend(close_settles=True)
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._mutation_journal import MutationTransaction
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    restore_calls = 0
    real_restore = MutationTransaction.restore

    def fail_first_restore(self, *args, **kwargs):
        nonlocal restore_calls
        restore_calls += 1
        if restore_calls == 1:
            return False
        return real_restore(self, *args, **kwargs)

    monkeypatch.setattr(
        MutationTransaction,
        "restore",
        fail_first_restore,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=corpus_dir,
            tier_id="t0",
            cwd=str(repo),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )

    assert caught.value is backend.terminal_error
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is caught.value
    assert len(backend.invocations) == 1
    assert backend.close_calls == 1
    assert restore_calls == 1
    assert repo.joinpath("target.py").read_text() == (
        "value = indeterminate\n"
    )
    assert repo.joinpath("failed_only.py").exists()
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["provider_cleanup_pending"] is False
    assert summary["mutation_cleanup_pending"] is True
    assert summary["journal_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True
    partial = summary["partial_summary"]
    assert partial["status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert cleanup_owner.owner.cleanup_complete is False

    cleanup_owner.retry_cleanup()
    # One explicit owner retry enters the public restore operation once, then
    # the journal reacquires workspace authority by recursively re-entering
    # the same method.  The pre-retry assertion above proves no implicit retry.
    assert restore_calls >= 2
    assert cleanup_owner.owner.cleanup_complete is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_indeterminate_secondary_then_rollback_failure_retains_live_owner(
    tmp_path,
    monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_spec(
        base,
        files_in_scope=("target.py", "failed_only.py"),
    ))
    backend = _RaisedMutationIndeterminateBackend(close_settles=True)
    backend.repo = repo
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )

    from crucible.tools import ToolMutationIndeterminate
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._mutation_journal import (
        MutationJournalError,
        MutationTransaction,
    )
    from crucible.workflows._provider_settlement import (
        ProviderSettlementCleanupError,
    )

    real_prepare = MutationTransaction.prepare_git_capture
    real_restore = MutationTransaction.restore
    prepare_failed = False
    restore_failed = False
    restore_entries = 0

    def fail_first_prepare(self) -> None:
        nonlocal prepare_failed
        if not prepare_failed:
            prepare_failed = True
            raise MutationJournalError("injected TMI capture failure")
        real_prepare(self)

    def fail_first_restore(self, *args, **kwargs):
        nonlocal restore_failed, restore_entries
        restore_entries += 1
        if not restore_failed:
            restore_failed = True
            return False
        return real_restore(self, *args, **kwargs)

    monkeypatch.setattr(
        MutationTransaction,
        "prepare_git_capture",
        fail_first_prepare,
    )
    monkeypatch.setattr(
        MutationTransaction,
        "restore",
        fail_first_restore,
    )

    with pytest.raises(ToolMutationIndeterminate) as caught:
        tier_workflow.run_tier(
            spec=corpus_dir,
            tier_id="t0",
            cwd=str(repo),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )

    assert caught.value is backend.terminal_error
    cleanup_owner = caught.value.__cause__
    assert isinstance(cleanup_owner, ProviderSettlementCleanupError)
    assert cleanup_owner.primary_error is caught.value
    assert prepare_failed is True
    assert restore_entries == 1
    assert len(backend.invocations) == 1
    assert repo.joinpath("target.py").read_text() == (
        "value = indeterminate\n"
    )
    assert repo.joinpath("failed_only.py").exists()
    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "incomplete_accounting"
    assert summary["provider_cleanup_pending"] is False
    assert summary["mutation_cleanup_pending"] is True
    assert summary["journal_cleanup_pending"] is True
    assert summary["workspace_cleanup_pending"] is True
    partial = summary["partial_summary"]
    assert partial["status"] == (
        "hard_fail_tool_mutation_indeterminate"
    )
    assert partial["pre_cleanup_secondary_error_kind"] == (
        "MutationJournalError"
    )
    assert cleanup_owner.owner.cleanup_complete is False

    cleanup_owner.retry_cleanup()
    assert restore_entries >= 2
    assert cleanup_owner.owner.cleanup_complete is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("failed_only.py").exists()


def test_incomplete_terminal_recovery_timeout_restores_retry_mutations(
    tmp_path,
    monkeypatch,
):
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="max_turns",
        recovery_terminal_kind="timeout",
    )
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert backend.pre_call_states == [
        ("value = inherited\n", False),
        ("value = inherited\n", False),
    ]
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()
    assert breach_called == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_timeout"
    assert phase["incomplete_terminal_recovery"]["retry_attempted"] is True

    corpus = Corpus(tmp_path / "corpus")
    from crucible.workflows.tier import attest_mutation_journal_settlement

    settlement = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="recovery-timeout-rollback-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []


def test_incomplete_accounting_restores_partial_without_paid_retry_or_breach(
    tmp_path,
    monkeypatch,
):
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="transport",
        accounting_complete=False,
    )
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()
    assert breach_called == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_transport"
    assert phase["incomplete_terminal_recovery"]["retry_attempted"] is False
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["accounting_debt_by_role"] == {
        "implementer": ["cost_complete", "token_usage_complete"],
    }

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    assert spec is not None
    from assay import Implementation

    implementation = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.accounting_debt == (
        "token_usage_complete",
        "cost_complete",
    )
    assert "incomplete_terminal_failed_diff" in implementation.artifacts
    from crucible.workflows.tier import attest_mutation_journal_settlement

    settlement = attest_mutation_journal_settlement(
        corpus=corpus,
        cwd=repo,
        tier_id="t0",
        operation_id="incomplete-accounting-recovery-test",
    )
    assert settlement["settled"] is True
    assert settlement["registry_clear"] is True
    assert settlement["remaining_journal_manifests"] == []


@pytest.mark.parametrize(
    ("error_kind", "expected_status"),
    [
        ("budget_exhausted", "slice_failed"),
        # A cancelled implementer was not judged. Its tier halts all the same.
        ("cancellation", "cancelled"),
        ("content_rejected", "slice_failed"),
        ("incomplete_evidence", "slice_failed"),
        ("stale_conversation", "slice_failed"),
        ("timeout", "slice_failed"),
    ],
)
def test_backend_terminal_partial_write_never_becomes_breach_seed(
    tmp_path,
    monkeypatch,
    error_kind,
    expected_status,
):
    backend = _NonAuthorizingTerminalBackend(error_kind=error_kind)
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert summary["status"] == expected_status
    assert summary["rollback_applied"] is True
    assert len(backend.invocations) == 1
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == f"hard_fail_{error_kind}"
    assert phase["incomplete_terminal_recovery"] == {
        "classification": f"hard_fail_{error_kind}",
        "restored_paths": ["target.py", "failed_only.py"],
        "restored_to_pre_call_snapshot": True,
        "retry_attempted": False,
    }

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    assert spec is not None
    from assay import Implementation

    implementation = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert implementation.classification == f"hard_fail_{error_kind}"
    assert implementation.accounting_debt == ()
    assert "incomplete_terminal_failed_diff" in implementation.artifacts


@pytest.mark.parametrize("fail_first_recovery", [False, True])
def test_declared_create_recovery_requires_exact_outstanding_path(
    tmp_path,
    monkeypatch,
    fail_first_recovery,
):
    backend = _ScopedDeclaredCreateBackend(
        fail_first_recovery=fail_first_recovery,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        creates_paths=(
            "repo/generated_one.py",
            "repo/generated_two.py",
        ),
    )

    assert rc == 0
    assert backend.dispatch_results[0].startswith("OK:")
    if fail_first_recovery:
        assert len(backend.invocations) == 4
        assert backend.dispatch_results[1].startswith(
            "ERROR: unknown tool 'apply_patch'"
        )
        assert "attempt 2 of 2" in backend.invocations[2].user
        expected_contents = (
            "CREATED_2 = True\n",
            "CREATED_3 = True\n",
        )
    else:
        assert len(backend.invocations) == 3
        expected_contents = (
            "CREATED_1 = True\n",
            "CREATED_2 = True\n",
        )
    assert len(backend.recovery_toolsets) == len(backend.invocations) - 1
    assert all(
        toolset.tool_names == {
            "read_source",
            "grep_symbol",
            "find_callers",
            "find_definition",
            "write_source",
        }
        for toolset in backend.recovery_toolsets
    )
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = partial\n"
    )
    assert repo.joinpath("generated_one.py").read_text(encoding="utf-8") == (
        expected_contents[0]
    )
    assert repo.joinpath("generated_two.py").read_text(encoding="utf-8") == (
        expected_contents[1]
    )
    assert not repo.joinpath("component.py").exists()
    assert not repo.joinpath("provider.py").exists()
    # Each fresh recovery generation is constructed directly with its exact
    # singleton writer profile. The closed generation is never reattached to
    # a broader surface after settlement.
    assert len(backend.attach_history) == 1 + len(backend.recovery_toolsets)
    assert backend.attach_history[-1].tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
        "write_source",
    }
    assert backend.toolset is backend.attach_history[-1]
    assert summary["phase_1_slices"][0]["classification"] == "success"
    assert summary["phase_1_slices"][0]["missing_create_paths"] == []


def test_declared_create_checkpoint_restores_direct_escape_before_correction(
    tmp_path,
    monkeypatch,
):
    backend = _DirectRecoveryEscapeThenCreateBackend()

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        creates_paths=("repo/generated_one.py",),
    )

    assert rc == 0
    assert len(backend.invocations) == 3
    assert summary["phase_1_slices"][0]["classification"] == "success"
    assert summary["phase_1_slices"][0]["missing_create_paths"] == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = partial\n"
    )
    assert repo.joinpath("generated_one.py").read_text(encoding="utf-8") == (
        "ONE = 1\n"
    )


def test_declared_create_does_not_reattach_closed_generation_when_backend_raises(
    tmp_path,
    monkeypatch,
):
    backend = _RaisingRecoveryBackend()

    with pytest.raises(
        RuntimeError,
        match="provider exploded during exact create",
    ):
        _run(
            tmp_path,
            monkeypatch,
            backend,
            creates_paths=("repo/generated_one.py",),
        )

    assert len(backend.attach_history) == 2
    assert backend.attach_history[1].tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
        "write_source",
    }
    assert backend.attach_history[-1].tool_names == {
        "read_source",
        "grep_symbol",
        "find_callers",
        "find_definition",
        "write_source",
    }
    assert backend.toolset is backend.attach_history[-1]
    assert tmp_path.joinpath("repo/target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )


def test_missing_declared_creates_exhaustion_rolls_back_before_breach(
    tmp_path,
    monkeypatch,
):
    backend = _MissingDeclaredCreatesBackend(
        tmp_path,
        complete_on_retry=False,
    )
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        creates_paths=(
            "repo/generated_one.py",
            "repo/generated_two.py",
        ),
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert "INCOMPLETE-TERMINAL RECOVERY" in backend.invocations[1].user
    assert summary["status"] == "slice_failed"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_missing_create_paths"
    )
    assert summary["phase_1_slices"][0]["missing_create_paths"] == [
        "generated_one.py",
        "generated_two.py",
    ]
    assert breach_called == []
    assert not repo.joinpath("generated_one.py").exists()
    assert not repo.joinpath("generated_two.py").exists()
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )


def test_declared_test_mutation_gets_one_focused_alternative_recovery(
    tmp_path,
    monkeypatch,
):
    backend = _DeclaredTestMutationBackend(
        recovery_test_path="tests/test_alternative.py",
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=(
            "target.py",
            "tests/test_primary.py",
            "tests/test_alternative.py",
        ),
        committed_files={
            "tests/test_primary.py": "def test_primary():\n    assert True\n",
            "tests/test_alternative.py": (
                "def test_alternative():\n    assert True\n"
            ),
        },
        # Pre-existing dirt in another declared test must not count as this
        # slice's local test evidence.
        inherited_mutations={
            "tests/test_primary.py": (
                "def test_inherited():\n    assert True\n"
            ),
        },
    )

    assert rc == 0
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 8
    assert "DECLARED TEST-MUTATION RECOVERY" in backend.invocations[1].user
    assert "at least ONE" in backend.invocations[1].user
    assert "- tests/test_primary.py" in backend.invocations[1].user
    assert "- tests/test_alternative.py" in backend.invocations[1].user
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "success"
    assert phase["declared_test_paths"] == [
        "tests/test_primary.py",
        "tests/test_alternative.py",
    ]
    assert phase["locally_mutated_test_paths"] == [
        "tests/test_alternative.py",
    ]
    assert repo.joinpath("tests/test_primary.py").read_text() == (
        "def test_inherited():\n    assert True\n"
    )


def test_one_initial_declared_test_mutation_satisfies_alternatives(
    tmp_path,
    monkeypatch,
):
    backend = _DeclaredTestMutationBackend(
        production_path="lib/browser_test_runner.js",
        initial_test_path="tests/test_alternative.py",
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=(
            "lib/browser_test_runner.js",
            "tests/test_primary.py",
            "tests/test_alternative.py",
        ),
        committed_files={
            "lib/browser_test_runner.js": "module.exports = {};\n",
            "tests/test_primary.py": "def test_primary():\n    assert True\n",
            "tests/test_alternative.py": (
                "def test_alternative():\n    assert True\n"
            ),
        },
    )

    assert rc == 0
    assert len(backend.invocations) == 1
    phase = summary["phase_1_slices"][0]
    # A production filename containing "test" is not preferred over paths
    # structurally located under tests/.
    assert phase["declared_test_paths"] == [
        "tests/test_primary.py",
        "tests/test_alternative.py",
    ]
    assert phase["locally_mutated_test_paths"] == [
        "tests/test_alternative.py",
    ]


def test_declared_test_mutation_exhaustion_rolls_back_before_breach(
    tmp_path,
    monkeypatch,
):
    backend = _DeclaredTestMutationBackend(complete_pricing=True)
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert summary["status"] == "slice_failed"
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert phase["declared_test_paths"] == ["tests/test_target.py"]
    assert phase["locally_mutated_test_paths"] == []
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert repo.joinpath("tests/test_target.py").read_text() == (
        "def test_target():\n    assert True\n"
    )
    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert "declared_test_evidence_pending" not in retained.artifacts


@pytest.mark.parametrize(
    ("gate_classification", "expected_disposition"),
    (
        ("success", "passed"),
        ("verify_failed", "failed"),
        ("runner_missing", "unavailable"),
    ),
)
def test_missing_declared_test_runs_one_non_authorizing_diagnostic_gate(
    tmp_path,
    monkeypatch,
    gate_classification,
    expected_disposition,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend(complete_pricing=True)
    gate_calls = []
    breach_called = []

    def diagnostic_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        assert tmp_path.joinpath("repo/target.py").read_text() == (
            "value = production\n"
        )
        if gate_classification == "success":
            return _successful_gate_outcome()
        return _failed_gate_outcome(gate_classification)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(diagnostic_gate),
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        inherited_mutations={
            "tests/test_target.py": "def test_inherited():\n    assert True\n",
        },
        skip_build_verify=False,
        build_retries=2,
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert gate_calls == ["s0"]
    assert breach_called == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert phase["build_verify"] == gate_classification
    evidence_ref = phase["declared_test_evidence_pending"]
    assert summary["rollback_applied"] is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert repo.joinpath("tests/test_target.py").read_text() == (
        "def test_inherited():\n    assert True\n"
    )

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    retained = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.classification == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert str(retained.artifacts["declared_test_evidence_pending"]) == (
        evidence_ref
    )
    assert "tier_verification_evidence" not in retained.artifacts
    evidence = tier_workflow._gate_evidence.load_verification_evidence(
        corpus,
        evidence_ref,
    )
    assert evidence.classification == gate_classification
    assert evidence.disposition == expected_disposition
    assert evidence.candidate_diff_sha256 == str(retained.diff_ref)
    assert corpus.get_artifact(retained.diff_ref)
    tier = spec.tier_dag.tiers[0]
    landed, pending = tier_workflow._partition.partition_for_resume(
        corpus,
        tier,
        spec=spec,
    )
    assert landed == {}
    assert tuple(item.id for item in pending) == ("s0",)


@pytest.mark.parametrize("mutation_path", ("target.py", "rogue.py"))
def test_declared_test_diagnostic_gate_mutation_restores_both_transactions(
    tmp_path,
    monkeypatch,
    mutation_path,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend(complete_pricing=True)
    gate_calls = []

    def mutating_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        tmp_path.joinpath("repo", mutation_path).write_text(
            "value = gate mutation\n",
            encoding="utf-8",
        )
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(mutating_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        skip_build_verify=False,
    )

    assert rc == 1
    assert gate_calls == ["s0"]
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert phase["build_verify"] == "hard_fail_gate_mutation"
    assert summary["rollback_applied"] is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("rogue.py").exists()
    corpus = Corpus(tmp_path / "corpus")
    evidence = tier_workflow._gate_evidence.load_verification_evidence(
        corpus,
        phase["declared_test_evidence_pending"],
    )
    assert evidence.classification == "hard_fail_gate_mutation"
    assert evidence.disposition == "failed"


def test_declared_test_diagnostic_requires_complete_provider_pricing(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend()
    gate_calls = []

    def forbidden_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        raise AssertionError("an unpriced candidate cannot run diagnostics")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(forbidden_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        skip_build_verify=False,
    )

    assert rc == 1
    assert gate_calls == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert phase["build_verify"] is None
    assert phase["declared_test_evidence_pending"] is None
    assert phase["declared_test_diagnostic"] is None
    assert summary["rollback_applied"] is True
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert "declared_test_evidence_pending" not in retained.artifacts


def test_declared_test_diagnostic_does_not_run_after_budget_overrun(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend(
        complete_pricing=True,
        cost_usd=0.6,
    )
    gate_calls = []

    def forbidden_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        raise AssertionError("an over-budget candidate cannot run diagnostics")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(forbidden_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        skip_build_verify=False,
        budget_usd=1.0,
    )

    assert rc == 1
    assert gate_calls == []
    assert summary["status"] == "budget_exhausted"
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert phase["build_verify"] is None
    assert phase["declared_test_evidence_pending"] is None
    assert phase["declared_test_diagnostic"] is None
    assert repo.joinpath("target.py").read_text() == "value = 1\n"


def test_declared_test_diagnostic_preserves_scope_escape_as_primary_failure(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend(
        scope_escape_path="rogue.py",
        complete_pricing=True,
        cost_usd=1.0,
    )
    gate_calls = []

    def forbidden_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        raise AssertionError("a scope-escaped candidate cannot run diagnostics")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(forbidden_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        skip_build_verify=False,
        budget_usd=1.0,
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert gate_calls == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_scope_escape"
    assert phase["build_verify"] is None
    assert phase["declared_test_evidence_pending"] is None
    assert phase["declared_test_diagnostic"] is None
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert not repo.joinpath("rogue.py").exists()

    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.classification == "hard_fail_scope_escape"
    assert retained.scope_escapes == ("rogue.py",)
    assert "declared_test_evidence_pending" not in retained.artifacts


def test_declared_test_diagnostic_rejects_out_of_region_candidate(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _DeclaredTestMutationBackend(complete_pricing=True)
    gate_calls = []
    original_target = (
        "# OPTIM-START\n"
        "inside = 1\n"
        "# OPTIM-END\n"
        "outside = 1\n"
    )

    def forbidden_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        raise AssertionError("an out-of-region candidate cannot run diagnostics")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(forbidden_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "tests/test_target.py"),
        committed_files={
            "target.py": original_target,
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
        mutation_regions=(
            MutationRegion(
                file="target.py",
                start_marker="OPTIM-START",
                end_marker="OPTIM-END",
            ),
        ),
        skip_build_verify=False,
        skip_mutation_scope=False,
    )

    assert rc == 1
    assert gate_calls == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_mutation_scope"
    assert phase["mutation_scope_violations"] > 0
    assert phase["declared_test_evidence_pending"] is None
    assert phase["declared_test_diagnostic"] is None
    assert summary["rollback_applied"] is True
    assert repo.joinpath("target.py").read_text() == original_target


def test_declared_test_diagnostic_halts_before_sibling_dispatch(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    repo.joinpath("tests").mkdir()
    repo.joinpath("tests/test_target.py").write_text(
        "def test_target():\n    assert True\n",
        encoding="utf-8",
    )
    repo.joinpath("sibling.py").write_text(
        "sibling = 1\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "tests/test_target.py", "sibling.py"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "add diagnostic fixtures"],
        cwd=repo,
        check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    Corpus(corpus_dir).put_spec(
        _two_slice_spec(
            base,
            first_files_in_scope=(
                "target.py",
                "tests/test_target.py",
            ),
        )
    )

    backend = _DeclaredTestMutationBackend(complete_pricing=True)
    backend.repo = repo
    gate_calls = []

    def diagnostic_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(diagnostic_gate),
    )

    rc = tier_workflow.run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        breach_policy=BreachPolicy(enabled=False),
    )

    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads(run_dir.joinpath("summary.json").read_text())
    assert rc == 1
    assert len(backend.invocations) == 2
    assert gate_calls == ["s0"]
    assert [row["slice_id"] for row in summary["phase_1_slices"]] == ["s0"]
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_missing_declared_test_mutation"
    )
    assert repo.joinpath("target.py").read_text() == "value = 1\n"
    assert repo.joinpath("sibling.py").read_text() == "sibling = 1\n"


@pytest.mark.parametrize(
    ("verify_mode", "files_in_scope", "expected_test_paths"),
    [
        ("tests", ("target.py",), []),
        ("skip", ("target.py", "tests/test_target.py"), []),
    ],
)
def test_declared_test_mutation_gate_ignores_inapplicable_directives(
    tmp_path,
    monkeypatch,
    verify_mode,
    files_in_scope,
    expected_test_paths,
):
    backend = _DeclaredTestMutationBackend()

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        verify_mode=verify_mode,
        files_in_scope=files_in_scope,
        committed_files={
            "tests/test_target.py": "def test_target():\n    assert True\n",
        },
    )

    assert rc == 0
    assert len(backend.invocations) == 1
    assert summary["phase_1_slices"][0]["declared_test_paths"] == (
        expected_test_paths
    )


def test_empty_retry_exhaustion_halts_before_breach(tmp_path, monkeypatch):
    backend = _AlwaysEmptyBackend()
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert summary["status"] == "slice_failed"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_empty_diff"
    )
    assert "empty_seed_deferred_to_breach" not in summary
    assert breach_called == []
    assert (_repo / "target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )


def test_implementer_blocked_empty_attempt_is_terminal_and_durably_attributed(
    tmp_path,
    monkeypatch,
):
    backend = _ImplementerBlockedBackend()

    rc, summary, repo = _run(tmp_path, monkeypatch, backend)

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "slice_failed"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    assert summary["costs"]["by_role"]["implementer"]["calls"] == 1
    assert summary["costs"]["total_input_tokens"] == 13
    assert summary["costs"]["total_output_tokens"] == 4
    assert summary["costs"]["total_usd"] == 0.25
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )

    corpus = Corpus(tmp_path / "corpus")
    retained_refs = corpus.latest_spec().implementation_refs
    assert len(retained_refs) == 1
    retained = corpus.get_record(retained_refs[0], as_type=Implementation)
    assert retained.classification == "hard_fail_implementer_blocked"
    assert retained.input_tokens == 13
    assert retained.output_tokens == 4
    assert retained.cost_usd == 0.25
    disposition = json.loads(corpus.get_artifact(
        retained.artifacts["implementer_terminal_disposition"],
    ))
    assert disposition == {
        "schema_version": "crucible.implementer-terminal.v1",
        "slice_id": "s0",
        "disposition": "blocked",
        "observed_classification": "hard_fail_empty_diff",
        "retained_classification": "hard_fail_implementer_blocked",
    }


def test_implementer_blocked_partial_write_is_rolled_back_without_breach(
    tmp_path,
    monkeypatch,
):
    backend = _ImplementerBlockedBackend(mutate=True)
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )


def test_implementer_blocked_stops_before_sibling_slice_dispatch(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows.tier import run_tier

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    repo.joinpath("sibling.py").write_text(
        "sibling = 1\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "sibling.py"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "add sibling"],
        cwd=repo,
        check=True,
    )
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    Corpus(corpus_dir).put_spec(_two_slice_spec(base))

    backend = _ImplementerBlockedBackend(repo=repo, mutate=True)
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda *args, **kwargs: backend,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc = run_tier(
        spec=corpus_dir,
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        build_retries=2,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    run_dir = next(
        path for path in (corpus_dir / "runs").iterdir() if "tier" in path.name
    )
    summary = json.loads(run_dir.joinpath("summary.json").read_text())
    assert rc == 1
    assert len(backend.invocations) == 1
    assert [row["slice_id"] for row in summary["phase_1_slices"]] == ["s0"]
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert repo.joinpath("sibling.py").read_text(encoding="utf-8") == (
        "sibling = 1\n"
    )


def test_empty_recovery_blocked_stops_after_the_second_call(
    tmp_path,
    monkeypatch,
):
    backend = _EmptyThenBlockedBackend()

    rc, summary, repo = _run(tmp_path, monkeypatch, backend)

    assert rc == 1
    assert len(backend.invocations) == 2
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )


@pytest.mark.parametrize(
    "message",
    [
        "BLOCKED",
        "BLOCKED concrete blocker",
        "BLOCKED: concrete blocker",
        "BLOCKED. Concrete blocker",
        "BLOCKED, concrete blocker",
        "BLOCKED; concrete blocker",
        "BLOCKED-concrete blocker",
        "BLOCKED–concrete blocker",
        "BLOCKED — concrete blocker",
    ],
)
def test_implementer_blocked_protocol_accepts_only_token_boundaries(message):
    from crucible.workflows import tier as tier_workflow

    payload = json.dumps([
        {"role": "assistant", "content": message},
    ]).encode("utf-8")
    corpus = SimpleNamespace(get_artifact=lambda _ref: payload)
    impl = SimpleNamespace(transcript_ref="sha256:transcript")

    assert tier_workflow._implementer_completion_disposition(
        corpus,
        impl,
    ) == tier_workflow._IMPLEMENTER_COMPLETION_BLOCKED


@pytest.mark.parametrize(
    "payload,expected",
    [
        (
            [
                {"role": "assistant", "content": "BLOCKED earlier thought"},
                {"role": "assistant", "content": "DONE final result"},
            ],
            "clear",
        ),
        (
            [
                {"role": "user", "content": "BLOCKED user text"},
                {"role": "assistant", "content": "DONE final result"},
            ],
            "clear",
        ),
        (
            [
                {"role": "assistant", "content": "BLOCKED earlier thought"},
                {"role": "tool", "content": "tool terminal"},
            ],
            "clear",
        ),
        ([{"role": "assistant", "content": "BLOCKEDNESS is a word"}], "clear"),
        ([{"role": "assistant", "content": "UNBLOCKED now"}], "clear"),
        ([{"role": "assistant", "content": "blocked lowercase"}], "clear"),
        ([{"role": "assistant", "content": "`BLOCKED` quoted"}], "clear"),
        ([{"role": "assistant", "content": 7}], "incomplete"),
        ([{"role": "user"}], "incomplete"),
        ([{"role": "user", "content": 7}], "incomplete"),
        ([{"role": "system"}], "incomplete"),
        ([{"role": "system", "content": 7}], "incomplete"),
        ([{"role": "tool"}], "incomplete"),
        ([{"role": "tool", "content": 7}], "incomplete"),
        ([{}], "incomplete"),
        ([{"role": 7, "content": "BLOCKED"}], "incomplete"),
        ([{"role": "bogus", "content": "BLOCKED"}], "incomplete"),
        ([7], "incomplete"),
        ({"role": "assistant", "content": "BLOCKED"}, "incomplete"),
    ],
)
def test_implementer_blocked_protocol_rejects_non_authoritative_shapes(
    payload,
    expected,
):
    from crucible.workflows import tier as tier_workflow

    corpus = SimpleNamespace(
        get_artifact=lambda _ref: json.dumps(payload).encode("utf-8"),
    )
    impl = SimpleNamespace(transcript_ref="sha256:transcript")

    assert tier_workflow._implementer_completion_disposition(
        corpus,
        impl,
    ) == expected


@pytest.mark.parametrize(
    "get_artifact",
    [
        lambda _ref: b"{",
        lambda _ref: (_ for _ in ()).throw(KeyError("missing")),
    ],
)
def test_implementer_completion_evidence_failure_is_incomplete(get_artifact):
    from crucible.workflows import tier as tier_workflow

    corpus = SimpleNamespace(get_artifact=get_artifact)
    impl = SimpleNamespace(transcript_ref="sha256:transcript")

    assert tier_workflow._implementer_completion_disposition(
        corpus,
        impl,
    ) == tier_workflow._IMPLEMENTER_COMPLETION_INCOMPLETE


def test_incomplete_completion_evidence_suppresses_paid_empty_retry(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _AlwaysEmptyBackend()
    monkeypatch.setattr(
        "crucible.workflows._implementer_completion."
        "implementer_completion_disposition",
        lambda *_args, **_kwargs: tier_workflow._IMPLEMENTER_COMPLETION_INCOMPLETE,
    )

    rc, summary, _repo = _run(tmp_path, monkeypatch, backend)

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_incomplete_evidence"
    )


def test_implementer_blocked_scope_escape_is_terminal_and_fully_rolled_back(
    tmp_path,
    monkeypatch,
):
    backend = _ImplementerBlockedBackend(mutate=True, scope_escape=True)

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        build_retries=2,
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_scope_escape"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert not repo.joinpath("rogue.py").exists()
    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.scope_escapes == ("rogue.py",)


def test_scope_escape_retry_blocked_stops_and_restores_exact_baseline(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _ScopeEscapeThenBlockedBackend()
    breach_called = []

    def unexpected_gate(*args, **kwargs):
        pytest.fail("BLOCKED scope retry reached build/verify")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        unexpected_gate,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        build_retries=2,
        skip_build_verify=False,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert "SCOPE-ESCAPE RETRY" in backend.invocations[1].user
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_implementer_blocked"
    assert phase["implementer_terminal_disposition"] == "blocked"
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert not repo.joinpath("rogue.py").exists()

    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.classification == "hard_fail_implementer_blocked"
    terminal = json.loads(corpus.get_artifact(
        retained.artifacts["implementer_terminal_disposition"],
    ))
    assert terminal["disposition"] == "blocked"


def test_incomplete_terminal_retry_blocked_stops_without_gate_or_breach(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind="max_turns",
        recovery_terminal_kind="end_turn",
        blocked_on_attempt=2,
    )
    breach_called = []

    def unexpected_gate(*args, **kwargs):
        pytest.fail("BLOCKED incomplete-terminal retry reached build/verify")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        unexpected_gate,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow."
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        files_in_scope=("target.py", "failed_only.py"),
        skip_build_verify=False,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert backend.invocations[1].max_turns == 24
    assert "INCOMPLETE-TERMINAL RECOVERY" in backend.invocations[1].user
    assert backend.pre_call_states == [
        ("value = inherited\n", False),
        ("value = inherited\n", False),
    ]
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "hard_fail_implementer_blocked"
    assert phase["implementer_terminal_disposition"] == "blocked"
    recovery = phase["incomplete_terminal_recovery"]
    assert recovery["classification"] == "hard_fail_max_turns"
    assert set(recovery["restored_paths"]) == {"failed_only.py", "target.py"}
    assert recovery["restored_to_pre_call_snapshot"] is True
    assert recovery["retry_attempted"] is True
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
    assert not repo.joinpath("failed_only.py").exists()

    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    assert retained.classification == "hard_fail_implementer_blocked"
    terminal = json.loads(corpus.get_artifact(
        retained.artifacts["implementer_terminal_disposition"],
    ))
    assert terminal["disposition"] == "blocked"


def test_incomplete_accounting_takes_precedence_over_blocked_protocol(
    tmp_path,
    monkeypatch,
):
    backend = _ImplementerBlockedBackend(
        completeness=_ACCOUNTING_DEBT_EXECUTION,
    )

    rc, summary, _repo = _run(tmp_path, monkeypatch, backend)

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "incomplete_accounting"
    assert summary["phase_1_slices"][0]["classification"] != (
        "hard_fail_implementer_blocked"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    corpus = Corpus(tmp_path / "corpus")
    retained = corpus.get_record(
        corpus.latest_spec().implementation_refs[-1],
        as_type=Implementation,
    )
    terminal_ref = retained.artifacts["implementer_terminal_disposition"]
    terminal = json.loads(corpus.get_artifact(terminal_ref))
    assert terminal["disposition"] == "blocked"
    assert terminal["retained_classification"] == (
        "hard_fail_incomplete_evidence"
    )


def test_budget_overrun_takes_precedence_over_blocked_protocol(
    tmp_path,
    monkeypatch,
):
    backend = _ImplementerBlockedBackend()

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        budget_usd=0.1,
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "budget_exhausted"
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"


@pytest.mark.parametrize("terminal_kind", ["max_turns", "transport"])
def test_provider_terminal_classification_precedes_blocked_without_recovery(
    tmp_path,
    monkeypatch,
    terminal_kind,
):
    backend = _IncompleteTerminalTwiceBackend(
        terminal_kind=terminal_kind,
        blocked_report=True,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py", "failed_only.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["phase_1_slices"][0]["classification"] == (
        f"hard_fail_{terminal_kind}"
    )
    assert summary["phase_1_slices"][0][
        "implementer_terminal_disposition"
    ] == "blocked"
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert not repo.joinpath("failed_only.py").exists()


def test_declared_create_recovery_blocked_stops_remaining_paths(
    tmp_path,
    monkeypatch,
):
    backend = _WriteThenBlockedBackend()

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        files_in_scope=("target.py",),
        creates_paths=("generated_one.py", "generated_two.py"),
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert not repo.joinpath("generated_one.py").exists()
    assert not repo.joinpath("generated_two.py").exists()
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )


def test_declared_test_recovery_blocked_is_not_overwritten(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _WriteThenBlockedBackend()
    gate_calls = []

    def forbidden_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        raise AssertionError("BLOCKED cannot authorize a diagnostic gate")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(forbidden_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        committed_files={"tests/test_target.py": "def test_seed():\n    pass\n"},
        files_in_scope=("target.py", "tests/test_target.py"),
        verify_mode="tests",
        skip_build_verify=False,
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert gate_calls == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert phase["declared_test_evidence_pending"] is None
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )
    assert repo.joinpath("tests/test_target.py").read_text(encoding="utf-8") == (
        "def test_seed():\n    pass\n"
    )


def test_build_retry_blocked_is_not_discarded_or_retried(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    backend = _WriteThenBlockedBackend()
    gate_calls = []

    def failing_gate(*args, **kwargs):
        gate_calls.append(kwargs["slice_id"])
        return _failed_gate_outcome("verify_failed")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(failing_gate),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        skip_build_verify=False,
        build_retries=2,
    )

    assert rc == 1
    assert len(backend.invocations) == 2
    assert gate_calls == ["s0"]
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_implementer_blocked"
    )
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = 1\n"
    )


def test_expected_empty_guard_tier_skips_diff_grounded_breach(
    tmp_path, monkeypatch,
):
    backend = _AlwaysEmptyBackend()
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 0
    assert len(backend.invocations) == 0
    assert "implementer" not in summary["costs"]["by_role"]
    assert summary["phase_3_breach"]["status"] == ("skipped_empty_union_diff")
    assert breach_called == []


def test_verification_only_tier_never_resolves_implementer_route(
    tmp_path,
    monkeypatch,
):
    def reject_resolution(*args, **kwargs):
        pytest.fail("verification-only tier must not resolve implementer")

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        backend_resolver=reject_resolution,
    )

    assert rc == 0
    assert summary["phase_1_slices"][0]["classification"] == ("soft_fail_no_changes")
    assert "implementer" not in summary["costs"]["by_role"]

    from crucible.workflows.tier import run_tier

    assert (
        run_tier(
            spec=tmp_path / "corpus",
            tier_id="t0",
            cwd=str(tmp_path / "repo"),
            max_turns=24,
            skip_build_verify=True,
            skip_smoke_check=True,
            skip_mutation_scope=True,
            skip_smoke_bench=True,
            skip_review=True,
        )
        == 0
    )
    run_dirs = sorted(
        (
            path
            for path in (tmp_path / "corpus" / "runs").iterdir()
            if "tier" in path.name
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    resumed = json.loads(
        run_dirs[-1].joinpath("summary.json").read_text(encoding="utf-8")
    )
    assert resumed["resumed_slices"] == ["s0"]
    assert resumed["phase_1_slices"][0]["resumed"] is True


def test_expected_empty_guard_rebinds_inherited_workspace_dirt(
    tmp_path,
    monkeypatch,
):
    backend = _AlwaysEmptyBackend()
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        inherited_dirty=True,
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 0
    assert len(backend.invocations) == 0
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "soft_fail_no_changes"
    assert phase["build_verify"] == "skipped_by_policy"
    assert phase["diff_ref"]
    assert summary["phase_2_union_diff_bytes"] == 0
    assert summary["phase_3_breach"]["status"] == "skipped_empty_union_diff"
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )


def test_verification_only_slice_runs_gate_before_review_evidence(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _AlwaysEmptyBackend()
    gate_calls = []

    def _passing_gate(*args, **kwargs):
        gate_calls.append((args, kwargs))
        return _successful_gate_outcome()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        _passing_gate,
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=False,
    )

    assert rc == 0
    assert len(backend.invocations) == 0
    assert len(gate_calls) == 2
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == "success"
    assert phase["build_verify"] == "success"

    artifact = tier_workflow._render_tier_completeness_artifact(
        _spec("0" * 40, no_surface=True).tier_dag.tiers[0],
        b"",
        verification_evidence=(phase,),
    ).decode("utf-8")
    assert "WORKFLOW-RECORDED VERIFICATION EVIDENCE" in artifact
    assert '"build_verify": "success"' in artifact


@pytest.mark.parametrize("classification", ["build_failed", "verify_failed"])
def test_verification_only_gate_failure_never_resolves_implementer(
    tmp_path,
    monkeypatch,
    classification,
):
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _AlwaysEmptyBackend()
    resolved_roles = []

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        lambda *args, **kwargs: BuildVerifyOutcome(
            classification,
            build=None,
            verify=None,
        ),
    )

    def resolve_backend(role, *args, **kwargs):
        resolved_roles.append(role)
        if role == "implementer":
            pytest.fail(
                "failed verification-only gate must not resolve Implementer"
            )
        return backend

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=False,
        backend_resolver=resolve_backend,
        build_retries=2,
    )

    assert rc != 0
    assert resolved_roles == []
    assert backend.invocations == []
    phase = summary["phase_1_slices"][0]
    assert phase["classification"] == classification
    assert phase["build_verify"] == classification
    assert "implementer" not in summary["costs"]["by_role"]


def test_verification_only_slice_passes_default_completeness_review(
    tmp_path,
    monkeypatch,
):
    from assay import ReviewCoverage, Verdict
    from crucible.common.policies import ReviewPolicy
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _AlwaysEmptyBackend()
    reviewed_artifacts: list[str] = []

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        lambda *args, **kwargs: _successful_gate_outcome(),
    )

    def passing_review(artifact, rubric, *, corpus, **kwargs):
        rendered = bytes(artifact)
        reviewed_artifacts.append(rendered.decode("utf-8"))
        artifact_ref = corpus.put_artifact(
            rendered,
            kind="reviewed_artifacts",
        )
        return Verdict(
            rubric=rubric.name,
            decision="pass",
            findings=(),
            artifact_ref=artifact_ref,
            artifact_commit=None,
            reasoning="Exact verification-only evidence is sufficient.",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(rendered.decode("utf-8")),
                artifact_chars_reviewed=len(rendered.decode("utf-8")),
            ),
            backend_name=backend.name,
            transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
            cost_usd=0.0,
            execution=None,
        )

    monkeypatch.setattr(tier_workflow, "review", passing_review)

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=False,
        skip_review=False,
        review_policy=ReviewPolicy(rubrics=("tier_completeness",)),
    )

    assert rc == 0
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert len(reviewed_artifacts) == 1
    artifact = reviewed_artifacts[0]
    assert "**verify_mode:** `tests`" in artifact
    assert "none (verification-only)" in artifact
    assert '"gate_status": "passed"' in artifact


def test_verification_only_review_retry_never_creates_write_authority(
    tmp_path,
    monkeypatch,
):
    from assay import ReviewCoverage, Verdict
    from crucible.common.policies import ReviewPolicy
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _AlwaysEmptyBackend()
    review_calls = []

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        lambda *args, **kwargs: _successful_gate_outcome(),
    )

    def revising_review(artifact, rubric, *, corpus, **kwargs):
        rendered = bytes(artifact)
        review_calls.append(rendered)
        return Verdict(
            rubric=rubric.name,
            decision="revise",
            findings=(),
            artifact_ref=corpus.put_artifact(
                rendered,
                kind="reviewed_artifacts",
            ),
            artifact_commit=None,
            reasoning="Fixture requests a bounded re-review.",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
                artifact_chars_total=len(rendered.decode("utf-8")),
                artifact_chars_reviewed=len(rendered.decode("utf-8")),
            ),
            backend_name=backend.name,
            transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
            cost_usd=0.0,
            execution=None,
        )

    def resolve_backend(role, *args, **kwargs):
        if role == "implementer":
            pytest.fail("review retry must not grant verification-only writes")
        return backend

    monkeypatch.setattr(tier_workflow, "review", revising_review)

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=False,
        skip_review=False,
        backend_resolver=resolve_backend,
        review_policy=ReviewPolicy(
            rubrics=("tier_completeness",),
            retry_budget=1,
        ),
    )

    assert rc != 0
    assert len(review_calls) == 2
    assert len(backend.invocations) == 0
    assert "implementer" not in summary["costs"]["by_role"]


@pytest.mark.parametrize(
    ("verification_first", "expected_gate_slices"),
    [
        (False, ["mutate", "verify_only", "verify_only"]),
        (
            True,
            ["verify_only", "mutate", "verify_only", "verify_only"],
        ),
    ],
)
def test_mixed_tier_binds_verification_only_gate_to_final_candidate(
    tmp_path,
    monkeypatch,
    verification_first,
    expected_gate_slices,
):
    from assay import Implementation
    from crucible.workflows import _gate_evidence
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _WriteTargetBackend()
    gate_slices: list[str] = []
    reviewed_artifacts: list[str] = []

    def passing_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(passing_gate),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )

    rc, summary, repo, corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=verification_first,
    )

    assert rc == 0
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert len(backend.invocations) == 1
    assert gate_slices == expected_gate_slices
    assert len(reviewed_artifacts) == 1

    tier = _mixed_spec(
        corpus.latest_spec().project_context.base_commit,
        verification_first=verification_first,
    ).tier_dag.tiers[0]
    candidate = tier_workflow._capture_current_tier_diff(str(repo), tier)
    candidate_digest = "sha256:" + hashlib.sha256(candidate).hexdigest()
    artifact = reviewed_artifacts[0]
    assert candidate_digest in artifact
    assert '"slice_id": "verify_only"' in artifact
    assert '"gate_status": "passed"' in artifact

    latest = corpus.latest_spec()
    assert latest is not None
    latest_by_id: dict[str, Implementation] = {}
    for ref in latest.implementation_refs:
        implementation = corpus.get_record(ref, as_type=Implementation)
        if implementation.slice_id in {"mutate", "verify_only"}:
            latest_by_id[implementation.slice_id] = implementation
    verification_impl = latest_by_id["verify_only"]
    evidence_ref = verification_impl.artifacts[
        _gate_evidence.TIER_VERIFICATION_ARTIFACT
    ]
    evidence = _gate_evidence.load_verification_evidence(
        corpus,
        str(evidence_ref),
    )
    assert evidence.candidate_diff_sha256 == candidate_digest
    assert evidence.satisfies_required_gate is True
    manifest_refs = {
        implementation.artifacts[
            _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
        ]
        for implementation in latest_by_id.values()
    }
    assert len(manifest_refs) == 1
    manifest_ref = next(iter(manifest_refs))
    manifest = _gate_evidence.load_tier_verification_manifest(
        corpus,
        str(manifest_ref),
    )
    assert manifest.candidate_diff_sha256 == candidate_digest
    assert manifest.obligation_slice_ids == ("verify_only",)
    assert manifest.disposition == "passed"
    current_implementations = tuple(
        latest_by_id[slice_.id] for slice_ in tier.slices
    )
    assert _gate_evidence.tier_verification_manifest_matches(
        corpus,
        manifest_ref=str(manifest_ref),
        tier_id="t0",
        spec_execution_authority_hash=(
            tier_workflow._partition.spec_execution_structure_hash(latest)
        ),
        expected_project_base_commit=latest.project_context.base_commit,
        current_implementations=current_implementations,
        dependency_implementations=(),
        expected_obligation_slice_ids=("verify_only",),
        require_gate=True,
    ) is True
    superseded = tuple(
        replace(implementation, classification="success_superseded")
        if implementation.slice_id == "mutate"
        else implementation
        for implementation in current_implementations
    )
    assert _gate_evidence.tier_verification_manifest_matches(
        corpus,
        manifest_ref=str(manifest_ref),
        tier_id="t0",
        spec_execution_authority_hash=(
            tier_workflow._partition.spec_execution_structure_hash(latest)
        ),
        expected_project_base_commit=latest.project_context.base_commit,
        current_implementations=superseded,
        dependency_implementations=(),
        expected_obligation_slice_ids=("verify_only",),
        require_gate=True,
    ) is False


def test_mixed_tier_normalizes_candidate_base_without_rewriting_project_base(
    tmp_path,
    monkeypatch,
):
    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    backend = _WriteTargetBackend()
    reviewed_artifacts: list[str] = []
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )

    rc, summary, repo, corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=False,
        abbreviated_base=True,
    )

    assert rc == 0
    full_base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    latest = corpus.latest_spec()
    assert latest.project_context.base_commit == full_base[:8]
    latest_by_id: dict[str, Implementation] = {}
    for ref in latest.implementation_refs:
        implementation = corpus.get_record(ref, as_type=Implementation)
        if implementation.slice_id in {"mutate", "verify_only"}:
            latest_by_id[implementation.slice_id] = implementation
    verification = latest_by_id["verify_only"]
    assert verification.base_commit == full_base
    manifest_ref = verification.artifacts[
        tier_workflow._gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
    ]
    manifest = tier_workflow._gate_evidence.load_tier_verification_manifest(
        corpus,
        str(manifest_ref),
    )
    assert manifest.base_commit == full_base
    assert manifest.project_base_commit == full_base[:8]
    evidence_ref = verification.artifacts[
        tier_workflow._gate_evidence.TIER_VERIFICATION_ARTIFACT
    ]
    evidence = tier_workflow._gate_evidence.load_verification_evidence(
        corpus,
        str(evidence_ref),
    )
    assert evidence.base_commit == full_base
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE


def test_stale_mixed_tier_replay_rejects_and_restores_gate_mutation(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _WriteTargetBackend()
    gate_slices: list[str] = []
    reviewed_artifacts: list[str] = []

    def mutating_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        if gate_slices == ["verify_only", "mutate", "verify_only"]:
            backend.repo.joinpath("target.py").write_text(
                "value = gate_mutation\n",
                encoding="utf-8",
            )
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(mutating_gate),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )

    rc, summary, repo, _corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=True,
    )

    assert rc != 0
    assert summary["error_kind"] == "final_candidate_verification_failed"
    assert len(backend.invocations) == 1
    assert gate_slices == ["verify_only", "mutate", "verify_only"]
    assert reviewed_artifacts == []
    assert "gate_mutation" not in repo.joinpath("target.py").read_text()


def test_mixed_tier_phase_gate_is_read_only_over_composed_candidate(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _WriteTargetBackend()
    gate_slices: list[str] = []
    reviewed_artifacts: list[str] = []

    def mutating_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        if kwargs["slice_id"] == "verify_only":
            backend.repo.joinpath("target.py").write_text(
                "value = gate_mutation\n",
                encoding="utf-8",
            )
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(mutating_gate),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )

    rc, summary, repo, _corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=False,
    )

    assert rc != 0
    assert summary["phase_1_slices"][-1]["classification"] == (
        "hard_fail_gate_mutation"
    )
    assert len(backend.invocations) == 1
    assert gate_slices == ["mutate", "verify_only"]
    assert reviewed_artifacts == []
    assert repo.joinpath("target.py").read_text() == "value = 2\n"


def test_manifest_candidate_change_before_lifecycle_fails_closed(
    tmp_path,
    monkeypatch,
):
    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    backend = _WriteTargetBackend()
    reviewed_artifacts: list[str] = []
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )
    real_matcher = (
        tier_workflow._gate_evidence.tier_verification_manifest_matches
    )

    def mutate_after_manifest_check(*args, **kwargs):
        matched = real_matcher(*args, **kwargs)
        if matched:
            backend.repo.joinpath("target.py").write_text(
                "value = post_manifest_mutation\n",
                encoding="utf-8",
            )
        return matched

    monkeypatch.setattr(
        tier_workflow._gate_evidence,
        "tier_verification_manifest_matches",
        mutate_after_manifest_check,
    )

    rc, summary, _repo, corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=False,
    )

    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == (
        "tier_verification_manifest_candidate_changed"
    )
    latest = corpus.latest_spec()
    assert latest is not None
    terminal = [
        corpus.get_record(ref, as_type=Implementation)
        for ref in latest.implementation_refs
    ]
    assert all(
        tier_workflow._gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
        not in (implementation.artifacts or {})
        for implementation in terminal
    )


def test_final_verification_evidence_refuses_concurrent_spec_rebase(
    tmp_path,
    monkeypatch,
):
    """A D1→D2-style evidence race cannot inherit an earlier gate result."""

    from crucible.workflows import tier as tier_workflow

    backend = _WriteTargetBackend()
    reviewed_artifacts: list[str] = []
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )
    original_commit = Corpus.commit
    injected = False

    def racing_commit(self, *args, **kwargs):
        nonlocal injected
        implementations = kwargs.get("implementations") or ()
        if (
            not injected
            and kwargs.get("transform") is not None
            and len(implementations) == 1
            and implementations[0].slice_id == "verify_only"
            and tier_workflow._gate_evidence.TIER_VERIFICATION_ARTIFACT
            in (implementations[0].artifacts or {})
        ):
            injected = True
            concurrent = replace(
                implementations[0],
                classification="success_concurrent_supersession",
            )
            original_commit(self, implementations=(concurrent,))
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "commit", racing_commit)

    rc, summary, _repo, _corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=False,
    )

    assert injected is True
    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == "final_candidate_verification_failed"
    assert "evidence closure changed" in summary["error_message"]
    assert len(backend.invocations) == 1


def test_final_rebase_rejects_changed_composed_dependency_projection(
    tmp_path,
    monkeypatch,
):
    """A D1 gate cannot publish a manifest that selects concurrent D2."""

    from assay import Implementation
    from assay.types.identity import ContentHash
    from crucible.workflows import tier as tier_workflow

    backend = _WriteTargetBackend()
    reviewed_artifacts: list[str] = []
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )
    _install_passing_completeness_review(
        monkeypatch,
        backend,
        reviewed_artifacts,
    )
    dependency_v1 = Implementation(
        slice_id="dependency_slice",
        base_commit="0" * 40,
        result_commit=None,
        diff_ref=ContentHash("sha256:" + "1" * 64),
        transcript_ref=ContentHash("sha256:" + "2" * 64),
        files_modified=("dependency.py",),
        classification="success",
        execution=tier_workflow.exec_ctx_now("tier-test"),
        scope_escapes=(),
    )
    dependency_v2 = replace(
        dependency_v1,
        classification="success_concurrent_supersession",
    )
    closure_reads: list[int] = []

    def changing_dependency_closure(*args, **kwargs):
        closure_reads.append(len(closure_reads) + 1)
        implementation = (
            dependency_v1 if len(closure_reads) == 1 else dependency_v2
        )
        return (("dependency", implementation),)

    monkeypatch.setattr(
        tier_workflow,
        "_dependency_implementation_bindings",
        changing_dependency_closure,
    )

    rc, summary, _repo, _corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=False,
    )

    assert rc != 0
    assert closure_reads == [1, 2]
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == "concurrent_execution_closure_change"
    assert "closure changed" in summary["error_message"]
    assert len(backend.invocations) == 1


def test_final_rebase_rejects_blindly_adopted_current_slice_record(
    tmp_path,
    monkeypatch,
):
    """A blind sibling persist cannot overwrite concurrent B2 with stale B1."""

    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    backend = _WriteTargetBackend()
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )
    original_commit = Corpus.commit
    verify_v1 = None
    verify_v2_ref = None

    def racing_commit(self, *args, **kwargs):
        nonlocal verify_v1, verify_v2_ref
        implementations = kwargs.get("implementations") or ()
        if (
            kwargs.get("transform") is None
            and len(implementations) == 1
        ):
            implementation = implementations[0]
            if implementation.slice_id == "verify_only":
                verify_v1 = implementation
            elif (
                implementation.slice_id == "mutate"
                and verify_v1 is not None
                and verify_v2_ref is None
            ):
                concurrent = replace(
                    verify_v1,
                    classification="success_concurrent_supersession",
                )
                injected = original_commit(
                    self,
                    implementations=(concurrent,),
                )
                verify_v2_ref = injected.refs.implementation_refs[0]
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(Corpus, "commit", racing_commit)

    rc, summary, _repo, corpus = _run_mixed(
        tmp_path,
        monkeypatch,
        backend,
        verification_first=True,
        skip_review=True,
    )

    assert verify_v2_ref is not None
    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == "concurrent_execution_closure_change"
    latest = corpus.latest_spec()
    raw_latest = tier_workflow._partition._latest_impl_per_slice(
        corpus,
        {"verify_only"},
        refs=latest.implementation_refs,
    )
    assert raw_latest["verify_only"].classification == (
        "success_concurrent_supersession"
    )
    assert verify_v2_ref in latest.implementation_refs


@pytest.mark.parametrize(
    ("tier_max_turns", "expected_breach_max_turns"),
    [(90, "90"), (None, "50")],
    ids=("explicit-90", "run-tier-default"),
)
def test_breach_exit_zero_requires_successful_typed_contract(
    tmp_path,
    monkeypatch,
    tier_max_turns,
    expected_breach_max_turns,
):
    from crucible.workflows import tier as tier_workflow

    fake_child_run = tmp_path / "fake-breach-child"
    fake_child_run.mkdir()
    breach_calls: list[list[str]] = []

    def fake_breach_main(argv, **_kwargs):
        breach_calls.append(list(argv))
        return 0

    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        fake_breach_main,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_run_dir",
        lambda *args, **kwargs: fake_child_run,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_outcome",
        lambda *args, **kwargs: SimpleNamespace(
            terminal_class=tier_workflow._outcome.TerminalClass.CONTRACT_ERROR,
            detail_status="contract_error",
            error_kind="invalid_implementation_refs",
            error_message="malformed implementation_refs",
            implementation_refs=(),
        ),
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        _WriteTargetBackend(),
        breach_policy=BreachPolicy(enabled=True),
        implementer_read_call_limit=7,
        response_retry_budget=5,
        max_turns=tier_max_turns,
    )

    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_BREACH_EXCEPTION
    assert "successful typed terminal contract" in summary["error_message"]
    assert len(breach_calls) == 1
    breach_argv = breach_calls[0]
    observed_base = breach_argv[
        breach_argv.index("--diff-base-treeish") + 1
    ]
    observed_read_limit = breach_argv[
        breach_argv.index("--implementer-read-call-limit") + 1
    ]
    assert breach_argv.count("--max-turns") == 1
    observed_max_turns = breach_argv[breach_argv.index("--max-turns") + 1]
    observed_response_retry_budget = breach_argv[
        breach_argv.index("--response-retry-budget") + 1
    ]
    expected_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert observed_base == expected_base
    assert observed_read_limit == "7"
    assert observed_max_turns == expected_breach_max_turns
    assert observed_response_retry_budget == "5"
    assert summary["response_retry_budget"] == 5


def test_breach_owned_refs_do_not_infer_ownership_from_plan_slice_prefix(
    tmp_path,
    monkeypatch,
):
    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    fake_child_run = tmp_path / "fake-breach-child"
    fake_child_run.mkdir()
    concurrent_ref = None

    def fake_breach_main(argv, **kwargs):
        nonlocal concurrent_ref
        corpus = Corpus(tmp_path / "corpus")
        latest = corpus.latest_spec()
        current = None
        for ref in latest.implementation_refs:
            implementation = corpus.get_record(ref, as_type=Implementation)
            if implementation.slice_id == "breach_current":
                current = implementation
        assert current is not None
        concurrent = replace(
            current,
            classification="success_concurrent_supersession",
        )
        committed = corpus.commit(implementations=(concurrent,))
        concurrent_ref = committed.refs.implementation_refs[0]
        return 0

    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "main",
        fake_breach_main,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_run_dir",
        lambda *args, **kwargs: fake_child_run,
    )
    monkeypatch.setattr(
        tier_workflow._outcome,
        "child_outcome",
        lambda *args, **kwargs: SimpleNamespace(
            terminal_class=tier_workflow._outcome.TerminalClass.SUCCEEDED,
            detail_status="no_attacks",
            error_kind=None,
            error_message=None,
            # The concurrent plan-slice ref is deliberately not saga-owned.
            implementation_refs=(),
        ),
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _WriteTargetBackend(),
        slice_id="breach_current",
        breach_policy=BreachPolicy(enabled=True),
    )

    assert concurrent_ref is not None
    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == "concurrent_execution_closure_change"
    corpus = Corpus(tmp_path / "corpus")
    latest = corpus.latest_spec()
    raw_latest = tier_workflow._partition._latest_impl_per_slice(
        corpus,
        {"breach_current"},
        refs=latest.implementation_refs,
    )
    assert raw_latest["breach_current"].classification == (
        "success_concurrent_supersession"
    )
    assert concurrent_ref in latest.implementation_refs


def test_persistent_workspace_proves_accepted_dependency_postimage(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, backend = _prepare_accepted_persistent_dependency(
        tmp_path,
        monkeypatch,
    )
    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
    )

    assert rc == 0
    assert len(backend.invocations) == 1
    assert gate_slices == ["dependent_verification", "dependent_verification"]
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert summary["persistent_workspace_evidence"]["passed"] is True
    assert summary["persistent_workspace_evidence"][
        "accepted_tier_ids"
    ] == ["t0"]


def test_persistent_workspace_rejects_live_dependency_drift_before_gate(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, backend = _prepare_accepted_persistent_dependency(
        tmp_path,
        monkeypatch,
    )
    repo.joinpath("target.py").write_text(
        "value = 'unbound dependency drift'\n",
        encoding="utf-8",
    )

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
    )

    assert rc == 2
    assert len(backend.invocations) == 1
    assert gate_slices == []
    assert summary["status"] == tier_workflow.STATUS_SETUP_ERROR
    assert summary["setup_stage"] == "mutation_recovery"
    assert summary["error_kind"] == "MutationJournalError"
    assert "positive Implementation" in summary["error_message"]
    assert (
        "postimage for 'target.py' is not live" in summary["error_message"]
    )
    assert "persistent_workspace_evidence" not in summary


def test_persistent_workspace_rejects_unrecorded_pending_create_before_provider(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, backend = _prepare_accepted_persistent_dependency(
        tmp_path,
        monkeypatch,
        successor_creates=True,
    )
    repo.joinpath("new.py").write_text(
        "unrecorded = True\n",
        encoding="utf-8",
    )

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
    )

    assert rc == 2
    assert len(backend.invocations) == 1
    assert gate_slices == []
    assert summary["status"] == tier_workflow.STATUS_SETUP_ERROR
    assert summary["error_kind"] == (
        "persistent_workspace_dependency_evidence_invalid"
    )
    assert summary["persistent_workspace_evidence"]["passed"] is False


def test_persistent_workspace_admits_clean_pending_create_before_provider(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, dependency_backend = (
        _prepare_accepted_persistent_dependency(
            tmp_path,
            monkeypatch,
            successor_creates=True,
        )
    )
    successor_backend = _CreateNewBackend()

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
        successor_backend=successor_backend,
    )

    assert rc == 0
    assert len(dependency_backend.invocations) == 1
    assert len(successor_backend.invocations) == 1
    assert gate_slices == ["dependent_verification"]
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert summary["persistent_workspace_evidence"]["passed"] is True
    assert repo.joinpath("new.py").read_text() == "created = True\n"


def test_persistent_workspace_rejects_unrecorded_recursive_scope_before_provider(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, dependency_backend = (
        _prepare_accepted_persistent_dependency(
            tmp_path,
            monkeypatch,
            successor_recursive=True,
        )
    )
    generated = repo / "generated"
    generated.mkdir()
    generated.joinpath("partial.py").write_text(
        "unrecorded = True\n",
        encoding="utf-8",
    )

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
    )

    assert rc == 2
    assert len(dependency_backend.invocations) == 1
    assert gate_slices == []
    assert summary["status"] == tier_workflow.STATUS_SETUP_ERROR
    assert summary["error_kind"] == (
        "persistent_workspace_dependency_evidence_invalid"
    )


def test_persistent_workspace_admits_clean_recursive_scope_before_provider(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow

    repo, corpus, dependency_backend = (
        _prepare_accepted_persistent_dependency(
            tmp_path,
            monkeypatch,
            successor_recursive=True,
        )
    )
    successor_backend = _CreateGeneratedBackend()

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
        successor_backend=successor_backend,
    )

    assert rc == 0
    assert len(dependency_backend.invocations) == 1
    assert len(successor_backend.invocations) == 1
    assert gate_slices == ["dependent_verification"]
    assert summary["status"] == tier_workflow.STATUS_TIER_COMPLETE
    assert summary["persistent_workspace_evidence"]["passed"] is True
    assert repo.joinpath("generated/partial.py").read_text() == (
        "created = True\n"
    )


def test_persistent_workspace_requires_dependency_tier_lifecycle(
    tmp_path,
    monkeypatch,
):
    from assay import Implementation
    from assay.primitives.implement import capture_workspace_delta
    from crucible.tools._context import ToolContext
    from crucible.tools.profiles import build_autonomous_implementer_profile
    from crucible.workflows import _partition
    from crucible.workflows import tier as tier_workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    base = _init_repo(repo)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(_persistent_dependency_spec(base))
    spec = corpus.latest_spec()
    assert spec is not None
    repo.joinpath("target.py").write_text("value = 2\n", encoding="utf-8")
    diff, files, observed_base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("target.py",),
    )
    dependency_tier = spec.tier_dag.tiers[0]
    dependency_slice = dependency_tier.slices[0]
    implementation = Implementation(
        slice_id=dependency_slice.id,
        base_commit=observed_base,
        result_commit=None,
        diff_ref=corpus.put_artifact(diff, kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=files,
        classification="success",
        execution=tier_workflow.exec_ctx_now("tier-test"),
        scope_escapes=(),
    )
    context = ToolContext(workspace_root=repo, repo_root=repo)
    profile = build_autonomous_implementer_profile(context)
    tool_digest = tier_workflow._tier_tool_authority_digests(
        context,
        dependency_tier,
        profile,
    )[dependency_slice.id]
    implementation = _partition.attach_resume_identity(
        corpus,
        spec,
        dependency_tier,
        dependency_slice,
        implementation,
        tool_authority_digest=tool_digest,
    )
    corpus.commit(implementations=(implementation,))

    rc, summary, gate_slices = _run_persistent_successor(
        repo,
        corpus,
        monkeypatch,
    )

    assert rc == 2
    assert gate_slices == []
    assert summary["status"] == tier_workflow.STATUS_SETUP_ERROR
    assert summary["setup_stage"] == "mutation_recovery"
    assert summary["error_kind"] == "MutationJournalError"
    assert (
        "not current-plan authority for cleanup"
        in summary["error_message"]
    )
    assert "mutation may still be present" in summary["error_message"]
    assert "persistent_workspace_evidence" not in summary


@pytest.mark.parametrize(
    ("disposition", "build_verify", "expected_status"),
    [
        ("waived", "skipped_by_policy", "skipped"),
        ("unavailable", "runner_missing", "unavailable"),
    ],
)
def test_completeness_renderer_never_promotes_nonpassing_gate_evidence(
    disposition,
    build_verify,
    expected_status,
):
    from crucible.workflows import tier as tier_workflow

    union_diff = b"exact candidate"
    digest = "sha256:" + hashlib.sha256(union_diff).hexdigest()
    tier = _spec("0" * 40, no_surface=True).tier_dag.tiers[0]
    artifact = tier_workflow._render_tier_completeness_artifact(
        tier,
        union_diff,
        verification_evidence=({
            "slice_id": "s0",
            "classification": build_verify,
            "build_verify": build_verify,
            "candidate_diff_sha256": digest,
            "verification_disposition": disposition,
        },),
    ).decode("utf-8")

    assert f'"gate_status": "{expected_status}"' in artifact
    assert '"gate_status": "passed"' not in artifact


def test_required_resume_replays_waived_verification_without_implementer(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    def reject_resolution(*args, **kwargs):
        pytest.fail("deterministic resume must not resolve Implementer")

    rc, _summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        verify_mode="tests",
        backend_resolver=reject_resolution,
        skip_build_verify=True,
    )
    assert rc == 0

    gate_slices: list[str] = []

    def passing_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(passing_gate),
    )
    runs_root = tmp_path / "corpus" / "runs"
    prior_tier_runs = {
        path.name for path in runs_root.iterdir() if "tier" in path.name
    }
    assert tier_workflow.run_tier(
        spec=tmp_path / "corpus",
        tier_id="t0",
        cwd=str(tmp_path / "repo"),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 0
    assert gate_slices == ["s0"]

    resumed_run_dirs = [
        path
        for path in runs_root.iterdir()
        if "tier" in path.name and path.name not in prior_tier_runs
    ]
    assert len(resumed_run_dirs) == 1
    resumed = json.loads(
        resumed_run_dirs[0].joinpath("summary.json").read_text()
    )
    assert resumed["resumed_slices"] == ["s0"]
    assert resumed["phase_1_slices"][0]["build_verify"] == "success"
    assert resumed["phase_1_slices"][0][
        "verification_disposition"
    ] == "passed"


def test_handoff_resume_preserves_operator_identity_through_manifest(
    tmp_path,
    monkeypatch,
):
    from crucible.workflows import _partition
    from crucible.workflows import tier as tier_workflow

    rc, _summary, repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=True,
    )
    assert rc == 0

    corpus = Corpus(tmp_path / "corpus")
    spec = corpus.latest_spec()
    assert spec is not None
    prior = corpus.get_record(
        spec.implementation_refs[-1],
        as_type=Implementation,
    )
    handoff = replace(
        prior,
        execution=replace(prior.execution, recipe="handoff"),
        artifacts={},
    )
    tier = spec.tier_dag.tiers[0]
    handoff = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        handoff,
        tool_authority_digest=None,
    )
    corpus.commit(implementations=(handoff,))

    def reject_resolution(*args, **kwargs):
        pytest.fail("Handoff verification resume must not resolve a backend")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )
    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(
            lambda *args, **kwargs: _successful_gate_outcome()
        ),
    )

    assert tier_workflow.run_tier(
        spec=tmp_path / "corpus",
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 0

    latest = corpus.latest_spec()
    retained = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    identity = json.loads(
        corpus.get_artifact(retained.artifacts["resume_identity"])
    )
    assert retained.execution.recipe == "handoff"
    assert identity["tool_authority_digest"] is None
    assert (
        tier_workflow._gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
        in retained.artifacts
    )

    landed, pending = _partition.partition_for_resume(
        corpus,
        latest.tier_dag.tiers[0],
        spec=latest,
        tool_authority_digests={"s0": "sha256:" + "c" * 64},
    )
    assert set(landed) == {"s0"}
    assert pending == ()


def test_tier_rejects_invalid_final_resume_identity_before_landing(
    tmp_path,
    monkeypatch,
):
    from assay import LifecycleEvent
    from crucible.workflows import _partition
    from crucible.workflows import tier as tier_workflow

    original_identity_matches = _partition._identity_matches

    def reject_manifest_binding(*args, **kwargs):
        implementation = args[4]
        if (
            tier_workflow._gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
            in (implementation.artifacts or {})
        ):
            return False
        return original_identity_matches(*args, **kwargs)

    monkeypatch.setattr(
        _partition,
        "_identity_matches",
        reject_manifest_binding,
    )

    rc, summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        skip_build_verify=True,
    )

    assert rc != 0
    assert summary["status"] == tier_workflow.STATUS_FINAL_BINDING_FAILED
    assert summary["error_kind"] == "final_resume_identity_invalid"
    corpus = Corpus(tmp_path / "corpus")
    latest = corpus.latest_spec()
    lifecycle_events = tuple(
        corpus.get_record(ref, as_type=LifecycleEvent)
        for ref in latest.lifecycle_event_refs
    )
    assert lifecycle_events
    assert all(event.kind != "landed" for event in lifecycle_events)


def test_verification_pending_replays_matching_bare_evidence(
    tmp_path,
    monkeypatch,
):
    """A stale closure signal outranks a matching per-slice candidate hash."""

    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    gate_slices: list[str] = []

    def passing_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(passing_gate),
    )

    def reject_resolution(*args, **kwargs):
        pytest.fail("verification replay must not resolve Implementer")

    rc, _summary, _repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        verify_mode="tests",
        backend_resolver=reject_resolution,
        skip_build_verify=False,
    )
    assert rc == 0
    assert gate_slices == ["s0", "s0"]

    corpus = Corpus(tmp_path / "corpus")
    latest = corpus.latest_spec()
    retained = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    assert tier_workflow._gate_evidence.TIER_VERIFICATION_ARTIFACT in (
        retained.artifacts
    )
    gate_slices.clear()
    monkeypatch.setattr(
        tier_workflow._partition,
        "partition_for_required_verification",
        lambda *args, **kwargs: SimpleNamespace(
            retained={"s0": retained},
            verification_pending={"s0": retained},
            implementation_pending=(),
        ),
    )

    assert tier_workflow.run_tier(
        spec=tmp_path / "corpus",
        tier_id="t0",
        cwd=str(tmp_path / "repo"),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 0
    # The passed bare evidence still matches the empty current-tier union, but
    # verification_pending represents a changed/missing closure manifest and
    # therefore forces one real deterministic replay.
    assert gate_slices == ["s0"]


def test_required_partition_replays_retained_mutation_without_implementer(
    tmp_path,
    monkeypatch,
):
    from assay import Implementation
    from crucible.workflows import tier as tier_workflow
    from crucible.workflows._build_verify import BuildVerifyOutcome

    backend = _WriteTargetBackend()
    rc, _summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        skip_build_verify=True,
    )
    assert rc == 0
    assert len(backend.invocations) == 1

    corpus = Corpus(tmp_path / "corpus")
    latest = corpus.latest_spec()
    retained = corpus.get_record(
        latest.implementation_refs[-1],
        as_type=Implementation,
    )
    monkeypatch.setattr(
        tier_workflow._partition,
        "partition_for_required_verification",
        lambda *args, **kwargs: SimpleNamespace(
            retained={"s0": retained},
            verification_pending={"s0": retained},
            implementation_pending=(),
        ),
    )

    def reject_resolution(*args, **kwargs):
        pytest.fail("verification-pending replay must not resolve Implementer")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )
    gate_slices: list[str] = []

    def passing_gate(*args, **kwargs):
        gate_slices.append(kwargs["slice_id"])
        return _successful_gate_outcome()

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(passing_gate),
    )
    assert tier_workflow.run_tier(
        spec=tmp_path / "corpus",
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    ) == 0

    assert gate_slices == ["s0"]
    assert len(backend.invocations) == 1
    run_dirs = sorted(
        path
        for path in (tmp_path / "corpus" / "runs").iterdir()
        if "tier" in path.name
    )
    resumed = json.loads(run_dirs[-1].joinpath("summary.json").read_text())
    assert resumed["verification_pending_slices"] == ["s0"]
    assert resumed["phase_1_slices"][0]["build_verify"] == "success"


@pytest.mark.parametrize("failure", ["runner_missing", "verify_failed"])
def test_failed_required_replay_remains_verification_pending_on_retry(
    tmp_path,
    monkeypatch,
    failure,
):
    """A deterministic gate failure never reacquires Implementer authority."""

    from assay import Implementation
    from crucible.workflows import tier as tier_workflow

    rc, _summary, repo = _run(
        tmp_path,
        monkeypatch,
        _AlwaysEmptyBackend(),
        no_surface=True,
        verify_mode="tests",
        skip_build_verify=True,
    )
    assert rc == 0

    def reject_resolution(*args, **kwargs):
        pytest.fail("verification replay must not resolve Implementer")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        reject_resolution,
    )
    gate_outcomes = [_failed_gate_outcome(failure)]

    def next_gate(*args, **kwargs):
        assert kwargs["slice_id"] == "s0"
        return gate_outcomes.pop(0)

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "build_and_verify",
        _contract_aware_build_verify_stub(next_gate),
    )
    run_kwargs = dict(
        spec=tmp_path / "corpus",
        tier_id="t0",
        cwd=str(repo),
        max_turns=24,
        skip_build_verify=False,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
    )
    assert tier_workflow.run_tier(**run_kwargs) != 0
    assert gate_outcomes == []

    corpus = Corpus(tmp_path / "corpus")
    latest = corpus.latest_spec()
    retained = None
    for ref in latest.implementation_refs:
        implementation = corpus.get_record(ref, as_type=Implementation)
        if implementation.slice_id == "s0":
            retained = implementation
    assert retained is not None
    assert retained.classification == "soft_fail_no_changes"
    failed_ref = retained.artifacts[
        tier_workflow._gate_evidence.TIER_VERIFICATION_ARTIFACT
    ]
    failed_evidence = tier_workflow._gate_evidence.load_verification_evidence(
        corpus,
        str(failed_ref),
    )
    assert failed_evidence.satisfies_required_gate is False
    retry_partition = (
        tier_workflow._partition.partition_for_required_verification(
            corpus,
            latest.tier_dag.tiers[0],
            spec=latest,
        )
    )
    assert set(retry_partition.verification_pending) == {"s0"}
    assert retry_partition.implementation_pending == ()

    gate_outcomes.append(_successful_gate_outcome())
    assert tier_workflow.run_tier(**run_kwargs) == 0
    assert gate_outcomes == []


def test_scoped_soft_no_change_never_seeds_breach_from_inherited_dirt(
    tmp_path,
    monkeypatch,
):
    backend = _AlwaysEmptyBackend()
    breach_called = []
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "crucible.workflows.tier.breach_workflow.main",
        lambda argv: breach_called.append(argv) or 0,
    )

    rc, summary, repo = _run(
        tmp_path,
        monkeypatch,
        backend,
        inherited_dirty=True,
        verify_mode="skip",
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert rc == 1
    assert len(backend.invocations) == 1
    assert summary["status"] == "slice_failed"
    assert summary["phase_1_slices"][0]["classification"] == (
        "soft_fail_no_changes"
    )
    assert breach_called == []
    assert repo.joinpath("target.py").read_text(encoding="utf-8") == (
        "value = inherited\n"
    )
