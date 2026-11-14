# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Contracts for test-owned interpreter trace instrumentation."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from tests import conftest as shared_conftest


ROOT = Path(__file__).resolve().parents[1]


_TRACE_OWNER_SCOPES = frozenset([('tests/test_mutation_journal_authority_regressions.py',
  ('test_inaccessible_recovery_nested_return_restores_mode_and_authority',)),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_nested_external_return_interrupt_preserves_parent_coordination',)),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_public_recovery_retains_lock_cleanup_before_entry_returns',)),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_readable_spec_authority_preowns_caller_return_interrupt',)),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_recovery_source_chain_sigint_at_nested_return_releases_every_owner',)),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_retained_nested_lock_closes_canonical_source_return_interrupt',)),
 ('tests/test_workflows_breach.py',
  ('test_breach_retained_nested_lock_return_interrupt_preserves_parent',)),
 ('tests/test_workflows_breach.py',
  ('test_breach_retained_recovery_lock_return_interrupt_releases_once',)),
 ('tests/test_workflows_handoff.py',
  ('test_handoff_loop_boundary_interrupt_publishes_precomposed_retry_owner',)),
 ('tests/test_workflows_handoff.py',
  ('test_handoff_loop_boundary_interrupt_without_provider_settles_parents',)),
 ('tests/test_workflows_handoff.py',
  ('test_handoff_retained_stack_closes_lock_at_context_return_boundary',)),
 ('tests/test_workflows_implement_transaction.py',
  ('test_implement_real_run_handle_transfer_interrupt_closes_all_owners',)),
 ('tests/test_workflows_implement_transaction.py',
  ('test_standalone_retained_lock_return_interrupt_releases_without_run',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_attestation_lock_return_interrupt_releases_authority',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_breach_child_transfer_return_interrupt_keeps_final_owner',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_direct_cleanup_factory_return_interrupt_uses_prebound_owner',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_outer_owner_return_interrupt_republishes_prebound_owner',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_publication_recognition_return_interrupt_uses_prebound_owner',)),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_retained_recovery_lock_return_interrupt_releases_before_run',))])

_ONE_SHOT_DISARM_SCOPES = frozenset([('tests/test_mutation_journal_authority_regressions.py',
  ('test_inaccessible_recovery_nested_return_restores_mode_and_authority', 'interrupt_return')),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_nested_external_return_interrupt_preserves_parent_coordination', 'interrupt_return')),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_public_recovery_retains_lock_cleanup_before_entry_returns', 'interrupt_return')),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_readable_spec_authority_preowns_caller_return_interrupt', 'interrupt_return')),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_recovery_source_chain_sigint_at_nested_return_releases_every_owner', 'interrupt_return')),
 ('tests/test_mutation_journal_authority_regressions.py',
  ('test_retained_nested_lock_closes_canonical_source_return_interrupt', 'interrupt_return')),
 ('tests/test_workflows_breach.py',
  ('test_breach_retained_nested_lock_return_interrupt_preserves_parent', 'interrupt_return')),
 ('tests/test_workflows_breach.py',
  ('test_breach_retained_recovery_lock_return_interrupt_releases_once', 'interrupt_return')),
 ('tests/test_workflows_handoff.py',
  ('test_handoff_retained_stack_closes_lock_at_context_return_boundary', 'interrupt_return')),
 ('tests/test_workflows_implement_transaction.py',
  ('test_implement_real_run_handle_transfer_interrupt_closes_all_owners', 'interrupt_transfer')),
 ('tests/test_workflows_implement_transaction.py',
  ('test_standalone_retained_lock_return_interrupt_releases_without_run', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_attestation_lock_return_interrupt_releases_authority', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_breach_child_transfer_return_interrupt_keeps_final_owner', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_direct_cleanup_factory_return_interrupt_uses_prebound_owner', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_outer_owner_return_interrupt_republishes_prebound_owner', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_publication_recognition_return_interrupt_uses_prebound_owner', 'interrupt_return')),
 ('tests/test_workflows_tier_mutation_transaction.py',
  ('test_tier_retained_recovery_lock_return_interrupt_releases_before_run', 'interrupt_return'))])

_MULTI_OVERRIDE_SCOPES = frozenset([])

_THREE_RESTORE_SCOPES = frozenset([])

_SPLIT_TRACE_OVERRIDE_SCOPES = frozenset([])

_SPLIT_TRACE_RESTORE_SCOPES = frozenset([])

_SPLIT_TRACE_CAPTURE_SCOPES = frozenset([])


class _TraceVisitor(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.scope: list[str] = []
        self.calls: list[tuple[str, tuple[str, tuple[str, ...]]]] = []
        self.previous_trace_captures: list[tuple[str, tuple[str, ...]]] = []

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_Assign(self, node: ast.Assign) -> None:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "previous_trace"
            and _is_sys_call(node.value, "gettrace")
        ):
            self.previous_trace_captures.append((self.path, tuple(self.scope)))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _is_sys_call(node, "settrace") and len(node.args) == 1:
            argument = node.args[0]
            if isinstance(argument, ast.Constant) and argument.value is None:
                kind = "disarm"
            elif isinstance(argument, ast.Name) and argument.id == "previous_trace":
                kind = "restore"
            else:
                kind = "override"
            self.calls.append((kind, (self.path, tuple(self.scope))))
        self.generic_visit(node)


def _is_sys_call(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sys"
        and node.func.attr == name
    )


def _trace_inventory() -> tuple[
    Counter[tuple[str, tuple[str, ...]]],
    Counter[tuple[str, tuple[str, ...]]],
    Counter[tuple[str, tuple[str, ...]]],
    Counter[tuple[str, tuple[str, ...]]],
]:
    calls: list[tuple[str, tuple[str, tuple[str, ...]]]] = []
    captures: list[tuple[str, tuple[str, ...]]] = []
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        relative = path.relative_to(ROOT).as_posix()
        visitor = _TraceVisitor(relative)
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        calls.extend(visitor.calls)
        captures.extend(visitor.previous_trace_captures)
    return (
        Counter(scope for kind, scope in calls if kind == "override"),
        Counter(scope for kind, scope in calls if kind == "restore"),
        Counter(scope for kind, scope in calls if kind == "disarm"),
        Counter(captures),
    )


def test_trace_injection_inventory_has_exact_restore_and_disarm_classes() -> None:
    overrides, restores, disarms, captures = _trace_inventory()
    expected_overrides = Counter({scope: 1 for scope in _TRACE_OWNER_SCOPES})
    expected_restores = Counter({scope: 1 for scope in _TRACE_OWNER_SCOPES})
    for scope in _MULTI_OVERRIDE_SCOPES:
        expected_overrides[scope] = 2
        expected_restores[scope] = 2
    for scope in _THREE_RESTORE_SCOPES:
        expected_restores[scope] = 3
    expected_overrides.update(_SPLIT_TRACE_OVERRIDE_SCOPES)
    expected_restores.update(_SPLIT_TRACE_RESTORE_SCOPES)
    expected_captures = Counter({scope: 1 for scope in _TRACE_OWNER_SCOPES})
    expected_captures.update(_SPLIT_TRACE_CAPTURE_SCOPES)

    assert overrides == expected_overrides
    assert restores == expected_restores
    assert disarms == Counter({scope: 1 for scope in _ONE_SHOT_DISARM_SCOPES})
    assert captures == expected_captures


@pytest.mark.parametrize("incoming_trace", (None, object()), ids=("none", "set"))
def test_trace_containment_guard_restores_drift_before_failing(
    monkeypatch: pytest.MonkeyPatch,
    incoming_trace: Any,
) -> None:
    state = {"trace": incoming_trace}
    installed: list[Any] = []

    monkeypatch.setattr(shared_conftest.sys, "gettrace", lambda: state["trace"])

    def settrace(value: Any) -> None:
        installed.append(value)
        state["trace"] = value

    monkeypatch.setattr(shared_conftest.sys, "settrace", settrace)
    guard = shared_conftest._contain_process_trace_state.__wrapped__()
    next(guard)
    state["trace"] = object()

    with pytest.raises(
        pytest.fail.Exception,
        match="test leaked sys.settrace state; restore the incoming tracer",
    ):
        next(guard)

    assert state["trace"] is incoming_trace
    assert installed == [incoming_trace]


@pytest.mark.parametrize("incoming_trace", (None, object()), ids=("none", "set"))
def test_trace_containment_guard_leaves_matching_state_untouched(
    monkeypatch: pytest.MonkeyPatch,
    incoming_trace: Any,
) -> None:
    installed: list[Any] = []
    monkeypatch.setattr(shared_conftest.sys, "gettrace", lambda: incoming_trace)
    monkeypatch.setattr(shared_conftest.sys, "settrace", installed.append)
    guard = shared_conftest._contain_process_trace_state.__wrapped__()
    next(guard)

    with pytest.raises(StopIteration):
        next(guard)

    assert installed == []


def test_trace_containment_guard_preserves_unrelated_test_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming_trace = object()
    installed: list[Any] = []
    monkeypatch.setattr(shared_conftest.sys, "gettrace", lambda: incoming_trace)
    monkeypatch.setattr(shared_conftest.sys, "settrace", installed.append)
    guard = shared_conftest._contain_process_trace_state.__wrapped__()
    next(guard)
    test_failure = RuntimeError("original test failure")

    with pytest.raises(RuntimeError) as caught:
        guard.throw(test_failure)

    assert caught.value is test_failure
    assert installed == []
