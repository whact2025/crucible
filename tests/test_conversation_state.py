# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import gc
import json
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
from assay import BackendFeatureSet, Usage

from crucible.backends.features import feature_profile_for_route
from crucible.workflows import _conversation as conversation
from crucible.workflows import _durable_state


_WORKFLOW_PROFILE = BackendFeatureSet(
    session_continuity="workflow_bound",
)
_RETRY_PROFILE = BackendFeatureSet(
    session_continuity="retry_only",
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _identity(
    *,
    profile: BackendFeatureSet = _WORKFLOW_PROFILE,
    prompt: str = "stable task prompt",
    tool_digest: str = _digest("a"),
    conversation_mode: str = "workflow_bound",
):
    return conversation.build_conversation_identity(
        role="implementer",
        backend="codex",
        model="gpt-5.6-sol",
        prompt_abi="assay.prompt.v2",
        conversation_mode=conversation_mode,
        route_identity={
            "routing_fingerprint": _digest("1"),
            "endpoint_identity": "organization-codex",
        },
        prompt_content=prompt,
        response_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
        },
        tool_authority_digest=tool_digest,
        feature_profile=profile,
        policy_identity={"max_turns": 8, "budget_usd": 5.0},
        runtime_identity={
            "adapter": "crucible.codex.v1",
            "sdk": "0.3.2",
        },
        trusted_context_identity={
            "plan": _digest("2"),
            "tier": "t0",
            "slice": "s0",
            "base_commit": "abc123",
            "workspace": _digest("3"),
        },
    )


def _key() -> conversation.ConversationStoreKey:
    return conversation.ConversationStoreKey(
        run_identity_digest=_digest("4"),
        workflow="tier",
        role="implementer",
        cycle="t0/s0/build-correction",
    )


def _controller(
    tmp_path: Path,
    *,
    identity=None,
    profile: BackendFeatureSet = _WORKFLOW_PROFILE,
    mode="workflow_bound",
    limits: conversation.ConversationLimits = (
        conversation.ConversationLimits()
    ),
) -> conversation.ConversationController:
    return conversation.ConversationController(
        store=conversation.ConversationHandleStore(
            tmp_path / "conversations"
        ),
        key=_key(),
        identity=identity or _identity(
            profile=profile,
            conversation_mode=mode,
        ),
        mode=mode,
        feature_profile=profile,
        limits=limits,
    )


def test_atomic_store_explicit_context_and_finalizer_lifecycle(
    tmp_path: Path,
) -> None:
    explicit = _durable_state.AtomicStateStore(tmp_path / "explicit")
    explicit_descriptor = explicit.root_descriptor
    explicit.close()
    explicit.close()
    assert explicit.closed is True
    with pytest.raises(OSError):
        os.fstat(explicit_descriptor)
    with pytest.raises(_durable_state.DurableStateError, match="closed"):
        explicit.load(_digest("0"))

    with _durable_state.AtomicStateStore(tmp_path / "context") as contextual:
        contextual_descriptor = contextual.root_descriptor
        assert contextual.closed is False
    assert contextual.closed is True
    with pytest.raises(OSError):
        os.fstat(contextual_descriptor)

    abandoned = _durable_state.AtomicStateStore(tmp_path / "finalizer")
    abandoned_descriptor = abandoned.root_descriptor
    abandoned_reference = weakref.ref(abandoned)
    del abandoned
    gc.collect()
    assert abandoned_reference() is None
    with pytest.raises(OSError):
        os.fstat(abandoned_descriptor)


def test_atomic_store_lease_contends_across_two_os_processes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cross-process"
    holder_source = """
import sys
from pathlib import Path
from crucible.workflows._durable_state import AtomicStateStore

with AtomicStateStore(Path(sys.argv[1])) as store:
    with store.nonblocking_lease(".cross-process.lease"):
        print("ready", flush=True)
        sys.stdin.read(1)
"""
    with _durable_state.AtomicStateStore(root) as contender:
        holder = subprocess.Popen(
            (sys.executable, "-c", holder_source, str(root)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline() == "ready\n"
            with pytest.raises(
                _durable_state.DurableStateError,
                match="already held",
            ):
                with contender.nonblocking_lease(".cross-process.lease"):
                    pytest.fail("a second process acquired the held lease")
        finally:
            if holder.poll() is None:
                assert holder.stdin is not None
                with contextlib.suppress(BrokenPipeError):
                    holder.stdin.write("x")
                    holder.stdin.flush()
            try:
                holder.wait(timeout=10)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait(timeout=5)
        assert holder.stderr is not None
        assert holder.returncode == 0, holder.stderr.read()


def test_conversation_controller_context_releases_store_without_logical_close(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    with controller as entered:
        assert entered is controller
        handle = controller.resume(scope="workflow_correction").handle
        assert handle.state == "active"

    assert controller.store.closed is True
    with pytest.raises(conversation.ConversationStoreError, match="closed"):
        controller.load()


def test_conversation_logical_close_remains_distinct_from_store_release(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.resume(scope="workflow_correction")
    handle = controller.close()

    assert handle.state == "closed"
    assert controller.store.closed is False
    controller.close_store()
    assert controller.store.closed is True


def test_codec_round_trip_is_closed_and_does_not_persist_semantic_bodies(
    tmp_path: Path,
) -> None:
    planted_secret = "sk-planted-do-not-persist"
    identity = _identity(prompt=f"task {planted_secret}")
    controller = _controller(tmp_path, identity=identity)
    controller.resume(scope="workflow_correction")
    operation = conversation.conversation_operation_digest(
        identity,
        operation_id="correction-1",
        content_identity={"feedback": planted_secret},
    )
    handle = controller.charge(
        operation_digest=operation,
        opaque_id="non-authorizing-thread-id",
        turns=1,
        context_bytes=10,
        usage=Usage(
            input_tokens=12,
            output_tokens=2,
            cost_usd=0.25,
            cache_creation_tokens=4,
            cache_read_tokens=5,
        ),
    )

    encoded = conversation.ConversationHandleCodec.encode(handle)
    decoded = conversation.ConversationHandleCodec.decode(encoded)
    assert decoded == handle
    encoded_usage = json.loads(encoded)["usage"]
    assert encoded_usage["cache_creation_tokens"] == 4
    assert encoded_usage["cache_read_tokens"] == 5
    assert decoded.usage.cache_creation_tokens == 4
    assert decoded.usage.cache_read_tokens == 5
    persisted = next(
        path.read_bytes()
        for path in (tmp_path / "conversations").glob("*.json")
    )
    assert planted_secret.encode() not in persisted
    assert b"response_schema" not in persisted
    assert b"api_key" not in persisted

    raw = json.loads(encoded)
    raw["api_key"] = planted_secret
    with pytest.raises(
        conversation.ConversationStoreError,
        match="closed schema",
    ):
        conversation.ConversationHandleCodec.decode(
            json.dumps(raw).encode()
        )


def test_codec_decodes_legacy_usage_without_cache_counters(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    identity = controller.identity
    controller.resume(scope="workflow_correction")
    handle = controller.charge(
        operation_digest=conversation.conversation_operation_digest(
            identity,
            operation_id="legacy-usage",
            content_identity={"feedback": "legacy"},
        ),
        opaque_id="legacy-thread-id",
        turns=1,
        context_bytes=10,
        usage=Usage(input_tokens=3, output_tokens=2, cost_usd=0.25),
    )
    legacy = conversation.ConversationHandleCodec.to_mapping(handle)
    assert set(legacy["usage"]) == {
        "input_tokens",
        "output_tokens",
        "cost_usd",
    }

    restored = conversation.ConversationHandleCodec.from_mapping(legacy)

    assert restored == handle
    assert restored.usage.cache_creation_tokens == 0
    assert restored.usage.cache_read_tokens == 0


def test_identity_binds_prompt_tool_profile_policy_runtime_and_context() -> None:
    original = _identity()
    assert _identity(prompt="different").fingerprint != original.fingerprint
    assert (
        _identity(tool_digest=_digest("b")).fingerprint
        != original.fingerprint
    )
    assert (
        _identity(profile=_RETRY_PROFILE).fingerprint
        != original.fingerprint
    )


def test_controller_rejects_unsupported_workflow_continuity_before_state_use(
    tmp_path: Path,
) -> None:
    store = conversation.ConversationHandleStore(tmp_path / "conversations")
    with pytest.raises(
        conversation.ConversationContinuityUnsupported,
        match="retry_only",
    ):
        conversation.ConversationController(
            store=store,
            key=_key(),
            identity=_identity(profile=_RETRY_PROFILE),
            mode="workflow_bound",
            feature_profile=_RETRY_PROFILE,
        )
    assert store.load(_key()) is None


@pytest.mark.parametrize(
    ("backend", "model", "supported"),
    [
        ("codex", "gpt-5.6-sol", True),
        ("claude", "aws/anthropic/bedrock-claude-opus-5", True),
        ("kimi", "kimi-k3.1-pro", False),
        ("openai_compat", "unqualified-model", False),
    ],
)
def test_builtin_route_workflow_continuity_admission_matches_profile(
    tmp_path: Path,
    backend: str,
    model: str,
    supported: bool,
) -> None:
    profile = feature_profile_for_route(backend, model=model)
    identity = conversation.build_conversation_identity(
        role="implementer",
        backend=backend,
        model=model,
        prompt_abi="assay.prompt.v2",
        conversation_mode="workflow_bound",
        route_identity={"backend": backend, "model": model},
        prompt_content="stable task",
        response_schema=None,
        tool_authority_digest=_digest("a"),
        feature_profile=profile,
        policy_identity={},
        runtime_identity={},
        trusted_context_identity={},
    )

    kwargs = {
        "store": conversation.ConversationHandleStore(tmp_path / backend),
        "key": _key(),
        "identity": identity,
        "mode": "workflow_bound",
        "feature_profile": profile,
    }
    if supported:
        controller = conversation.ConversationController(**kwargs)
        assert controller.resume(scope="workflow_correction").action == "fresh"
    else:
        with pytest.raises(conversation.ConversationContinuityUnsupported):
            conversation.ConversationController(**kwargs)


def test_none_retry_and_workflow_modes_are_enforced(tmp_path: Path) -> None:
    none_profile = BackendFeatureSet(session_continuity="none")
    none = _controller(
        tmp_path / "none",
        identity=_identity(
            profile=none_profile,
            conversation_mode="none",
        ),
        profile=none_profile,
        mode="none",
    )
    assert none.resume(scope="workflow_correction").action == "fresh"

    retry = _controller(
        tmp_path / "retry",
        identity=_identity(
            profile=_RETRY_PROFILE,
            conversation_mode="retry_only",
        ),
        profile=_RETRY_PROFILE,
        mode="retry_only",
    )
    retry.resume(scope="transport_retry")
    retry.charge(
        operation_digest=_digest("c"),
        opaque_id="retry-thread",
        turns=1,
    )
    assert retry.resume(scope="transport_retry").action == "reuse"
    with pytest.raises(
        conversation.ConversationReuseRejected,
        match="workflow correction",
    ):
        retry.resume(scope="workflow_correction")

    workflow = _controller(tmp_path / "workflow")
    workflow.resume(scope="workflow_correction")
    workflow.charge(
        operation_digest=_digest("d"),
        opaque_id="workflow-thread",
        turns=1,
    )
    assert workflow.resume(scope="workflow_correction").action == "reuse"


def test_cross_fingerprint_and_mode_reuse_are_rejected(tmp_path: Path) -> None:
    original = _controller(tmp_path)
    original.resume(scope="workflow_correction")
    original.charge(
        operation_digest=_digest("d"),
        opaque_id="thread-1",
        turns=1,
    )

    changed = _controller(
        tmp_path,
        identity=_identity(prompt="different stable prompt"),
    )
    with pytest.raises(
        conversation.ConversationReuseRejected,
        match="fingerprint mismatch",
    ):
        changed.resume(scope="workflow_correction")

    changed_mode = _controller(
        tmp_path,
        mode="retry_only",
    )
    with pytest.raises(
        conversation.ConversationReuseRejected,
        match="fingerprint mismatch",
    ):
        changed_mode.resume(scope="transport_retry")


def test_idempotent_charge_survives_controller_reconstruction(
    tmp_path: Path,
) -> None:
    identity = _identity()
    operation = conversation.conversation_operation_digest(
        identity,
        operation_id="build-correction-1",
        content_identity={"build_output_digest": _digest("e")},
    )
    first = _controller(tmp_path, identity=identity)
    first.resume(scope="workflow_correction")
    charged = first.charge(
        operation_digest=operation,
        opaque_id="thread-1",
        turns=2,
        context_bytes=100,
        usage=Usage(input_tokens=20, output_tokens=10, cost_usd=1.5),
    )

    reconstructed = _controller(tmp_path, identity=identity)
    replayed = reconstructed.charge(
        operation_digest=operation,
        opaque_id="thread-1",
        turns=200,
        context_bytes=10_000,
        usage=Usage(input_tokens=2000, output_tokens=1000, cost_usd=150),
    )

    assert replayed == charged
    assert replayed.turns_used == 2
    assert replayed.usage.cost_usd == 1.5


def test_begin_marker_survives_restart_and_requires_exact_recovery(
    tmp_path: Path,
) -> None:
    identity = _identity()
    operation = conversation.conversation_operation_digest(
        identity,
        operation_id="provider-call-1",
        content_identity={"request": _digest("e")},
    )
    controller = _controller(tmp_path, identity=identity)
    controller.resume(scope="workflow_correction")

    begun = controller.begin_operation(
        operation_digest=operation,
        started_at="2026-07-28T12:00:00Z",
    )
    assert begun.pending_operation is not None
    assert begun.pending_operation.operation_digest == operation
    assert controller.begin_operation(
        operation_digest=operation,
        started_at="ignored-on-idempotent-replay",
    ) == begun

    restarted = _controller(tmp_path, identity=identity)
    with pytest.raises(
        conversation.ConversationOperationRecoveryRequired,
        match="unresolved pending",
    ):
        restarted.resume(scope="workflow_correction")
    with pytest.raises(
        conversation.ConversationOperationRecoveryRequired,
    ):
        restarted.begin_operation(
            operation_digest=_digest("f"),
            started_at="2026-07-28T12:01:00Z",
        )

    request = restarted.transport_request(operation_digest=operation)
    assert request.action == "fresh"
    assert request.operation_digest == operation
    assert request.identity_fingerprint == identity.fingerprint


def test_begin_commit_is_atomic_idempotent_and_clears_pending(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.resume(scope="workflow_correction")
    operation = _digest("e")
    controller.begin_operation(
        operation_digest=operation,
        started_at="2026-07-28T12:00:00Z",
    )

    committed = controller.commit_operation(
        operation_digest=operation,
        opaque_id="thread-1",
        turns=2,
        context_bytes=100,
        usage=Usage(input_tokens=20, output_tokens=10, cost_usd=1.5),
    )
    replayed = _controller(tmp_path).commit_operation(
        operation_digest=operation,
        opaque_id="different-ignored-id",
        turns=200,
        context_bytes=10_000,
        usage=Usage(input_tokens=2000, output_tokens=1000, cost_usd=150),
    )

    assert committed.pending_operation is None
    assert replayed == committed
    assert replayed.turns_used == 2
    assert replayed.usage.cost_usd == 1.5


def test_indeterminate_pending_fails_closed_until_evidence_resolution(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.resume(scope="workflow_correction")
    operation = _digest("e")
    controller.begin_operation(
        operation_digest=operation,
        started_at="2026-07-28T12:00:00Z",
    )
    marked = controller.mark_pending_indeterminate(
        operation_digest=operation,
    )
    assert marked.pending_operation is not None
    assert marked.pending_operation.state == "indeterminate"

    with pytest.raises(
        conversation.ConversationOperationRecoveryRequired,
    ):
        controller.transport_request(operation_digest=operation)
    with pytest.raises(
        conversation.ConversationOperationRecoveryRequired,
    ):
        controller.commit_operation(operation_digest=operation)
    with pytest.raises(ValueError, match="resolution_evidence_digest"):
        controller.resolve_pending_not_started(
            operation_digest=operation,
            resolution_evidence_digest="operator-says-safe",
            resolved_at="2026-07-28T12:02:00Z",
            actor="operator:test",
        )

    resolved = controller.resolve_pending_not_started(
        operation_digest=operation,
        resolution_evidence_digest=_digest("a"),
        resolved_at="2026-07-28T12:02:00Z",
        actor="operator:test",
    )
    assert resolved.pending_operation is None
    assert len(resolved.recovery_resolutions) == 1
    resolution = resolved.recovery_resolutions[0]
    assert resolution.operation_digest == operation
    assert resolution.evidence_digest == _digest("a")
    assert resolution.resolved_at == "2026-07-28T12:02:00Z"
    assert resolution.actor == "operator:test"
    restarted = _controller(tmp_path)
    assert restarted.load().recovery_resolutions == (resolution,)
    assert controller.resume(scope="workflow_correction").action == "fresh"

    controller.begin_operation(
        operation_digest=operation,
        started_at="2026-07-28T12:03:00Z",
    )
    with pytest.raises(
        conversation.ConversationOperationConflict,
        match="replayed recovery resolution",
    ):
        controller.resolve_pending_not_started(
            operation_digest=operation,
            resolution_evidence_digest=_digest("a"),
            resolved_at="2026-07-28T12:02:00Z",
            actor="operator:test",
        )
    with pytest.raises(
        conversation.ConversationOperationConflict,
        match="conflicting recovery resolution",
    ):
        controller.resolve_pending_not_started(
            operation_digest=operation,
            resolution_evidence_digest=_digest("b"),
            resolved_at="2026-07-28T12:04:00Z",
            actor="operator:other",
        )


def test_stale_fresh_fallback_preserves_every_cumulative_ledger(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.resume(scope="workflow_correction")
    controller.charge(
        operation_digest=_digest("e"),
        opaque_id="expired-thread",
        turns=3,
        context_bytes=80,
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.75),
    )

    decision = controller.resume(
        scope="workflow_correction",
        provider_reports_stale=True,
        allow_stale_fallback=True,
    )

    assert decision.action == "fresh"
    assert decision.handle.opaque_id is None
    assert decision.handle.turns_used == 3
    assert decision.handle.context_bytes == 80
    assert decision.handle.usage == Usage(
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.75,
    )
    assert controller.remaining_budget() == (
        conversation.RemainingConversationBudget(
            turns=None,
            context_bytes=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
        )
    )


def test_budget_crossing_is_persisted_before_rejection(tmp_path: Path) -> None:
    limits = conversation.ConversationLimits(
        max_turns=2,
        max_context_bytes=20,
        max_total_tokens=10,
        max_cost_usd=1.0,
    )
    controller = _controller(tmp_path, limits=limits)
    controller.resume(scope="workflow_correction")

    with pytest.raises(
        conversation.ConversationBudgetExceeded,
        match="turns=3 exceeded",
    ):
        controller.charge(
            operation_digest=_digest("f"),
            opaque_id="thread-1",
            turns=3,
            context_bytes=25,
            usage=Usage(
                input_tokens=7,
                output_tokens=5,
                cost_usd=1.25,
            ),
        )

    persisted = controller.load()
    assert persisted is not None
    assert persisted.turns_used == 3
    assert persisted.context_bytes == 25
    assert persisted.usage.cost_usd == 1.25
    with pytest.raises(conversation.ConversationBudgetExceeded):
        controller.resume(scope="workflow_correction")


def test_orphan_staged_record_is_discarded_during_recovery(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    initial = controller.resume(scope="workflow_correction").handle
    store = controller.store
    record_path, _lock_path, staged_path = store._paths(_key())  # noqa: SLF001
    assert record_path.exists()
    staged_path.write_bytes(b"uncommitted future image")

    recovered = store.load(_key())

    assert recovered == initial
    assert not staged_path.exists()


def test_failed_atomic_replace_preserves_prior_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    initial = controller.resume(scope="workflow_correction").handle

    def fail_replace(_source, _target, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(_durable_state.os, "replace", fail_replace)
    with pytest.raises(
        conversation.ConversationStoreError,
        match="atomically",
    ):
        controller.charge(
            operation_digest=_digest("1"),
            opaque_id="thread-1",
            turns=1,
        )

    assert controller.store.load(_key()) == initial
