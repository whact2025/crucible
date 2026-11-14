# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from crucible.workflows import _control


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _identity(
    *,
    run: str = "1",
    route: str = "2",
    authority: str = "3",
) -> _control.ControlIdentity:
    return _control.ControlIdentity(
        run_fingerprint=_digest(run),
        route_fingerprint=_digest(route),
        authority_fingerprint=_digest(authority),
    )


def _controller(
    tmp_path: Path,
    *,
    identity: _control.ControlIdentity | None = None,
) -> _control.WorkflowControlController:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return _control.WorkflowControlController(
        store=_control.WorkflowControlStore.for_run(run_dir),
        identity=identity or _identity(),
    )


def test_control_controller_context_releases_held_store_descriptor(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    with controller as entered:
        assert entered is controller
        assert controller.initialize().identity == _identity()

    assert controller.store.closed is True
    controller.close()
    with pytest.raises(_control.WorkflowControlStoreError, match="closed"):
        controller.load()


def test_restart_preserves_heartbeat_pause_progress_and_resume(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path)
    initial = first.initialize()
    heartbeat = first.heartbeat(
        owner="worker-1",
        at="2026-07-28T12:00:00Z",
        expected_sequence=0,
    )
    assert heartbeat.control_revision == initial.control_revision
    assert heartbeat.record_revision == 1

    paused = first.pause(expected_control_revision=0)
    progressed = first.set_progress(
        blocker="Waiting for a qualified route",
        next_action="Run the route diagnostic",
        actor="worker-1",
        updated_at="2026-07-28T12:01:00Z",
        expected_control_revision=paused.control_revision,
    )

    restarted = _controller(tmp_path)
    recovered = restarted.load()
    assert recovered == progressed
    assert recovered is not None
    assert recovered.desired_state == "paused"
    assert recovered.heartbeat_sequence == 1
    assert recovered.blocker == "Waiting for a qualified route"
    assert recovered.next_action == "Run the route diagnostic"

    resumed = restarted.resume(
        expected_control_revision=recovered.control_revision,
    )
    assert resumed.desired_state == "running"
    with pytest.raises(_control.StaleControlRevision):
        restarted.pause(expected_control_revision=0)


def test_exact_run_route_and_authority_identity_is_enforced(
    tmp_path: Path,
) -> None:
    original = _controller(tmp_path)
    original.initialize()

    for changed in (
        _identity(run="4"),
        _identity(route="4"),
        _identity(authority="4"),
    ):
        with pytest.raises(_control.WorkflowControlIdentityMismatch):
            _controller(tmp_path, identity=changed).initialize()


def test_concurrent_steering_writers_have_one_monotonic_winner(
    tmp_path: Path,
) -> None:
    first = _controller(tmp_path)
    second = _controller(tmp_path)
    first.initialize()
    barrier = threading.Barrier(3)
    winners: list[str] = []
    stale: list[str] = []

    def write(controller, instruction: str) -> None:
        barrier.wait()
        try:
            controller.steer(
                instruction=instruction,
                actor="operator",
                created_at="2026-07-28T12:00:00Z",
                expected_steering_revision=0,
            )
        except _control.StaleSteeringRevision:
            stale.append(instruction)
        else:
            winners.append(instruction)

    threads = (
        threading.Thread(target=write, args=(first, "choose-a")),
        threading.Thread(target=write, args=(second, "choose-b")),
    )
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert len(winners) == 1
    assert len(stale) == 1
    state = first.load()
    assert state is not None
    assert state.steering_revision == 1
    assert state.steering is not None
    assert state.steering.instruction == winners[0]


def test_stale_and_replayed_approvals_fail_closed(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    controller.initialize()
    controller.steer(
        instruction="Apply the narrow patch",
        actor="operator",
        created_at="2026-07-28T12:00:00Z",
        expected_steering_revision=0,
    )
    request = controller.request_approval(
        request_id="approval-1",
        action="write source",
        resource_digest=_digest("4"),
        rationale="Required to complete the accepted slice",
        requested_by="implementer",
        requested_at="2026-07-28T12:01:00Z",
        expected_steering_revision=1,
    )
    assert controller.request_approval(
        request_id="approval-1",
        action="write source",
        resource_digest=_digest("4"),
        rationale="Required to complete the accepted slice",
        requested_by="implementer",
        requested_at="2026-07-28T12:01:00Z",
        expected_steering_revision=1,
    ) == request
    with pytest.raises(_control.ApprovalConflict):
        controller.request_approval(
            request_id="approval-1",
            action="delete repository",
            resource_digest=_digest("4"),
            rationale="Different replay",
            requested_by="implementer",
            requested_at="2026-07-28T12:01:00Z",
            expected_steering_revision=1,
        )

    controller.steer(
        instruction="Apply a different patch",
        actor="operator",
        created_at="2026-07-28T12:02:00Z",
        expected_steering_revision=1,
    )
    with pytest.raises(_control.StaleSteeringRevision):
        controller.decide_approval(
            decision_id="decision-1",
            request_id=request.request_id,
            request_fingerprint=request.fingerprint,
            decision="approved",
            reason="Looks good",
            decided_by="operator",
            decided_at="2026-07-28T12:03:00Z",
            expected_steering_revision=1,
        )

    current = controller.load()
    assert current is not None
    request2 = controller.request_approval(
        request_id="approval-2",
        action="write source",
        resource_digest=_digest("5"),
        rationale="Matches steering revision two",
        requested_by="implementer",
        requested_at="2026-07-28T12:04:00Z",
        expected_steering_revision=2,
    )
    decision = controller.decide_approval(
        decision_id="decision-2",
        request_id=request2.request_id,
        request_fingerprint=request2.fingerprint,
        decision="approved",
        reason="Exact request accepted",
        decided_by="operator",
        decided_at="2026-07-28T12:05:00Z",
        expected_steering_revision=2,
    )
    assert controller.decide_approval(
        decision_id="decision-2",
        request_id=request2.request_id,
        request_fingerprint=request2.fingerprint,
        decision="approved",
        reason="Exact request accepted",
        decided_by="operator",
        decided_at="2026-07-28T12:05:00Z",
        expected_steering_revision=2,
    ) == decision
    with pytest.raises(_control.ApprovalConflict):
        controller.decide_approval(
            decision_id="decision-3",
            request_id=request2.request_id,
            request_fingerprint=request2.fingerprint,
            decision="denied",
            reason="Conflicting replay",
            decided_by="operator",
            decided_at="2026-07-28T12:06:00Z",
            expected_steering_revision=2,
        )
    request3 = controller.request_approval(
        request_id="approval-3",
        action="write another source file",
        resource_digest=_digest("6"),
        rationale="A distinct request under the same steering revision",
        requested_by="implementer",
        requested_at="2026-07-28T12:07:00Z",
        expected_steering_revision=2,
    )
    with pytest.raises(
        _control.ApprovalConflict,
        match="decision ID is already bound",
    ):
        controller.decide_approval(
            decision_id="decision-2",
            request_id=request3.request_id,
            request_fingerprint=request3.fingerprint,
            decision="approved",
            reason="Must not reuse a globally unique decision ID",
            decided_by="operator",
            decided_at="2026-07-28T12:08:00Z",
            expected_steering_revision=2,
        )


def test_cancellation_is_durable_before_fanout_and_reaches_every_child(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    initial = controller.initialize()
    registry = _control.CancellationRegistry(controller)
    calls: list[str] = []

    def successful() -> None:
        state = controller.load()
        assert state is not None
        assert state.desired_state == "cancelled"
        calls.append("success")

    def failing() -> None:
        state = controller.load()
        assert state is not None
        assert state.desired_state == "cancelled"
        calls.append("failure")
        raise RuntimeError("provider cancel failed")

    registry.register(
        participant_id="child-success",
        kind="readonly-child",
        registration_digest=_digest("4"),
        registered_at="2026-07-28T12:00:00Z",
        cancel=successful,
    )
    registry.register(
        participant_id="child-failure",
        kind="writer-child",
        registration_digest=_digest("5"),
        registered_at="2026-07-28T12:00:00Z",
        cancel=failing,
    )
    cancelled = registry.request_cancel(
        expected_control_revision=initial.control_revision,
        requested_at="2026-07-28T12:01:00Z",
    )

    assert sorted(calls) == ["failure", "success"]
    assert cancelled.desired_state == "cancelled"
    statuses = {
        participant.participant_id: participant.state
        for participant in cancelled.participants
    }
    assert statuses == {
        "child-success": "cancel_signaled",
        "child-failure": "cancel_failed",
    }

    retry_calls: list[str] = []
    restarted = _control.CancellationRegistry(_controller(tmp_path))
    recovered = restarted.register(
        participant_id="child-failure",
        kind="writer-child",
        registration_digest=_digest("5"),
        registered_at="2026-07-28T12:02:00Z",
        cancel=lambda: retry_calls.append("retried"),
    )
    assert retry_calls == ["retried"]
    assert recovered.state == "cancel_signaled"


def test_post_routing_binding_projects_pause_steering_and_cancel(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "bound-run"
    run_dir.mkdir()
    cancel_event = threading.Event()
    binding = _control.bind_workflow_control(
        run_dir,
        identity=_identity(),
        owner="tier-worker",
        heartbeat_at="2026-07-28T12:00:00Z",
        cancel_event=cancel_event,
    )
    calls: list[str] = []
    binding.cancellations.register(
        participant_id="backend-1",
        kind="provider",
        registration_digest=_digest("4"),
        registered_at="2026-07-28T12:00:00Z",
        cancel=lambda: calls.append("provider-cancelled"),
    )
    state = binding.controller.load()
    assert state is not None
    steered = binding.controller.steer(
        instruction="Stop after the current validation step",
        actor="operator",
        created_at="2026-07-28T12:01:00Z",
        expected_steering_revision=0,
    )
    paused = binding.controller.pause(
        expected_control_revision=steered.control_revision,
    )
    checkpoint = binding.checkpoint(
        observed_steering_revision=0,
        at="2026-07-28T12:02:00Z",
    )
    assert checkpoint.should_pause is True
    assert checkpoint.steering is not None
    assert checkpoint.steering.revision == 1

    resumed = binding.controller.resume(
        expected_control_revision=paused.control_revision,
    )
    binding.controller.request_cancel(
        expected_control_revision=resumed.control_revision,
    )
    cancelled = binding.checkpoint(
        observed_steering_revision=checkpoint.steering_revision,
        at="2026-07-28T12:03:00Z",
    )
    assert cancelled.should_cancel is True
    assert cancel_event.is_set()
    assert calls == ["provider-cancelled"]


def test_corruption_symlinks_and_orphan_staged_images_fail_closed(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    initial = controller.initialize()
    store = controller.store
    record_path, _lock_path, staged_path = store._atomic.paths(  # noqa: SLF001
        _control._CONTROL_STORE_KEY,  # noqa: SLF001
    )
    staged_path.write_bytes(b"uncommitted future control image")
    assert store.load() == initial
    assert not staged_path.exists()

    original = record_path.read_bytes()
    envelope = json.loads(original)
    envelope["state"]["desired_state"] = "cancelled"
    record_path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(
        _control.WorkflowControlStoreError,
        match="checksum mismatch",
    ):
        store.load()

    record_path.write_bytes(original)
    record_path.unlink()
    target = tmp_path / "attacker-control.json"
    target.write_bytes(original)
    record_path.symlink_to(target)
    with pytest.raises(_control.WorkflowControlStoreError):
        store.load()


def test_control_store_rejects_symlink_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        _control.WorkflowControlStoreError,
        match="real directory",
    ):
        _control.WorkflowControlStore(link)
