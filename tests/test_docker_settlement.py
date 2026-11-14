# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hermetic adversarial tests for daemon-side Docker settlement."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytest

from crucible.common.docker_settlement import (
    DockerContainerInspection,
    DockerDaemonIdentity,
    DockerSettlementArmStatus,
    DockerSettlementBlockedError,
    DockerSettlementIssueCode,
    DockerSettlementRegistrationError,
    DockerSettlementSession,
    DockerSettlementStatus,
)


_A = "a" * 64
_B = "b" * 64


def _identity(*, engine: str = "engine-a", generation: str = "daemon-1"):
    return DockerDaemonIdentity.from_components(
        endpoint="unix:///run/docker.sock",
        engine_id=engine,
        generation=generation,
        configuration={"root": "/var/lib/docker", "api": "1.51"},
    )


class _FakeControlPlane:
    def __init__(self) -> None:
        self.identity = _identity()
        self.identity_error: Exception | None = None
        self.identity_overrides: list[DockerDaemonIdentity | Exception] = []
        self.containers: dict[str, dict[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self.remove_failures: set[str] = set()
        self.remove_calls: list[str] = []
        self.list_overrides: list[Sequence[str] | Exception] = []
        self.inspect_overrides: dict[
            str, list[DockerContainerInspection | None | Exception]
        ] = {}
        self.late_container: tuple[str, dict[str, str]] | None = None

    def daemon_identity(self, *, deadline_monotonic: float):
        if self.identity_overrides:
            result = self.identity_overrides.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if self.identity_error is not None:
            raise self.identity_error
        return self.identity

    def list_container_ids(
        self,
        *,
        labels: Mapping[str, str],
        limit: int,
        deadline_monotonic: float,
    ) -> Sequence[str]:
        if self.list_overrides:
            result = self.list_overrides.pop(0)
            if isinstance(result, Exception):
                raise result
            return result[:limit]
        return tuple(sorted(
            container_id
            for container_id, actual in self.containers.items()
            if all(actual.get(key) == value for key, value in labels.items())
        ))[:limit]

    def inspect_container(
        self,
        reference: str,
        *,
        deadline_monotonic: float,
    ) -> DockerContainerInspection | None:
        overrides = self.inspect_overrides.get(reference)
        if overrides:
            result = overrides.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        canonical = self.aliases.get(reference, reference)
        labels = self.containers.get(canonical)
        if labels is None:
            return None
        return DockerContainerInspection.from_parts(canonical, labels)

    def remove_container(
        self,
        container_id: str,
        *,
        deadline_monotonic: float,
    ) -> None:
        self.remove_calls.append(container_id)
        if container_id in self.remove_failures:
            raise RuntimeError("daemon refused removal")
        self.containers.pop(container_id, None)
        if self.late_container is not None:
            late_id, late_labels = self.late_container
            self.late_container = None
            self.containers[late_id] = late_labels


def _session(
    control: _FakeControlPlane,
    *,
    attempts: int = 8,
    max_container_ids: int = 256,
):
    return DockerSettlementSession(
        control,
        max_poll_attempts=attempts,
        max_container_ids=max_container_ids,
        poll_interval_s=0,
        monotonic=lambda: 10.0,
        sleep=lambda _seconds: None,
    )


def _arm(session: DockerSettlementSession):
    record = session.arm(
        run_generation="run-a",
        arm_generation="arm-a",
        deadline_monotonic=20.0,
    )
    assert record.status == DockerSettlementArmStatus.ARMED
    return record


def _settle(session: DockerSettlementSession):
    return session.settle(
        process_tree_proven=True,
        process_proof_ref="process-proof:run-a",
        deadline_monotonic=20.0,
    )


@pytest.mark.parametrize("poll_interval", [math.nan, math.inf, -math.inf])
def test_nonfinite_poll_interval_is_rejected_before_arm(poll_interval: float):
    with pytest.raises(ValueError, match="finite and non-negative"):
        DockerSettlementSession(
            _FakeControlPlane(),
            poll_interval_s=poll_interval,
        )


def test_settlement_registers_short_reference_and_removes_by_canonical_id():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.aliases["short-a"] = _A

    registered = session.register_container(
        "short-a", deadline_monotonic=20.0,
    )
    terminal = _settle(session)

    assert registered.container_id == _A
    assert terminal.status == DockerSettlementStatus.SETTLED
    assert terminal.known_container_ids == (_A,)
    assert terminal.removed_container_ids == (_A,)
    assert control.remove_calls == [_A]
    assert terminal.clean_observations == 2
    assert len(arm.initial_inventories) == 2
    assert len(terminal.inventory_trace) == terminal.poll_attempts


@pytest.mark.parametrize(
    ("replacement", "changed_field"),
    [
        (_identity(engine="engine-b"), "identity"),
        (_identity(generation="daemon-2"), "restart generation"),
    ],
)
def test_daemon_identity_drift_or_restart_is_uncertain_without_removal(
    replacement: DockerDaemonIdentity,
    changed_field: str,
):
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    session.register_container(_A, deadline_monotonic=20.0)
    control.identity = replacement

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN, changed_field
    assert DockerSettlementIssueCode.DAEMON_IDENTITY_DRIFT in {
        issue.code for issue in terminal.issues
    }
    assert control.remove_calls == []
    assert _A in control.containers


def test_relabelled_registered_container_is_removed_but_remains_uncertain():
    control = _FakeControlPlane()
    session = _session(control, attempts=3)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    session.register_container(_A, deadline_monotonic=20.0)
    control.containers[_A] = {
        "io.crucible.managed": "true",
        "io.crucible.arm-generation": "somebody-else",
    }

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.LABEL_DRIFT in {
        issue.code for issue in terminal.issues
    }
    assert control.remove_calls == [_A]
    assert _A not in control.containers


def test_late_created_matching_container_is_discovered_and_removed():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.late_container = (_B, arm.labels)

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.SETTLED
    assert terminal.known_container_ids == (_A, _B)
    assert terminal.removed_container_ids == (_A, _B)
    assert control.remove_calls == [_A, _B]


def test_removal_failure_retains_uncertain_terminal():
    control = _FakeControlPlane()
    session = _session(control, attempts=3)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.remove_failures.add(_A)

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.REMOVE_FAILED in {
        issue.code for issue in terminal.issues
    }
    assert DockerSettlementIssueCode.ABSENCE_NOT_PROVEN in {
        issue.code for issue in terminal.issues
    }
    assert _A in control.containers


def test_daemon_query_failure_is_uncertain_and_does_not_remove_by_guess():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    session.register_container(_A, deadline_monotonic=20.0)
    control.identity_error = RuntimeError("daemon connection closed")

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.DAEMON_QUERY_FAILED in {
        issue.code for issue in terminal.issues
    }
    assert control.remove_calls == []
    assert _A in control.containers


def test_daemon_restart_during_absence_poll_is_uncertain():
    control = _FakeControlPlane()
    original = control.identity
    restarted = _identity(generation="daemon-2")
    # Arm identity, arm confirmation, settle identity, first-poll identity,
    # then a changed restart generation before the first clean observation.
    control.identity_overrides.extend([
        original,
        original,
        original,
        original,
        restarted,
        restarted,
    ])
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert terminal.removed_container_ids == (_A,)
    assert DockerSettlementIssueCode.DAEMON_IDENTITY_DRIFT in {
        issue.code for issue in terminal.issues
    }


def test_list_query_failure_is_uncertain():
    control = _FakeControlPlane()
    session = _session(control)
    _arm(session)
    control.list_overrides.append(RuntimeError("list unavailable"))

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.LIST_QUERY_FAILED in {
        issue.code for issue in terminal.issues
    }


def test_inspect_query_failure_is_uncertain():
    control = _FakeControlPlane()
    session = _session(control, attempts=3)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.inspect_overrides[_A] = [RuntimeError("inspect unavailable")]

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INSPECT_QUERY_FAILED in {
        issue.code for issue in terminal.issues
    }


def test_canonical_inspect_mismatch_is_uncertain():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.inspect_overrides[_A] = [
        DockerContainerInspection.from_parts(_B, arm.labels),
    ]

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.CANONICAL_ID_MISMATCH in {
        issue.code for issue in terminal.issues
    }


def test_noncanonical_list_id_is_uncertain():
    control = _FakeControlPlane()
    session = _session(control)
    _arm(session)
    control.list_overrides.extend([("short-id",), (), ()])

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVALID_CONTAINER_ID in {
        issue.code for issue in terminal.issues
    }


def test_inventory_limit_is_bounded_and_uncertain():
    control = _FakeControlPlane()
    session = _session(control, max_container_ids=1)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.containers[_B] = arm.labels

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVENTORY_LIMIT_EXCEEDED in {
        issue.code for issue in terminal.issues
    }
    assert control.remove_calls == [_A, _B]
    assert control.containers == {}


def test_inspect_list_disagreement_is_uncertain_even_if_later_absent():
    control = _FakeControlPlane()
    session = _session(control, attempts=3)
    _arm(session)
    # The arm consumes two empty inventories. The next list reports an object
    # that inspect authoritatively says is absent; later passes converge.
    control.list_overrides.extend([(_A,), (), ()])

    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVENTORY_DISAGREEMENT in {
        issue.code for issue in terminal.issues
    }
    assert terminal.clean_observations == 2


def test_uncertain_terminal_blocks_rearm_and_survives_session_restore():
    control = _FakeControlPlane()
    session = _session(control, attempts=2)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    control.remove_failures.add(_A)
    terminal = _settle(session)
    assert terminal.status == DockerSettlementStatus.UNCERTAIN

    with pytest.raises(DockerSettlementBlockedError, match="uncertain"):
        session.arm(
            run_generation="run-b",
            arm_generation="arm-b",
            deadline_monotonic=20.0,
        )

    restored = DockerSettlementSession(
        control,
        max_poll_attempts=2,
        required_clean_observations=2,
        poll_interval_s=0,
        monotonic=lambda: 10.0,
        prior_terminal=terminal,
    )
    with pytest.raises(DockerSettlementBlockedError, match="uncertain"):
        restored.arm(
            run_generation="run-c",
            arm_generation="arm-c",
            deadline_monotonic=20.0,
        )


@pytest.mark.parametrize("deadline", [math.nan, math.inf, -math.inf])
def test_nonfinite_arm_deadline_is_typed_uncertain_and_restart_blocking(
    deadline: float,
):
    control = _FakeControlPlane()
    session = _session(control)

    arm = session.arm(
        run_generation="run-invalid-deadline",
        arm_generation="arm-invalid-deadline",
        deadline_monotonic=deadline,
    )

    assert arm.status == DockerSettlementArmStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVALID_DEADLINE in {
        issue.code for issue in arm.issues
    }
    assert arm.daemon_identity is None
    assert arm.initial_inventories == ()
    restored = DockerSettlementSession(
        control,
        monotonic=lambda: 10.0,
        prior_arm=arm,
    )
    with pytest.raises(DockerSettlementBlockedError):
        restored.arm(
            run_generation="run-after-restart",
            arm_generation="arm-after-restart",
            deadline_monotonic=20.0,
        )


def test_active_arm_record_blocks_rearm_after_process_restart():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)

    restored = DockerSettlementSession(
        control,
        monotonic=lambda: 10.0,
        prior_arm=arm,
    )
    assert restored.blocked
    with pytest.raises(DockerSettlementBlockedError):
        restored.arm(
            run_generation="run-after-crash",
            arm_generation="arm-after-crash",
            deadline_monotonic=20.0,
        )


@pytest.mark.parametrize("deadline", [math.nan, math.inf])
def test_nonfinite_settlement_deadline_is_uncertain(deadline: float):
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels
    session.register_container(_A, deadline_monotonic=20.0)

    terminal = session.settle(
        process_tree_proven=True,
        process_proof_ref="process-proof:run-a",
        deadline_monotonic=deadline,
    )

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVALID_DEADLINE in {
        issue.code for issue in terminal.issues
    }
    assert control.remove_calls == []
    assert _A in control.containers


def test_nonfinite_registration_deadline_is_retained_as_uncertain():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels

    with pytest.raises(DockerSettlementRegistrationError):
        session.register_container(_A, deadline_monotonic=math.nan)
    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.INVALID_DEADLINE in {
        issue.code for issue in terminal.issues
    }
    assert terminal.removed_container_ids == (_A,)


def test_missing_process_proof_is_uncertain_after_container_cleanup():
    control = _FakeControlPlane()
    session = _session(control)
    arm = _arm(session)
    control.containers[_A] = arm.labels

    terminal = session.settle(
        process_tree_proven=False,
        process_proof_ref=None,
        deadline_monotonic=20.0,
    )

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.PROCESS_PROOF_FAILED in {
        issue.code for issue in terminal.issues
    }
    assert terminal.removed_container_ids == (_A,)


def test_failed_registration_is_retained_in_terminal_evidence():
    control = _FakeControlPlane()
    session = _session(control)
    _arm(session)

    with pytest.raises(DockerSettlementRegistrationError):
        session.register_container("missing", deadline_monotonic=20.0)
    terminal = _settle(session)

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert DockerSettlementIssueCode.REGISTRATION_FAILED in {
        issue.code for issue in terminal.issues
    }
