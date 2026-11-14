# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durability and recovery tests for workflow Docker settlement binding."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from crucible.common import docker_workflow_settlement as settlement_module
from crucible.common.docker_settlement import (
    DockerContainerInspection,
    DockerDaemonIdentity,
    DockerSettlementStatus,
)
from crucible.common.process_settlement import (
    CGROUP_SETTLEMENT_SCHEMA,
    ProcessSettlementProof,
)
from crucible.common.docker_workflow_settlement import (
    DOCKER_SETTLEMENT_ARM_ENV,
    DOCKER_SETTLEMENT_DEADLINE_ENV,
    DOCKER_SETTLEMENT_NONCE_ENV,
    DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA,
    DOCKER_WORKFLOW_SETTLEMENT_SCHEMA,
    DockerSettlementBridgeError,
    DockerSettlementRegistrationClient,
    DockerSettlementRegistrationServer,
    DockerWorkflowRecoveryRequired,
    DockerWorkflowSettlement,
    DockerWorkflowSettlementError,
    DockerWorkflowSettlementReceipt,
)


_A = "a" * 64
_B = "b" * 64


def _canonical_test_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _process_settlement_proof(
    *,
    returncode: int = 0,
) -> ProcessSettlementProof:
    admission_sha256 = "sha256:" + _B
    supervisor_pid = 102
    supervisor_start_token = 1002
    supervisor = _canonical_test_json(
        {
            "cgroup_admission_sha256": admission_sha256,
            "error": None,
            "returncode": returncode,
            "schema_version": "crucible.process-supervisor.v3",
            "settled": True,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_token": supervisor_start_token,
            "target_pid": 103,
            "target_start_token": 1003,
        }
    )
    cgroup = _canonical_test_json(
        {
            "admission_outcome": "committed",
            "admission_sha256": admission_sha256,
            "controller_cgroup": "/crucible-test.slice",
            "controller_identity": "1:2:3",
            "controller_pid": 101,
            "controller_start_token": 1001,
            "kill_requested": True,
            "populated": 0,
            "removed": True,
            "schema_version": CGROUP_SETTLEMENT_SCHEMA,
            "supervisor_pid": supervisor_pid,
            "supervisor_start_token": supervisor_start_token,
            "target_identity": "1:2:4",
        }
    )
    return ProcessSettlementProof.compose(
        supervisor_canonical_bytes=supervisor,
        cgroup_canonical_bytes=cgroup,
        effective_returncode=returncode,
    )


def _identity() -> DockerDaemonIdentity:
    return DockerDaemonIdentity.from_components(
        endpoint="unix:///run/docker.sock",
        engine_id="engine-a",
        generation="boot-a:pid-42:start-7",
        configuration={"api": "1.41", "root": "/var/lib/docker"},
    )


class _ControlPlane:
    def __init__(self) -> None:
        self.identity = _identity()
        self.containers: dict[str, dict[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self.identity_calls = 0
        self.deadlines: list[float] = []

    def daemon_identity(self, *, deadline_monotonic: float):
        self.identity_calls += 1
        self.deadlines.append(deadline_monotonic)
        return self.identity

    def list_container_ids(
        self,
        *,
        labels: Mapping[str, str],
        limit: int,
        deadline_monotonic: float,
    ) -> Sequence[str]:
        self.deadlines.append(deadline_monotonic)
        return tuple(
            container_id
            for container_id, actual in sorted(self.containers.items())
            if all(actual.get(key) == value for key, value in labels.items())
        )[:limit]

    def inspect_container(
        self,
        reference: str,
        *,
        deadline_monotonic: float,
    ) -> DockerContainerInspection | None:
        self.deadlines.append(deadline_monotonic)
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
        self.deadlines.append(deadline_monotonic)
        self.containers.pop(container_id, None)


def _arm(tmp_path, control: _ControlPlane) -> DockerWorkflowSettlement:
    return DockerWorkflowSettlement.arm_new(
        state_path=tmp_path / "docker-settlement.json",
        run_generation="run-a",
        arm_generation="arm-a",
        control_plane=control,
        operation_timeout_s=30,
    )


def _settled_authority(
    tmp_path: Path,
) -> tuple[DockerWorkflowSettlement, ProcessSettlementProof]:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    control.containers[_A] = authority.labels
    authority.register_container(_A)
    proof = _process_settlement_proof()
    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref=proof.sha256,
    )
    assert terminal.status is DockerSettlementStatus.SETTLED
    return authority, proof


def test_common_workflow_settlement_import_is_assay_and_provider_free(
    tmp_path: Path,
) -> None:
    source_parent = Path(__file__).resolve().parents[2]
    script = """
import sys

sys.path.insert(0, sys.argv[1])
import crucible.common.docker_workflow_settlement  # noqa: F401

forbidden = sorted(
    name
    for name in sys.modules
    if name == "assay"
    or name.startswith("assay.")
    or name == "anthropic"
    or name.startswith("anthropic.")
    or name == "openai"
    or name.startswith("openai.")
    or name.startswith("crucible.backends.")
    or name.startswith("crucible.workflows.")
)
if forbidden:
    raise SystemExit("eager Docker-settlement imports: " + ", ".join(forbidden))
"""
    result = subprocess.run(
        (sys.executable, "-I", "-B", "-c", script, str(source_parent)),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_arm_is_durable_before_any_container_creation(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)

    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DOCKER_WORKFLOW_SETTLEMENT_SCHEMA
    assert payload["arm"]["status"] == "armed"
    assert payload["terminal"] is None
    assert payload["registrations"] == []
    assert payload["arm_fingerprint"] == authority.arm.fingerprint


def test_arm_rejects_unproved_directory_fsync_before_child_authority(
    tmp_path,
    monkeypatch,
) -> None:
    control = _ControlPlane()
    original_fsync = settlement_module.os.fsync

    def reject_directory_fsync(descriptor: int) -> None:
        metadata = settlement_module.os.fstat(descriptor)
        if settlement_module.stat.S_ISDIR(metadata.st_mode):
            raise OSError("injected directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(settlement_module.os, "fsync", reject_directory_fsync)

    with pytest.raises(
        DockerWorkflowSettlementError,
        match="could not be persisted",
    ):
        _arm(tmp_path, control)

    # The rename can be visible in this process, but it was not proved durable
    # and therefore cannot release container-creation authority.
    assert tmp_path.joinpath("docker-settlement.json").is_file()


def test_registration_persists_canonical_id_and_exact_labels(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    control.containers[_A] = authority.labels
    control.aliases["short-a"] = _A

    observed = authority.register_container("short-a")

    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert observed.container_id == _A
    assert payload["registrations"] == [observed.to_payload()]


def test_settled_terminal_allows_the_next_arm(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    assert authority.terminal_fingerprint is None
    control.containers[_A] = authority.labels
    authority.register_container(_A)

    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "1" * 64,
    )
    assert terminal.status == DockerSettlementStatus.SETTLED
    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert payload["terminal"] == terminal.to_payload()
    assert authority.terminal_fingerprint == payload["terminal_fingerprint"]

    replacement = DockerWorkflowSettlement.arm_new(
        state_path=authority.state_path,
        run_generation="run-b",
        arm_generation="arm-b",
        control_plane=control,
    )
    assert replacement.arm.run_generation == "run-b"


def test_failed_terminal_persistence_exposes_no_terminal_fingerprint(
    tmp_path,
    monkeypatch,
) -> None:
    authority = _arm(tmp_path, _ControlPlane())
    proof = _process_settlement_proof()
    original_write = settlement_module._write_strict_atomic_json

    def fail_terminal(path, payload):
        if payload["terminal"] is not None:
            raise OSError("injected terminal fsync failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        settlement_module,
        "_write_strict_atomic_json",
        fail_terminal,
    )

    with pytest.raises(
        DockerWorkflowSettlementError,
        match="could not be persisted",
    ):
        authority.settle(
            process_tree_proven=True,
            process_proof_ref=proof.sha256,
        )

    assert authority.terminal is not None
    assert authority.terminal_fingerprint is None
    with pytest.raises(
        DockerWorkflowSettlementError,
        match="clean durable terminal",
    ):
        authority.settlement_receipt(process_settlement=proof)


def test_unterminated_arm_blocks_before_the_next_daemon_query(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    calls_after_arm = control.identity_calls

    with pytest.raises(
        DockerWorkflowRecoveryRequired,
        match="unterminated Docker generation",
    ):
        DockerWorkflowSettlement.arm_new(
            state_path=authority.state_path,
            run_generation="run-b",
            arm_generation="arm-b",
            control_plane=control,
        )

    assert control.identity_calls == calls_after_arm


def test_explicit_interrupted_recovery_reuses_exact_arm_and_registered_ids(
    tmp_path,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    control.containers[_A] = authority.labels
    authority.register_container(_A)
    proof = "sha256:" + "7" * 64

    recovered = DockerWorkflowSettlement.recover_interrupted(
        state_path=authority.state_path,
        process_proof_ref=proof,
        control_plane=control,
        recovery_deadline_monotonic=time.monotonic() + 30,
    )

    assert recovered.terminal is not None
    assert recovered.terminal.status is DockerSettlementStatus.SETTLED
    assert recovered.terminal.process_proof_ref == proof
    assert _A not in control.containers
    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert payload["arm_fingerprint"] == authority.arm.fingerprint
    assert payload["terminal"] == recovered.terminal.to_payload()
    assert payload["registrations"] == [
        {"container_id": _A, "labels": authority.labels}
    ]


def test_uncertain_terminal_is_persisted_and_blocks_the_next_arm(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    terminal = authority.settle(
        process_tree_proven=False,
        process_proof_ref=None,
    )
    assert terminal.status == DockerSettlementStatus.UNCERTAIN

    with pytest.raises(
        DockerWorkflowRecoveryRequired,
        match="uncertain Docker settlement",
    ):
        DockerWorkflowSettlement.arm_new(
            state_path=authority.state_path,
            run_generation="run-b",
            arm_generation="arm-b",
            control_plane=control,
        )


def test_terminal_bound_to_another_arm_blocks_recovery(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "2" * 64,
    )
    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    payload["terminal"]["arm_fingerprint"] = "sha256:" + "f" * 64
    authority.state_path.write_bytes(settlement_module._canonical_json_bytes(payload))

    with pytest.raises(
        DockerWorkflowRecoveryRequired,
        match="invalid or changed",
    ):
        DockerWorkflowSettlement.arm_new(
            state_path=authority.state_path,
            run_generation="run-b",
            arm_generation="arm-b",
            control_plane=control,
        )


@pytest.mark.parametrize(
    "reserved",
    [
        "io.crucible.settlement",
        "IO.CRUCIBLE.SETTLEMENT",
        "io.crucible.arm-generation",
    ],
)
def test_managed_labels_cannot_override_reserved_arm_labels(
    tmp_path,
    reserved: str,
) -> None:
    authority = _arm(tmp_path, _ControlPlane())

    with pytest.raises(DockerWorkflowSettlementError, match="settlement label"):
        authority.labels_for({reserved: "attacker"})


def test_malformed_prior_journal_blocks_before_daemon_access(tmp_path) -> None:
    path = tmp_path / "docker-settlement.json"
    path.write_text('{"schema_version":', encoding="utf-8")
    control = _ControlPlane()

    with pytest.raises(DockerWorkflowRecoveryRequired, match="malformed"):
        DockerWorkflowSettlement.arm_new(
            state_path=path,
            run_generation="run-b",
            arm_generation="arm-b",
            control_plane=control,
        )

    assert control.identity_calls == 0


def test_one_absolute_deadline_is_reused_for_arm_register_and_settle(
    tmp_path,
) -> None:
    control = _ControlPlane()

    def monotonic() -> float:
        return 100.0

    authority = DockerWorkflowSettlement.arm_new(
        state_path=tmp_path / "docker-settlement.json",
        run_generation="run-a",
        arm_generation="arm-a",
        control_plane=control,
        deadline_monotonic=150.0,
        monotonic=monotonic,
    )
    control.containers[_A] = authority.labels
    authority.register_container(_A)
    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "9" * 64,
    )

    assert terminal.status == DockerSettlementStatus.SETTLED
    assert authority.deadline_monotonic == 150.0
    assert control.deadlines
    assert set(control.deadlines) == {150.0}
    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert payload["deadline_monotonic"] == "150"


def test_parent_bridge_fsyncs_canonical_registration_before_reply(
    tmp_path,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)

    with DockerSettlementRegistrationServer(authority) as server:
        client = DockerSettlementRegistrationClient.from_environment(
            server.environment,
        )
        assert client is not None
        merged = client.labels_for({"io.crucible.managed": "true"})
        assert all(merged[key] == value for key, value in authority.labels.items())
        control.containers[_A] = merged
        control.aliases["short-a"] = _A

        observed = client.register_container("short-a")
        persisted = json.loads(authority.state_path.read_text(encoding="utf-8"))

        assert observed.container_id == _A
        assert persisted["registrations"] == [observed.to_payload()]

    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "8" * 64,
    )
    assert terminal.status == DockerSettlementStatus.SETTLED


def test_lost_post_registration_ack_latches_parent_uncertainty(
    tmp_path,
    monkeypatch,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    server = DockerSettlementRegistrationServer(authority).start()
    client = DockerSettlementRegistrationClient.from_environment(
        server.environment,
    )
    assert client is not None
    control.containers[_A] = client.labels_for({"io.crucible.managed": "true"})

    def reject_valid_inspection(_response):
        raise DockerSettlementBridgeError("injected child ACK loss")

    monkeypatch.setattr(
        client,
        "_inspection_from_response",
        reject_valid_inspection,
    )
    with pytest.raises(DockerSettlementBridgeError, match="ACK loss"):
        client.register_container(_A)
    with pytest.raises(
        DockerSettlementBridgeError,
        match="registration bridge failed",
    ):
        server.close()

    persisted = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert persisted["registrations"][0]["container_id"] == _A
    assert any(
        issue.startswith("registration_bridge_failed:")
        for issue in persisted["workflow_issues"]
    )
    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "4" * 64,
    )
    assert terminal.status == DockerSettlementStatus.UNCERTAIN


def test_bridge_environment_binds_nonce_arm_and_absolute_deadline(
    tmp_path,
) -> None:
    authority = _arm(tmp_path, _ControlPlane())

    with DockerSettlementRegistrationServer(
        authority,
        socket_name="crucible-test-bridge",
        nonce="7" * 64,
    ) as server:
        assert server._listener is not None
        assert server._listener.family == settlement_module.socket.AF_UNIX
        assert (
            server._listener.type & settlement_module.socket.SOCK_SEQPACKET
            == settlement_module.socket.SOCK_SEQPACKET
        )
        address = server._listener.getsockname()
        assert address.startswith(b"\x00" if isinstance(address, bytes) else "\x00")
        environment = server.environment
        assert environment[DOCKER_SETTLEMENT_NONCE_ENV] == "7" * 64
        assert environment[DOCKER_SETTLEMENT_ARM_ENV] == authority.arm.fingerprint
        assert environment[DOCKER_SETTLEMENT_DEADLINE_ENV] == format(
            authority.deadline_monotonic,
            ".17g",
        )
        client = DockerSettlementRegistrationClient.from_environment(environment)
        assert client is not None
        assert client.arm_fingerprint == authority.arm.fingerprint
        assert client.deadline_monotonic == authority.deadline_monotonic


def test_wrong_bridge_nonce_rejects_without_mutating_registration(
    tmp_path,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)

    with DockerSettlementRegistrationServer(authority) as server:
        hostile = dict(server.environment)
        hostile[DOCKER_SETTLEMENT_NONCE_ENV] = "0" * 64
        with pytest.raises(DockerSettlementBridgeError):
            DockerSettlementRegistrationClient.from_environment(hostile)

    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert payload["registrations"] == []
    assert authority.workflow_issues == ()


def test_partial_or_missing_bridge_environment_fails_before_transport() -> None:
    with pytest.raises(DockerSettlementBridgeError, match="absent"):
        DockerSettlementRegistrationClient.from_environment({})
    assert (
        DockerSettlementRegistrationClient.from_environment({}, required=False) is None
    )
    with pytest.raises(DockerSettlementBridgeError, match="incomplete"):
        DockerSettlementRegistrationClient.from_environment(
            {
                DOCKER_SETTLEMENT_NONCE_ENV: "0" * 64,
            }
        )


def test_bridge_registration_rejection_is_terminal_uncertainty(tmp_path) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)

    with DockerSettlementRegistrationServer(authority) as server:
        client = DockerSettlementRegistrationClient.from_environment(
            server.environment,
        )
        assert client is not None
        # The reference is authoritatively absent. The parent session retains
        # REGISTRATION_FAILED even though the bridge stays healthy.
        with pytest.raises(
            DockerSettlementBridgeError,
            match="registration was rejected",
        ):
            client.register_container("missing-container")

    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "6" * 64,
    )
    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert any(issue.code.value == "registration_failed" for issue in terminal.issues)


def test_expired_bridge_deadline_rejects_child_construction(tmp_path) -> None:
    control = _ControlPlane()
    now = [10.0]

    def monotonic() -> float:
        return now[0]

    authority = DockerWorkflowSettlement.arm_new(
        state_path=tmp_path / "docker-settlement.json",
        run_generation="run-a",
        arm_generation="arm-a",
        control_plane=control,
        deadline_monotonic=20.0,
        monotonic=monotonic,
    )
    server = DockerSettlementRegistrationServer(
        authority,
        monotonic=monotonic,
    ).start()
    try:
        environment = server.environment
        now[0] = 20.0
        with pytest.raises(DockerSettlementBridgeError, match="environment"):
            DockerSettlementRegistrationClient.from_environment(
                environment,
                monotonic=monotonic,
            )
    finally:
        server.close()


def test_registration_persistence_failure_latches_uncertain_terminal(
    tmp_path,
    monkeypatch,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    control.containers[_A] = authority.labels
    original_write = settlement_module._write_strict_atomic_json
    failed = False

    def fail_first_registration(path, payload):
        nonlocal failed
        if payload["registrations"] and payload["terminal"] is None and not failed:
            failed = True
            raise OSError("injected registration fsync failure")
        return original_write(path, payload)

    monkeypatch.setattr(
        settlement_module,
        "_write_strict_atomic_json",
        fail_first_registration,
    )
    with DockerSettlementRegistrationServer(authority) as server:
        client = DockerSettlementRegistrationClient.from_environment(
            server.environment,
        )
        assert client is not None
        with pytest.raises(
            DockerSettlementBridgeError,
            match="registration was rejected",
        ):
            client.register_container(_A)

    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="sha256:" + "5" * 64,
    )
    payload = json.loads(authority.state_path.read_text(encoding="utf-8"))
    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert payload["terminal"]["status"] == "uncertain"
    assert any(
        issue.startswith("persistence_failed:") for issue in payload["workflow_issues"]
    )


def test_unbound_process_proof_ref_cleans_but_cannot_settle_positive(
    tmp_path,
) -> None:
    authority = _arm(tmp_path, _ControlPlane())

    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref="caller-asserted-not-a-digest",
    )

    assert terminal.status == DockerSettlementStatus.UNCERTAIN
    assert "process_proof_ref_invalid" in authority.workflow_issues


def test_receipt_retains_exact_durable_journal_and_round_trips_closed_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, proof = _settled_authority(tmp_path)
    opened: list[tuple[object, int, int | None]] = []
    original_open = settlement_module.os.open

    def observe_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        opened.append((path, flags, dir_fd))
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(settlement_module.os, "open", observe_open)
    receipt = authority.settlement_receipt(process_settlement=proof)

    retained = authority.state_path.read_bytes()
    assert receipt.journal_canonical_bytes == retained
    assert len(retained) <= settlement_module._MAX_JOURNAL_BYTES
    assert receipt.journal_sha256 == (
        "sha256:" + hashlib.sha256(retained).hexdigest()
    )
    assert receipt.process_settlement == proof
    assert receipt.process_settlement_record_digest == proof.sha256
    assert receipt.terminal.process_proof_ref == proof.sha256
    assert receipt.terminal_fingerprint == authority.terminal_fingerprint
    assert [item.container_id for item in receipt.registrations] == [_A]

    leaf_opens = [
        (flags, dir_fd)
        for path, flags, dir_fd in opened
        if path == authority.state_path.name
    ]
    assert len(leaf_opens) == 2
    assert all(type(dir_fd) is int for _flags, dir_fd in leaf_opens)
    assert all(
        flags & settlement_module.os.O_NOFOLLOW
        for flags, _dir_fd in leaf_opens
    )

    payload = receipt.to_payload()
    assert payload["schema_version"] == DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA
    assert payload["journal"]["canonical_json"].encode("utf-8") == retained
    assert payload["journal"]["sha256"] == receipt.journal_sha256
    assert payload["binding"]["terminal_fingerprint"] == (
        authority.terminal_fingerprint
    )
    assert payload["authority"] == {
        "scope": "provider-free-docker-settlement-evidence-only",
        "verifier_outcome_authority_conferred": False,
        "production_execution_authority_conferred": False,
        "launch_authority_conferred": False,
    }
    reopened = DockerWorkflowSettlementReceipt.from_payload(
        json.loads(json.dumps(payload))
    )
    assert reopened == receipt
    assert reopened.fingerprint == receipt.fingerprint


def test_receipt_requires_a_clean_persisted_terminal(tmp_path: Path) -> None:
    proof = _process_settlement_proof()
    authority = _arm(tmp_path, _ControlPlane())

    with pytest.raises(
        DockerWorkflowSettlementError,
        match="clean durable terminal",
    ):
        authority.settlement_receipt(process_settlement=proof)

    terminal = authority.settle(
        process_tree_proven=False,
        process_proof_ref=None,
    )
    assert terminal.status is DockerSettlementStatus.UNCERTAIN
    with pytest.raises(
        DockerWorkflowSettlementError,
        match="clean durable terminal",
    ):
        authority.settlement_receipt(process_settlement=proof)


def test_receipt_requires_exact_cgroup_proof_bound_by_terminal(
    tmp_path: Path,
) -> None:
    authority, proof = _settled_authority(tmp_path)
    wrong_proof = _process_settlement_proof(returncode=1)
    supervisor_only = ProcessSettlementProof.compose(
        supervisor_canonical_bytes=_canonical_test_json(
            {
                "error": None,
                "returncode": 0,
                "schema_version": "crucible.process-supervisor.v1",
                "settled": True,
            }
        ),
        cgroup_canonical_bytes=None,
        effective_returncode=0,
    )

    with pytest.raises(
        DockerWorkflowSettlementError,
        match="does not bind",
    ):
        authority.settlement_receipt(process_settlement=wrong_proof)
    with pytest.raises(
        DockerWorkflowSettlementError,
        match="delegated-cgroup",
    ):
        authority.settlement_receipt(process_settlement=supervisor_only)

    class ProcessProofSubclass(ProcessSettlementProof):
        pass

    subclass = ProcessProofSubclass(
        supervisor_canonical_bytes=proof.supervisor_canonical_bytes,
        cgroup_canonical_bytes=proof.cgroup_canonical_bytes,
        combined_canonical_bytes=proof.combined_canonical_bytes,
    )
    with pytest.raises(
        DockerWorkflowSettlementError,
        match="exact delegated-cgroup",
    ):
        authority.settlement_receipt(process_settlement=subclass)


def test_receipt_reread_rejects_hardlink_and_symlink_aliases(
    tmp_path: Path,
) -> None:
    authority, proof = _settled_authority(tmp_path)
    alias = tmp_path / "journal-alias.json"
    os.link(authority.state_path, alias)
    with pytest.raises(DockerWorkflowSettlementError, match="single-link"):
        authority.settlement_receipt(process_settlement=proof)

    alias.unlink()
    target = tmp_path / "journal-target.json"
    authority.state_path.rename(target)
    authority.state_path.symlink_to(target.name)
    with pytest.raises(DockerWorkflowSettlementError, match="regular file"):
        authority.settlement_receipt(process_settlement=proof)


def test_receipt_reread_fails_closed_when_barrier_or_bytes_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, proof = _settled_authority(tmp_path)

    def reject_fsync(_descriptor: int) -> None:
        raise OSError("injected receipt fsync failure")

    monkeypatch.setattr(settlement_module.os, "fsync", reject_fsync)
    with pytest.raises(DockerWorkflowSettlementError, match="durably reread"):
        authority.settlement_receipt(process_settlement=proof)

    monkeypatch.undo()
    journal = json.loads(authority.state_path.read_bytes())
    journal["deadline_monotonic"] = "999"
    authority.state_path.write_bytes(settlement_module._canonical_json_bytes(journal))
    with pytest.raises(DockerWorkflowSettlementError, match="differs from its terminal"):
        authority.settlement_receipt(process_settlement=proof)


def test_receipt_from_payload_rejects_digest_semantic_and_authority_tampering(
    tmp_path: Path,
) -> None:
    authority, proof = _settled_authority(tmp_path)
    payload = authority.settlement_receipt(process_settlement=proof).to_payload()

    def clone() -> dict[str, object]:
        return json.loads(json.dumps(payload))

    variants: list[dict[str, object]] = []

    unknown = clone()
    unknown["unexpected"] = None
    variants.append(unknown)

    journal_digest = clone()
    journal_digest["journal"]["sha256"] = "sha256:" + "0" * 64
    variants.append(journal_digest)

    process_digest = clone()
    process_digest["process_settlement"]["record_digest"] = "sha256:" + "0" * 64
    variants.append(process_digest)

    process_preimage = clone()
    process_preimage["process_settlement"]["evidence"]["combined"]["sha256"] = (
        "sha256:" + "0" * 64
    )
    variants.append(process_preimage)

    binding = clone()
    binding["binding"]["terminal_fingerprint"] = "sha256:" + "0" * 64
    variants.append(binding)

    authority_claim = clone()
    authority_claim["authority"]["launch_authority_conferred"] = True
    variants.append(authority_claim)

    changed_registration = clone()
    changed_journal = json.loads(
        changed_registration["journal"]["canonical_json"]
    )
    changed_journal["registrations"][0]["labels"][
        "io.crucible.run-generation"
    ] = "run-b"
    changed_bytes = settlement_module._canonical_json_bytes(changed_journal)
    changed_registration["journal"]["canonical_json"] = changed_bytes.decode()
    changed_registration["journal"]["sha256"] = (
        "sha256:" + hashlib.sha256(changed_bytes).hexdigest()
    )
    variants.append(changed_registration)

    changed_process_binding = clone()
    changed_journal = json.loads(
        changed_process_binding["journal"]["canonical_json"]
    )
    changed_journal["terminal"]["process_proof_ref"] = "sha256:" + "f" * 64
    changed_journal["terminal_fingerprint"] = settlement_module.canonical_fingerprint(
        changed_journal["terminal"]
    )
    changed_bytes = settlement_module._canonical_json_bytes(changed_journal)
    changed_process_binding["journal"]["canonical_json"] = changed_bytes.decode()
    changed_process_binding["journal"]["sha256"] = (
        "sha256:" + hashlib.sha256(changed_bytes).hexdigest()
    )
    variants.append(changed_process_binding)

    changed_cleanup_accounting = clone()
    changed_journal = json.loads(
        changed_cleanup_accounting["journal"]["canonical_json"]
    )
    changed_journal["terminal"]["removed_container_ids"] = []
    changed_journal["terminal_fingerprint"] = settlement_module.canonical_fingerprint(
        changed_journal["terminal"]
    )
    changed_bytes = settlement_module._canonical_json_bytes(changed_journal)
    changed_cleanup_accounting["journal"]["canonical_json"] = (
        changed_bytes.decode()
    )
    changed_cleanup_accounting["journal"]["sha256"] = (
        "sha256:" + hashlib.sha256(changed_bytes).hexdigest()
    )
    variants.append(changed_cleanup_accounting)

    for variant in variants:
        with pytest.raises(ValueError):
            DockerWorkflowSettlementReceipt.from_payload(variant)


def test_receipt_rejects_journal_above_existing_fixed_bound() -> None:
    with pytest.raises(ValueError, match="fixed byte bound"):
        DockerWorkflowSettlementReceipt(
            journal_canonical_bytes=b"x"
            * (settlement_module._MAX_JOURNAL_BYTES + 1),
            process_settlement=_process_settlement_proof(),
        )


def test_receipt_allows_terminal_discovery_beyond_registration_set(
    tmp_path: Path,
) -> None:
    control = _ControlPlane()
    authority = _arm(tmp_path, control)
    control.containers[_A] = authority.labels
    proof = _process_settlement_proof()
    terminal = authority.settle(
        process_tree_proven=True,
        process_proof_ref=proof.sha256,
    )

    receipt = authority.settlement_receipt(process_settlement=proof)
    assert receipt.registrations == ()
    assert terminal.known_container_ids == (_A,)
    assert receipt.terminal == terminal
