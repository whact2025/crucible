# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Focused contracts for benchmark crash-safety/runtime primitives."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from crucible.common.benchmark_runtime import (
    BudgetReservationError,
    BudgetReservations,
    ImmutableImageError,
    TaskLeaseBusy,
    canonical_fingerprint,
    image_verification_policy_identity,
    managed_container_name,
    managed_container_labels,
    reconcile_orphaned_containers,
    resolve_immutable_image,
    run_process_tree,
    task_execution_lease,
)
from crucible.products.swebench.predictions import (
    Prediction,
    merge_predictions_jsonl,
)


def test_canonical_fingerprint_is_order_independent_and_complete():
    left = canonical_fingerprint({"b": [2, 3], "a": {"x", "y"}})
    right = canonical_fingerprint({"a": {"y", "x"}, "b": [2, 3]})
    changed = canonical_fingerprint({"b": [2, 4], "a": {"x", "y"}})
    assert left == right
    assert changed != left
    assert left.startswith("sha256:")


def test_task_lease_rejects_concurrent_owner(tmp_path: Path):
    with task_execution_lease(tmp_path, namespace="test", task_id="a/../b"):
        with pytest.raises(TaskLeaseBusy):
            with task_execution_lease(
                tmp_path, namespace="test", task_id="a/../b",
            ):
                pytest.fail("unreachable")
    lease_files = list((tmp_path / ".benchmark-leases" / "test").iterdir())
    assert len(lease_files) == 1
    assert len(lease_files[0].stem) == 64
    int(lease_files[0].stem, 16)


def test_budget_reservations_are_atomic_and_settle_unused_capacity():
    ledger = BudgetReservations(10.0, spent=2.0)
    first = ledger.reserve(5.0)
    second = ledger.reserve(5.0)
    assert first.amount == 5.0
    assert second.amount == 3.0
    assert ledger.available == 0.0

    ledger.settle(first, 1.5)
    assert ledger.spent == pytest.approx(3.5)
    assert ledger.available == pytest.approx(3.5)
    ledger.settle(second, 3.0)
    assert ledger.spent == pytest.approx(6.5)


def test_budget_settlement_surfaces_worker_ceiling_violation():
    ledger = BudgetReservations(1.0)
    reservation = ledger.reserve(1.0)
    with pytest.raises(BudgetReservationError, match="beyond reservation"):
        ledger.settle(reservation, 1.01)
    # Actual known spend is retained for terminal reporting.
    assert ledger.spent == pytest.approx(1.01)


def test_resolve_immutable_image_uses_repository_digest(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", raising=False)
    calls: list[list[str]] = []

    def run_docker(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps([
                "registry/repo@sha256:" + "b" * 64,
            ]),
            stderr="",
        )

    resolved = resolve_immutable_image(
        "registry/repo:latest", run_docker=run_docker,
    )
    assert resolved == "registry/repo@sha256:" + "b" * 64
    assert calls[0][:3] == ["image", "inspect", "--format"]


def test_resolve_immutable_image_preserves_registry_port(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", raising=False)

    def run_docker(args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps([
                "localhost:5000/repo@sha256:" + "c" * 64,
                "other/repo@sha256:" + "a" * 64,
            ]),
            stderr="",
        )

    assert resolve_immutable_image(
        "localhost:5000/repo:latest",
        run_docker=run_docker,
    ) == "localhost:5000/repo@sha256:" + "c" * 64


def test_resolve_immutable_image_rejects_other_repository_digest(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", raising=False)

    def run_docker(args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(["attacker/repo@sha256:" + "d" * 64]),
            stderr="",
        )

    with pytest.raises(ImmutableImageError, match="repository trust domain"):
        resolve_immutable_image(
            "trusted/repo:latest",
            run_docker=run_docker,
        )


def test_image_verification_policy_identity_tracks_command(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", raising=False)
    assert image_verification_policy_identity() == "disabled"
    monkeypatch.setenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", "cosign verify --offline")
    first = image_verification_policy_identity()
    monkeypatch.setenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", "cosign verify --key policy.pub")
    assert image_verification_policy_identity() != first


def test_managed_container_name_is_scope_bound_and_dataset_safe():
    first = managed_container_name(
        prefix="bench", scope="/spec/a", task_id="../../task/one",
    )
    same = managed_container_name(
        prefix="bench", scope="/spec/a", task_id="../../task/one",
    )
    other_scope = managed_container_name(
        prefix="bench", scope="/spec/b", task_id="../../task/one",
    )
    assert first == same
    assert first != other_scope
    assert "/" not in first and ".." not in first


def test_resolve_immutable_image_rejects_malformed_supplied_digest(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_IMAGE_VERIFY_COMMAND", raising=False)
    with pytest.raises(ImmutableImageError, match="invalid immutable image digest"):
        resolve_immutable_image(
            "registry/repo@sha256:not-a-digest",
            run_docker=lambda *args, **kwargs: pytest.fail("must not inspect"),
        )


def test_process_timeout_terminates_complete_process_group():
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_process_tree(
            ["sh", "-c", "sleep 60 & echo $!; wait"], timeout=0.1,
        )
    child_pid = int((caught.value.output or b"").splitlines()[0])
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            stat_record = Path(f"/proc/{child_pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            break
        # In a container whose PID 1 does not promptly reap orphans, the
        # terminated grandchild can remain as a zombie. It is no longer
        # executing and cannot escape the process-group timeout.
        if stat_record.rsplit(")", 1)[-1].split()[0] == "Z":
            break
        time.sleep(0.02)
    else:
        pytest.fail(f"child process {child_pid} remained live after timeout")


def test_orphan_reconciliation_removes_only_dead_same_host_owner(monkeypatch):
    scope = "/spec"
    labels = managed_container_labels(scope=scope, task_id="task", phase="author")
    labels["io.crucible.owner-pid"] = "999999999"
    calls: list[list[str]] = []

    def run_docker(args, **kwargs):
        calls.append(args)
        if args[0] == "ps":
            stdout = "dead-container\n"
        elif args[0] == "inspect":
            stdout = json.dumps(labels)
        else:
            stdout = ""
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout, stderr="",
        )

    monkeypatch.setattr(os, "kill", lambda pid, sig: (_ for _ in ()).throw(
        ProcessLookupError(),
    ))
    assert reconcile_orphaned_containers(
        scope=scope, run_docker=run_docker,
    ) == ("dead-container",)
    assert ["rm", "-f", "dead-container"] in calls


def test_orphan_reconciliation_does_not_trust_reused_pid(monkeypatch):
    import crucible.common.benchmark_runtime as runtime

    scope = "/spec"
    labels = managed_container_labels(scope=scope, task_id="task", phase="author")
    labels["io.crucible.owner-start"] = "old-start-token"
    calls: list[list[str]] = []

    def run_docker(args, **kwargs):
        calls.append(args)
        stdout = "container\n" if args[0] == "ps" else (
            json.dumps(labels) if args[0] == "inspect" else ""
        )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout, stderr="",
        )

    monkeypatch.setattr(os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(runtime, "_pid_start_token", lambda pid: "new-start-token")
    assert reconcile_orphaned_containers(
        scope=scope, run_docker=run_docker,
    ) == ("container",)


def test_prediction_merge_repairs_torn_tail_and_preserves_siblings(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    merge_predictions_jsonl((
        Prediction("a", "old", "m1"),
        Prediction("b", "keep", "m2"),
    ), path)
    with path.open("ab") as stream:
        stream.write(b'{"instance_id":"torn"')
    count = merge_predictions_jsonl((Prediction("a", "new", "m3"),), path)
    assert count == 2
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"instance_id": "a", "model_patch": "new", "model_name_or_path": "m3"},
        {"instance_id": "b", "model_patch": "keep", "model_name_or_path": "m2"},
    ]
