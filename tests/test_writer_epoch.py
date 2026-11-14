# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Adversarial durability tests for writer-family epoch authority."""

from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest


def _sha(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _record_sha(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _epoch_owner(kind: str, identifier: str, *, process_id: int = 4242):
    from crucible.workflows._writer_epoch import WriterEpochOwner

    return WriterEpochOwner(
        kind,
        identifier,
        process_id,
        987654321,
        "00000000-1111-2222-3333-444444444444",
    )


class _Storage:
    def __init__(
        self,
        transaction_id: str,
        parent_transaction_id: str | None,
        *,
        record: object = None,
        log: list[tuple[str, str]] | None = None,
    ) -> None:
        from crucible.workflows._writer_epoch import (
            initial_writer_epoch_record,
        )

        self.transaction_id = transaction_id
        self.parent_transaction_id = parent_transaction_id
        self.active = True
        self.record = copy.deepcopy(
            initial_writer_epoch_record() if record is None else record
        )
        self.log = log if log is not None else []
        self.fail_before_state: str | None = None
        self.fail_after_state: str | None = None
        self.return_unproved = False

    def read_writer_epoch_record(self) -> object:
        return copy.deepcopy(self.record)

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        if self.record != expected:
            raise RuntimeError("compare-and-write conflict")
        state = str(replacement["state"])
        if self.fail_before_state == state:
            self.fail_before_state = None
            raise OSError(f"fail before {state}")
        self.record = copy.deepcopy(replacement)
        self.log.append((self.transaction_id, state))
        if self.fail_after_state == state:
            self.fail_after_state = None
            raise OSError(f"fail after durable {state}")
        if self.return_unproved:
            return {"unproved": True}
        return copy.deepcopy(self.record)


def _family(*, log: list[tuple[str, str]] | None = None):
    root = _Storage("root", None, log=log)
    child_a = _Storage("child-a", "root", log=log)
    child_b = _Storage("child-b", "root", log=log)
    grandchild = _Storage("grandchild", "child-a", log=log)
    return root, child_a, child_b, grandchild


def _owner_boundary(
    *,
    owner_ids: tuple[str, ...] = ("backend-a", "backend-b"),
    worker_ids: tuple[str, ...] = ("lane-a", "lane-b"),
):
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochOwner,
    )

    return (
        _epoch_owner("scheduler", "writer-fanout"),
        WriterEpochBoundary(
            "scheduler",
            "provider-and-executor",
            "scheduler_join",
            owner_ids,
            worker_ids,
        ),
    )


def _arm(members):
    from crucible.workflows._writer_epoch import arm_writer_epoch_family

    owner, boundary = _owner_boundary()
    return arm_writer_epoch_family(
        tuple(members),
        owner=owner,
        boundary=boundary,
    )


def _scheduler_receipt(lease, *, outcome="succeeded"):
    from crucible.workflows._writer_epoch import scheduler_settlement_receipt

    return scheduler_settlement_receipt(
        lease,
        settled_owner_ids=("backend-b", "backend-a"),
        joined_worker_ids=("lane-b", "lane-a"),
        evidence_digest=_sha("joined"),
        outcome=outcome,
    )


def test_owner_and_boundary_fields_are_exact_and_ids_are_canonical() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochOwner,
    )

    owner = WriterEpochOwner(
        "provider",
        "implementer",
        4242,
        987654321,
        "00000000-1111-2222-3333-444444444444",
    )
    boundary = WriterEpochBoundary(
        "provider",
        "before-capture",
        "provider_close_drain",
        ("z-owner", "a-owner"),
        ("z-worker", "a-worker"),
    )
    assert owner.to_record() == {
        "kind": "provider",
        "identifier": "implementer",
        "process_id": 4242,
        "process_start_time_ticks": 987654321,
        "boot_id": "00000000-1111-2222-3333-444444444444",
    }
    assert boundary.expected_owner_ids == ("a-owner", "z-owner")
    assert boundary.expected_worker_ids == ("a-worker", "z-worker")
    assert boundary.to_record()["settlement_kind"] == "provider_close_drain"
    with pytest.raises(ValueError, match="at least one expected writer"):
        WriterEpochBoundary("provider", "empty", "provider_close_drain", (), ())
    with pytest.raises(ValueError, match="duplicates"):
        WriterEpochBoundary(
            "provider",
            "duplicate",
            "provider_close_drain",
            ("owner", "owner"),
            (),
        )


def test_trusted_owner_capture_records_pid_start_time_and_boot_id() -> None:
    from crucible.workflows._writer_epoch import capture_writer_epoch_owner

    owner = capture_writer_epoch_owner("provider", "implementer")
    assert owner.process_id == os.getpid()
    assert owner.process_start_time_ticks > 0
    assert len(owner.boot_id) == 36
    assert owner.to_record()["process_id"] == os.getpid()


@pytest.mark.parametrize(
    ("process_id", "start_time", "boot_id"),
    [
        (999_999_999, 1, "00000000-1111-2222-3333-444444444444"),
        (4242, 999_999_999, "00000000-1111-2222-3333-444444444444"),
        (4242, 987654321, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
    ],
    ids=("pid-death", "pid-reuse", "boot-id-change"),
)
def test_process_diagnostics_never_imply_settlement(
    process_id: int,
    start_time: int,
    boot_id: str,
) -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        WriterEpochBoundary,
        WriterEpochOwner,
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
        require_writer_epoch_quiescence,
    )

    root = _Storage("root", None)
    lease = arm_writer_epoch_family(
        (root,),
        owner=WriterEpochOwner(
            "provider",
            "diagnostic-only",
            process_id,
            start_time,
            boot_id,
        ),
        boundary=WriterEpochBoundary(
            "provider",
            "before-capture",
            "provider_close_drain",
            ("backend",),
            (),
        ),
    )
    assert root.record["owner"] == lease.owner.to_record()
    assert inspect_writer_epoch_family((root,)).state == "armed"
    with pytest.raises(WriterEpochBlockedError, match="state=armed"):
        require_writer_epoch_quiescence((root,))


def test_family_canonicalizes_topology_not_caller_order() -> None:
    from crucible.workflows._writer_epoch import (
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
    )

    log: list[tuple[str, str]] = []
    root, child_a, child_b, grandchild = _family(log=log)
    owner, boundary = _owner_boundary()
    lease = arm_writer_epoch_family(
        (grandchild, child_b, root, child_a),
        owner=owner,
        boundary=boundary,
    )
    assert lease.member_ids == ("root", "child-a", "child-b", "grandchild")
    assert log == [
        ("root", "armed"),
        ("child-a", "armed"),
        ("child-b", "armed"),
        ("grandchild", "armed"),
    ]
    assert (
        inspect_writer_epoch_family((child_b, grandchild, child_a, root)).state
        == "armed"
    )


def test_prepare_writer_epoch_arm_performs_no_durable_writes() -> None:
    from crucible.workflows._writer_epoch import (
        inspect_writer_epoch_family,
        prepare_writer_epoch_arm,
        retry_writer_epoch_arm,
    )

    log: list[tuple[str, str]] = []
    root, child_a, child_b, grandchild = _family(log=log)
    owner, boundary = _owner_boundary()
    lease = prepare_writer_epoch_arm(
        (grandchild, child_b, root, child_a),
        owner=owner,
        boundary=boundary,
    )

    assert lease.member_ids == ("root", "child-a", "child-b", "grandchild")
    assert log == []
    assert inspect_writer_epoch_family(lease.members).state == "never_armed"

    retry_writer_epoch_arm(lease)
    assert inspect_writer_epoch_family(lease.members).state == "armed"
    assert log == [
        ("root", "armed"),
        ("child-a", "armed"),
        ("child-b", "armed"),
        ("grandchild", "armed"),
    ]


@pytest.mark.parametrize(
    "members, message",
    [
        (lambda: (_Storage("root", None), _Storage("root", None)), "duplicate"),
        (
            lambda: (
                _Storage("root", None),
                _Storage("child", "foreign"),
            ),
            "foreign-parent",
        ),
        (
            lambda: (
                _Storage("root", None),
                _Storage("a", "b"),
                _Storage("b", "a"),
            ),
            "disconnected or cyclic",
        ),
        (
            lambda: (_Storage("a", None), _Storage("b", None)),
            "exactly one parentless root",
        ),
    ],
)
def test_family_rejects_duplicate_disconnected_and_foreign_members(
    members,
    message: str,
) -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochFamilyError,
        inspect_writer_epoch_family,
    )

    with pytest.raises(WriterEpochFamilyError, match=message):
        inspect_writer_epoch_family(tuple(members()))


def test_missing_legacy_state_is_unknown_and_blocks_settlement() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        inspect_writer_epoch_family,
        require_writer_epoch_quiescence,
    )

    legacy = _Storage("root", None, record={})
    inspection = inspect_writer_epoch_family((legacy,))
    assert inspection.state == "unknown"
    assert "malformed" in str(inspection.reason)
    with pytest.raises(WriterEpochBlockedError, match="state=unknown"):
        require_writer_epoch_quiescence((legacy,))


def test_never_armed_is_safe_only_when_explicitly_allowed() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        require_writer_epoch_quiescence,
    )

    root = _Storage("root", None)
    assert require_writer_epoch_quiescence((root,)).state == "never_armed"
    with pytest.raises(WriterEpochBlockedError, match="never_armed"):
        require_writer_epoch_quiescence((root,), allow_never_armed=False)


def test_partial_root_first_arm_blocks_and_exact_retry_finishes() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochArmError,
        inspect_writer_epoch_family,
    )

    log: list[tuple[str, str]] = []
    root, child_a, child_b, grandchild = _family(log=log)
    child_a.fail_before_state = "armed"
    with pytest.raises(WriterEpochArmError) as captured:
        _arm((child_b, grandchild, root, child_a))
    assert log == [("root", "armed")]
    assert (
        inspect_writer_epoch_family((root, child_a, child_b, grandchild)).state
        == "partial"
    )

    lease = captured.value.retry_arm()
    assert inspect_writer_epoch_family(lease.members).state == "armed"
    assert log == [
        ("root", "armed"),
        ("child-a", "armed"),
        ("child-b", "armed"),
        ("grandchild", "armed"),
    ]


def test_arm_retry_accepts_write_that_failed_after_durable_publish() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochArmError,
        inspect_writer_epoch_family,
    )

    root, child_a, child_b, grandchild = _family()
    child_a.fail_after_state = "armed"
    with pytest.raises(WriterEpochArmError) as captured:
        _arm((root, child_a, child_b, grandchild))
    lease = captured.value.retry_arm()
    assert inspect_writer_epoch_family(lease.members).state == "armed"


def test_new_arm_rejects_partial_or_already_armed_family() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        arm_writer_epoch_family,
    )

    root, child_a, child_b, grandchild = _family()
    lease = _arm((root, child_a, child_b, grandchild))
    owner, boundary = _owner_boundary()
    with pytest.raises(WriterEpochBlockedError, match="non-quiescent"):
        arm_writer_epoch_family(
            lease.members,
            owner=owner,
            boundary=boundary,
        )


def test_arm_rejects_generation_that_races_between_inspection_and_snapshot() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        arm_writer_epoch_family,
    )

    already_armed = _Storage("root", None)
    foreign_lease = _arm((already_armed,))

    class RacingStorage(_Storage):
        def __init__(self) -> None:
            super().__init__("root", None)
            self.reads = 0

        def read_writer_epoch_record(self) -> object:
            self.reads += 1
            if self.reads == 2:
                self.record = copy.deepcopy(already_armed.record)
            return super().read_writer_epoch_record()

    racing = RacingStorage()
    owner, boundary = _owner_boundary()
    with pytest.raises(WriterEpochBlockedError, match="raced, armed, or partial"):
        arm_writer_epoch_family(
            (racing,),
            owner=owner,
            boundary=boundary,
        )
    assert racing.record["epoch_id"] == foreign_lease.epoch_id


def test_scheduler_receipt_cannot_bless_owner_or_worker_subset() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        scheduler_settlement_receipt,
    )

    lease = _arm(_family())
    with pytest.raises(WriterEpochBlockedError, match="expected owners"):
        scheduler_settlement_receipt(
            lease,
            settled_owner_ids=("backend-a",),
            joined_worker_ids=("lane-a", "lane-b"),
            evidence_digest=_sha("subset-owner"),
        )
    with pytest.raises(WriterEpochBlockedError, match="expected workers"):
        scheduler_settlement_receipt(
            lease,
            settled_owner_ids=("backend-a", "backend-b"),
            joined_worker_ids=("lane-a",),
            evidence_digest=_sha("subset-worker"),
        )


def test_process_supervisor_receipt_cannot_bless_a_boundary_subset() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        WriterEpochBoundary,
        arm_writer_epoch_family,
        process_supervisor_settlement_receipt,
    )

    lease = arm_writer_epoch_family(
        _family(),
        owner=_epoch_owner("process-supervisor", "subset"),
        boundary=WriterEpochBoundary(
            "process-supervisor",
            "subset",
            "process_supervisor",
            ("backend-a", "backend-b"),
            ("lane-a", "lane-b"),
        ),
    )
    with pytest.raises(WriterEpochBlockedError, match="expected owners"):
        process_supervisor_settlement_receipt(
            lease,
            settled_owner_ids=("backend-a",),
            joined_worker_ids=("lane-a", "lane-b"),
            evidence_digest=_sha("supervisor-subset-owner"),
            outcome="failed",
        )
    with pytest.raises(WriterEpochBlockedError, match="expected workers"):
        process_supervisor_settlement_receipt(
            lease,
            settled_owner_ids=("backend-a", "backend-b"),
            joined_worker_ids=("lane-a",),
            evidence_digest=_sha("supervisor-subset-worker"),
            outcome="failed",
        )


def test_stale_process_supervisor_receipt_cannot_settle_a_new_generation() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochBlockedError,
        arm_writer_epoch_family,
        process_supervisor_settlement_receipt,
        quiesce_writer_epoch_family,
    )

    family = _family()
    first = arm_writer_epoch_family(
        family,
        owner=_epoch_owner("process-supervisor", "first-generation"),
        boundary=WriterEpochBoundary(
            "process-supervisor",
            "first-generation",
            "process_supervisor",
            ("backend-a", "backend-b"),
            ("lane-a", "lane-b"),
        ),
    )
    stale = process_supervisor_settlement_receipt(
        first,
        settled_owner_ids=("backend-a", "backend-b"),
        joined_worker_ids=("lane-a", "lane-b"),
        evidence_digest=_sha("supervisor-first-generation"),
        outcome="succeeded",
    )
    quiesce_writer_epoch_family(first, stale)
    second = arm_writer_epoch_family(
        tuple(family),
        owner=_epoch_owner("process-supervisor", "second-generation"),
        boundary=WriterEpochBoundary(
            "process-supervisor",
            "second-generation",
            "process_supervisor",
            ("backend-a", "backend-b"),
            ("lane-a", "lane-b"),
        ),
    )
    with pytest.raises(WriterEpochBlockedError, match="exact lease"):
        quiesce_writer_epoch_family(second, stale)


def test_process_supervisor_receipt_cannot_settle_a_different_lease() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochBlockedError,
        arm_writer_epoch_family,
        process_supervisor_settlement_receipt,
        quiesce_writer_epoch_family,
    )

    boundary = WriterEpochBoundary(
        "process-supervisor",
        "subprocess-runner",
        "process_supervisor",
        ("runner-owner",),
        ("runner-worker",),
    )
    first = arm_writer_epoch_family(
        (_Storage("supervisor-a", None),),
        owner=_epoch_owner("process-supervisor", "first"),
        boundary=boundary,
    )
    second = arm_writer_epoch_family(
        (_Storage("supervisor-b", None),),
        owner=_epoch_owner("process-supervisor", "second"),
        boundary=boundary,
    )
    wrong = process_supervisor_settlement_receipt(
        second,
        settled_owner_ids=("runner-owner",),
        joined_worker_ids=("runner-worker",),
        evidence_digest=_sha("wrong-supervisor-lease"),
        outcome="succeeded",
    )
    with pytest.raises(WriterEpochBlockedError, match="exact lease"):
        quiesce_writer_epoch_family(first, wrong)


@pytest.mark.parametrize(
    "boundary_kind",
    (
        "provider_close_drain",
        "scheduler_join",
        "process_supervisor",
        "synchronous_callback",
    ),
)
@pytest.mark.parametrize(
    "receipt_kind",
    (
        "provider_close_drain",
        "scheduler_join",
        "process_supervisor",
        "synchronous_callback",
    ),
)
def test_receipt_builder_kind_must_match_exact_boundary(
    boundary_kind: str,
    receipt_kind: str,
) -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        WriterEpochBoundary,
        arm_writer_epoch_family,
        process_supervisor_settlement_receipt,
        provider_settlement_receipt,
        scheduler_settlement_receipt,
        synchronous_callback_settlement_receipt,
    )

    lease = arm_writer_epoch_family(
        (_Storage("root", None),),
        owner=_epoch_owner("category-matrix", boundary_kind),
        boundary=WriterEpochBoundary(
            "category-matrix",
            boundary_kind,
            boundary_kind,
            ("owner",),
            ("worker",),
        ),
    )
    builders = {
        "provider_close_drain": provider_settlement_receipt,
        "scheduler_join": scheduler_settlement_receipt,
        "process_supervisor": process_supervisor_settlement_receipt,
        "synchronous_callback": synchronous_callback_settlement_receipt,
    }

    def build():
        return builders[receipt_kind](
            lease,
            settled_owner_ids=("owner",),
            joined_worker_ids=("worker",),
            evidence_digest=_sha(f"{boundary_kind}:{receipt_kind}"),
            outcome="succeeded",
        )

    if receipt_kind == boundary_kind:
        assert build().kind == boundary_kind
    else:
        with pytest.raises(WriterEpochBlockedError, match="exact boundary"):
            build()


def test_forged_cross_kind_settlement_record_is_unknown() -> None:
    from crucible.workflows._writer_epoch import (
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    root = _Storage("root", None)
    lease = _arm((root,))
    quiesce_writer_epoch_family(lease, _scheduler_receipt(lease))
    forged = copy.deepcopy(root.record)
    forged["boundary"]["settlement_kind"] = "provider_close_drain"
    settlement = forged["settlement"]
    settlement["boundary"]["settlement_kind"] = "provider_close_drain"
    unsigned = dict(settlement)
    unsigned.pop("receipt_digest")
    settlement["receipt_digest"] = _record_sha(unsigned)
    root.record = forged

    inspection = inspect_writer_epoch_family((root,))
    assert inspection.state == "unknown"
    assert "conflicts with its boundary" in str(inspection.reason)


def test_quiescence_is_children_before_root_and_receipt_is_durable() -> None:
    from crucible.workflows._writer_epoch import (
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    log: list[tuple[str, str]] = []
    family = _family(log=log)
    lease = _arm(family)
    log.clear()
    receipt = _scheduler_receipt(lease)
    quiesce_writer_epoch_family(lease, receipt)
    assert log == [
        ("grandchild", "quiesced"),
        ("child-b", "quiesced"),
        ("child-a", "quiesced"),
        ("root", "quiesced"),
    ]
    inspection = inspect_writer_epoch_family(tuple(reversed(family)))
    assert inspection.state == "quiesced"
    for member in family:
        assert member.record["settlement"] == receipt.to_record()


def test_partial_quiescence_keeps_root_armed_and_typed_retry_finishes() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochQuiescenceError,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    log: list[tuple[str, str]] = []
    root, child_a, child_b, grandchild = _family(log=log)
    lease = _arm((root, child_a, child_b, grandchild))
    receipt = _scheduler_receipt(lease, outcome="failed")
    log.clear()
    child_b.fail_before_state = "quiesced"
    with pytest.raises(WriterEpochQuiescenceError) as captured:
        quiesce_writer_epoch_family(lease, receipt)
    assert log == [("grandchild", "quiesced")]
    assert root.record["state"] == "armed"
    assert inspect_writer_epoch_family(lease.members).state == "partial"

    captured.value.retry_cleanup()
    assert inspect_writer_epoch_family(lease.members).state == "quiesced"
    assert root.record["settlement"]["outcome"] == "failed"


def test_quiescence_retry_accepts_post_publish_storage_error() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochQuiescenceError,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    root, child_a, child_b, grandchild = _family()
    lease = _arm((root, child_a, child_b, grandchild))
    receipt = _scheduler_receipt(lease)
    grandchild.fail_after_state = "quiesced"
    with pytest.raises(WriterEpochQuiescenceError) as captured:
        quiesce_writer_epoch_family(lease, receipt)
    captured.value.retry_cleanup()
    assert inspect_writer_epoch_family(lease.members).state == "quiesced"


def test_quiescence_requires_every_member_active_before_each_cas() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochQuiescenceError,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    log: list[tuple[str, str]] = []
    root, child_a, child_b, grandchild = _family(log=log)
    lease = _arm((root, child_a, child_b, grandchild))
    receipt = _scheduler_receipt(lease)
    log.clear()
    original_write = grandchild.compare_and_write_writer_epoch_record

    def quiesce_grandchild_then_retire_sibling(expected, replacement):
        result = original_write(expected, replacement)
        if replacement["state"] == "quiesced":
            child_b.active = False
        return result

    grandchild.compare_and_write_writer_epoch_record = (
        quiesce_grandchild_then_retire_sibling
    )
    with pytest.raises(WriterEpochQuiescenceError, match="inactive family"):
        quiesce_writer_epoch_family(lease, receipt)
    assert log == [("grandchild", "quiesced")]
    assert root.record["state"] == "armed"
    assert inspect_writer_epoch_family(lease.members).state == "partial"


def test_wrong_receipt_cannot_quiesce_an_exact_lease() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        WriterEpochBoundary,
        WriterEpochOwner,
        arm_writer_epoch_family,
        quiesce_writer_epoch_family,
        scheduler_settlement_receipt,
    )

    first = arm_writer_epoch_family(
        (_Storage("root-a", None),),
        owner=_epoch_owner("scheduler", "first"),
        boundary=WriterEpochBoundary(
            "scheduler",
            "first",
            "scheduler_join",
            ("backend",),
            ("lane",),
        ),
    )
    second = arm_writer_epoch_family(
        (_Storage("root-b", None),),
        owner=_epoch_owner("scheduler", "second"),
        boundary=WriterEpochBoundary(
            "scheduler",
            "second",
            "scheduler_join",
            ("backend",),
            ("lane",),
        ),
    )
    wrong = scheduler_settlement_receipt(
        second,
        settled_owner_ids=("backend",),
        joined_worker_ids=("lane",),
        evidence_digest=_sha("wrong"),
    )
    with pytest.raises(WriterEpochBlockedError, match="exact lease"):
        quiesce_writer_epoch_family(first, wrong)


def test_repeated_generations_are_monotonic_on_same_active_family() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochOwner,
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
        scheduler_settlement_receipt,
    )

    family = _family()
    first = _arm(family)
    quiesce_writer_epoch_family(first, _scheduler_receipt(first))
    second = arm_writer_epoch_family(
        tuple(reversed(family)),
        owner=_epoch_owner("runner", "build-verify"),
        boundary=WriterEpochBoundary(
            "runner",
            "build-verify",
            "scheduler_join",
            ("process-owner",),
            ("build", "verify"),
        ),
    )
    assert second.generation == first.generation + 1
    assert second.epoch_id != first.epoch_id
    receipt = scheduler_settlement_receipt(
        second,
        settled_owner_ids=("process-owner",),
        joined_worker_ids=("verify", "build"),
        evidence_digest=_sha("runner-joined"),
    )
    quiesce_writer_epoch_family(second, receipt)
    inspection = inspect_writer_epoch_family(second.members)
    assert inspection.state == "quiesced"
    assert inspection.generation == 2


@pytest.mark.parametrize("outcome", ("succeeded", "failed"))
def test_singleton_integration_callback_return_can_settle(outcome: str) -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochOwner,
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
        synchronous_callback_settlement_receipt,
    )

    integration = _Storage("integration", None)
    lease = arm_writer_epoch_family(
        (integration,),
        owner=_epoch_owner("scheduler", "serial-integration"),
        boundary=WriterEpochBoundary(
            "integration_gate",
            "union-gate",
            "synchronous_callback",
            (),
            ("integration-gate-callback",),
        ),
    )
    receipt = synchronous_callback_settlement_receipt(
        lease,
        settled_owner_ids=(),
        joined_worker_ids=("integration-gate-callback",),
        evidence_digest=_sha(f"gate-{outcome}"),
        outcome=outcome,
    )
    quiesce_writer_epoch_family(lease, receipt)
    assert inspect_writer_epoch_family((integration,)).state == "quiesced"


def test_integration_callback_exception_has_no_receipt_and_remains_armed() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        WriterEpochOwner,
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
    )

    integration = _Storage("integration", None)
    lease = arm_writer_epoch_family(
        (integration,),
        owner=_epoch_owner("scheduler", "serial-integration"),
        boundary=WriterEpochBoundary(
            "integration_gate",
            "union-gate",
            "synchronous_callback",
            (),
            ("integration-gate-callback",),
        ),
    )

    def gate() -> None:
        raise RuntimeError("callback may have launched descendants")

    with pytest.raises(RuntimeError, match="descendants"):
        gate()
    assert inspect_writer_epoch_family(lease.members).state == "armed"


def test_tampered_receipt_is_unknown_and_cannot_be_rearmed() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
        WriterEpochOwner,
        arm_writer_epoch_family,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
    )

    family = _family()
    lease = _arm(family)
    quiesce_writer_epoch_family(lease, _scheduler_receipt(lease))
    family[1].record["settlement"]["evidence_digest"] = _sha("tampered")
    inspection = inspect_writer_epoch_family(family)
    assert inspection.state == "unknown"
    with pytest.raises(WriterEpochBlockedError, match="non-quiescent"):
        arm_writer_epoch_family(
            family,
            owner=_epoch_owner("scheduler", "next"),
            boundary=lease.boundary,
        )


def test_inactive_member_cannot_be_armed() -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBlockedError,
    )

    root = _Storage("root", None)
    root.active = False
    with pytest.raises(WriterEpochBlockedError, match="not active"):
        _arm((root,))
