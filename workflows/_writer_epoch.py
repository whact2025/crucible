# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable writer-family epochs for mutation-journal settlement.

An invocation future returning is not proof that every process, tool transport,
or backend worker capable of mutating a checkout has stopped.  This module
provides the small state machine used to bind that proof to an exact mutation-
journal family before Git capture, rollback, commit, or terminal cleanup.

The persistence adapter deliberately has compare-and-write semantics.  Its
implementation must durably write, fsync, and reread the journal manifest
before returning.  Arming is root-before-members, so any interrupted arm leaves
the family root blocking settlement.  Quiescence is children-before-root, so
the root remains armed until every child has durably accepted the same typed
receipt.  Missing legacy state and every malformed or partial family are
UNKNOWN and fail closed; PID or boot identity is never settlement evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import secrets
import stat
from typing import Literal, Protocol, cast, runtime_checkable


WRITER_EPOCH_MANIFEST_KEY = "writer_epoch"
WRITER_EPOCH_SCHEMA = "crucible.writer-epoch.v1"
WRITER_SETTLEMENT_SCHEMA = "crucible.writer-settlement-receipt.v1"
WRITER_TERMINAL_PROOF_SCHEMA = "crucible.writer-terminal-proof.v1"

_PORTABLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_EPOCH_ID_RE = re.compile(r"epoch:[0-9a-f]{64}\Z")
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BOOT_ID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}\Z"
)

WriterEpochState = Literal[
    "never_armed",
    "armed",
    "quiesced",
    "unknown",
    "partial",
]
WriterSettlementKind = Literal[
    "provider_close_drain",
    "scheduler_join",
    "process_supervisor",
    "synchronous_callback",
]
WriterSettlementOutcome = Literal["succeeded", "failed"]
WriterTerminalState = Literal["committed", "rolled_back"]


class WriterEpochError(RuntimeError):
    """Base class for fail-closed writer-epoch failures."""


class WriterEpochFamilyError(WriterEpochError):
    """The supplied storage members do not form one exact rooted family."""


class WriterEpochBlockedError(WriterEpochError):
    """Writer settlement is unsafe because family state is not authoritative."""


class WriterEpochStorageError(WriterEpochError):
    """Durable writer-epoch persistence did not satisfy its contract."""


@runtime_checkable
class WriterEpochStorage(Protocol):
    """Descriptor-safe persistence supplied by a mutation journal.

    ``compare_and_write_writer_epoch_record`` is a durable compare-and-write:
    it must compare against a freshly read current manifest value, persist the
    replacement atomically, fsync the required files/directories, reread it
    through journal authority, and return that reread value.
    """

    @property
    def transaction_id(self) -> str: ...

    @property
    def parent_transaction_id(self) -> str | None: ...

    @property
    def active(self) -> bool: ...

    def read_writer_epoch_record(self) -> object: ...

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object: ...


def _validate_portable_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _PORTABLE_ID_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a portable 1-128 character identifier")
    return value


def _canonical_ids(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(
        _validate_portable_id(item, label=f"{label} member") for item in value
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class WriterEpochOwner:
    """Workflow owner plus diagnostic process provenance.

    Process ID, Linux ``/proc`` start time, and boot ID help an operator
    identify the process that armed a retained epoch.  They are never liveness
    leases: death, PID reuse, or a changed boot ID cannot quiesce an epoch.
    Only a typed receipt covering the complete boundary can do that.
    """

    kind: str
    identifier: str
    process_id: int
    process_start_time_ticks: int
    boot_id: str

    def __post_init__(self) -> None:
        _validate_portable_id(self.kind, label="WriterEpochOwner.kind")
        _validate_portable_id(
            self.identifier,
            label="WriterEpochOwner.identifier",
        )
        if (
            isinstance(self.process_id, bool)
            or not isinstance(self.process_id, int)
            or self.process_id <= 0
        ):
            raise ValueError("WriterEpochOwner.process_id must be positive")
        if (
            isinstance(self.process_start_time_ticks, bool)
            or not isinstance(self.process_start_time_ticks, int)
            or self.process_start_time_ticks <= 0
        ):
            raise ValueError(
                "WriterEpochOwner.process_start_time_ticks must be positive"
            )
        if (
            not isinstance(self.boot_id, str)
            or _BOOT_ID_RE.fullmatch(self.boot_id) is None
        ):
            raise ValueError("WriterEpochOwner.boot_id must be a lowercase UUID")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "process_id": self.process_id,
            "process_start_time_ticks": self.process_start_time_ticks,
            "boot_id": self.boot_id,
        }


def _read_proc_diagnostic(path: str, *, max_bytes: int) -> str:
    """Read one bounded, non-aliased procfs diagnostic file."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise WriterEpochStorageError(
            f"cannot open writer owner diagnostic {path}: {exc}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise WriterEpochStorageError(
                f"writer owner diagnostic is not a regular procfs file: {path}"
            )
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(4096, max_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > max_bytes:
                raise WriterEpochStorageError(
                    f"writer owner diagnostic exceeds its ceiling: {path}"
                )
            chunks.append(chunk)
        finished = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (
            finished.st_dev,
            finished.st_ino,
        ):
            raise WriterEpochStorageError(
                f"writer owner diagnostic changed while reading: {path}"
            )
        try:
            return b"".join(chunks).decode("ascii")
        except UnicodeDecodeError as exc:
            raise WriterEpochStorageError(
                f"writer owner diagnostic is not ASCII: {path}"
            ) from exc
    finally:
        os.close(descriptor)


def capture_writer_epoch_owner(kind: str, identifier: str) -> WriterEpochOwner:
    """Capture trusted Linux process diagnostics for a new epoch owner."""

    process_id = os.getpid()
    boot_id_before = _read_proc_diagnostic(
        "/proc/sys/kernel/random/boot_id",
        max_bytes=128,
    ).strip()
    process_stat = _read_proc_diagnostic(
        "/proc/self/stat",
        max_bytes=64 * 1024,
    ).strip()
    boot_id_after = _read_proc_diagnostic(
        "/proc/sys/kernel/random/boot_id",
        max_bytes=128,
    ).strip()
    if boot_id_before != boot_id_after:
        raise WriterEpochStorageError(
            "system boot identity changed during writer owner capture"
        )
    closing_parenthesis = process_stat.rfind(")")
    if closing_parenthesis <= 0:
        raise WriterEpochStorageError("/proc/self/stat is malformed")
    try:
        recorded_pid = int(process_stat[: process_stat.index(" ")])
        # The suffix begins with field 3 (state), so field 22 (starttime) is
        # zero-based suffix index 19.
        suffix = process_stat[closing_parenthesis + 2 :].split()
        start_time_ticks = int(suffix[19])
    except (IndexError, ValueError) as exc:
        raise WriterEpochStorageError("/proc/self/stat is malformed") from exc
    if recorded_pid != process_id or start_time_ticks <= 0:
        raise WriterEpochStorageError(
            "writer owner process identity changed during capture"
        )
    return WriterEpochOwner(
        kind=kind,
        identifier=identifier,
        process_id=process_id,
        process_start_time_ticks=start_time_ticks,
        boot_id=boot_id_before,
    )


@dataclass(frozen=True, slots=True)
class WriterEpochBoundary:
    """Exact writer population that must settle before quiescence.

    Owner IDs name close/drain-capable resource owners.  Worker IDs name
    futures, processes, synchronous callbacks, or other execution boundaries
    that must join or return.  Receipt builders require exact equality with
    both canonical sets, so a subset can never settle the epoch.
    """

    kind: str
    identifier: str
    settlement_kind: WriterSettlementKind
    expected_owner_ids: tuple[str, ...]
    expected_worker_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_portable_id(self.kind, label="WriterEpochBoundary.kind")
        _validate_portable_id(
            self.identifier,
            label="WriterEpochBoundary.identifier",
        )
        if self.settlement_kind not in {
            "provider_close_drain",
            "scheduler_join",
            "process_supervisor",
            "synchronous_callback",
        }:
            raise ValueError("WriterEpochBoundary.settlement_kind is unsupported")
        owners = _canonical_ids(
            self.expected_owner_ids,
            label="WriterEpochBoundary.expected_owner_ids",
        )
        workers = _canonical_ids(
            self.expected_worker_ids,
            label="WriterEpochBoundary.expected_worker_ids",
        )
        if not owners and not workers:
            raise ValueError(
                "WriterEpochBoundary requires at least one expected writer"
            )
        object.__setattr__(self, "expected_owner_ids", owners)
        object.__setattr__(self, "expected_worker_ids", workers)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "settlement_kind": self.settlement_kind,
            "expected_owner_ids": list(self.expected_owner_ids),
            "expected_worker_ids": list(self.expected_worker_ids),
        }


@dataclass(frozen=True, slots=True)
class _FamilyMember:
    transaction_id: str
    parent_transaction_id: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "parent_transaction_id": self.parent_transaction_id,
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _clone_record(value: object) -> object:
    return json.loads(_canonical_json(value))


def _canonical_family(
    members: tuple[WriterEpochStorage, ...],
    *,
    require_active: bool,
) -> tuple[tuple[WriterEpochStorage, ...], tuple[_FamilyMember, ...]]:
    if not isinstance(members, tuple):
        raise TypeError("writer epoch members must be a tuple")
    if not members:
        raise WriterEpochFamilyError("writer epoch family must not be empty")
    if len({id(member) for member in members}) != len(members):
        raise WriterEpochFamilyError(
            "writer epoch family contains a duplicate storage object"
        )

    by_id: dict[str, WriterEpochStorage] = {}
    parents: dict[str, str | None] = {}
    for member in members:
        if not isinstance(member, WriterEpochStorage):
            raise TypeError("writer epoch member does not implement WriterEpochStorage")
        transaction_id = _validate_portable_id(
            member.transaction_id,
            label="writer epoch transaction_id",
        )
        if transaction_id in by_id:
            raise WriterEpochFamilyError(
                f"duplicate writer epoch transaction: {transaction_id!r}"
            )
        parent = member.parent_transaction_id
        if parent is not None:
            parent = _validate_portable_id(
                parent,
                label="writer epoch parent_transaction_id",
            )
        if require_active and member.active is not True:
            raise WriterEpochBlockedError(
                f"writer epoch member is not active: {transaction_id!r}"
            )
        by_id[transaction_id] = member
        parents[transaction_id] = parent

    roots = [
        transaction_id for transaction_id, parent in parents.items() if parent is None
    ]
    if len(roots) != 1:
        raise WriterEpochFamilyError(
            "writer epoch family requires exactly one parentless root"
        )
    root = roots[0]
    foreign = sorted(
        {
            parent
            for transaction_id, parent in parents.items()
            if transaction_id != root and parent is not None and parent not in by_id
        }
    )
    if foreign:
        raise WriterEpochFamilyError(
            f"writer epoch family contains foreign-parent members: {foreign!r}"
        )

    depths: dict[str, int] = {root: 0}
    unresolved = set(by_id).difference({root})
    while unresolved:
        progressed = False
        for transaction_id in sorted(unresolved):
            parent = parents[transaction_id]
            if parent in depths:
                depths[transaction_id] = depths[parent] + 1
                unresolved.remove(transaction_id)
                progressed = True
                break
        if not progressed:
            raise WriterEpochFamilyError(
                "writer epoch family is disconnected or cyclic"
            )

    ordered_ids = tuple(sorted(by_id, key=lambda item: (depths[item], item)))
    ordered_members = tuple(by_id[item] for item in ordered_ids)
    family = tuple(_FamilyMember(item, parents[item]) for item in ordered_ids)
    return ordered_members, family


def _family_record(family: tuple[_FamilyMember, ...]) -> list[object]:
    return [member.to_record() for member in family]


def _family_digest(family: tuple[_FamilyMember, ...]) -> str:
    return _digest({"family": _family_record(family)})


def _validate_family_topology(
    family: tuple[_FamilyMember, ...],
) -> tuple[_FamilyMember, ...]:
    """Validate one canonical root-first family without storage objects."""

    if not family:
        raise ValueError("writer epoch family must not be empty")
    by_id: dict[str, _FamilyMember] = {}
    for member in family:
        if member.transaction_id in by_id:
            raise ValueError("writer epoch family contains duplicate members")
        by_id[member.transaction_id] = member
    roots = tuple(
        member.transaction_id
        for member in family
        if member.parent_transaction_id is None
    )
    if len(roots) != 1:
        raise ValueError("writer epoch family requires exactly one root")
    root = roots[0]
    depths: dict[str, int] = {root: 0}
    unresolved = set(by_id).difference({root})
    while unresolved:
        progressed = False
        for transaction_id in sorted(unresolved):
            parent = by_id[transaction_id].parent_transaction_id
            if parent in depths:
                depths[transaction_id] = depths[cast(str, parent)] + 1
                unresolved.remove(transaction_id)
                progressed = True
                break
        if not progressed:
            raise ValueError("writer epoch family is disconnected or cyclic")
    canonical = tuple(sorted(
        family,
        key=lambda member: (
            depths[member.transaction_id],
            member.transaction_id,
        ),
    ))
    if family != canonical:
        raise ValueError("writer epoch family order is noncanonical")
    return family


def _family_from_record(value: object) -> tuple[_FamilyMember, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("writer epoch family record is malformed")
    family: list[_FamilyMember] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "transaction_id",
            "parent_transaction_id",
        }:
            raise ValueError("writer epoch family record is malformed")
        transaction_id = _validate_portable_id(
            item["transaction_id"],
            label="writer epoch family transaction_id",
        )
        parent = item["parent_transaction_id"]
        if parent is not None:
            parent = _validate_portable_id(
                parent,
                label="writer epoch family parent_transaction_id",
            )
        family.append(_FamilyMember(transaction_id, parent))
    return _validate_family_topology(tuple(family))


def initial_writer_epoch_record() -> dict[str, object]:
    """Return capture-time v1 state for a journal that has never dispatched."""

    return {
        "schema": WRITER_EPOCH_SCHEMA,
        "generation": 0,
        "state": "never_armed",
        "epoch_id": None,
        "family_digest": None,
        "family": [],
        "owner": None,
        "boundary": None,
        "settlement": None,
    }


def _owner_from_record(value: object) -> WriterEpochOwner:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "identifier",
        "process_id",
        "process_start_time_ticks",
        "boot_id",
    }:
        raise ValueError("writer epoch owner is malformed")
    return WriterEpochOwner(
        kind=value["kind"],
        identifier=value["identifier"],
        process_id=value["process_id"],
        process_start_time_ticks=value["process_start_time_ticks"],
        boot_id=value["boot_id"],
    )


def _boundary_from_record(value: object) -> WriterEpochBoundary:
    if not isinstance(value, dict) or set(value) != {
        "kind",
        "identifier",
        "settlement_kind",
        "expected_owner_ids",
        "expected_worker_ids",
    }:
        raise ValueError("writer epoch boundary is malformed")
    owners = value["expected_owner_ids"]
    workers = value["expected_worker_ids"]
    if not isinstance(owners, list) or not isinstance(workers, list):
        raise ValueError("writer epoch boundary writer IDs are malformed")
    return WriterEpochBoundary(
        kind=value["kind"],
        identifier=value["identifier"],
        settlement_kind=value["settlement_kind"],
        expected_owner_ids=tuple(owners),
        expected_worker_ids=tuple(workers),
    )


@dataclass(frozen=True, slots=True)
class WriterSettlementReceipt:
    """Typed, epoch/family-bound proof that every expected writer settled."""

    kind: WriterSettlementKind
    outcome: WriterSettlementOutcome
    epoch_id: str
    generation: int
    family_digest: str
    owner: WriterEpochOwner
    boundary: WriterEpochBoundary
    settled_owner_ids: tuple[str, ...]
    joined_worker_ids: tuple[str, ...]
    evidence_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if self.kind not in {
            "provider_close_drain",
            "scheduler_join",
            "process_supervisor",
            "synchronous_callback",
        }:
            raise ValueError("writer settlement kind is unsupported")
        if self.outcome not in {"succeeded", "failed"}:
            raise ValueError("writer settlement outcome is unsupported")
        if self.kind != self.boundary.settlement_kind:
            raise ValueError("writer settlement kind conflicts with its boundary")
        if (
            not isinstance(self.epoch_id, str)
            or _EPOCH_ID_RE.fullmatch(self.epoch_id) is None
        ):
            raise ValueError("writer settlement epoch_id is malformed")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation <= 0
        ):
            raise ValueError("writer settlement generation is malformed")
        for label, digest in (
            ("family_digest", self.family_digest),
            ("evidence_digest", self.evidence_digest),
            ("receipt_digest", self.receipt_digest),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"writer settlement {label} is malformed")
        settled = _canonical_ids(
            self.settled_owner_ids,
            label="WriterSettlementReceipt.settled_owner_ids",
        )
        joined = _canonical_ids(
            self.joined_worker_ids,
            label="WriterSettlementReceipt.joined_worker_ids",
        )
        object.__setattr__(self, "settled_owner_ids", settled)
        object.__setattr__(self, "joined_worker_ids", joined)
        if settled != self.boundary.expected_owner_ids:
            raise ValueError(
                "writer settlement receipt omits or adds an expected owner"
            )
        if joined != self.boundary.expected_worker_ids:
            raise ValueError(
                "writer settlement receipt omits or adds an expected worker"
            )
        if self.receipt_digest != _digest(self._unsigned_record()):
            raise ValueError("writer settlement receipt digest is invalid")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema": WRITER_SETTLEMENT_SCHEMA,
            "kind": self.kind,
            "outcome": self.outcome,
            "epoch_id": self.epoch_id,
            "generation": self.generation,
            "family_digest": self.family_digest,
            "owner": self.owner.to_record(),
            "boundary": self.boundary.to_record(),
            "settled_owner_ids": list(self.settled_owner_ids),
            "joined_worker_ids": list(self.joined_worker_ids),
            "evidence_digest": self.evidence_digest,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "receipt_digest": self.receipt_digest}


def _receipt_from_record(value: object) -> WriterSettlementReceipt:
    expected_keys = {
        "schema",
        "kind",
        "outcome",
        "epoch_id",
        "generation",
        "family_digest",
        "owner",
        "boundary",
        "settled_owner_ids",
        "joined_worker_ids",
        "evidence_digest",
        "receipt_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("writer settlement receipt is malformed")
    if value.get("schema") != WRITER_SETTLEMENT_SCHEMA:
        raise ValueError("writer settlement receipt schema is unsupported")
    settled = value["settled_owner_ids"]
    joined = value["joined_worker_ids"]
    if not isinstance(settled, list) or not isinstance(joined, list):
        raise ValueError("writer settlement receipt writer IDs are malformed")
    return WriterSettlementReceipt(
        kind=value["kind"],
        outcome=value["outcome"],
        epoch_id=value["epoch_id"],
        generation=value["generation"],
        family_digest=value["family_digest"],
        owner=_owner_from_record(value["owner"]),
        boundary=_boundary_from_record(value["boundary"]),
        settled_owner_ids=tuple(settled),
        joined_worker_ids=tuple(joined),
        evidence_digest=value["evidence_digest"],
        receipt_digest=value["receipt_digest"],
    )


@dataclass(frozen=True, slots=True)
class _ParsedEpochRecord:
    generation: int
    state: Literal["never_armed", "armed", "quiesced"]
    epoch_id: str | None
    family_digest: str | None
    family: tuple[_FamilyMember, ...]
    owner: WriterEpochOwner | None
    boundary: WriterEpochBoundary | None
    settlement: WriterSettlementReceipt | None
    raw: dict[str, object]


def _parse_epoch_record(value: object) -> _ParsedEpochRecord:
    expected_keys = {
        "schema",
        "generation",
        "state",
        "epoch_id",
        "family_digest",
        "family",
        "owner",
        "boundary",
        "settlement",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("writer epoch record is missing or malformed")
    if value.get("schema") != WRITER_EPOCH_SCHEMA:
        raise ValueError("writer epoch schema is unsupported")
    generation = value.get("generation")
    state = value.get("state")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
        or state not in {"never_armed", "armed", "quiesced"}
    ):
        raise ValueError("writer epoch generation/state is malformed")
    if state == "never_armed":
        if value != initial_writer_epoch_record():
            raise ValueError("never-armed writer epoch record is malformed")
        return _ParsedEpochRecord(
            generation=0,
            state="never_armed",
            epoch_id=None,
            family_digest=None,
            family=(),
            owner=None,
            boundary=None,
            settlement=None,
            raw=dict(value),
        )
    epoch_id = value.get("epoch_id")
    family_digest = value.get("family_digest")
    raw_family = value.get("family")
    if (
        generation <= 0
        or not isinstance(epoch_id, str)
        or _EPOCH_ID_RE.fullmatch(epoch_id) is None
        or not isinstance(family_digest, str)
        or _SHA256_RE.fullmatch(family_digest) is None
        or not isinstance(raw_family, list)
        or not raw_family
    ):
        raise ValueError("armed writer epoch authority is malformed")
    family_tuple = _family_from_record(raw_family)
    if family_digest != _family_digest(family_tuple):
        raise ValueError("writer epoch family digest is invalid")
    owner = _owner_from_record(value.get("owner"))
    boundary = _boundary_from_record(value.get("boundary"))
    settlement_value = value.get("settlement")
    if state == "armed":
        if settlement_value is not None:
            raise ValueError("armed writer epoch cannot carry settlement")
        settlement = None
    else:
        settlement = _receipt_from_record(settlement_value)
        if (
            settlement.epoch_id != epoch_id
            or settlement.generation != generation
            or settlement.family_digest != family_digest
            or settlement.owner != owner
            or settlement.boundary != boundary
        ):
            raise ValueError("writer epoch settlement binding is invalid")
    return _ParsedEpochRecord(
        generation=generation,
        state=state,
        epoch_id=epoch_id,
        family_digest=family_digest,
        family=family_tuple,
        owner=owner,
        boundary=boundary,
        settlement=settlement,
        raw=dict(value),
    )


@dataclass(frozen=True, slots=True)
class WriterEpochTerminalProof:
    """Root-retained proof for one children-first terminal removal.

    The proof embeds the exact epoch record and canonical family that were
    checked while the retiring member's terminal journal inode was still
    held.  It is not a liveness lease: it authorizes only that member's
    removal from that exact prior family and generation.
    """

    transaction_id: str
    parent_transaction_id: str | None
    terminal_state: WriterTerminalState
    epoch_record: dict[str, object]
    family: tuple[tuple[str, str | None], ...]
    family_digest: str
    terminal_manifest_digest: str
    proof_digest: str

    def __post_init__(self) -> None:
        _validate_portable_id(
            self.transaction_id,
            label="WriterEpochTerminalProof.transaction_id",
        )
        if self.parent_transaction_id is not None:
            _validate_portable_id(
                self.parent_transaction_id,
                label="WriterEpochTerminalProof.parent_transaction_id",
            )
        if self.terminal_state not in {"committed", "rolled_back"}:
            raise ValueError("writer terminal proof state is unsupported")
        for label, digest in (
            ("family_digest", self.family_digest),
            ("terminal_manifest_digest", self.terminal_manifest_digest),
            ("proof_digest", self.proof_digest),
        ):
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError(f"writer terminal proof {label} is malformed")
        members = _validate_family_topology(tuple(
            _FamilyMember(transaction_id, parent)
            for transaction_id, parent in self.family
        ))
        if self.family_digest != _family_digest(members):
            raise ValueError("writer terminal proof family digest is invalid")
        member = next(
            (
                candidate
                for candidate in members
                if candidate.transaction_id == self.transaction_id
            ),
            None,
        )
        if member is None or member.parent_transaction_id != (
            self.parent_transaction_id
        ):
            raise ValueError("writer terminal proof member binding is invalid")
        parsed = _parse_epoch_record(self.epoch_record)
        if parsed.state not in {"never_armed", "quiesced"}:
            raise ValueError("writer terminal proof epoch is not settled")
        if parsed.state == "quiesced" and parsed.family != members:
            raise ValueError("writer terminal proof epoch family differs")
        if self.proof_digest != _digest(self._unsigned_record()):
            raise ValueError("writer terminal proof digest is invalid")

    def _unsigned_record(self) -> dict[str, object]:
        return {
            "schema": WRITER_TERMINAL_PROOF_SCHEMA,
            "transaction_id": self.transaction_id,
            "parent_transaction_id": self.parent_transaction_id,
            "terminal_state": self.terminal_state,
            "epoch_record": _clone_record(self.epoch_record),
            "family": [
                {
                    "transaction_id": transaction_id,
                    "parent_transaction_id": parent,
                }
                for transaction_id, parent in self.family
            ],
            "family_digest": self.family_digest,
            "terminal_manifest_digest": self.terminal_manifest_digest,
        }

    def to_record(self) -> dict[str, object]:
        return {**self._unsigned_record(), "proof_digest": self.proof_digest}


def build_writer_epoch_terminal_proof(
    *,
    transaction_id: str,
    parent_transaction_id: str | None,
    terminal_state: WriterTerminalState,
    epoch_record: object,
    family: tuple[tuple[str, str | None], ...],
    terminal_manifest: object,
) -> WriterEpochTerminalProof:
    """Build one digest-bound proof while the terminal member is still held."""

    parsed = _parse_epoch_record(epoch_record)
    members = _validate_family_topology(tuple(
        _FamilyMember(member_id, parent) for member_id, parent in family
    ))
    if parsed.state == "quiesced" and parsed.family != members:
        raise WriterEpochBlockedError(
            "terminal proof family differs from the settled epoch"
        )
    canonical_family = tuple(
        (member.transaction_id, member.parent_transaction_id)
        for member in members
    )
    unsigned = {
        "schema": WRITER_TERMINAL_PROOF_SCHEMA,
        "transaction_id": transaction_id,
        "parent_transaction_id": parent_transaction_id,
        "terminal_state": terminal_state,
        "epoch_record": _clone_record(parsed.raw),
        "family": _family_record(members),
        "family_digest": _family_digest(members),
        "terminal_manifest_digest": _digest({
            "terminal_manifest": terminal_manifest,
        }),
    }
    return WriterEpochTerminalProof(
        transaction_id=transaction_id,
        parent_transaction_id=parent_transaction_id,
        terminal_state=terminal_state,
        epoch_record=cast(dict[str, object], _clone_record(parsed.raw)),
        family=canonical_family,
        family_digest=cast(str, unsigned["family_digest"]),
        terminal_manifest_digest=cast(
            str,
            unsigned["terminal_manifest_digest"],
        ),
        proof_digest=_digest(unsigned),
    )


def parse_writer_epoch_terminal_proof(
    value: object,
) -> WriterEpochTerminalProof:
    """Parse and authenticate one root-retained terminal proof record."""

    expected_keys = {
        "schema",
        "transaction_id",
        "parent_transaction_id",
        "terminal_state",
        "epoch_record",
        "family",
        "family_digest",
        "terminal_manifest_digest",
        "proof_digest",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("writer terminal proof is malformed")
    if value.get("schema") != WRITER_TERMINAL_PROOF_SCHEMA:
        raise ValueError("writer terminal proof schema is unsupported")
    members = _family_from_record(value.get("family"))
    epoch_record = value.get("epoch_record")
    if not isinstance(epoch_record, dict):
        raise ValueError("writer terminal proof epoch record is malformed")
    return WriterEpochTerminalProof(
        transaction_id=value["transaction_id"],
        parent_transaction_id=value["parent_transaction_id"],
        terminal_state=value["terminal_state"],
        epoch_record=cast(dict[str, object], _clone_record(epoch_record)),
        family=tuple(
            (member.transaction_id, member.parent_transaction_id)
            for member in members
        ),
        family_digest=value["family_digest"],
        terminal_manifest_digest=value["terminal_manifest_digest"],
        proof_digest=value["proof_digest"],
    )


def _validate_generation_zero_family_transition(
    *,
    epoch_record: object,
    previous_family: tuple[tuple[str, str | None], ...],
    replacement_family: tuple[tuple[str, str | None], ...],
    terminal_proofs: tuple[object, ...],
) -> None:
    """Authenticate a generation-zero projection before its first arm.

    A generation-zero epoch record intentionally carries no family.  Mutation
    journals therefore retain a separate monotonic family projection.  The
    projection is durable authority, not advisory registry state: omitting a
    historical member is allowed only after the retained root has recorded
    that member's exact children-first terminal proof.
    """

    parsed = _parse_epoch_record(epoch_record)
    if parsed.state != "never_armed" or parsed.generation != 0:
        raise WriterEpochBlockedError(
            "generation-zero family transition requires a pristine epoch"
        )
    try:
        previous = _validate_family_topology(tuple(
            _FamilyMember(transaction_id, parent)
            for transaction_id, parent in previous_family
        ))
        replacement = _validate_family_topology(tuple(
            _FamilyMember(transaction_id, parent)
            for transaction_id, parent in replacement_family
        ))
    except (TypeError, ValueError) as exc:
        raise WriterEpochBlockedError(
            f"generation-zero family projection is UNKNOWN: {exc}"
        ) from exc
    previous_by_id = {
        member.transaction_id: member for member in previous
    }
    replacement_by_id = {
        member.transaction_id: member for member in replacement
    }
    previous_root = previous[0]
    replacement_root = replacement[0]
    if (
        previous_root.transaction_id != replacement_root.transaction_id
        or previous_root.parent_transaction_id is not None
        or replacement_root.parent_transaction_id is not None
    ):
        raise WriterEpochBlockedError(
            "generation-zero family transition changed its retained root"
        )
    for transaction_id in previous_by_id.keys() & replacement_by_id.keys():
        if previous_by_id[transaction_id].parent_transaction_id != (
            replacement_by_id[transaction_id].parent_transaction_id
        ):
            raise WriterEpochBlockedError(
                "generation-zero retained member was reparented: "
                f"{transaction_id!r}"
            )
    try:
        parsed_proofs = tuple(
            parse_writer_epoch_terminal_proof(proof)
            for proof in terminal_proofs
        )
    except (TypeError, ValueError) as exc:
        raise WriterEpochBlockedError(
            f"generation-zero terminal proof set is UNKNOWN: {exc}"
        ) from exc
    if len({proof.transaction_id for proof in parsed_proofs}) != len(
        parsed_proofs
    ):
        raise WriterEpochBlockedError(
            "generation-zero terminal proof set contains duplicate members"
        )
    proofs = {
        proof.transaction_id: proof
        for proof in parsed_proofs
        if proof.epoch_record == parsed.raw
    }
    removed = previous_by_id.keys() - replacement_by_id.keys()
    expected_family = tuple(
        (member.transaction_id, member.parent_transaction_id)
        for member in previous
    )
    expected_digest = _family_digest(previous)
    for transaction_id in sorted(removed):
        proof = proofs.get(transaction_id)
        if (
            proof is None
            or proof.parent_transaction_id
            != previous_by_id[transaction_id].parent_transaction_id
            or proof.family != expected_family
            or proof.family_digest != expected_digest
        ):
            raise WriterEpochBlockedError(
                "generation-zero family rollover lacks an exact terminal "
                f"proof for {transaction_id!r}"
            )


@dataclass(frozen=True, slots=True)
class WriterEpochFamilyInspection:
    """Read-only classification of one canonical journal family."""

    state: WriterEpochState
    member_ids: tuple[str, ...]
    family_digest: str
    generation: int | None = None
    epoch_id: str | None = None
    reason: str | None = None


def inspect_writer_epoch_family(
    members: tuple[WriterEpochStorage, ...],
) -> WriterEpochFamilyInspection:
    """Classify exact family state without repairing or normalizing it."""

    ordered, family = _canonical_family(members, require_active=False)
    expected_digest = _family_digest(family)
    member_ids = tuple(member.transaction_id for member in ordered)
    parsed: list[_ParsedEpochRecord] = []
    for member in ordered:
        try:
            parsed.append(_parse_epoch_record(member.read_writer_epoch_record()))
        except BaseException as exc:  # noqa: BLE001 - UNKNOWN is fail-closed
            return WriterEpochFamilyInspection(
                state="unknown",
                member_ids=member_ids,
                family_digest=expected_digest,
                reason=(f"{member.transaction_id}: {type(exc).__name__}: {exc}"),
            )
    if all(record.state == "never_armed" for record in parsed):
        return WriterEpochFamilyInspection(
            state="never_armed",
            member_ids=member_ids,
            family_digest=expected_digest,
            generation=0,
        )
    if any(record.state == "never_armed" for record in parsed):
        return WriterEpochFamilyInspection(
            state="partial",
            member_ids=member_ids,
            family_digest=expected_digest,
            reason="family mixes never-armed and generation-bound records",
        )
    expected_family = tuple(family)
    if any(
        record.family != expected_family or record.family_digest != expected_digest
        for record in parsed
    ):
        return WriterEpochFamilyInspection(
            state="partial",
            member_ids=member_ids,
            family_digest=expected_digest,
            reason="family membership differs from durable epoch authority",
        )
    first_raw = _canonical_json(parsed[0].raw)
    if any(_canonical_json(record.raw) != first_raw for record in parsed[1:]):
        return WriterEpochFamilyInspection(
            state="partial",
            member_ids=member_ids,
            family_digest=expected_digest,
            reason="family members carry different epoch records",
        )
    state = parsed[0].state
    return WriterEpochFamilyInspection(
        state=state,
        member_ids=member_ids,
        family_digest=expected_digest,
        generation=parsed[0].generation,
        epoch_id=parsed[0].epoch_id,
    )


def require_writer_epoch_quiescence(
    members: tuple[WriterEpochStorage, ...],
    *,
    allow_never_armed: bool = True,
) -> WriterEpochFamilyInspection:
    """Require a state safe for Git/source capture or terminal settlement."""

    inspection = inspect_writer_epoch_family(members)
    safe = inspection.state == "quiesced" or (
        allow_never_armed and inspection.state == "never_armed"
    )
    if not safe:
        raise WriterEpochBlockedError(
            "writer epoch family is not quiescent: "
            f"state={inspection.state} reason={inspection.reason or 'none'}"
        )
    return inspection


def _armed_record(
    *,
    generation: int,
    epoch_id: str,
    family: tuple[_FamilyMember, ...],
    owner: WriterEpochOwner,
    boundary: WriterEpochBoundary,
) -> dict[str, object]:
    return {
        "schema": WRITER_EPOCH_SCHEMA,
        "generation": generation,
        "state": "armed",
        "epoch_id": epoch_id,
        "family_digest": _family_digest(family),
        "family": _family_record(family),
        "owner": owner.to_record(),
        "boundary": boundary.to_record(),
        "settlement": None,
    }


@dataclass(frozen=True, slots=True)
class WriterEpochLease:
    """In-memory retry authority for one exact durable generation."""

    epoch_id: str
    generation: int
    family_digest: str
    owner: WriterEpochOwner
    boundary: WriterEpochBoundary
    members: tuple[WriterEpochStorage, ...]
    _family: tuple[_FamilyMember, ...]
    _previous_records: tuple[object, ...]

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(member.transaction_id for member in self.members)

    def armed_record(self) -> dict[str, object]:
        return _armed_record(
            generation=self.generation,
            epoch_id=self.epoch_id,
            family=self._family,
            owner=self.owner,
            boundary=self.boundary,
        )


class WriterEpochArmError(WriterEpochStorageError):
    """Exact owner for retrying an interrupted root-first arm."""

    def __init__(self, lease: WriterEpochLease, error: BaseException) -> None:
        self.lease = lease
        self.error = error
        super().__init__(
            "writer epoch arm remains partial and blocking: "
            f"{type(error).__name__}: {error}"
        )

    def retry_arm(self) -> WriterEpochLease:
        return retry_writer_epoch_arm(self.lease)


def _write_and_reprove(
    member: WriterEpochStorage,
    expected: object,
    replacement: dict[str, object],
) -> None:
    returned = member.compare_and_write_writer_epoch_record(
        _clone_record(expected),
        cast(dict[str, object], _clone_record(replacement)),
    )
    if returned != replacement:
        raise WriterEpochStorageError(
            f"writer epoch storage returned an unproved write for "
            f"{member.transaction_id!r}"
        )
    reread = member.read_writer_epoch_record()
    if reread != replacement:
        raise WriterEpochStorageError(
            f"writer epoch durable reread changed for {member.transaction_id!r}"
        )


def _read_root_terminal_proofs(
    root: WriterEpochStorage,
) -> tuple[WriterEpochTerminalProof, ...]:
    """Read optional journal-adapter proofs without widening the protocol."""

    reader = getattr(root, "read_writer_epoch_terminal_proofs", None)
    if reader is None or not callable(reader):
        return ()
    raw = reader()
    if not isinstance(raw, tuple):
        raise WriterEpochBlockedError(
            "writer terminal proof reader must return a tuple"
        )
    try:
        proofs = tuple(parse_writer_epoch_terminal_proof(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise WriterEpochBlockedError(
            f"writer terminal proof set is UNKNOWN: {exc}"
        ) from exc
    if len({proof.transaction_id for proof in proofs}) != len(proofs):
        raise WriterEpochBlockedError(
            "writer terminal proof set contains duplicate members"
        )
    return proofs


def prepare_writer_epoch_arm(
    members: tuple[WriterEpochStorage, ...],
    *,
    owner: WriterEpochOwner,
    boundary: WriterEpochBoundary,
) -> WriterEpochLease:
    """Validate and reserve one exact generation without durable writes.

    The returned lease is complete retry authority. Callers which can receive
    asynchronous exceptions retain it before invoking
    :func:`retry_writer_epoch_arm`, so no exception can land after the first
    durable write but before the lease is reachable.
    """

    if not isinstance(owner, WriterEpochOwner):
        raise TypeError("owner must be WriterEpochOwner")
    if not isinstance(boundary, WriterEpochBoundary):
        raise TypeError("boundary must be WriterEpochBoundary")
    ordered, family = _canonical_family(members, require_active=True)
    inspection = inspect_writer_epoch_family(ordered)
    if inspection.state not in {"never_armed", "quiesced", "partial"}:
        raise WriterEpochBlockedError(
            "cannot arm writer epoch from non-quiescent family state: "
            f"{inspection.state} ({inspection.reason or 'no detail'})"
        )
    previous = tuple(
        _clone_record(member.read_writer_epoch_record()) for member in ordered
    )
    try:
        parsed_previous = tuple(_parse_epoch_record(record) for record in previous)
    except (TypeError, ValueError) as exc:
        raise WriterEpochBlockedError(
            f"cannot arm writer epoch from UNKNOWN snapshot: {exc}"
        ) from exc
    snapshot_is_never_armed = all(
        record.state == "never_armed" for record in parsed_previous
    )
    snapshot_is_exact_quiesced = (
        all(record.state == "quiesced" for record in parsed_previous)
        and all(record.family == family for record in parsed_previous)
        and all(
            record.family_digest == _family_digest(family) for record in parsed_previous
        )
        and all(
            _canonical_json(record.raw) == _canonical_json(parsed_previous[0].raw)
            for record in parsed_previous[1:]
        )
    )
    generation: int
    if snapshot_is_never_armed:
        prearm_validator = getattr(
            ordered[0],
            "validate_writer_epoch_prearm_family",
            None,
        )
        if prearm_validator is not None:
            if not callable(prearm_validator):
                raise WriterEpochBlockedError(
                    "writer epoch pre-arm validator is not callable"
                )
            prearm_validator(
                previous[0],
                tuple(
                    (member.transaction_id, member.parent_transaction_id)
                    for member in family
                ),
            )
        generation = 1
    elif snapshot_is_exact_quiesced:
        generation = parsed_previous[0].generation + 1
    else:
        # Strict family rollover: the surviving root's last quiesced record is
        # the prior-generation certificate; retained members must carry that
        # byte-identical record, added members must be pristine generation
        # zero, and every removed prior member needs a root-retained terminal
        # proof from children-first cleanup.  The lease below writes this root
        # first, so a crash during rollover leaves every frontier blocked.
        root_previous = parsed_previous[0]
        new_root = family[0]
        if (
            new_root.parent_transaction_id is not None
            or root_previous.state != "quiesced"
            or not root_previous.family
            or root_previous.family[0].transaction_id
            != new_root.transaction_id
            or root_previous.family[0].parent_transaction_id is not None
        ):
            raise WriterEpochBlockedError(
                "cannot arm writer epoch from a raced, armed, or partial snapshot"
            )
        old_family = root_previous.family
        old_by_id = {
            member.transaction_id: member for member in old_family
        }
        new_by_id = {
            member.transaction_id: member for member in family
        }
        root_raw = _canonical_json(root_previous.raw)
        for member, record in zip(ordered, parsed_previous, strict=True):
            old_member = old_by_id.get(member.transaction_id)
            if old_member is None:
                if record.state != "never_armed" or record.generation != 0:
                    raise WriterEpochBlockedError(
                        "new writer-family member is not pristine generation zero: "
                        f"{member.transaction_id!r}"
                    )
                continue
            if (
                old_member.parent_transaction_id
                != new_by_id[member.transaction_id].parent_transaction_id
                or record.state != "quiesced"
                or _canonical_json(record.raw) != root_raw
            ):
                raise WriterEpochBlockedError(
                    "retained writer-family member differs from the prior "
                    f"quiesced certificate: {member.transaction_id!r}"
                )
        removed = set(old_by_id).difference(new_by_id)
        proofs = {
            proof.transaction_id: proof
            for proof in _read_root_terminal_proofs(ordered[0])
            if proof.epoch_record == root_previous.raw
        }
        missing_proofs = removed.difference(proofs)
        if missing_proofs:
            raise WriterEpochBlockedError(
                "writer-family rollover lacks authenticated terminal proofs: "
                f"{sorted(missing_proofs)!r}"
            )
        expected_old_family = tuple(
            (member.transaction_id, member.parent_transaction_id)
            for member in old_family
        )
        for transaction_id in removed:
            proof = proofs[transaction_id]
            old_member = old_by_id[transaction_id]
            if (
                proof.parent_transaction_id
                != old_member.parent_transaction_id
                or proof.family != expected_old_family
                or proof.family_digest != root_previous.family_digest
            ):
                raise WriterEpochBlockedError(
                    "writer-family rollover terminal proof differs from prior "
                    f"authority: {transaction_id!r}"
                )
        generation = root_previous.generation + 1
    epoch_id = "epoch:" + secrets.token_hex(32)
    return WriterEpochLease(
        epoch_id=epoch_id,
        generation=generation,
        family_digest=_family_digest(family),
        owner=owner,
        boundary=boundary,
        members=ordered,
        _family=family,
        _previous_records=previous,
    )


def arm_writer_epoch_family(
    members: tuple[WriterEpochStorage, ...],
    *,
    owner: WriterEpochOwner,
    boundary: WriterEpochBoundary,
) -> WriterEpochLease:
    """Prepare and durably arm one fresh generation root first."""

    lease = prepare_writer_epoch_arm(
        members,
        owner=owner,
        boundary=boundary,
    )
    try:
        return retry_writer_epoch_arm(lease)
    except WriterEpochArmError:
        raise
    except BaseException as exc:  # noqa: BLE001 - retain exact lease
        raise WriterEpochArmError(lease, exc) from exc


def retry_writer_epoch_arm(lease: WriterEpochLease) -> WriterEpochLease:
    """Idempotently finish only the exact generation carried by ``lease``."""

    if not isinstance(lease, WriterEpochLease):
        raise TypeError("lease must be WriterEpochLease")
    replacement = lease.armed_record()
    try:
        for member, previous in zip(
            lease.members,
            lease._previous_records,
            strict=True,
        ):
            if member.active is not True:
                raise WriterEpochBlockedError(
                    f"writer epoch member is not active: {member.transaction_id!r}"
                )
            current = member.read_writer_epoch_record()
            if current == replacement:
                continue
            if current != previous:
                raise WriterEpochBlockedError(
                    "writer epoch arm retry encountered foreign state: "
                    f"{member.transaction_id!r}"
                )
            _write_and_reprove(member, current, replacement)
        inspection = inspect_writer_epoch_family(lease.members)
        if (
            inspection.state != "armed"
            or inspection.epoch_id != lease.epoch_id
            or inspection.generation != lease.generation
            or inspection.family_digest != lease.family_digest
        ):
            raise WriterEpochStorageError(
                "writer epoch family did not reach exact armed state"
            )
        return lease
    except WriterEpochArmError:
        raise
    except BaseException as exc:  # noqa: BLE001 - retain exact lease
        raise WriterEpochArmError(lease, exc) from exc


def _settlement_receipt(
    lease: WriterEpochLease,
    *,
    kind: WriterSettlementKind,
    outcome: WriterSettlementOutcome,
    settled_owner_ids: tuple[str, ...],
    joined_worker_ids: tuple[str, ...],
    evidence_digest: str,
) -> WriterSettlementReceipt:
    if not isinstance(lease, WriterEpochLease):
        raise TypeError("lease must be WriterEpochLease")
    if kind != lease.boundary.settlement_kind:
        raise WriterEpochBlockedError(
            "writer settlement kind does not match the exact boundary"
        )
    owners = _canonical_ids(
        settled_owner_ids,
        label="settled_owner_ids",
    )
    workers = _canonical_ids(
        joined_worker_ids,
        label="joined_worker_ids",
    )
    if owners != lease.boundary.expected_owner_ids:
        raise WriterEpochBlockedError(
            "writer settlement does not cover the exact expected owners"
        )
    if workers != lease.boundary.expected_worker_ids:
        raise WriterEpochBlockedError(
            "writer settlement does not cover the exact expected workers"
        )
    if outcome not in {"succeeded", "failed"}:
        raise ValueError("writer settlement outcome is unsupported")
    if (
        not isinstance(evidence_digest, str)
        or _SHA256_RE.fullmatch(evidence_digest) is None
    ):
        raise ValueError("writer settlement evidence_digest is malformed")
    unsigned = {
        "schema": WRITER_SETTLEMENT_SCHEMA,
        "kind": kind,
        "outcome": outcome,
        "epoch_id": lease.epoch_id,
        "generation": lease.generation,
        "family_digest": lease.family_digest,
        "owner": lease.owner.to_record(),
        "boundary": lease.boundary.to_record(),
        "settled_owner_ids": list(owners),
        "joined_worker_ids": list(workers),
        "evidence_digest": evidence_digest,
    }
    return WriterSettlementReceipt(
        kind=kind,
        outcome=outcome,
        epoch_id=lease.epoch_id,
        generation=lease.generation,
        family_digest=lease.family_digest,
        owner=lease.owner,
        boundary=lease.boundary,
        settled_owner_ids=owners,
        joined_worker_ids=workers,
        evidence_digest=evidence_digest,
        receipt_digest=_digest(unsigned),
    )


def provider_settlement_receipt(
    lease: WriterEpochLease,
    *,
    settled_owner_ids: tuple[str, ...],
    joined_worker_ids: tuple[str, ...] = (),
    evidence_digest: str,
    outcome: WriterSettlementOutcome = "succeeded",
) -> WriterSettlementReceipt:
    """Bind trusted provider close/drain evidence to an exact epoch."""

    return _settlement_receipt(
        lease,
        kind="provider_close_drain",
        outcome=outcome,
        settled_owner_ids=settled_owner_ids,
        joined_worker_ids=joined_worker_ids,
        evidence_digest=evidence_digest,
    )


def scheduler_settlement_receipt(
    lease: WriterEpochLease,
    *,
    settled_owner_ids: tuple[str, ...],
    joined_worker_ids: tuple[str, ...],
    evidence_digest: str,
    outcome: WriterSettlementOutcome = "succeeded",
) -> WriterSettlementReceipt:
    """Bind complete provider closure plus executor drain/join evidence."""

    return _settlement_receipt(
        lease,
        kind="scheduler_join",
        outcome=outcome,
        settled_owner_ids=settled_owner_ids,
        joined_worker_ids=joined_worker_ids,
        evidence_digest=evidence_digest,
    )


def process_supervisor_settlement_receipt(
    lease: WriterEpochLease,
    *,
    settled_owner_ids: tuple[str, ...],
    joined_worker_ids: tuple[str, ...],
    evidence_digest: str,
    outcome: WriterSettlementOutcome,
) -> WriterSettlementReceipt:
    """Bind an exact supervised-process termination and join boundary."""

    return _settlement_receipt(
        lease,
        kind="process_supervisor",
        outcome=outcome,
        settled_owner_ids=settled_owner_ids,
        joined_worker_ids=joined_worker_ids,
        evidence_digest=evidence_digest,
    )


def synchronous_callback_settlement_receipt(
    lease: WriterEpochLease,
    *,
    settled_owner_ids: tuple[str, ...],
    joined_worker_ids: tuple[str, ...],
    evidence_digest: str,
    outcome: WriterSettlementOutcome,
) -> WriterSettlementReceipt:
    """Bind an exact trusted synchronous callback return.

    Callers must invoke this only after the callback returns its typed result.
    An exception produces no receipt and deliberately leaves the lease armed.
    """

    return _settlement_receipt(
        lease,
        kind="synchronous_callback",
        outcome=outcome,
        settled_owner_ids=settled_owner_ids,
        joined_worker_ids=joined_worker_ids,
        evidence_digest=evidence_digest,
    )


class WriterEpochQuiescenceError(WriterEpochStorageError):
    """Exact retry owner for interrupted children-before-root settlement."""

    def __init__(
        self,
        lease: WriterEpochLease,
        receipt: WriterSettlementReceipt,
        error: BaseException,
    ) -> None:
        self.lease = lease
        self.receipt = receipt
        self.error = error
        super().__init__(
            "writer epoch quiescence remains partial and blocking: "
            f"{type(error).__name__}: {error}"
        )

    def retry_cleanup(self) -> None:
        quiesce_writer_epoch_family(self.lease, self.receipt)


def _quiesced_record(
    lease: WriterEpochLease,
    receipt: WriterSettlementReceipt,
) -> dict[str, object]:
    record = lease.armed_record()
    record["state"] = "quiesced"
    record["settlement"] = receipt.to_record()
    return record


def quiesce_writer_epoch_family(
    lease: WriterEpochLease,
    receipt: WriterSettlementReceipt,
) -> None:
    """Durably settle children before the root; exact retries are idempotent."""

    if not isinstance(lease, WriterEpochLease):
        raise TypeError("lease must be WriterEpochLease")
    if not isinstance(receipt, WriterSettlementReceipt):
        raise TypeError("receipt must be WriterSettlementReceipt")
    if (
        receipt.epoch_id != lease.epoch_id
        or receipt.generation != lease.generation
        or receipt.family_digest != lease.family_digest
        or receipt.owner != lease.owner
        or receipt.boundary != lease.boundary
    ):
        raise WriterEpochBlockedError(
            "writer settlement receipt does not bind the exact lease"
        )
    armed = lease.armed_record()
    quiesced = _quiesced_record(lease, receipt)
    try:

        def require_all_members_active() -> None:
            inactive = tuple(
                member.transaction_id
                for member in lease.members
                if member.active is not True
            )
            if inactive:
                raise WriterEpochBlockedError(
                    f"writer epoch cannot quiesce an inactive family: {inactive!r}"
                )

        require_all_members_active()
        # Reject foreign/malformed state before making this attempt partial.
        for member in lease.members:
            current = member.read_writer_epoch_record()
            if current != armed and current != quiesced:
                raise WriterEpochBlockedError(
                    "writer epoch quiescence encountered foreign state: "
                    f"{member.transaction_id!r}"
                )
        # Canonical members are root-first, therefore reverse order keeps the
        # root armed until every descendant has accepted the receipt.
        for member in reversed(lease.members):
            # A sibling becoming terminal is a family-wide authority change,
            # not permission to finish settling the remaining records.
            require_all_members_active()
            current = member.read_writer_epoch_record()
            if current == quiesced:
                continue
            _write_and_reprove(member, current, quiesced)
        inspection = inspect_writer_epoch_family(lease.members)
        if (
            inspection.state != "quiesced"
            or inspection.epoch_id != lease.epoch_id
            or inspection.generation != lease.generation
            or inspection.family_digest != lease.family_digest
        ):
            raise WriterEpochStorageError(
                "writer epoch family did not reach exact quiesced state"
            )
    except WriterEpochQuiescenceError:
        raise
    except BaseException as exc:  # noqa: BLE001 - retain lease + receipt
        raise WriterEpochQuiescenceError(lease, receipt, exc) from exc


__all__ = [
    "WRITER_EPOCH_MANIFEST_KEY",
    "WRITER_EPOCH_SCHEMA",
    "WRITER_SETTLEMENT_SCHEMA",
    "WRITER_TERMINAL_PROOF_SCHEMA",
    "WriterEpochArmError",
    "WriterEpochBlockedError",
    "WriterEpochBoundary",
    "WriterEpochError",
    "WriterEpochFamilyError",
    "WriterEpochFamilyInspection",
    "WriterEpochLease",
    "WriterEpochOwner",
    "WriterEpochQuiescenceError",
    "WriterEpochStorage",
    "WriterEpochStorageError",
    "WriterEpochTerminalProof",
    "WriterSettlementReceipt",
    "arm_writer_epoch_family",
    "build_writer_epoch_terminal_proof",
    "capture_writer_epoch_owner",
    "initial_writer_epoch_record",
    "inspect_writer_epoch_family",
    "prepare_writer_epoch_arm",
    "process_supervisor_settlement_receipt",
    "provider_settlement_receipt",
    "quiesce_writer_epoch_family",
    "parse_writer_epoch_terminal_proof",
    "require_writer_epoch_quiescence",
    "retry_writer_epoch_arm",
    "scheduler_settlement_receipt",
    "synchronous_callback_settlement_receipt",
]
