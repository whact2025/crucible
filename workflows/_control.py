# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable, fingerprint-bound workflow control state.

This module coordinates long-running autonomous workflows. It keeps heartbeat, pause/resume/cancel intent, steering,
blocker/next-action progress, participant cancellation state, and typed
approval events in one crash-atomic record.  Every record is bound to the
exact run, route, and tool-authority fingerprints.

Control writes use an optimistic semantic revision inside the same per-run
file lock used for atomic publication.  Heartbeats and cancellation delivery
bookkeeping advance the record revision without invalidating an operator's
semantic revision. Workflow-owned control bindings coordinate these records
with each provider boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ._durable_state import AtomicStateStore, DurableStateError


CONTROL_STATE_SCHEMA_VERSION = "crucible.workflow-control.v1"
CONTROL_STORE_SCHEMA_VERSION = "crucible.workflow-control-store.v1"
CONTROL_STATE_DIRNAME = ".crucible-control"
_MAX_RECORD_BYTES = 8 * 1024 * 1024
_SHA256_PREFIX = "sha256:"

DesiredState: TypeAlias = Literal[
    "running",
    "paused",
    "cancelled",
    "completed",
]
ParticipantState: TypeAlias = Literal[
    "active",
    "cancel_signaled",
    "cancel_failed",
    "drained",
]
ApprovalDecision: TypeAlias = Literal["approved", "denied"]

_DESIRED_STATES = frozenset({
    "running",
    "paused",
    "cancelled",
    "completed",
})
_PARTICIPANT_STATES = frozenset({
    "active",
    "cancel_signaled",
    "cancel_failed",
    "drained",
})
_APPROVAL_DECISIONS = frozenset({"approved", "denied"})


class WorkflowControlError(RuntimeError):
    """Base error for unsafe or conflicting workflow control state."""


class WorkflowControlStoreError(WorkflowControlError):
    """The durable control record is corrupt, unsafe, or unavailable."""


class WorkflowControlIdentityMismatch(WorkflowControlError):
    """A controller attempted to use state owned by another exact run."""


class StaleControlRevision(WorkflowControlError):
    """A semantic control writer lost an optimistic concurrency race."""


class StaleSteeringRevision(WorkflowControlError):
    """A steering or approval action targets an obsolete instruction."""


class ApprovalConflict(WorkflowControlError):
    """An approval ID, binding, or decision conflicts with durable history."""


class ParticipantConflict(WorkflowControlError):
    """A cancellation participant ID was rebound to different authority."""


def _require_digest(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != len(_SHA256_PREFIX) + 64
        or not value.startswith(_SHA256_PREFIX)
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(
            f"{field} must be 'sha256:' followed by 64 lowercase "
            "hexadecimal digits"
        )
    return value


def _require_text(
    value: str,
    *,
    field: str,
    max_bytes: int,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not allow_empty and not value.strip())
        or len(value.encode("utf-8")) > max_bytes
        or any(ord(character) < 0x20 for character in value)
    ):
        qualifier = "control-free UTF-8 string"
        if not allow_empty:
            qualifier = "non-empty, " + qualifier
        raise ValueError(
            f"{field} must be a {qualifier} no larger than "
            f"{max_bytes} bytes"
        )
    return value


def _require_exact_nonnegative_int(value: int, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be an exact integer >= 0")
    return value


def _canonical_json(value: Any, *, field: str) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key")
            result[key] = _canonical_json(
                child,
                field=f"{field}.{key}",
            )
        return result
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
    ):
        return [
            _canonical_json(child, field=f"{field}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(
        f"{field} contains non-JSON value {type(value).__name__}"
    )


def control_value_digest(value: Any, *, domain: str) -> str:
    """Return one domain-separated SHA-256 digest."""

    _require_text(
        domain,
        field="digest domain",
        max_bytes=1024,
    )
    payload = {
        "domain": domain,
        "value": _canonical_json(value, field=domain),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _SHA256_PREFIX + hashlib.sha256(encoded).hexdigest()


_CONTROL_STORE_KEY = control_value_digest(
    CONTROL_STATE_SCHEMA_VERSION,
    domain="workflow_control.singleton_store_key",
)


@dataclass(frozen=True, slots=True)
class ControlIdentity:
    """The exact run, backend route, and tool authority being controlled."""

    run_fingerprint: str
    route_fingerprint: str
    authority_fingerprint: str

    def __post_init__(self) -> None:
        for field_name in (
            "run_fingerprint",
            "route_fingerprint",
            "authority_fingerprint",
        ):
            _require_digest(
                getattr(self, field_name),
                field=f"ControlIdentity.{field_name}",
            )

    @property
    def fingerprint(self) -> str:
        return control_value_digest(
            {
                "run_fingerprint": self.run_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "authority_fingerprint": self.authority_fingerprint,
            },
            domain="workflow_control.identity",
        )


@dataclass(frozen=True, slots=True)
class SteeringRecord:
    revision: int
    instruction: str
    instruction_digest: str
    actor: str
    created_at: str

    def __post_init__(self) -> None:
        if (
            type(self.revision) is not int
            or self.revision < 1
        ):
            raise ValueError("SteeringRecord.revision must be an integer >= 1")
        _require_text(
            self.instruction,
            field="SteeringRecord.instruction",
            max_bytes=64 * 1024,
        )
        expected = control_value_digest(
            self.instruction,
            domain="workflow_control.steering_instruction",
        )
        if self.instruction_digest != expected:
            raise ValueError(
                "SteeringRecord.instruction_digest does not match instruction"
            )
        _require_text(
            self.actor,
            field="SteeringRecord.actor",
            max_bytes=1024,
        )
        _require_text(
            self.created_at,
            field="SteeringRecord.created_at",
            max_bytes=128,
        )


@dataclass(frozen=True, slots=True)
class ProgressRecord:
    blocker: str | None
    next_action: str | None
    actor: str
    updated_at: str

    def __post_init__(self) -> None:
        for field_name in ("blocker", "next_action"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(
                    value,
                    field=f"ProgressRecord.{field_name}",
                    max_bytes=64 * 1024,
                )
        _require_text(
            self.actor,
            field="ProgressRecord.actor",
            max_bytes=1024,
        )
        _require_text(
            self.updated_at,
            field="ProgressRecord.updated_at",
            max_bytes=128,
        )


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    participant_id: str
    kind: str
    registration_digest: str
    state: ParticipantState
    registered_at: str
    updated_at: str
    error_digest: str | None = None

    def __post_init__(self) -> None:
        _require_text(
            self.participant_id,
            field="ParticipantRecord.participant_id",
            max_bytes=1024,
        )
        _require_text(
            self.kind,
            field="ParticipantRecord.kind",
            max_bytes=1024,
        )
        _require_digest(
            self.registration_digest,
            field="ParticipantRecord.registration_digest",
        )
        if self.state not in _PARTICIPANT_STATES:
            raise ValueError(
                "ParticipantRecord.state must be one of "
                f"{sorted(_PARTICIPANT_STATES)}"
            )
        _require_text(
            self.registered_at,
            field="ParticipantRecord.registered_at",
            max_bytes=128,
        )
        _require_text(
            self.updated_at,
            field="ParticipantRecord.updated_at",
            max_bytes=128,
        )
        if self.error_digest is not None:
            _require_digest(
                self.error_digest,
                field="ParticipantRecord.error_digest",
            )
        if self.state != "cancel_failed" and self.error_digest is not None:
            raise ValueError(
                "ParticipantRecord.error_digest is allowed only for "
                "cancel_failed state"
            )


@dataclass(frozen=True, slots=True)
class ApprovalRequestEvent:
    """An identity- and steering-bound request for one privileged action."""

    sequence: int
    request_id: str
    control_identity_fingerprint: str
    run_fingerprint: str
    route_fingerprint: str
    authority_fingerprint: str
    steering_revision: int
    action: str
    resource_digest: str
    rationale: str
    requested_by: str
    requested_at: str
    event_kind: Literal["approval_request"] = "approval_request"

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError(
                "ApprovalRequestEvent.sequence must be an integer >= 1"
            )
        _require_text(
            self.request_id,
            field="ApprovalRequestEvent.request_id",
            max_bytes=1024,
        )
        for field_name in (
            "control_identity_fingerprint",
            "run_fingerprint",
            "route_fingerprint",
            "authority_fingerprint",
            "resource_digest",
        ):
            _require_digest(
                getattr(self, field_name),
                field=f"ApprovalRequestEvent.{field_name}",
            )
        _require_exact_nonnegative_int(
            self.steering_revision,
            field="ApprovalRequestEvent.steering_revision",
        )
        _require_text(
            self.action,
            field="ApprovalRequestEvent.action",
            max_bytes=4096,
        )
        _require_text(
            self.rationale,
            field="ApprovalRequestEvent.rationale",
            max_bytes=64 * 1024,
        )
        _require_text(
            self.requested_by,
            field="ApprovalRequestEvent.requested_by",
            max_bytes=1024,
        )
        _require_text(
            self.requested_at,
            field="ApprovalRequestEvent.requested_at",
            max_bytes=128,
        )
        if self.event_kind != "approval_request":
            raise ValueError(
                "ApprovalRequestEvent.event_kind must be approval_request"
            )

    @property
    def fingerprint(self) -> str:
        return control_value_digest(
            {
                "request_id": self.request_id,
                "control_identity_fingerprint": (
                    self.control_identity_fingerprint
                ),
                "run_fingerprint": self.run_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "authority_fingerprint": self.authority_fingerprint,
                "steering_revision": self.steering_revision,
                "action": self.action,
                "resource_digest": self.resource_digest,
                "rationale": self.rationale,
                "requested_by": self.requested_by,
                "requested_at": self.requested_at,
            },
            domain="workflow_control.approval_request",
        )


@dataclass(frozen=True, slots=True)
class ApprovalDecisionEvent:
    """One immutable decision over an exact approval request fingerprint."""

    sequence: int
    decision_id: str
    request_id: str
    request_fingerprint: str
    control_identity_fingerprint: str
    run_fingerprint: str
    route_fingerprint: str
    authority_fingerprint: str
    steering_revision: int
    decision: ApprovalDecision
    reason: str
    decided_by: str
    decided_at: str
    event_kind: Literal["approval_decision"] = "approval_decision"

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError(
                "ApprovalDecisionEvent.sequence must be an integer >= 1"
            )
        for field_name in ("decision_id", "request_id"):
            _require_text(
                getattr(self, field_name),
                field=f"ApprovalDecisionEvent.{field_name}",
                max_bytes=1024,
            )
        for field_name in (
            "request_fingerprint",
            "control_identity_fingerprint",
            "run_fingerprint",
            "route_fingerprint",
            "authority_fingerprint",
        ):
            _require_digest(
                getattr(self, field_name),
                field=f"ApprovalDecisionEvent.{field_name}",
            )
        _require_exact_nonnegative_int(
            self.steering_revision,
            field="ApprovalDecisionEvent.steering_revision",
        )
        if self.decision not in _APPROVAL_DECISIONS:
            raise ValueError(
                "ApprovalDecisionEvent.decision must be approved or denied"
            )
        _require_text(
            self.reason,
            field="ApprovalDecisionEvent.reason",
            max_bytes=64 * 1024,
        )
        _require_text(
            self.decided_by,
            field="ApprovalDecisionEvent.decided_by",
            max_bytes=1024,
        )
        _require_text(
            self.decided_at,
            field="ApprovalDecisionEvent.decided_at",
            max_bytes=128,
        )
        if self.event_kind != "approval_decision":
            raise ValueError(
                "ApprovalDecisionEvent.event_kind must be approval_decision"
            )

    @property
    def fingerprint(self) -> str:
        return control_value_digest(
            {
                "decision_id": self.decision_id,
                "request_id": self.request_id,
                "request_fingerprint": self.request_fingerprint,
                "control_identity_fingerprint": (
                    self.control_identity_fingerprint
                ),
                "run_fingerprint": self.run_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "authority_fingerprint": self.authority_fingerprint,
                "steering_revision": self.steering_revision,
                "decision": self.decision,
                "reason": self.reason,
                "decided_by": self.decided_by,
                "decided_at": self.decided_at,
            },
            domain="workflow_control.approval_decision",
        )


ApprovalEvent: TypeAlias = ApprovalRequestEvent | ApprovalDecisionEvent


@dataclass(frozen=True, slots=True)
class WorkflowControlState:
    identity: ControlIdentity
    record_revision: int = 0
    control_revision: int = 0
    desired_state: DesiredState = "running"
    heartbeat_sequence: int = 0
    heartbeat_at: str | None = None
    heartbeat_owner: str | None = None
    steering_revision: int = 0
    steering: SteeringRecord | None = None
    progress: ProgressRecord | None = None
    participants: tuple[ParticipantRecord, ...] = ()
    approval_events: tuple[ApprovalEvent, ...] = ()
    schema_version: str = CONTROL_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ControlIdentity):
            raise TypeError(
                "WorkflowControlState.identity must be ControlIdentity"
            )
        for field_name in (
            "record_revision",
            "control_revision",
            "heartbeat_sequence",
            "steering_revision",
        ):
            _require_exact_nonnegative_int(
                getattr(self, field_name),
                field=f"WorkflowControlState.{field_name}",
            )
        if self.control_revision > self.record_revision:
            raise ValueError(
                "WorkflowControlState.control_revision cannot exceed "
                "record_revision"
            )
        if self.desired_state not in _DESIRED_STATES:
            raise ValueError(
                "WorkflowControlState.desired_state must be one of "
                f"{sorted(_DESIRED_STATES)}"
            )
        if self.heartbeat_sequence == 0:
            if self.heartbeat_at is not None or self.heartbeat_owner is not None:
                raise ValueError(
                    "zero heartbeat sequence cannot have heartbeat metadata"
                )
        else:
            assert self.heartbeat_at is not None
            assert self.heartbeat_owner is not None
            _require_text(
                self.heartbeat_at,
                field="WorkflowControlState.heartbeat_at",
                max_bytes=128,
            )
            _require_text(
                self.heartbeat_owner,
                field="WorkflowControlState.heartbeat_owner",
                max_bytes=1024,
            )
        if self.steering_revision == 0:
            if self.steering is not None:
                raise ValueError(
                    "zero steering revision cannot have a steering record"
                )
        elif (
            self.steering is None
            or self.steering.revision != self.steering_revision
        ):
            raise ValueError(
                "steering record must match the current steering revision"
            )
        if self.progress is not None and not isinstance(
            self.progress,
            ProgressRecord,
        ):
            raise TypeError(
                "WorkflowControlState.progress must be ProgressRecord or None"
            )
        if not isinstance(self.participants, tuple):
            raise TypeError(
                "WorkflowControlState.participants must be a tuple"
            )
        participant_ids: set[str] = set()
        for participant in self.participants:
            if not isinstance(participant, ParticipantRecord):
                raise TypeError(
                    "WorkflowControlState participant must be "
                    "ParticipantRecord"
                )
            if participant.participant_id in participant_ids:
                raise ValueError("participant IDs must be unique")
            participant_ids.add(participant.participant_id)
        if not isinstance(self.approval_events, tuple):
            raise TypeError(
                "WorkflowControlState.approval_events must be a tuple"
            )
        requests: dict[str, ApprovalRequestEvent] = {}
        decided_requests: set[str] = set()
        decision_ids: set[str] = set()
        for sequence, event in enumerate(self.approval_events, start=1):
            if not isinstance(
                event,
                (ApprovalRequestEvent, ApprovalDecisionEvent),
            ):
                raise TypeError(
                    "WorkflowControlState approval event has unknown type"
                )
            if event.sequence != sequence:
                raise ValueError(
                    "approval event sequence must be contiguous and ordered"
                )
            self._validate_event_identity(event)
            if isinstance(event, ApprovalRequestEvent):
                if event.request_id in requests:
                    raise ValueError("approval request IDs must be unique")
                requests[event.request_id] = event
            else:
                request = requests.get(event.request_id)
                if request is None:
                    raise ValueError(
                        "approval decision must follow its request"
                    )
                if event.request_fingerprint != request.fingerprint:
                    raise ValueError(
                        "approval decision request fingerprint mismatch"
                    )
                if event.request_id in decided_requests:
                    raise ValueError(
                        "approval request cannot have multiple decisions"
                    )
                if event.decision_id in decision_ids:
                    raise ValueError("approval decision IDs must be unique")
                decided_requests.add(event.request_id)
                decision_ids.add(event.decision_id)
        if self.schema_version != CONTROL_STATE_SCHEMA_VERSION:
            raise ValueError(
                "WorkflowControlState.schema_version must be "
                f"{CONTROL_STATE_SCHEMA_VERSION!r}"
            )

    def _validate_event_identity(self, event: ApprovalEvent) -> None:
        expected = self.identity
        if (
            event.control_identity_fingerprint != expected.fingerprint
            or event.run_fingerprint != expected.run_fingerprint
            or event.route_fingerprint != expected.route_fingerprint
            or event.authority_fingerprint != expected.authority_fingerprint
        ):
            raise ValueError(
                "approval event does not match workflow control identity"
            )

    @property
    def blocker(self) -> str | None:
        return None if self.progress is None else self.progress.blocker

    @property
    def next_action(self) -> str | None:
        return None if self.progress is None else self.progress.next_action


class WorkflowControlCodec:
    """Strict canonical JSON codec for workflow control state."""

    _IDENTITY_FIELDS = frozenset({
        "run_fingerprint",
        "route_fingerprint",
        "authority_fingerprint",
    })
    _STEERING_FIELDS = frozenset({
        "revision",
        "instruction",
        "instruction_digest",
        "actor",
        "created_at",
    })
    _PROGRESS_FIELDS = frozenset({
        "blocker",
        "next_action",
        "actor",
        "updated_at",
    })
    _PARTICIPANT_FIELDS = frozenset({
        "participant_id",
        "kind",
        "registration_digest",
        "state",
        "registered_at",
        "updated_at",
        "error_digest",
    })
    _REQUEST_FIELDS = frozenset({
        "event_kind",
        "sequence",
        "request_id",
        "control_identity_fingerprint",
        "run_fingerprint",
        "route_fingerprint",
        "authority_fingerprint",
        "steering_revision",
        "action",
        "resource_digest",
        "rationale",
        "requested_by",
        "requested_at",
    })
    _DECISION_FIELDS = frozenset({
        "event_kind",
        "sequence",
        "decision_id",
        "request_id",
        "request_fingerprint",
        "control_identity_fingerprint",
        "run_fingerprint",
        "route_fingerprint",
        "authority_fingerprint",
        "steering_revision",
        "decision",
        "reason",
        "decided_by",
        "decided_at",
    })
    _STATE_FIELDS = frozenset({
        "schema_version",
        "identity",
        "record_revision",
        "control_revision",
        "desired_state",
        "heartbeat_sequence",
        "heartbeat_at",
        "heartbeat_owner",
        "steering_revision",
        "steering",
        "progress",
        "participants",
        "approval_events",
    })

    @staticmethod
    def _event_to_mapping(event: ApprovalEvent) -> dict[str, Any]:
        if isinstance(event, ApprovalRequestEvent):
            return {
                "event_kind": event.event_kind,
                "sequence": event.sequence,
                "request_id": event.request_id,
                "control_identity_fingerprint": (
                    event.control_identity_fingerprint
                ),
                "run_fingerprint": event.run_fingerprint,
                "route_fingerprint": event.route_fingerprint,
                "authority_fingerprint": event.authority_fingerprint,
                "steering_revision": event.steering_revision,
                "action": event.action,
                "resource_digest": event.resource_digest,
                "rationale": event.rationale,
                "requested_by": event.requested_by,
                "requested_at": event.requested_at,
            }
        return {
            "event_kind": event.event_kind,
            "sequence": event.sequence,
            "decision_id": event.decision_id,
            "request_id": event.request_id,
            "request_fingerprint": event.request_fingerprint,
            "control_identity_fingerprint": (
                event.control_identity_fingerprint
            ),
            "run_fingerprint": event.run_fingerprint,
            "route_fingerprint": event.route_fingerprint,
            "authority_fingerprint": event.authority_fingerprint,
            "steering_revision": event.steering_revision,
            "decision": event.decision,
            "reason": event.reason,
            "decided_by": event.decided_by,
            "decided_at": event.decided_at,
        }

    @classmethod
    def to_mapping(cls, state: WorkflowControlState) -> dict[str, Any]:
        if not isinstance(state, WorkflowControlState):
            raise TypeError("state must be WorkflowControlState")
        steering = state.steering
        progress = state.progress
        return {
            "schema_version": state.schema_version,
            "identity": {
                "run_fingerprint": state.identity.run_fingerprint,
                "route_fingerprint": state.identity.route_fingerprint,
                "authority_fingerprint": state.identity.authority_fingerprint,
            },
            "record_revision": state.record_revision,
            "control_revision": state.control_revision,
            "desired_state": state.desired_state,
            "heartbeat_sequence": state.heartbeat_sequence,
            "heartbeat_at": state.heartbeat_at,
            "heartbeat_owner": state.heartbeat_owner,
            "steering_revision": state.steering_revision,
            "steering": (
                None
                if steering is None
                else {
                    "revision": steering.revision,
                    "instruction": steering.instruction,
                    "instruction_digest": steering.instruction_digest,
                    "actor": steering.actor,
                    "created_at": steering.created_at,
                }
            ),
            "progress": (
                None
                if progress is None
                else {
                    "blocker": progress.blocker,
                    "next_action": progress.next_action,
                    "actor": progress.actor,
                    "updated_at": progress.updated_at,
                }
            ),
            "participants": [
                {
                    "participant_id": participant.participant_id,
                    "kind": participant.kind,
                    "registration_digest": (
                        participant.registration_digest
                    ),
                    "state": participant.state,
                    "registered_at": participant.registered_at,
                    "updated_at": participant.updated_at,
                    "error_digest": participant.error_digest,
                }
                for participant in state.participants
            ],
            "approval_events": [
                cls._event_to_mapping(event)
                for event in state.approval_events
            ],
        }

    @classmethod
    def encode(cls, state: WorkflowControlState) -> bytes:
        return json.dumps(
            cls.to_mapping(state),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkflowControlStoreError(
                    f"control JSON contains duplicate field {key!r}"
                )
            result[key] = value
        return result

    @classmethod
    def parse_json(cls, payload: bytes) -> Any:
        if not isinstance(payload, bytes):
            raise TypeError("control payload must be bytes")
        if len(payload) > _MAX_RECORD_BYTES:
            raise WorkflowControlStoreError(
                f"control record exceeds {_MAX_RECORD_BYTES} bytes"
            )
        try:
            return json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=cls._reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    WorkflowControlStoreError(
                        f"control JSON contains invalid constant {value}"
                    )
                ),
            )
        except WorkflowControlStoreError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkflowControlStoreError(
                "control record is not strict UTF-8 JSON"
            ) from exc

    @classmethod
    def _event_from_mapping(cls, raw: Any) -> ApprovalEvent:
        if not isinstance(raw, Mapping):
            raise WorkflowControlStoreError(
                "approval event must be a JSON object"
            )
        event_kind = raw.get("event_kind")
        try:
            if event_kind == "approval_request":
                if set(raw) != cls._REQUEST_FIELDS:
                    raise WorkflowControlStoreError(
                        "approval request fields do not match closed schema"
                    )
                return ApprovalRequestEvent(**raw)
            if event_kind == "approval_decision":
                if set(raw) != cls._DECISION_FIELDS:
                    raise WorkflowControlStoreError(
                        "approval decision fields do not match closed schema"
                    )
                return ApprovalDecisionEvent(**raw)
        except (TypeError, ValueError) as exc:
            raise WorkflowControlStoreError(
                f"invalid approval event: {exc}"
            ) from exc
        raise WorkflowControlStoreError(
            f"unknown approval event kind {event_kind!r}"
        )

    @classmethod
    def from_mapping(cls, raw: Any) -> WorkflowControlState:
        if not isinstance(raw, Mapping) or set(raw) != cls._STATE_FIELDS:
            raise WorkflowControlStoreError(
                "control state fields do not match the closed schema"
            )
        identity_raw = raw["identity"]
        if (
            not isinstance(identity_raw, Mapping)
            or set(identity_raw) != cls._IDENTITY_FIELDS
        ):
            raise WorkflowControlStoreError(
                "control identity fields do not match the closed schema"
            )
        steering_raw = raw["steering"]
        progress_raw = raw["progress"]
        participants_raw = raw["participants"]
        events_raw = raw["approval_events"]
        if not isinstance(participants_raw, list):
            raise WorkflowControlStoreError(
                "control participants must be an array"
            )
        if not isinstance(events_raw, list):
            raise WorkflowControlStoreError(
                "control approval_events must be an array"
            )
        try:
            identity = ControlIdentity(**identity_raw)
            steering = None
            if steering_raw is not None:
                if (
                    not isinstance(steering_raw, Mapping)
                    or set(steering_raw) != cls._STEERING_FIELDS
                ):
                    raise WorkflowControlStoreError(
                        "steering fields do not match the closed schema"
                    )
                steering = SteeringRecord(**steering_raw)
            progress = None
            if progress_raw is not None:
                if (
                    not isinstance(progress_raw, Mapping)
                    or set(progress_raw) != cls._PROGRESS_FIELDS
                ):
                    raise WorkflowControlStoreError(
                        "progress fields do not match the closed schema"
                    )
                progress = ProgressRecord(**progress_raw)
            participants: list[ParticipantRecord] = []
            for participant_raw in participants_raw:
                if (
                    not isinstance(participant_raw, Mapping)
                    or set(participant_raw) != cls._PARTICIPANT_FIELDS
                ):
                    raise WorkflowControlStoreError(
                        "participant fields do not match the closed schema"
                    )
                participants.append(ParticipantRecord(**participant_raw))
            return WorkflowControlState(
                schema_version=raw["schema_version"],
                identity=identity,
                record_revision=raw["record_revision"],
                control_revision=raw["control_revision"],
                desired_state=raw["desired_state"],
                heartbeat_sequence=raw["heartbeat_sequence"],
                heartbeat_at=raw["heartbeat_at"],
                heartbeat_owner=raw["heartbeat_owner"],
                steering_revision=raw["steering_revision"],
                steering=steering,
                progress=progress,
                participants=tuple(participants),
                approval_events=tuple(
                    cls._event_from_mapping(event)
                    for event in events_raw
                ),
            )
        except WorkflowControlStoreError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorkflowControlStoreError(
                f"invalid workflow control state: {exc}"
            ) from exc

    @classmethod
    def decode(cls, payload: bytes) -> WorkflowControlState:
        return cls.from_mapping(cls.parse_json(payload))


class WorkflowControlStore:
    """One locked, checksummed control record for a run directory."""

    _ENVELOPE_FIELDS = frozenset({
        "schema_version",
        "state_digest",
        "state",
    })

    def __init__(self, root: Path | str) -> None:
        try:
            self._atomic = AtomicStateStore(
                root,
                max_record_bytes=_MAX_RECORD_BYTES,
            )
        except DurableStateError as exc:
            raise WorkflowControlStoreError(
                f"workflow control store is unavailable: {exc}"
            ) from exc
        self.root = self._atomic.root

    @property
    def closed(self) -> bool:
        return self._atomic.closed

    def close(self) -> None:
        """Release the retained workflow-control root descriptor."""

        try:
            self._atomic.close()
        except DurableStateError as exc:
            raise WorkflowControlStoreError(str(exc)) from exc

    def __enter__(self) -> WorkflowControlStore:
        self._atomic.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @classmethod
    def for_run(cls, run_dir: Path | str) -> WorkflowControlStore:
        return cls(Path(run_dir) / CONTROL_STATE_DIRNAME)

    @staticmethod
    def _record_bytes(state: WorkflowControlState) -> bytes:
        state_payload = WorkflowControlCodec.encode(state)
        envelope = {
            "schema_version": CONTROL_STORE_SCHEMA_VERSION,
            "state_digest": (
                _SHA256_PREFIX + hashlib.sha256(state_payload).hexdigest()
            ),
            "state": WorkflowControlCodec.to_mapping(state),
        }
        return json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def _decode_record(cls, payload: bytes) -> WorkflowControlState:
        raw = WorkflowControlCodec.parse_json(payload)
        if not isinstance(raw, Mapping) or set(raw) != cls._ENVELOPE_FIELDS:
            raise WorkflowControlStoreError(
                "control store envelope fields do not match closed schema"
            )
        if raw["schema_version"] != CONTROL_STORE_SCHEMA_VERSION:
            raise WorkflowControlStoreError(
                "unsupported control store schema version"
            )
        state = WorkflowControlCodec.from_mapping(raw["state"])
        actual_digest = (
            _SHA256_PREFIX
            + hashlib.sha256(WorkflowControlCodec.encode(state)).hexdigest()
        )
        if raw["state_digest"] != actual_digest:
            raise WorkflowControlStoreError(
                "workflow control state checksum mismatch"
            )
        return state

    def load(self) -> WorkflowControlState | None:
        try:
            payload = self._atomic.load(_CONTROL_STORE_KEY)
        except DurableStateError as exc:
            raise WorkflowControlStoreError(str(exc)) from exc
        return None if payload is None else self._decode_record(payload)

    def update(
        self,
        transform: Callable[
            [WorkflowControlState | None],
            WorkflowControlState,
        ],
    ) -> WorkflowControlState:
        if not callable(transform):
            raise TypeError("control transform must be callable")
        result: WorkflowControlState | None = None

        def _transform(payload: bytes | None) -> bytes:
            nonlocal result
            current = (
                None if payload is None else self._decode_record(payload)
            )
            updated = transform(current)
            if not isinstance(updated, WorkflowControlState):
                raise TypeError(
                    "control transform must return WorkflowControlState"
                )
            result = updated
            return self._record_bytes(updated)

        try:
            self._atomic.update(_CONTROL_STORE_KEY, _transform)
        except DurableStateError as exc:
            raise WorkflowControlStoreError(str(exc)) from exc
        assert result is not None
        return result


class WorkflowControlController:
    """Atomic mutations over one exact workflow control identity."""

    def __init__(
        self,
        *,
        store: WorkflowControlStore,
        identity: ControlIdentity,
    ) -> None:
        self.store = store
        self.identity = identity

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> WorkflowControlController:
        self.store.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _validate(
        self,
        state: WorkflowControlState,
    ) -> WorkflowControlState:
        if state.identity != self.identity:
            raise WorkflowControlIdentityMismatch(
                "workflow control identity does not match exact "
                "run/route/authority fingerprints"
            )
        return state

    def initialize(self) -> WorkflowControlState:
        def _initialize(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            if current is None:
                return WorkflowControlState(identity=self.identity)
            return self._validate(current)

        return self.store.update(_initialize)

    def load(self) -> WorkflowControlState | None:
        state = self.store.load()
        return None if state is None else self._validate(state)

    def _require_current(
        self,
        current: WorkflowControlState | None,
    ) -> WorkflowControlState:
        if current is None:
            raise WorkflowControlError(
                "workflow control state must be initialized first"
            )
        return self._validate(current)

    @staticmethod
    def _require_control_revision(
        state: WorkflowControlState,
        expected: int,
    ) -> None:
        _require_exact_nonnegative_int(
            expected,
            field="expected_control_revision",
        )
        if state.control_revision != expected:
            raise StaleControlRevision(
                "stale control revision: "
                f"expected {expected}, current {state.control_revision}"
            )

    def heartbeat(
        self,
        *,
        owner: str,
        at: str,
        expected_sequence: int | None = None,
    ) -> WorkflowControlState:
        _require_text(owner, field="heartbeat owner", max_bytes=1024)
        _require_text(at, field="heartbeat time", max_bytes=128)
        if expected_sequence is not None:
            _require_exact_nonnegative_int(
                expected_sequence,
                field="expected heartbeat sequence",
            )

        def _heartbeat(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            state = self._require_current(current)
            if (
                expected_sequence is not None
                and state.heartbeat_sequence != expected_sequence
            ):
                raise StaleControlRevision(
                    "stale heartbeat sequence: "
                    f"expected {expected_sequence}, "
                    f"current {state.heartbeat_sequence}"
                )
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                heartbeat_sequence=state.heartbeat_sequence + 1,
                heartbeat_at=at,
                heartbeat_owner=owner,
            )

        return self.store.update(_heartbeat)

    def pause(
        self,
        *,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        return self._set_desired_state(
            "paused",
            expected_control_revision=expected_control_revision,
        )

    def resume(
        self,
        *,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        return self._set_desired_state(
            "running",
            expected_control_revision=expected_control_revision,
        )

    def request_cancel(
        self,
        *,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        return self._set_desired_state(
            "cancelled",
            expected_control_revision=expected_control_revision,
        )

    def complete(
        self,
        *,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        return self._set_desired_state(
            "completed",
            expected_control_revision=expected_control_revision,
        )

    def _set_desired_state(
        self,
        desired_state: DesiredState,
        *,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        if desired_state not in _DESIRED_STATES:
            raise ValueError(f"unknown desired state {desired_state!r}")

        def _transition(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            state = self._require_current(current)
            if state.desired_state == desired_state:
                return state
            self._require_control_revision(
                state,
                expected_control_revision,
            )
            if state.desired_state in {"cancelled", "completed"}:
                raise WorkflowControlError(
                    f"terminal control state {state.desired_state!r} "
                    "cannot transition"
                )
            if desired_state == "completed" and state.desired_state == "paused":
                raise WorkflowControlError(
                    "paused workflow must resume or cancel before completion"
                )
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                control_revision=state.control_revision + 1,
                desired_state=desired_state,
            )

        return self.store.update(_transition)

    def steer(
        self,
        *,
        instruction: str,
        actor: str,
        created_at: str,
        expected_steering_revision: int,
    ) -> WorkflowControlState:
        _require_text(
            instruction,
            field="steering instruction",
            max_bytes=64 * 1024,
        )
        _require_text(actor, field="steering actor", max_bytes=1024)
        _require_text(
            created_at,
            field="steering created_at",
            max_bytes=128,
        )
        _require_exact_nonnegative_int(
            expected_steering_revision,
            field="expected_steering_revision",
        )
        instruction_digest = control_value_digest(
            instruction,
            domain="workflow_control.steering_instruction",
        )

        def _steer(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            state = self._require_current(current)
            existing = state.steering
            if (
                existing is not None
                and existing.instruction_digest == instruction_digest
                and existing.actor == actor
            ):
                return state
            if state.steering_revision != expected_steering_revision:
                raise StaleSteeringRevision(
                    "stale steering revision: "
                    f"expected {expected_steering_revision}, "
                    f"current {state.steering_revision}"
                )
            if state.desired_state in {"cancelled", "completed"}:
                raise WorkflowControlError(
                    "terminal workflow cannot accept steering"
                )
            revision = state.steering_revision + 1
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                control_revision=state.control_revision + 1,
                steering_revision=revision,
                steering=SteeringRecord(
                    revision=revision,
                    instruction=instruction,
                    instruction_digest=instruction_digest,
                    actor=actor,
                    created_at=created_at,
                ),
            )

        return self.store.update(_steer)

    def set_progress(
        self,
        *,
        blocker: str | None,
        next_action: str | None,
        actor: str,
        updated_at: str,
        expected_control_revision: int,
    ) -> WorkflowControlState:
        progress = ProgressRecord(
            blocker=blocker,
            next_action=next_action,
            actor=actor,
            updated_at=updated_at,
        )

        def _progress(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            state = self._require_current(current)
            if state.progress == progress:
                return state
            self._require_control_revision(
                state,
                expected_control_revision,
            )
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                control_revision=state.control_revision + 1,
                progress=progress,
            )

        return self.store.update(_progress)

    def request_approval(
        self,
        *,
        request_id: str,
        action: str,
        resource_digest: str,
        rationale: str,
        requested_by: str,
        requested_at: str,
        expected_steering_revision: int,
    ) -> ApprovalRequestEvent:
        result: ApprovalRequestEvent | None = None

        def _request(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            nonlocal result
            state = self._require_current(current)
            candidate = ApprovalRequestEvent(
                sequence=len(state.approval_events) + 1,
                request_id=request_id,
                control_identity_fingerprint=self.identity.fingerprint,
                run_fingerprint=self.identity.run_fingerprint,
                route_fingerprint=self.identity.route_fingerprint,
                authority_fingerprint=self.identity.authority_fingerprint,
                steering_revision=expected_steering_revision,
                action=action,
                resource_digest=resource_digest,
                rationale=rationale,
                requested_by=requested_by,
                requested_at=requested_at,
            )
            for event in state.approval_events:
                if (
                    isinstance(event, ApprovalRequestEvent)
                    and event.request_id == request_id
                ):
                    if event.fingerprint != candidate.fingerprint:
                        raise ApprovalConflict(
                            "approval request ID was replayed with "
                            "different content or binding"
                        )
                    result = event
                    return state
            if state.steering_revision != expected_steering_revision:
                raise StaleSteeringRevision(
                    "approval request targets stale steering revision"
                )
            if state.desired_state in {"cancelled", "completed"}:
                raise ApprovalConflict(
                    "terminal workflow cannot request approval"
                )
            result = candidate
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                control_revision=state.control_revision + 1,
                approval_events=(*state.approval_events, candidate),
            )

        self.store.update(_request)
        assert result is not None
        return result

    def decide_approval(
        self,
        *,
        decision_id: str,
        request_id: str,
        request_fingerprint: str,
        decision: ApprovalDecision,
        reason: str,
        decided_by: str,
        decided_at: str,
        expected_steering_revision: int,
    ) -> ApprovalDecisionEvent:
        result: ApprovalDecisionEvent | None = None

        def _decide(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            nonlocal result
            state = self._require_current(current)
            request: ApprovalRequestEvent | None = None
            existing_decision: ApprovalDecisionEvent | None = None
            decision_id_owner: ApprovalDecisionEvent | None = None
            for event in state.approval_events:
                if (
                    isinstance(event, ApprovalRequestEvent)
                    and event.request_id == request_id
                ):
                    request = event
                elif (
                    isinstance(event, ApprovalDecisionEvent)
                    and event.request_id == request_id
                ):
                    existing_decision = event
                if (
                    isinstance(event, ApprovalDecisionEvent)
                    and event.decision_id == decision_id
                ):
                    decision_id_owner = event
            if request is None:
                raise ApprovalConflict("approval request does not exist")
            if request.fingerprint != request_fingerprint:
                raise ApprovalConflict(
                    "approval decision request fingerprint mismatch"
                )
            candidate = ApprovalDecisionEvent(
                sequence=len(state.approval_events) + 1,
                decision_id=decision_id,
                request_id=request_id,
                request_fingerprint=request_fingerprint,
                control_identity_fingerprint=self.identity.fingerprint,
                run_fingerprint=self.identity.run_fingerprint,
                route_fingerprint=self.identity.route_fingerprint,
                authority_fingerprint=self.identity.authority_fingerprint,
                steering_revision=expected_steering_revision,
                decision=decision,
                reason=reason,
                decided_by=decided_by,
                decided_at=decided_at,
            )
            if existing_decision is not None:
                if existing_decision.fingerprint != candidate.fingerprint:
                    raise ApprovalConflict(
                        "approval was already decided differently"
                    )
                result = existing_decision
                return state
            if decision_id_owner is not None:
                raise ApprovalConflict(
                    "approval decision ID is already bound to another request"
                )
            if (
                state.steering_revision != expected_steering_revision
                or request.steering_revision != expected_steering_revision
            ):
                raise StaleSteeringRevision(
                    "approval decision targets stale steering revision"
                )
            if state.desired_state in {"cancelled", "completed"}:
                raise ApprovalConflict(
                    "terminal workflow cannot accept an approval decision"
                )
            result = candidate
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                control_revision=state.control_revision + 1,
                approval_events=(*state.approval_events, candidate),
            )

        self.store.update(_decide)
        assert result is not None
        return result

    def register_participant(
        self,
        *,
        participant_id: str,
        kind: str,
        registration_digest: str,
        registered_at: str,
    ) -> ParticipantRecord:
        result: ParticipantRecord | None = None

        def _register(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            nonlocal result
            state = self._require_current(current)
            candidate = ParticipantRecord(
                participant_id=participant_id,
                kind=kind,
                registration_digest=registration_digest,
                state="active",
                registered_at=registered_at,
                updated_at=registered_at,
            )
            for participant in state.participants:
                if participant.participant_id != participant_id:
                    continue
                if (
                    participant.kind != kind
                    or participant.registration_digest
                    != registration_digest
                ):
                    raise ParticipantConflict(
                        "participant ID was rebound to different registration"
                    )
                result = participant
                return state
            result = candidate
            return dataclasses.replace(
                state,
                record_revision=state.record_revision + 1,
                participants=(*state.participants, candidate),
            )

        self.store.update(_register)
        assert result is not None
        return result

    def update_participant(
        self,
        *,
        participant_id: str,
        state: ParticipantState,
        updated_at: str,
        error_digest: str | None = None,
    ) -> ParticipantRecord:
        if state not in _PARTICIPANT_STATES:
            raise ValueError(f"unknown participant state {state!r}")
        result: ParticipantRecord | None = None

        def _update(
            current: WorkflowControlState | None,
        ) -> WorkflowControlState:
            nonlocal result
            control = self._require_current(current)
            updated: list[ParticipantRecord] = []
            found = False
            for participant in control.participants:
                if participant.participant_id != participant_id:
                    updated.append(participant)
                    continue
                found = True
                candidate = dataclasses.replace(
                    participant,
                    state=state,
                    updated_at=updated_at,
                    error_digest=error_digest,
                )
                result = candidate
                updated.append(candidate)
            if not found:
                raise ParticipantConflict(
                    f"unknown cancellation participant {participant_id!r}"
                )
            if tuple(updated) == control.participants:
                return control
            return dataclasses.replace(
                control,
                record_revision=control.record_revision + 1,
                participants=tuple(updated),
            )

        self.store.update(_update)
        assert result is not None
        return result


class CancellationRegistry:
    """At-least-once cancellation fan-out with durable participant state."""

    def __init__(
        self,
        controller: WorkflowControlController,
    ) -> None:
        self.controller = controller
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._inflight: set[str] = set()
        self._delivered: set[str] = set()

    def register(
        self,
        *,
        participant_id: str,
        kind: str,
        registration_digest: str,
        registered_at: str,
        cancel: Callable[[], None],
    ) -> ParticipantRecord:
        if not callable(cancel):
            raise TypeError("cancel callback must be callable")
        participant = self.controller.register_participant(
            participant_id=participant_id,
            kind=kind,
            registration_digest=registration_digest,
            registered_at=registered_at,
        )
        with self._lock:
            self._callbacks[participant_id] = cancel
        current = self.controller.load()
        if current is not None and current.desired_state == "cancelled":
            self._signal_one(participant_id, updated_at=registered_at)
            refreshed = self.controller.load()
            assert refreshed is not None
            return next(
                item
                for item in refreshed.participants
                if item.participant_id == participant_id
            )
        return participant

    def request_cancel(
        self,
        *,
        expected_control_revision: int,
        requested_at: str,
    ) -> WorkflowControlState:
        """Persist cancellation intent before invoking any callback."""

        self.controller.request_cancel(
            expected_control_revision=expected_control_revision,
        )
        self.fan_out(updated_at=requested_at)
        refreshed = self.controller.load()
        assert refreshed is not None
        return refreshed

    def propagate_if_requested(
        self,
        *,
        updated_at: str,
    ) -> WorkflowControlState | None:
        state = self.controller.load()
        if state is None or state.desired_state != "cancelled":
            return state
        self.fan_out(updated_at=updated_at)
        return self.controller.load()

    def fan_out(self, *, updated_at: str) -> dict[str, str]:
        state = self.controller.load()
        if state is None or state.desired_state != "cancelled":
            raise WorkflowControlError(
                "cancellation must be durable before fan-out"
            )
        with self._lock:
            participant_ids = tuple(self._callbacks)
        results: dict[str, str] = {}
        for participant_id in participant_ids:
            results[participant_id] = self._signal_one(
                participant_id,
                updated_at=updated_at,
            )
        return results

    def _signal_one(
        self,
        participant_id: str,
        *,
        updated_at: str,
    ) -> str:
        with self._lock:
            if participant_id in self._delivered:
                return "cancel_signaled"
            if participant_id in self._inflight:
                return "inflight"
            callback = self._callbacks.get(participant_id)
            if callback is None:
                return "unregistered"
            self._inflight.add(participant_id)
        try:
            callback()
        except Exception as exc:  # noqa: BLE001
            error_digest = control_value_digest(
                {
                    "type": (
                        f"{type(exc).__module__}.{type(exc).__qualname__}"
                    ),
                },
                domain="workflow_control.cancellation_error",
            )
            self.controller.update_participant(
                participant_id=participant_id,
                state="cancel_failed",
                updated_at=updated_at,
                error_digest=error_digest,
            )
            return "cancel_failed"
        else:
            self.controller.update_participant(
                participant_id=participant_id,
                state="cancel_signaled",
                updated_at=updated_at,
            )
            with self._lock:
                self._delivered.add(participant_id)
            return "cancel_signaled"
        finally:
            with self._lock:
                self._inflight.discard(participant_id)

    def drain(
        self,
        *,
        participant_id: str,
        drained_at: str,
    ) -> ParticipantRecord:
        with self._lock:
            self._callbacks.pop(participant_id, None)
            self._delivered.discard(participant_id)
        return self.controller.update_participant(
            participant_id=participant_id,
            state="drained",
            updated_at=drained_at,
        )


@dataclass(frozen=True, slots=True)
class WorkflowControlCheckpoint:
    """One cooperative boundary observed by a workflow between side effects."""

    control_revision: int
    steering_revision: int
    desired_state: DesiredState
    should_pause: bool
    should_cancel: bool
    steering: SteeringRecord | None


class WorkflowControlBinding:
    """Post-routing workflow binding for heartbeat and cooperative control.

    Workflows create this only after route and tool authority admission.  A
    checkpoint is taken before backend construction/invocation and before
    mutation or integration boundaries.  It projects durable cancellation
    into the workflow's existing ``threading.Event`` and fans cancellation to
    every registered provider/child callback.
    """

    def __init__(
        self,
        *,
        controller: WorkflowControlController,
        owner: str,
        cancel_event: threading.Event | None = None,
    ) -> None:
        _require_text(
            owner,
            field="WorkflowControlBinding.owner",
            max_bytes=1024,
        )
        if cancel_event is not None and not isinstance(
            cancel_event,
            threading.Event,
        ):
            raise TypeError(
                "WorkflowControlBinding.cancel_event must be "
                "threading.Event or None"
            )
        self.controller = controller
        self.owner = owner
        self.cancel_event = cancel_event
        self.cancellations = CancellationRegistry(controller)

    def close(self) -> None:
        self.controller.close()

    def __enter__(self) -> WorkflowControlBinding:
        self.controller.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def heartbeat(self, *, at: str) -> WorkflowControlState:
        return self.controller.heartbeat(owner=self.owner, at=at)

    def checkpoint(
        self,
        *,
        observed_steering_revision: int,
        at: str,
    ) -> WorkflowControlCheckpoint:
        """Observe pause/steering and propagate durable cancellation."""

        _require_exact_nonnegative_int(
            observed_steering_revision,
            field="observed_steering_revision",
        )
        _require_text(at, field="checkpoint time", max_bytes=128)
        state = self.controller.load()
        if state is None:
            raise WorkflowControlError(
                "workflow control state must be initialized first"
            )
        should_cancel = state.desired_state == "cancelled"
        if should_cancel:
            if self.cancel_event is not None:
                self.cancel_event.set()
            self.cancellations.propagate_if_requested(updated_at=at)
            refreshed = self.controller.load()
            if refreshed is not None:
                state = refreshed
        steering = (
            state.steering
            if state.steering_revision > observed_steering_revision
            else None
        )
        return WorkflowControlCheckpoint(
            control_revision=state.control_revision,
            steering_revision=state.steering_revision,
            desired_state=state.desired_state,
            should_pause=state.desired_state == "paused",
            should_cancel=should_cancel,
            steering=steering,
        )


def bind_workflow_control(
    run_dir: Path | str,
    *,
    identity: ControlIdentity,
    owner: str,
    heartbeat_at: str,
    cancel_event: threading.Event | None = None,
) -> WorkflowControlBinding:
    """Initialize the explicit post-routing control binding for one run."""

    controller = WorkflowControlController(
        store=WorkflowControlStore.for_run(run_dir),
        identity=identity,
    )
    try:
        controller.initialize()
        binding = WorkflowControlBinding(
            controller=controller,
            owner=owner,
            cancel_event=cancel_event,
        )
        binding.heartbeat(at=heartbeat_at)
    except BaseException:
        controller.close()
        raise
    return binding


__all__ = [
    "ApprovalConflict",
    "ApprovalDecision",
    "ApprovalDecisionEvent",
    "ApprovalEvent",
    "ApprovalRequestEvent",
    "CONTROL_STATE_DIRNAME",
    "CONTROL_STATE_SCHEMA_VERSION",
    "CONTROL_STORE_SCHEMA_VERSION",
    "CancellationRegistry",
    "ControlIdentity",
    "DesiredState",
    "ParticipantConflict",
    "ParticipantRecord",
    "ParticipantState",
    "ProgressRecord",
    "StaleControlRevision",
    "StaleSteeringRevision",
    "SteeringRecord",
    "WorkflowControlCodec",
    "WorkflowControlBinding",
    "WorkflowControlCheckpoint",
    "WorkflowControlController",
    "WorkflowControlError",
    "WorkflowControlIdentityMismatch",
    "WorkflowControlState",
    "WorkflowControlStore",
    "WorkflowControlStoreError",
    "bind_workflow_control",
    "control_value_digest",
]
