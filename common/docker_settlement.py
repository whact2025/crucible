# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed Docker generation settlement for benchmark workflows.

Docker containers are owned by a daemon, not by the process tree that asked
the daemon to create them.  Proving the caller's local descendants quiescent
therefore does not prove that its containers are gone.  This module provides
the separate, product-neutral state machine used to bind and settle that
daemon-side authority.

The correctness core intentionally consumes a typed control-plane protocol
instead of parsing human Docker CLI errors.  An adapter must distinguish an
authoritative absent-container result from a daemon/query failure and must
provide a restart-sensitive daemon generation.  Docker's persistent engine
ID alone is not restart-sensitive and is insufficient for that field.
"""

from __future__ import annotations

import dataclasses
import math
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Protocol

from crucible.common.benchmark_runtime import canonical_fingerprint


DOCKER_SETTLEMENT_ARM_SCHEMA = "crucible.docker-settlement-arm.v1"
DOCKER_SETTLEMENT_TERMINAL_SCHEMA = "crucible.docker-settlement-terminal.v1"
DOCKER_SETTLEMENT_LABEL = "io.crucible.settlement"
DOCKER_RUN_GENERATION_LABEL = "io.crucible.run-generation"
DOCKER_ARM_GENERATION_LABEL = "io.crucible.arm-generation"
DOCKER_DAEMON_BINDING_LABEL = "io.crucible.daemon-binding"

_SETTLEMENT_LABEL_KEYS = frozenset({
    DOCKER_SETTLEMENT_LABEL,
    DOCKER_RUN_GENERATION_LABEL,
    DOCKER_ARM_GENERATION_LABEL,
    DOCKER_DAEMON_BINDING_LABEL,
})

_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_GENERATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_LABEL_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class DockerSettlementArmStatus(StrEnum):
    """Whether a daemon-side generation was safely armed."""

    ARMED = "armed"
    UNCERTAIN = "uncertain"


class DockerSettlementStatus(StrEnum):
    """Closed terminal states for one Docker settlement generation."""

    SETTLED = "settled"
    UNCERTAIN = "uncertain"


class DockerSettlementIssueCode(StrEnum):
    """Stable machine-readable reasons that prevent positive settlement."""

    DEADLINE_EXPIRED = "deadline_expired"
    INVALID_DEADLINE = "invalid_deadline"
    DAEMON_QUERY_FAILED = "daemon_query_failed"
    DAEMON_IDENTITY_DRIFT = "daemon_identity_drift"
    GENERATION_COLLISION = "generation_collision"
    INVENTORY_LIMIT_EXCEEDED = "inventory_limit_exceeded"
    LIST_QUERY_FAILED = "list_query_failed"
    INSPECT_QUERY_FAILED = "inspect_query_failed"
    INVENTORY_DISAGREEMENT = "inventory_disagreement"
    INVALID_CONTAINER_ID = "invalid_container_id"
    CANONICAL_ID_MISMATCH = "canonical_id_mismatch"
    LABEL_DRIFT = "label_drift"
    REGISTRATION_FAILED = "registration_failed"
    REMOVE_FAILED = "remove_failed"
    PROCESS_PROOF_FAILED = "process_proof_failed"
    ABSENCE_NOT_PROVEN = "absence_not_proven"


class DockerSettlementError(RuntimeError):
    """Base error raised for invalid or unavailable settlement authority."""


class DockerSettlementBlockedError(DockerSettlementError):
    """A prior active or uncertain generation forbids another arm."""


class DockerSettlementRegistrationError(DockerSettlementError):
    """A created container could not be bound to the active generation."""


def _bounded_detail(value: object) -> str:
    rendered = f"{type(value).__name__}: {value}"
    return rendered[-1_000:]


def _validate_generation(value: str, *, field: str) -> str:
    if not _GENERATION_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be a 1-128 character Docker-label-safe token"
        )
    return value


def _freeze_labels(labels: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    frozen: list[tuple[str, str]] = []
    for key, value in labels.items():
        if not isinstance(key, str) or not _LABEL_KEY_RE.fullmatch(key):
            raise ValueError(f"invalid Docker label key {key!r}")
        if not isinstance(value, str) or "\x00" in value or len(value) > 4_096:
            raise ValueError(f"invalid Docker label value for {key!r}")
        frozen.append((key, value))
    if len({key for key, _value in frozen}) != len(frozen):
        raise ValueError("Docker labels contain duplicate keys")
    return tuple(sorted(frozen))


def _label_dict(labels: Sequence[tuple[str, str]]) -> dict[str, str]:
    return dict(labels)


def new_docker_generation(prefix: str) -> str:
    """Return a fresh, Docker-label-safe opaque generation identifier."""

    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", prefix).strip("-.:")
    if not normalized:
        normalized = "generation"
    return f"{normalized[:88]}-{secrets.token_hex(16)}"


@dataclasses.dataclass(frozen=True)
class DockerDaemonIdentity:
    """Identity and restart-sensitive generation of one Docker daemon.

    ``engine_id`` identifies the daemon's persistent data domain.
    ``generation`` must change whenever the serving daemon restarts or its
    control plane is replaced.  ``configuration_fingerprint`` binds all other
    adapter-selected facts that affect the authority boundary (for example
    endpoint/context, root directory, API version, and security mode).
    """

    endpoint_fingerprint: str
    engine_id: str
    generation: str
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.endpoint_fingerprint):
            raise ValueError("Docker daemon endpoint_fingerprint is invalid")
        if not self.engine_id.strip() or len(self.engine_id) > 256:
            raise ValueError("Docker daemon engine_id is required")
        if not self.generation.strip() or len(self.generation) > 256:
            raise ValueError("Docker daemon restart generation is required")
        if not _SHA256_RE.fullmatch(self.configuration_fingerprint):
            raise ValueError(
                "Docker daemon configuration_fingerprint must be sha256:<hex>"
            )

    @classmethod
    def from_components(
        cls,
        *,
        endpoint: str,
        engine_id: str,
        generation: str,
        configuration: Mapping[str, object],
    ) -> "DockerDaemonIdentity":
        """Build an identity while hashing adapter-specific configuration."""

        return cls(
            # Endpoint strings can contain credentials. Retain only their
            # canonical identity in terminal provenance.
            endpoint_fingerprint=canonical_fingerprint(endpoint),
            engine_id=engine_id,
            generation=generation,
            configuration_fingerprint=canonical_fingerprint(configuration),
        )

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())

    def to_payload(self) -> dict[str, str]:
        return {
            "endpoint_fingerprint": self.endpoint_fingerprint,
            "engine_id": self.engine_id,
            "generation": self.generation,
            "configuration_fingerprint": self.configuration_fingerprint,
        }


@dataclasses.dataclass(frozen=True)
class DockerContainerInspection:
    """One exact Docker container observation returned by the control plane."""

    container_id: str
    labels: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not _CONTAINER_ID_RE.fullmatch(self.container_id):
            raise ValueError(
                "Docker inspection must carry a canonical 64-hex container ID"
            )
        supplied = dict(self.labels)
        object.__setattr__(
            self,
            "labels",
            _freeze_labels({
                key: value
                for key, value in supplied.items()
                if key in _SETTLEMENT_LABEL_KEYS
            }),
        )

    @classmethod
    def from_parts(
        cls,
        container_id: str,
        labels: Mapping[str, str],
    ) -> "DockerContainerInspection":
        return cls(
            container_id=container_id,
            labels=_freeze_labels({
                key: value
                for key, value in labels.items()
                if key in _SETTLEMENT_LABEL_KEYS
            }),
        )

    def has_labels(self, expected: Mapping[str, str]) -> bool:
        actual = _label_dict(self.labels)
        return all(actual.get(key) == value for key, value in expected.items())

    def to_payload(self) -> dict[str, object]:
        return {
            "container_id": self.container_id,
            "labels": _label_dict(self.labels),
        }


class DockerSettlementControlPlane(Protocol):
    """Typed Docker authority consumed by :class:`DockerSettlementSession`.

    Every method must honor ``deadline_monotonic``.  ``inspect_container``
    returns ``None`` only for an authoritative not-found response; daemon,
    permission, timeout, and decoding failures must raise instead.
    """

    def daemon_identity(
        self,
        *,
        deadline_monotonic: float,
    ) -> DockerDaemonIdentity: ...

    def list_container_ids(
        self,
        *,
        labels: Mapping[str, str],
        limit: int,
        deadline_monotonic: float,
    ) -> Sequence[str]: ...

    def inspect_container(
        self,
        reference: str,
        *,
        deadline_monotonic: float,
    ) -> DockerContainerInspection | None: ...

    def remove_container(
        self,
        container_id: str,
        *,
        deadline_monotonic: float,
    ) -> None: ...


@dataclasses.dataclass(frozen=True)
class DockerSettlementIssue:
    """One immutable reason an arm or terminal cannot be trusted."""

    code: DockerSettlementIssueCode
    operation: str
    detail: str
    container_id: str | None = None

    def to_payload(self) -> dict[str, str | None]:
        return {
            "code": self.code.value,
            "operation": self.operation,
            "detail": self.detail,
            "container_id": self.container_id,
        }


@dataclasses.dataclass(frozen=True)
class DockerInventory:
    """One bounded list-plus-inspect observation of a generation."""

    listed_container_ids: tuple[str, ...]
    inspections: tuple[DockerContainerInspection, ...]
    absent_container_ids: tuple[str, ...]
    complete: bool

    @classmethod
    def empty(cls, *, complete: bool = False) -> "DockerInventory":
        return cls((), (), (), complete)

    def to_payload(self) -> dict[str, object]:
        return {
            "listed_container_ids": list(self.listed_container_ids),
            "inspections": [item.to_payload() for item in self.inspections],
            "absent_container_ids": list(self.absent_container_ids),
            "complete": self.complete,
        }


@dataclasses.dataclass(frozen=True)
class DockerSettlementArmRecord:
    """Typed evidence for one attempted daemon-side generation arm."""

    status: DockerSettlementArmStatus
    run_generation: str
    arm_generation: str
    generation_labels: tuple[tuple[str, str], ...]
    daemon_identity: DockerDaemonIdentity | None
    initial_inventories: tuple[DockerInventory, ...]
    issues: tuple[DockerSettlementIssue, ...]

    @property
    def labels(self) -> dict[str, str]:
        """Labels every container created under this arm must receive."""

        return _label_dict(self.generation_labels)

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())

    @property
    def initial_inventory(self) -> DockerInventory:
        """Return the final initial observation for concise callers."""

        if self.initial_inventories:
            return self.initial_inventories[-1]
        return DockerInventory.empty()

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": DOCKER_SETTLEMENT_ARM_SCHEMA,
            "status": self.status.value,
            "run_generation": self.run_generation,
            "arm_generation": self.arm_generation,
            "generation_labels": self.labels,
            "daemon_identity": (
                self.daemon_identity.to_payload()
                if self.daemon_identity is not None else None
            ),
            "initial_inventories": [
                inventory.to_payload()
                for inventory in self.initial_inventories
            ],
            "issues": [issue.to_payload() for issue in self.issues],
        }


@dataclasses.dataclass(frozen=True)
class DockerSettlementTerminal:
    """Terminal proof, or explicit uncertainty, for one Docker arm."""

    status: DockerSettlementStatus
    arm_fingerprint: str
    daemon_identity: DockerDaemonIdentity | None
    process_tree_proven: bool
    process_proof_ref: str | None
    pre_cleanup_inventory: DockerInventory
    post_cleanup_inventory: DockerInventory
    inventory_trace: tuple[DockerInventory, ...]
    known_container_ids: tuple[str, ...]
    removed_container_ids: tuple[str, ...]
    poll_attempts: int
    clean_observations: int
    issues: tuple[DockerSettlementIssue, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": DOCKER_SETTLEMENT_TERMINAL_SCHEMA,
            "status": self.status.value,
            "arm_fingerprint": self.arm_fingerprint,
            "daemon_identity": (
                self.daemon_identity.to_payload()
                if self.daemon_identity is not None else None
            ),
            "process_tree_proven": self.process_tree_proven,
            "process_proof_ref": self.process_proof_ref,
            "pre_cleanup_inventory": self.pre_cleanup_inventory.to_payload(),
            "post_cleanup_inventory": self.post_cleanup_inventory.to_payload(),
            "inventory_trace": [
                inventory.to_payload() for inventory in self.inventory_trace
            ],
            "known_container_ids": list(self.known_container_ids),
            "removed_container_ids": list(self.removed_container_ids),
            "poll_attempts": self.poll_attempts,
            "clean_observations": self.clean_observations,
            "issues": [issue.to_payload() for issue in self.issues],
        }


def _deduplicate_issues(
    issues: Sequence[DockerSettlementIssue],
) -> tuple[DockerSettlementIssue, ...]:
    retained: list[DockerSettlementIssue] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for issue in issues:
        identity = (
            issue.code.value,
            issue.operation,
            issue.detail,
            issue.container_id,
        )
        if identity not in seen:
            seen.add(identity)
            retained.append(issue)
    return tuple(retained)


class DockerSettlementSession:
    """Arm, register, and settle one Docker generation at a time.

    An uncertain arm or terminal is an irreversible latch for this session.
    Constructing a replacement session after a process restart must pass its
    previously persisted unterminated arm or terminal through ``prior_arm`` or
    ``prior_terminal``. Either an unterminated arm or an uncertain terminal
    then continues to block new work until operator recovery creates a
    separate, explicitly authorized lineage.
    """

    def __init__(
        self,
        control_plane: DockerSettlementControlPlane,
        *,
        max_poll_attempts: int = 8,
        required_clean_observations: int = 2,
        max_container_ids: int = 256,
        poll_interval_s: float = 0.05,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        prior_arm: DockerSettlementArmRecord | None = None,
        prior_terminal: DockerSettlementTerminal | None = None,
    ) -> None:
        if max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be positive")
        if required_clean_observations < 2:
            raise ValueError("at least two clean observations are required")
        if required_clean_observations > max_poll_attempts:
            raise ValueError(
                "required_clean_observations cannot exceed max_poll_attempts"
            )
        if (
            not isinstance(poll_interval_s, (int, float))
            or not math.isfinite(poll_interval_s)
            or poll_interval_s < 0
        ):
            raise ValueError("poll_interval_s must be finite and non-negative")
        if max_container_ids < 1:
            raise ValueError("max_container_ids must be positive")
        if prior_arm is not None and prior_terminal is not None:
            raise ValueError(
                "restore either an unterminated Docker arm or its terminal, "
                "not both"
            )
        self._control_plane = control_plane
        self._max_poll_attempts = max_poll_attempts
        self._required_clean_observations = required_clean_observations
        self._max_container_ids = max_container_ids
        self._poll_interval_s = poll_interval_s
        self._monotonic = monotonic
        self._sleep = sleep
        self._active_arm: DockerSettlementArmRecord | None = None
        self._known_container_ids: set[str] = set()
        self._pending_issues: list[DockerSettlementIssue] = []
        self._blocked_terminal = (
            prior_terminal
            if prior_terminal is not None
            and prior_terminal.status == DockerSettlementStatus.UNCERTAIN
            else None
        )
        # An unterminated armed generation cannot be reconstructed from its
        # arm record alone: registered full IDs and pending query evidence may
        # have existed only in the crashed process. Retain it as a blocking
        # record for explicit recovery instead of silently re-arming.
        self._blocked_arm: DockerSettlementArmRecord | None = prior_arm
        self._lock = threading.Lock()

    @property
    def active_arm(self) -> DockerSettlementArmRecord | None:
        with self._lock:
            return self._active_arm

    @property
    def blocked(self) -> bool:
        with self._lock:
            return self._blocked_terminal is not None or self._blocked_arm is not None

    def _deadline_issue(
        self,
        operation: str,
        deadline_monotonic: float,
    ) -> DockerSettlementIssue:
        if not isinstance(deadline_monotonic, (int, float)) or not math.isfinite(
            deadline_monotonic
        ):
            return DockerSettlementIssue(
                DockerSettlementIssueCode.INVALID_DEADLINE,
                operation,
                "the shared Docker settlement deadline must be finite",
            )
        return DockerSettlementIssue(
            DockerSettlementIssueCode.DEADLINE_EXPIRED,
            operation,
            "the shared monotonic Docker settlement deadline expired",
        )

    def _deadline_unavailable(self, deadline_monotonic: float) -> bool:
        return (
            not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
            or self._monotonic() >= deadline_monotonic
        )

    def _probe_identity(
        self,
        *,
        deadline_monotonic: float,
        operation: str,
    ) -> tuple[DockerDaemonIdentity | None, list[DockerSettlementIssue]]:
        if self._deadline_unavailable(deadline_monotonic):
            return None, [self._deadline_issue(operation, deadline_monotonic)]
        try:
            identity = self._control_plane.daemon_identity(
                deadline_monotonic=deadline_monotonic,
            )
        except Exception as exc:  # noqa: BLE001 - typed uncertain evidence
            return None, [DockerSettlementIssue(
                DockerSettlementIssueCode.DAEMON_QUERY_FAILED,
                operation,
                _bounded_detail(exc),
            )]
        if not isinstance(identity, DockerDaemonIdentity):
            return None, [DockerSettlementIssue(
                DockerSettlementIssueCode.DAEMON_QUERY_FAILED,
                operation,
                "control plane returned an untyped daemon identity",
            )]
        return identity, []

    @staticmethod
    def _identity_drift_issue(
        *,
        operation: str,
        expected: DockerDaemonIdentity,
        actual: DockerDaemonIdentity,
    ) -> DockerSettlementIssue:
        return DockerSettlementIssue(
            DockerSettlementIssueCode.DAEMON_IDENTITY_DRIFT,
            operation,
            "Docker daemon identity or restart generation changed: "
            f"expected={expected.fingerprint}, actual={actual.fingerprint}",
        )

    def _observe_inventory(
        self,
        *,
        labels: Mapping[str, str],
        known_container_ids: set[str],
        deadline_monotonic: float,
        operation: str,
    ) -> tuple[DockerInventory, list[DockerSettlementIssue]]:
        issues: list[DockerSettlementIssue] = []
        if self._deadline_unavailable(deadline_monotonic):
            return DockerInventory.empty(), [
                self._deadline_issue(operation, deadline_monotonic)
            ]
        try:
            raw_listed = tuple(self._control_plane.list_container_ids(
                labels=labels,
                limit=self._max_container_ids + 1,
                deadline_monotonic=deadline_monotonic,
            ))
        except Exception as exc:  # noqa: BLE001 - typed uncertain evidence
            return DockerInventory.empty(), [DockerSettlementIssue(
                DockerSettlementIssueCode.LIST_QUERY_FAILED,
                operation,
                _bounded_detail(exc),
            )]

        listed: list[str] = []
        if len(raw_listed) > self._max_container_ids:
            issues.append(DockerSettlementIssue(
                DockerSettlementIssueCode.INVENTORY_LIMIT_EXCEEDED,
                operation,
                "Docker generation inventory exceeded the configured bound",
            ))
        for candidate in raw_listed[:self._max_container_ids]:
            if not isinstance(candidate, str) or not _CONTAINER_ID_RE.fullmatch(
                candidate
            ):
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.INVALID_CONTAINER_ID,
                    operation,
                    "Docker list did not return a canonical 64-hex ID",
                    candidate if isinstance(candidate, str) else None,
                ))
                continue
            listed.append(candidate)
        if len(set(listed)) != len(listed):
            issues.append(DockerSettlementIssue(
                DockerSettlementIssueCode.INVENTORY_DISAGREEMENT,
                operation,
                "Docker list returned duplicate container IDs",
            ))
        listed_ids = tuple(sorted(set(listed)))
        requested_ids = sorted(set(listed_ids) | known_container_ids)
        inspections: list[DockerContainerInspection] = []
        absent: list[str] = []
        for requested_id in requested_ids:
            if self._deadline_unavailable(deadline_monotonic):
                issues.append(self._deadline_issue(
                    operation, deadline_monotonic,
                ))
                break
            try:
                inspected = self._control_plane.inspect_container(
                    requested_id,
                    deadline_monotonic=deadline_monotonic,
                )
            except Exception as exc:  # noqa: BLE001 - uncertain evidence
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.INSPECT_QUERY_FAILED,
                    operation,
                    _bounded_detail(exc),
                    requested_id,
                ))
                continue
            if inspected is None:
                absent.append(requested_id)
                if requested_id in listed_ids:
                    issues.append(DockerSettlementIssue(
                        DockerSettlementIssueCode.INVENTORY_DISAGREEMENT,
                        operation,
                        "Docker list reported an ID that inspect reported absent",
                        requested_id,
                    ))
                continue
            if not isinstance(inspected, DockerContainerInspection):
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.INSPECT_QUERY_FAILED,
                    operation,
                    "control plane returned an untyped container inspection",
                    requested_id,
                ))
                continue
            if inspected.container_id != requested_id:
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.CANONICAL_ID_MISMATCH,
                    operation,
                    "inspect resolved a different canonical container ID",
                    requested_id,
                ))
                continue
            inspections.append(inspected)
            if not inspected.has_labels(labels):
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.LABEL_DRIFT,
                    operation,
                    "container no longer carries the exact arm labels",
                    requested_id,
                ))
            elif requested_id not in listed_ids:
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.INVENTORY_DISAGREEMENT,
                    operation,
                    "inspect found an arm-labeled container omitted by list",
                    requested_id,
                ))
        complete = not any(
            issue.code in {
                DockerSettlementIssueCode.DEADLINE_EXPIRED,
                DockerSettlementIssueCode.INVALID_DEADLINE,
                DockerSettlementIssueCode.LIST_QUERY_FAILED,
                DockerSettlementIssueCode.INSPECT_QUERY_FAILED,
                DockerSettlementIssueCode.INVALID_CONTAINER_ID,
                DockerSettlementIssueCode.CANONICAL_ID_MISMATCH,
                DockerSettlementIssueCode.INVENTORY_LIMIT_EXCEEDED,
                DockerSettlementIssueCode.INVENTORY_DISAGREEMENT,
                DockerSettlementIssueCode.LABEL_DRIFT,
            }
            for issue in issues
        )
        return DockerInventory(
            listed_container_ids=listed_ids,
            inspections=tuple(sorted(
                inspections, key=lambda item: item.container_id,
            )),
            absent_container_ids=tuple(sorted(set(absent))),
            complete=complete,
        ), issues

    def arm(
        self,
        *,
        run_generation: str,
        arm_generation: str,
        deadline_monotonic: float,
    ) -> DockerSettlementArmRecord:
        """Arm one empty, daemon-bound label generation.

        Two clean initial inventories prevent reusing a generation that was
        already active.  An uncertain result latches the session and callers
        must not create a container from its labels.
        """

        run_generation = _validate_generation(
            run_generation, field="run_generation",
        )
        arm_generation = _validate_generation(
            arm_generation, field="arm_generation",
        )
        with self._lock:
            if self._blocked_terminal is not None or self._blocked_arm is not None:
                raise DockerSettlementBlockedError(
                    "an uncertain Docker settlement blocks the next arm"
                )
            if self._active_arm is not None:
                raise DockerSettlementBlockedError(
                    "the active Docker generation has not reached a terminal"
                )

            identity, issues = self._probe_identity(
                deadline_monotonic=deadline_monotonic,
                operation="arm_daemon_identity",
            )
            labels: dict[str, str] = {
                DOCKER_SETTLEMENT_LABEL: "true",
                DOCKER_RUN_GENERATION_LABEL: run_generation,
                DOCKER_ARM_GENERATION_LABEL: arm_generation,
            }
            if identity is not None:
                labels[DOCKER_DAEMON_BINDING_LABEL] = identity.fingerprint.split(
                    ":", 1,
                )[1]
            initial_inventories: list[DockerInventory] = []
            if identity is not None and not issues:
                for index in range(2):
                    observed, observed_issues = self._observe_inventory(
                        labels=labels,
                        known_container_ids=set(),
                        deadline_monotonic=deadline_monotonic,
                        operation=f"arm_inventory_{index + 1}",
                    )
                    initial_inventories.append(observed)
                    issues.extend(observed_issues)
                    if observed.listed_container_ids or observed.inspections:
                        issues.append(DockerSettlementIssue(
                            DockerSettlementIssueCode.GENERATION_COLLISION,
                            f"arm_inventory_{index + 1}",
                            "the requested Docker arm generation is not empty",
                        ))
                        break
                confirmed, confirmed_issues = self._probe_identity(
                    deadline_monotonic=deadline_monotonic,
                    operation="arm_daemon_identity_confirm",
                )
                issues.extend(confirmed_issues)
                if confirmed is not None and confirmed != identity:
                    issues.append(self._identity_drift_issue(
                        operation="arm_daemon_identity_confirm",
                        expected=identity,
                        actual=confirmed,
                    ))

            status = (
                DockerSettlementArmStatus.ARMED
                if not issues else DockerSettlementArmStatus.UNCERTAIN
            )
            record = DockerSettlementArmRecord(
                status=status,
                run_generation=run_generation,
                arm_generation=arm_generation,
                generation_labels=_freeze_labels(labels),
                daemon_identity=identity,
                initial_inventories=tuple(initial_inventories),
                issues=_deduplicate_issues(issues),
            )
            if status == DockerSettlementArmStatus.ARMED:
                self._active_arm = record
                self._known_container_ids.clear()
                self._pending_issues.clear()
            else:
                self._blocked_arm = record
            return record

    def register_container(
        self,
        reference: str,
        *,
        deadline_monotonic: float,
    ) -> DockerContainerInspection:
        """Resolve one created container reference to its full canonical ID.

        Workflows must call this immediately after every successful Docker
        create/run.  Registration lets settlement retain the exact object even
        if it later disappears from the generation-label list.
        """

        with self._lock:
            arm = self._active_arm
            if arm is None or arm.daemon_identity is None:
                raise DockerSettlementBlockedError(
                    "no armed Docker generation is available for registration"
                )
            issues: list[DockerSettlementIssue] = []
            actual_identity, probe_issues = self._probe_identity(
                deadline_monotonic=deadline_monotonic,
                operation="register_daemon_identity",
            )
            issues.extend(probe_issues)
            if (
                actual_identity is not None
                and actual_identity != arm.daemon_identity
            ):
                issues.append(self._identity_drift_issue(
                    operation="register_daemon_identity",
                    expected=arm.daemon_identity,
                    actual=actual_identity,
                ))
            inspected: DockerContainerInspection | None = None
            if not issues:
                if self._deadline_unavailable(deadline_monotonic):
                    issues.append(self._deadline_issue(
                        "register_inspect", deadline_monotonic,
                    ))
                else:
                    try:
                        inspected = self._control_plane.inspect_container(
                            reference,
                            deadline_monotonic=deadline_monotonic,
                        )
                    except Exception as exc:  # noqa: BLE001 - retain evidence
                        issues.append(DockerSettlementIssue(
                            DockerSettlementIssueCode.INSPECT_QUERY_FAILED,
                            "register_inspect",
                            _bounded_detail(exc),
                        ))
            if inspected is None and not issues:
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.REGISTRATION_FAILED,
                    "register_inspect",
                    "the created container was absent during registration",
                ))
            if inspected is not None:
                if not isinstance(inspected, DockerContainerInspection):
                    issues.append(DockerSettlementIssue(
                        DockerSettlementIssueCode.INSPECT_QUERY_FAILED,
                        "register_inspect",
                        "control plane returned an untyped inspection",
                    ))
                    inspected = None
                elif not inspected.has_labels(arm.labels):
                    issues.append(DockerSettlementIssue(
                        DockerSettlementIssueCode.LABEL_DRIFT,
                        "register_inspect",
                        "created container is missing the exact arm labels",
                        inspected.container_id,
                    ))
                else:
                    if (
                        inspected.container_id not in self._known_container_ids
                        and len(self._known_container_ids)
                        >= self._max_container_ids
                    ):
                        issues.append(DockerSettlementIssue(
                            DockerSettlementIssueCode.INVENTORY_LIMIT_EXCEEDED,
                            "register_inspect",
                            "registered Docker container inventory exceeded "
                            "the configured bound",
                            inspected.container_id,
                        ))
                    else:
                        self._known_container_ids.add(inspected.container_id)
            if issues:
                self._pending_issues.extend(issues)
                raise DockerSettlementRegistrationError(
                    "Docker container registration is uncertain: "
                    + "; ".join(issue.code.value for issue in issues)
                )
            assert inspected is not None
            return inspected

    def settle(
        self,
        *,
        process_tree_proven: bool,
        process_proof_ref: str | None,
        deadline_monotonic: float,
    ) -> DockerSettlementTerminal:
        """Remove the exact generation and prove repeated daemon-side absence."""

        with self._lock:
            arm = self._active_arm
            if arm is None or arm.daemon_identity is None:
                raise DockerSettlementBlockedError(
                    "no armed Docker generation is available for settlement"
                )
            issues = list(self._pending_issues)
            if not process_tree_proven or not (
                isinstance(process_proof_ref, str) and process_proof_ref.strip()
            ):
                issues.append(DockerSettlementIssue(
                    DockerSettlementIssueCode.PROCESS_PROOF_FAILED,
                    "process_tree_settlement",
                    "a positive, reference-bound process-tree proof is required",
                ))

            current_identity, identity_issues = self._probe_identity(
                deadline_monotonic=deadline_monotonic,
                operation="settle_daemon_identity",
            )
            issues.extend(identity_issues)
            daemon_stable = current_identity == arm.daemon_identity
            if current_identity is not None and not daemon_stable:
                issues.append(self._identity_drift_issue(
                    operation="settle_daemon_identity",
                    expected=arm.daemon_identity,
                    actual=current_identity,
                ))

            pre_inventory = DockerInventory.empty()
            post_inventory = DockerInventory.empty()
            inventory_trace: list[DockerInventory] = []
            removed: list[str] = []
            attempts = 0
            clean_observations = 0
            if daemon_stable:
                for attempt in range(1, self._max_poll_attempts + 1):
                    attempts = attempt
                    observed_identity, observed_identity_issues = (
                        self._probe_identity(
                            deadline_monotonic=deadline_monotonic,
                            operation=f"settle_poll_{attempt}_daemon_identity",
                        )
                    )
                    issues.extend(observed_identity_issues)
                    if observed_identity is None:
                        break
                    if observed_identity != arm.daemon_identity:
                        issues.append(self._identity_drift_issue(
                            operation=f"settle_poll_{attempt}_daemon_identity",
                            expected=arm.daemon_identity,
                            actual=observed_identity,
                        ))
                        break
                    inventory, inventory_issues = self._observe_inventory(
                        labels=arm.labels,
                        known_container_ids=set(self._known_container_ids),
                        deadline_monotonic=deadline_monotonic,
                        operation=f"settle_poll_{attempt}_inventory",
                    )
                    if attempt == 1:
                        pre_inventory = inventory
                    post_inventory = inventory
                    inventory_trace.append(inventory)
                    issues.extend(inventory_issues)
                    for inspection in inventory.inspections:
                        if inspection.has_labels(arm.labels):
                            if (
                                inspection.container_id
                                not in self._known_container_ids
                                and len(self._known_container_ids)
                                >= self._max_container_ids
                            ):
                                issues.append(DockerSettlementIssue(
                                    DockerSettlementIssueCode.INVENTORY_LIMIT_EXCEEDED,
                                    f"settle_poll_{attempt}_inventory",
                                    "Docker container inventory exceeded the "
                                    "configured bound across polls",
                                    inspection.container_id,
                                ))
                            else:
                                self._known_container_ids.add(
                                    inspection.container_id
                                )

                    # Exact registered IDs remain cleanup authority even if
                    # their labels drift: an extant canonical Docker ID cannot
                    # be rebound to another container. Retain LABEL_DRIFT as
                    # terminal uncertainty, but do not leave that known daemon
                    # object running. Newly discovered objects still require
                    # the exact arm labels before removal.
                    cleanup_candidates = tuple(
                        inspection
                        for inspection in inventory.inspections
                        if (
                            inspection.container_id in self._known_container_ids
                            or inspection.has_labels(arm.labels)
                        )
                    )
                    for inspection in cleanup_candidates:
                        try:
                            self._control_plane.remove_container(
                                inspection.container_id,
                                deadline_monotonic=deadline_monotonic,
                            )
                        except Exception as exc:  # noqa: BLE001 - evidence
                            issues.append(DockerSettlementIssue(
                                DockerSettlementIssueCode.REMOVE_FAILED,
                                f"settle_poll_{attempt}_remove",
                                _bounded_detail(exc),
                                inspection.container_id,
                            ))
                        else:
                            if inspection.container_id not in removed:
                                removed.append(inspection.container_id)

                    known_absent = self._known_container_ids.issubset(
                        set(inventory.absent_container_ids)
                    )
                    clean = (
                        inventory.complete
                        and not inventory.listed_container_ids
                        and not inventory.inspections
                        and known_absent
                    )
                    clean_observations = clean_observations + 1 if clean else 0
                    if clean_observations >= self._required_clean_observations:
                        break
                    if self._deadline_unavailable(deadline_monotonic):
                        issues.append(self._deadline_issue(
                            f"settle_poll_{attempt}_wait",
                            deadline_monotonic,
                        ))
                        break
                    if self._poll_interval_s:
                        remaining = deadline_monotonic - self._monotonic()
                        if remaining <= 0:
                            issues.append(self._deadline_issue(
                                f"settle_poll_{attempt}_wait",
                                deadline_monotonic,
                            ))
                            break
                        self._sleep(min(self._poll_interval_s, remaining))

                if clean_observations < self._required_clean_observations:
                    issues.append(DockerSettlementIssue(
                        DockerSettlementIssueCode.ABSENCE_NOT_PROVEN,
                        "settle_terminal",
                        "bounded inspect/list polling did not prove stable absence",
                    ))

            terminal_identity, terminal_identity_issues = self._probe_identity(
                deadline_monotonic=deadline_monotonic,
                operation="settle_terminal_daemon_identity",
            )
            issues.extend(terminal_identity_issues)
            if (
                terminal_identity is not None
                and terminal_identity != arm.daemon_identity
            ):
                issues.append(self._identity_drift_issue(
                    operation="settle_terminal_daemon_identity",
                    expected=arm.daemon_identity,
                    actual=terminal_identity,
                ))

            final_issues = _deduplicate_issues(issues)
            status = (
                DockerSettlementStatus.SETTLED
                if not final_issues
                and clean_observations >= self._required_clean_observations
                else DockerSettlementStatus.UNCERTAIN
            )
            terminal = DockerSettlementTerminal(
                status=status,
                arm_fingerprint=arm.fingerprint,
                daemon_identity=terminal_identity,
                process_tree_proven=process_tree_proven,
                process_proof_ref=process_proof_ref,
                pre_cleanup_inventory=pre_inventory,
                post_cleanup_inventory=post_inventory,
                inventory_trace=tuple(inventory_trace),
                known_container_ids=tuple(sorted(self._known_container_ids)),
                removed_container_ids=tuple(removed),
                poll_attempts=attempts,
                clean_observations=clean_observations,
                issues=final_issues,
            )
            self._active_arm = None
            self._pending_issues.clear()
            if terminal.status == DockerSettlementStatus.UNCERTAIN:
                self._blocked_terminal = terminal
            else:
                self._known_container_ids.clear()
            return terminal


__all__ = [
    "DOCKER_ARM_GENERATION_LABEL",
    "DOCKER_DAEMON_BINDING_LABEL",
    "DOCKER_RUN_GENERATION_LABEL",
    "DOCKER_SETTLEMENT_ARM_SCHEMA",
    "DOCKER_SETTLEMENT_LABEL",
    "DOCKER_SETTLEMENT_TERMINAL_SCHEMA",
    "DockerContainerInspection",
    "DockerDaemonIdentity",
    "DockerInventory",
    "DockerSettlementArmRecord",
    "DockerSettlementArmStatus",
    "DockerSettlementBlockedError",
    "DockerSettlementControlPlane",
    "DockerSettlementError",
    "DockerSettlementIssue",
    "DockerSettlementIssueCode",
    "DockerSettlementRegistrationError",
    "DockerSettlementSession",
    "DockerSettlementStatus",
    "DockerSettlementTerminal",
    "new_docker_generation",
]
