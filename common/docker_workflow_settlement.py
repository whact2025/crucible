# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable parent/child binding for daemon-side Docker settlement.

The cohort controller owns the Docker settlement session.  DeepSWE runs in a
separate, supervised process, so it cannot truthfully settle its own complete
process tree.  A small authenticated ``AF_UNIX``/``SOCK_SEQPACKET`` bridge lets
that child obtain the exact arm labels and synchronously register every
successful Docker create/run reference with the parent-owned session.

The parent persists the arm before starting the child.  A registration reply
is sent only after the reference has been resolved to a canonical Engine ID
and the updated journal has been fsynced.  After the trusted outer process
supervisor proves the child tree gone, the parent closes this bridge and calls
``settle`` with that proof.  No child-side success assertion substitutes for
the outer proof.

An unterminated, uncertain, malformed, or tampered prior journal blocks a new
arm.  Cleanup-only orphan reconciliation is deliberately not treated as proof
and cannot clear this latch. The protocol lives in ``crucible.common`` so a
sealed product runner can import it without importing the Assay-backed workflow
package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.common.benchmark_runtime import canonical_fingerprint
from crucible.common.docker_engine_control_plane import (
    UnixDockerEngineControlPlane,
)
from crucible.common.docker_settlement import (
    DOCKER_SETTLEMENT_ARM_SCHEMA,
    DOCKER_SETTLEMENT_TERMINAL_SCHEMA,
    DockerContainerInspection,
    DockerDaemonIdentity,
    DockerInventory,
    DockerSettlementArmRecord,
    DockerSettlementArmStatus,
    DockerSettlementControlPlane,
    DockerSettlementIssue,
    DockerSettlementIssueCode,
    DockerSettlementSession,
    DockerSettlementStatus,
    DockerSettlementTerminal,
)
from crucible.common.process_settlement import ProcessSettlementProof

DOCKER_WORKFLOW_SETTLEMENT_SCHEMA = "crucible.docker-workflow-settlement.v2"
DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA = (
    "crucible.docker-workflow-settlement-receipt.v1"
)
DOCKER_SETTLEMENT_BRIDGE_SCHEMA = "crucible.docker-settlement-bridge.v1"
DOCKER_WORKFLOW_SETTLEMENT_PARTIAL_PREFIX = ".partial-"

DOCKER_SETTLEMENT_SOCKET_ENV = "CRUCIBLE_DOCKER_SETTLEMENT_SOCKET"
DOCKER_SETTLEMENT_NONCE_ENV = "CRUCIBLE_DOCKER_SETTLEMENT_NONCE"
DOCKER_SETTLEMENT_ARM_ENV = "CRUCIBLE_DOCKER_SETTLEMENT_ARM_FINGERPRINT"
DOCKER_SETTLEMENT_DEADLINE_ENV = "CRUCIBLE_DOCKER_SETTLEMENT_DEADLINE"
DOCKER_SETTLEMENT_ENVIRONMENT = frozenset(
    {
        DOCKER_SETTLEMENT_SOCKET_ENV,
        DOCKER_SETTLEMENT_NONCE_ENV,
        DOCKER_SETTLEMENT_ARM_ENV,
        DOCKER_SETTLEMENT_DEADLINE_ENV,
    }
)

_MAX_JOURNAL_BYTES = 16 * 1024 * 1024
_MAX_BRIDGE_PACKET_BYTES = 16 * 1024
_MAX_REFERENCE_BYTES = 512
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_SOCKET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,91}$")
_SETTLEMENT_LABEL_KEYS = frozenset(
    {
        "io.crucible.settlement",
        "io.crucible.run-generation",
        "io.crucible.arm-generation",
        "io.crucible.daemon-binding",
    }
)
_ARM_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "run_generation",
        "arm_generation",
        "generation_labels",
        "daemon_identity",
        "initial_inventories",
        "issues",
    }
)
_TERMINAL_PAYLOAD_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "arm_fingerprint",
        "daemon_identity",
        "process_tree_proven",
        "process_proof_ref",
        "pre_cleanup_inventory",
        "post_cleanup_inventory",
        "inventory_trace",
        "known_container_ids",
        "removed_container_ids",
        "poll_attempts",
        "clean_observations",
        "issues",
    }
)
_PEER_CREDENTIALS = struct.Struct("3i")


class DockerWorkflowSettlementError(RuntimeError):
    """A durable Docker workflow authority could not be established."""


class DockerWorkflowRecoveryRequired(DockerWorkflowSettlementError):
    """Prior Docker settlement evidence blocks a new workflow arm."""


class DockerSettlementBridgeError(DockerWorkflowSettlementError):
    """The authenticated parent/child registration bridge failed closed."""


class _DockerSettlementBridgeAuthorityError(DockerSettlementBridgeError):
    """A post-registration bridge failure makes settlement uncertain."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def docker_workflow_settlement_partial_leaf(leaf: str) -> str:
    """Return the sole code-owned interrupted-publication name for a journal."""

    if (
        not isinstance(leaf, str)
        or leaf in {"", ".", ".."}
        or "/" in leaf
        or "\x00" in leaf
        or leaf.startswith(DOCKER_WORKFLOW_SETTLEMENT_PARTIAL_PREFIX)
    ):
        raise ValueError("Docker settlement journal leaf is invalid")
    return DOCKER_WORKFLOW_SETTLEMENT_PARTIAL_PREFIX + leaf


def _write_strict_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace one settlement journal with proved durability.

    The general workflow artifact writer deliberately treats directory-fsync
    support as best effort.  A Docker arm/registration acknowledgment is a
    crash-recovery authority, so this boundary must fail closed unless both
    the new file bytes and the containing-directory rename are synchronously
    durable.  The already-bound evidence directory is opened once and every
    mutation is descriptor-relative.
    """

    destination = Path(path)
    leaf = destination.name
    if leaf in {"", ".", ".."} or "/" in leaf or "\x00" in leaf:
        raise ValueError("Docker settlement journal leaf is invalid")
    encoded = _canonical_json_bytes(payload)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(destination.parent, directory_flags)
    temporary = docker_workflow_settlement_partial_leaf(leaf)
    replaced = False
    try:
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        file_fd = os.open(temporary, file_flags, 0o600, dir_fd=directory_fd)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(file_fd, view)
                if written <= 0:
                    raise OSError("Docker settlement journal write made no progress")
                view = view[written:]
            os.fsync(file_fd)
        finally:
            os.close(file_fd)
        os.replace(
            temporary,
            leaf,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        replaced = True
        # This error is intentionally propagated.  Without it, a successful
        # ACK could outlive only in cache while the blocking rename is lost on
        # a crash.
        os.fsync(directory_fd)
    finally:
        if not replaced:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _decode_packet(payload: bytes) -> dict[str, Any]:
    if not payload or len(payload) > _MAX_BRIDGE_PACKET_BYTES:
        raise DockerSettlementBridgeError(
            "Docker settlement bridge packet is empty or oversized"
        )
    try:
        parsed = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DockerSettlementBridgeError(
            "Docker settlement bridge packet is malformed"
        ) from exc
    if not isinstance(parsed, dict) or _canonical_json_bytes(parsed) != payload:
        raise DockerSettlementBridgeError(
            "Docker settlement bridge packet is not canonical JSON"
        )
    return parsed


def _bounded_detail(exc: BaseException) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    return detail[-1_000:]


def _validate_absolute_deadline(
    value: float,
    *,
    monotonic: Callable[[], float],
    require_future: bool,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            "Docker settlement deadline must be a finite monotonic timestamp"
        )
    deadline = float(value)
    if require_future and monotonic() >= deadline:
        raise ValueError("Docker settlement deadline has already expired")
    return deadline


def _remaining(
    deadline_monotonic: float,
    *,
    monotonic: Callable[[], float],
) -> float:
    remaining = deadline_monotonic - monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise DockerSettlementBridgeError("Docker settlement bridge deadline expired")
    return remaining


def _merge_arm_labels(
    labels: Mapping[str, str],
    arm_labels: Mapping[str, str],
) -> dict[str, str]:
    supplied = dict(labels)
    collision = sorted(_SETTLEMENT_LABEL_KEYS.intersection(supplied))
    if collision:
        raise DockerWorkflowSettlementError(
            "caller supplied reserved Docker settlement labels: " + ", ".join(collision)
        )
    lowered = {key.lower() for key in supplied}
    reserved_lowered = {key.lower() for key in _SETTLEMENT_LABEL_KEYS}
    if lowered.intersection(reserved_lowered):
        raise DockerWorkflowSettlementError(
            "caller supplied a case-variant Docker settlement label"
        )
    return {**supplied, **dict(arm_labels)}


def _read_prior_journal(path: Path) -> Mapping[str, Any] | None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DockerWorkflowRecoveryRequired(
            f"Docker settlement journal cannot be inspected: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal is not a regular file"
        )
    if metadata.st_size > _MAX_JOURNAL_BYTES:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal exceeds its size bound"
        )
    try:
        raw = path.read_bytes()
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal is unreadable or malformed"
        ) from exc
    if not isinstance(parsed, dict):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal must be a JSON object"
        )
    return parsed


def _recovery_daemon_from_payload(value: object) -> DockerDaemonIdentity:
    if not isinstance(value, dict) or set(value) != {
        "configuration_fingerprint",
        "endpoint_fingerprint",
        "engine_id",
        "generation",
    }:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery daemon identity differs"
        )
    try:
        result = DockerDaemonIdentity(**value)
    except (TypeError, ValueError) as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery daemon identity differs"
        ) from exc
    if result.to_payload() != value:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery daemon identity changed"
        )
    return result


def _recovery_inspection_from_payload(value: object) -> DockerContainerInspection:
    if (
        not isinstance(value, dict)
        or set(value) != {"container_id", "labels"}
        or not isinstance(value.get("container_id"), str)
        or not isinstance(value.get("labels"), dict)
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in value["labels"].items()
        )
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inspection differs"
        )
    try:
        result = DockerContainerInspection.from_parts(
            value["container_id"],
            value["labels"],
        )
    except ValueError as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inspection differs"
        ) from exc
    if result.to_payload() != value:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inspection changed"
        )
    return result


def _recovery_inventory_from_payload(value: object) -> DockerInventory:
    if not isinstance(value, dict) or set(value) != {
        "absent_container_ids",
        "complete",
        "inspections",
        "listed_container_ids",
    }:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inventory differs"
        )
    listed = value["listed_container_ids"]
    absent = value["absent_container_ids"]
    inspections = value["inspections"]
    if (
        not isinstance(listed, list)
        or not isinstance(absent, list)
        or not isinstance(inspections, list)
        or not isinstance(value["complete"], bool)
        or any(
            not isinstance(item, str) or not _CONTAINER_ID_RE.fullmatch(item)
            for item in (*listed, *absent)
        )
        or listed != sorted(set(listed))
        or absent != sorted(set(absent))
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inventory differs"
        )
    admitted = tuple(_recovery_inspection_from_payload(item) for item in inspections)
    if tuple(item.container_id for item in admitted) != tuple(
        sorted({item.container_id for item in admitted})
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inspections differ"
        )
    result = DockerInventory(
        listed_container_ids=tuple(listed),
        inspections=admitted,
        absent_container_ids=tuple(absent),
        complete=value["complete"],
    )
    if result.to_payload() != value:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery inventory changed"
        )
    return result


def _recovery_issue_from_payload(value: object) -> DockerSettlementIssue:
    if not isinstance(value, dict) or set(value) != {
        "code",
        "container_id",
        "detail",
        "operation",
    }:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery arm issue differs"
        )
    container_id = value["container_id"]
    if (
        not isinstance(value["code"], str)
        or not isinstance(value["operation"], str)
        or not value["operation"]
        or len(value["operation"]) > 256
        or not isinstance(value["detail"], str)
        or not value["detail"]
        or len(value["detail"]) > 1_000
        or (
            container_id is not None
            and (
                not isinstance(container_id, str)
                or not _CONTAINER_ID_RE.fullmatch(container_id)
            )
        )
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery arm issue differs"
        )
    try:
        code = DockerSettlementIssueCode(value["code"])
    except ValueError as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery arm issue differs"
        ) from exc
    result = DockerSettlementIssue(
        code=code,
        operation=value["operation"],
        detail=value["detail"],
        container_id=container_id,
    )
    if result.to_payload() != value:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery arm issue changed"
        )
    return result


def _admit_prework_journal_header(prior: Mapping[str, Any]) -> str:
    expected_keys = {
        "schema_version",
        "deadline_monotonic",
        "arm_fingerprint",
        "arm",
        "registrations",
        "workflow_issues",
        "terminal_fingerprint",
        "terminal",
    }
    deadline = prior.get("deadline_monotonic")
    if (
        set(prior) != expected_keys
        or prior.get("schema_version") != DOCKER_WORKFLOW_SETTLEMENT_SCHEMA
        or not isinstance(deadline, str)
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery journal header differs"
        )
    try:
        parsed_deadline = float(deadline)
    except ValueError as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery retained deadline differs"
        ) from exc
    if (
        not math.isfinite(parsed_deadline)
        or format(parsed_deadline, ".17g") != deadline
        or not isinstance(prior.get("arm_fingerprint"), str)
        or not _SHA256_RE.fullmatch(prior["arm_fingerprint"])
        or (
            (prior.get("terminal") is None)
            != (prior.get("terminal_fingerprint") is None)
        )
        or (
            prior.get("terminal_fingerprint") is not None
            and (
                not isinstance(prior["terminal_fingerprint"], str)
                or not _SHA256_RE.fullmatch(prior["terminal_fingerprint"])
            )
        )
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery journal header differs"
        )
    return deadline


def _uncertain_prework_arm_from_journal(
    prior: Mapping[str, Any],
) -> DockerSettlementArmRecord:
    _admit_prework_journal_header(prior)
    if prior.get("registrations") != [] or prior.get("workflow_issues") != []:
        raise DockerWorkflowRecoveryRequired("Docker prework recovery journal differs")
    arm_value = prior.get("arm")
    if (
        not isinstance(arm_value, dict)
        or set(arm_value) != _ARM_PAYLOAD_KEYS
        or arm_value.get("schema_version") != DOCKER_SETTLEMENT_ARM_SCHEMA
        or arm_value.get("status") != DockerSettlementArmStatus.UNCERTAIN.value
        or not isinstance(arm_value.get("run_generation"), str)
        or not arm_value["run_generation"]
        or len(arm_value["run_generation"]) > 128
        or not isinstance(arm_value.get("arm_generation"), str)
        or not arm_value["arm_generation"]
        or len(arm_value["arm_generation"]) > 128
        or not isinstance(arm_value.get("generation_labels"), dict)
        or not isinstance(arm_value.get("initial_inventories"), list)
        or len(arm_value["initial_inventories"]) > 2
        or not isinstance(arm_value.get("issues"), list)
        or not arm_value["issues"]
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain arm differs"
        )
    labels = arm_value["generation_labels"]
    daemon_value = arm_value.get("daemon_identity")
    daemon = (
        None if daemon_value is None else _recovery_daemon_from_payload(daemon_value)
    )
    expected_label_keys = _SETTLEMENT_LABEL_KEYS - (
        {"io.crucible.daemon-binding"} if daemon is None else set()
    )
    if (
        set(labels) != expected_label_keys
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in labels.items()
        )
        or labels.get("io.crucible.settlement") != "true"
        or labels.get("io.crucible.run-generation") != arm_value["run_generation"]
        or labels.get("io.crucible.arm-generation") != arm_value["arm_generation"]
        or (
            daemon is not None
            and labels.get("io.crucible.daemon-binding")
            != daemon.fingerprint.split(":", 1)[1]
        )
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain labels differ"
        )
    inventories = tuple(
        _recovery_inventory_from_payload(item)
        for item in arm_value["initial_inventories"]
    )
    if daemon is None and inventories:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery daemonless inventory differs"
        )
    if any(
        (
            set(item.absent_container_ids)
            | {inspection.container_id for inspection in item.inspections}
        )
        - set(item.listed_container_ids)
        or any(not inspection.has_labels(labels) for inspection in item.inspections)
        for item in inventories
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery initial inventory differs"
        )
    issues = tuple(_recovery_issue_from_payload(item) for item in arm_value["issues"])
    arm = DockerSettlementArmRecord(
        status=DockerSettlementArmStatus.UNCERTAIN,
        run_generation=arm_value["run_generation"],
        arm_generation=arm_value["arm_generation"],
        generation_labels=tuple(sorted(labels.items())),
        daemon_identity=daemon,
        initial_inventories=inventories,
        issues=issues,
    )
    if arm.to_payload() != arm_value or prior.get("arm_fingerprint") != arm.fingerprint:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain arm changed"
        )
    return arm


def _armed_prework_arm_from_journal(
    prior: Mapping[str, Any],
) -> DockerSettlementArmRecord:
    _admit_prework_journal_header(prior)
    if prior.get("registrations") != [] or prior.get("workflow_issues") != []:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery armed journal differs"
        )
    arm_value = prior.get("arm")
    if (
        not isinstance(arm_value, dict)
        or set(arm_value) != _ARM_PAYLOAD_KEYS
        or arm_value.get("schema_version") != DOCKER_SETTLEMENT_ARM_SCHEMA
        or arm_value.get("status") != DockerSettlementArmStatus.ARMED.value
        or arm_value.get("issues") != []
        or not isinstance(arm_value.get("run_generation"), str)
        or not arm_value["run_generation"]
        or len(arm_value["run_generation"]) > 128
        or not isinstance(arm_value.get("arm_generation"), str)
        or not arm_value["arm_generation"]
        or len(arm_value["arm_generation"]) > 128
        or not isinstance(arm_value.get("generation_labels"), dict)
        or not isinstance(arm_value.get("initial_inventories"), list)
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery armed arm differs"
        )
    daemon = _recovery_daemon_from_payload(arm_value.get("daemon_identity"))
    labels = arm_value["generation_labels"]
    if (
        set(labels) != _SETTLEMENT_LABEL_KEYS
        or any(
            not isinstance(key, str) or not isinstance(item, str)
            for key, item in labels.items()
        )
        or labels.get("io.crucible.settlement") != "true"
        or labels.get("io.crucible.run-generation") != arm_value["run_generation"]
        or labels.get("io.crucible.arm-generation") != arm_value["arm_generation"]
        or labels.get("io.crucible.daemon-binding")
        != daemon.fingerprint.split(":", 1)[1]
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery armed labels differ"
        )
    inventories = tuple(
        _recovery_inventory_from_payload(item)
        for item in arm_value["initial_inventories"]
    )
    if len(inventories) != 2 or any(
        not item.complete
        or item.listed_container_ids
        or item.inspections
        or item.absent_container_ids
        for item in inventories
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery armed initial inventory differs"
        )
    arm = DockerSettlementArmRecord(
        status=DockerSettlementArmStatus.ARMED,
        run_generation=arm_value["run_generation"],
        arm_generation=arm_value["arm_generation"],
        generation_labels=tuple(sorted(labels.items())),
        daemon_identity=daemon,
        initial_inventories=inventories,
        issues=(),
    )
    if arm.to_payload() != arm_value or prior.get("arm_fingerprint") != arm.fingerprint:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery armed arm changed"
        )
    return arm


def _uncertain_prework_terminal_from_journal(
    prior: Mapping[str, Any],
    *,
    arm: DockerSettlementArmRecord,
) -> DockerSettlementTerminal:
    value = prior.get("terminal")
    fingerprint = prior.get("terminal_fingerprint")
    if (
        not isinstance(value, dict)
        or set(value) != _TERMINAL_PAYLOAD_KEYS
        or value.get("schema_version") != DOCKER_SETTLEMENT_TERMINAL_SCHEMA
        or value.get("status") != DockerSettlementStatus.UNCERTAIN.value
        or value.get("arm_fingerprint") != arm.fingerprint
        or not isinstance(value.get("process_tree_proven"), bool)
        or (
            value.get("process_proof_ref") is not None
            and (
                not isinstance(value["process_proof_ref"], str)
                or not _SHA256_RE.fullmatch(value["process_proof_ref"])
            )
        )
        or not isinstance(value.get("issues"), list)
        or not value["issues"]
        or not isinstance(fingerprint, str)
        or not _SHA256_RE.fullmatch(fingerprint)
        or canonical_fingerprint(value) != fingerprint
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain terminal differs"
        )
    daemon_value = value.get("daemon_identity")
    daemon = (
        None if daemon_value is None else _recovery_daemon_from_payload(daemon_value)
    )
    pre = _recovery_inventory_from_payload(value.get("pre_cleanup_inventory"))
    post = _recovery_inventory_from_payload(value.get("post_cleanup_inventory"))
    trace_value = value.get("inventory_trace")
    known = value.get("known_container_ids")
    removed = value.get("removed_container_ids")
    if (
        not isinstance(trace_value, list)
        or len(trace_value) > 8
        or not isinstance(known, list)
        or not isinstance(removed, list)
        or len(known) > 256
        or any(
            not isinstance(item, str) or not _CONTAINER_ID_RE.fullmatch(item)
            for item in (*known, *removed)
        )
        or known != sorted(set(known))
        or removed != sorted(set(removed))
        or set(removed) - set(known)
        or not isinstance(value.get("poll_attempts"), int)
        or isinstance(value["poll_attempts"], bool)
        or not 0 <= value["poll_attempts"] <= 8
        or not isinstance(value.get("clean_observations"), int)
        or isinstance(value["clean_observations"], bool)
        or not 0 <= value["clean_observations"] <= 2
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain terminal inventory differs"
        )
    trace = tuple(_recovery_inventory_from_payload(item) for item in trace_value)
    issues = tuple(_recovery_issue_from_payload(item) for item in value["issues"])
    terminal = DockerSettlementTerminal(
        status=DockerSettlementStatus.UNCERTAIN,
        arm_fingerprint=arm.fingerprint,
        daemon_identity=daemon,
        process_tree_proven=value["process_tree_proven"],
        process_proof_ref=value["process_proof_ref"],
        pre_cleanup_inventory=pre,
        post_cleanup_inventory=post,
        inventory_trace=trace,
        known_container_ids=tuple(known),
        removed_container_ids=tuple(removed),
        poll_attempts=value["poll_attempts"],
        clean_observations=value["clean_observations"],
        issues=issues,
    )
    if terminal.to_payload() != value:
        raise DockerWorkflowRecoveryRequired(
            "Docker prework recovery uncertain terminal changed"
        )
    return terminal


def _blocked_prework_from_journal(
    prior: Mapping[str, Any],
) -> tuple[DockerSettlementArmRecord, DockerSettlementTerminal | None]:
    arm_value = prior.get("arm")
    status = arm_value.get("status") if isinstance(arm_value, dict) else None
    if status == DockerSettlementArmStatus.UNCERTAIN.value:
        arm = _uncertain_prework_arm_from_journal(prior)
        if (
            prior.get("terminal") is not None
            or prior.get("terminal_fingerprint") is not None
        ):
            raise DockerWorkflowRecoveryRequired(
                "Docker prework uncertain arm terminal differs"
            )
        return arm, None
    if status == DockerSettlementArmStatus.ARMED.value:
        arm = _armed_prework_arm_from_journal(prior)
        return arm, _uncertain_prework_terminal_from_journal(prior, arm=arm)
    raise DockerWorkflowRecoveryRequired(
        "Docker prework recovery blocking shape differs"
    )


def _blocked_prework_container_ids(
    arm: DockerSettlementArmRecord,
    terminal: DockerSettlementTerminal | None,
) -> tuple[str, ...]:
    retained: set[str] = set()

    def add_inventory(inventory: DockerInventory) -> None:
        retained.update(inventory.listed_container_ids)
        retained.update(inventory.absent_container_ids)
        retained.update(item.container_id for item in inventory.inspections)

    for inventory in arm.initial_inventories:
        add_inventory(inventory)
    if terminal is not None:
        add_inventory(terminal.pre_cleanup_inventory)
        add_inventory(terminal.post_cleanup_inventory)
        for inventory in terminal.inventory_trace:
            add_inventory(inventory)
        retained.update(terminal.known_container_ids)
        retained.update(terminal.removed_container_ids)
    result = tuple(sorted(retained))
    if len(result) > 256:
        raise DockerWorkflowRecoveryRequired(
            "Docker blocked prework container inventory exceeds its bound"
        )
    return result


def _admit_settled_prework_journal(
    prior: Mapping[str, Any],
    *,
    arm: DockerSettlementArmRecord,
    process_proof_ref: str | None = None,
    expected_container_ids: tuple[str, ...] | None = None,
) -> tuple[DockerSettlementArmRecord, DockerSettlementTerminal]:
    _admit_prework_journal_header(prior)
    terminal_value = prior.get("terminal")
    terminal_fingerprint = prior.get("terminal_fingerprint")
    if (
        not isinstance(terminal_value, dict)
        or set(terminal_value) != _TERMINAL_PAYLOAD_KEYS
        or terminal_value.get("schema_version") != DOCKER_SETTLEMENT_TERMINAL_SCHEMA
        or terminal_value.get("status") != DockerSettlementStatus.SETTLED.value
        or terminal_value.get("arm_fingerprint") != arm.fingerprint
        or terminal_value.get("process_tree_proven") is not True
        or not isinstance(terminal_value.get("process_proof_ref"), str)
        or not _SHA256_RE.fullmatch(terminal_value["process_proof_ref"])
        or terminal_value["process_proof_ref"] == "sha256:" + "0" * 64
        or (
            process_proof_ref is not None
            and terminal_value["process_proof_ref"] != process_proof_ref
        )
        or terminal_value.get("issues") != []
        or not isinstance(terminal_fingerprint, str)
        or not _SHA256_RE.fullmatch(terminal_fingerprint)
        or canonical_fingerprint(terminal_value) != terminal_fingerprint
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework terminal differs"
        )
    daemon = _recovery_daemon_from_payload(terminal_value.get("daemon_identity"))
    if arm.daemon_identity is not None and daemon != arm.daemon_identity:
        raise DockerWorkflowRecoveryRequired("Docker recovered prework daemon differs")
    pre = _recovery_inventory_from_payload(terminal_value.get("pre_cleanup_inventory"))
    post = _recovery_inventory_from_payload(
        terminal_value.get("post_cleanup_inventory")
    )
    trace_value = terminal_value.get("inventory_trace")
    known = terminal_value.get("known_container_ids")
    removed = terminal_value.get("removed_container_ids")
    if (
        not isinstance(trace_value, list)
        or not 2 <= len(trace_value) <= 8
        or not isinstance(known, list)
        or not isinstance(removed, list)
        or len(known) > 256
        or any(
            not isinstance(item, str) or not _CONTAINER_ID_RE.fullmatch(item)
            for item in (*known, *removed)
        )
        or known != sorted(set(known))
        or removed != sorted(set(removed))
        or set(removed) - set(known)
        or (
            expected_container_ids is not None
            and tuple(known) != expected_container_ids
        )
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework container inventory differs"
        )
    trace = tuple(_recovery_inventory_from_payload(item) for item in trace_value)
    observed_ids = {
        item
        for inventory in trace
        for item in (
            *inventory.listed_container_ids,
            *inventory.absent_container_ids,
            *(entry.container_id for entry in inventory.inspections),
        )
    }
    if (
        pre != trace[0]
        or post != trace[-1]
        or trace[-2:] != (post, post)
        or any(
            not inspection.has_labels(arm.labels)
            for inventory in trace
            for inspection in inventory.inspections
        )
        or observed_ids - set(known)
        or post.listed_container_ids
        or post.inspections
        or not post.complete
        or set(post.absent_container_ids) != set(known)
        or terminal_value.get("poll_attempts") != len(trace)
        or terminal_value.get("clean_observations") != 2
        or isinstance(terminal_value.get("poll_attempts"), bool)
        or isinstance(terminal_value.get("clean_observations"), bool)
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework cleanup proof differs"
        )
    terminal = DockerSettlementTerminal(
        status=DockerSettlementStatus.SETTLED,
        arm_fingerprint=arm.fingerprint,
        daemon_identity=daemon,
        process_tree_proven=True,
        process_proof_ref=terminal_value["process_proof_ref"],
        pre_cleanup_inventory=pre,
        post_cleanup_inventory=post,
        inventory_trace=trace,
        known_container_ids=tuple(known),
        removed_container_ids=tuple(removed),
        poll_attempts=terminal_value["poll_attempts"],
        clean_observations=terminal_value["clean_observations"],
        issues=(),
    )
    if terminal.to_payload() != terminal_value:
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework terminal changed"
        )
    return arm, terminal


def _decode_canonical_settled_journal(
    journal_canonical_bytes: bytes,
) -> dict[str, Any]:
    """Decode one exact, bounded v2 journal preimage."""

    if (
        type(journal_canonical_bytes) is not bytes
        or not journal_canonical_bytes
        or len(journal_canonical_bytes) > _MAX_JOURNAL_BYTES
    ):
        raise ValueError(
            "Docker settlement journal is empty or exceeds its fixed byte bound"
        )
    try:
        parsed = json.loads(
            journal_canonical_bytes.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        canonical = _canonical_json_bytes(parsed)
    except (TypeError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Docker settlement journal preimage is malformed") from exc
    if type(parsed) is not dict or canonical != journal_canonical_bytes:
        raise ValueError(
            "Docker settlement journal preimage is not one canonical JSON object"
        )
    return parsed


def _settled_receipt_components(
    journal_canonical_bytes: bytes,
    *,
    process_settlement: ProcessSettlementProof,
) -> tuple[
    DockerSettlementArmRecord,
    tuple[DockerContainerInspection, ...],
    DockerSettlementTerminal,
]:
    """Reconstruct every authority-bearing component retained by a receipt."""

    if type(process_settlement) is not ProcessSettlementProof:
        raise ValueError("Docker settlement receipt process proof type is invalid")
    try:
        validated_process = ProcessSettlementProof.from_evidence(
            dict(process_settlement.evidence)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Docker settlement receipt process proof is invalid") from exc
    if validated_process != process_settlement or not validated_process.cgroup_contained:
        raise ValueError(
            "Docker settlement receipt requires an exact delegated-cgroup process proof"
        )

    prior = _decode_canonical_settled_journal(journal_canonical_bytes)
    if prior.get("workflow_issues") != []:
        raise ValueError("Docker settlement receipt journal is not clean")

    # The existing arm recovery validator deliberately accepts only an empty
    # pre-terminal registration list.  Reuse it against an exact projection,
    # then separately reconstruct the retained settled registration sequence.
    arm_projection = dict(prior)
    arm_projection["registrations"] = []
    arm_projection["workflow_issues"] = []
    try:
        arm = _armed_prework_arm_from_journal(arm_projection)
    except DockerWorkflowRecoveryRequired as exc:
        raise ValueError("Docker settlement receipt arm is invalid") from exc
    arm_value = prior.get("arm")
    if (
        type(arm_value) is not dict
        or _canonical_json_bytes(arm.to_payload())
        != _canonical_json_bytes(arm_value)
    ):
        raise ValueError("Docker settlement receipt arm preimage changed")

    registration_values = prior.get("registrations")
    if type(registration_values) is not list:
        raise ValueError("Docker settlement receipt registrations are invalid")
    try:
        registrations = tuple(
            _recovery_inspection_from_payload(value)
            for value in registration_values
        )
    except DockerWorkflowRecoveryRequired as exc:
        raise ValueError("Docker settlement receipt registrations are invalid") from exc
    registration_ids = tuple(item.container_id for item in registrations)
    if (
        len(registrations) > 256
        or registration_ids != tuple(sorted(set(registration_ids)))
        or any(not item.has_labels(arm.labels) for item in registrations)
        or [item.to_payload() for item in registrations] != registration_values
    ):
        raise ValueError("Docker settlement receipt registrations changed")

    try:
        _admitted_arm, terminal = _admit_settled_prework_journal(
            prior,
            arm=arm,
            process_proof_ref=validated_process.sha256,
        )
    except DockerWorkflowRecoveryRequired as exc:
        raise ValueError("Docker settlement receipt terminal is invalid") from exc
    terminal_value = prior.get("terminal")
    reconstructed_known = set(registration_ids)
    reconstructed_removed: list[str] = []
    for inventory in terminal.inventory_trace:
        listed = set(inventory.listed_container_ids)
        inspected = {item.container_id for item in inventory.inspections}
        absent = set(inventory.absent_container_ids)
        requested = listed | reconstructed_known
        if (
            not inventory.complete
            or listed != inspected
            or inspected.intersection(absent)
            or inspected | absent != requested
        ):
            raise ValueError(
                "Docker settlement receipt terminal inventory cannot be reconstructed"
            )
        for inspection in inventory.inspections:
            reconstructed_known.add(inspection.container_id)
            if inspection.container_id not in reconstructed_removed:
                reconstructed_removed.append(inspection.container_id)
    if (
        type(terminal_value) is not dict
        or _canonical_json_bytes(terminal.to_payload())
        != _canonical_json_bytes(terminal_value)
        or tuple(sorted(reconstructed_known)) != terminal.known_container_ids
        or tuple(reconstructed_removed) != terminal.removed_container_ids
    ):
        raise ValueError("Docker settlement receipt terminal preimage changed")
    return arm, registrations, terminal


@dataclass(frozen=True, slots=True)
class DockerWorkflowSettlementReceipt:
    """Full canonical Docker journal bound to exact outer process evidence.

    This is a provider-free evidence codec, not execution or launch authority.
    Live acquisition remains owned by :class:`DockerWorkflowSettlement`, which
    exposes a receipt only after its clean terminal has been durably persisted
    and descriptor-relatively reread.
    """

    journal_canonical_bytes: bytes
    process_settlement: ProcessSettlementProof

    def __post_init__(self) -> None:
        if type(self) is not DockerWorkflowSettlementReceipt:
            raise ValueError("Docker settlement receipt subclasses are invalid")
        _settled_receipt_components(
            self.journal_canonical_bytes,
            process_settlement=self.process_settlement,
        )

    @property
    def journal_sha256(self) -> str:
        return _sha256_digest(self.journal_canonical_bytes)

    @property
    def process_settlement_record_digest(self) -> str:
        return self.process_settlement.sha256

    @property
    def arm(self) -> DockerSettlementArmRecord:
        arm, _registrations, _terminal = _settled_receipt_components(
            self.journal_canonical_bytes,
            process_settlement=self.process_settlement,
        )
        return arm

    @property
    def registrations(self) -> tuple[DockerContainerInspection, ...]:
        _arm, registrations, _terminal = _settled_receipt_components(
            self.journal_canonical_bytes,
            process_settlement=self.process_settlement,
        )
        return registrations

    @property
    def terminal(self) -> DockerSettlementTerminal:
        _arm, _registrations, terminal = _settled_receipt_components(
            self.journal_canonical_bytes,
            process_settlement=self.process_settlement,
        )
        return terminal

    @property
    def terminal_fingerprint(self) -> str:
        return canonical_fingerprint(self.terminal.to_payload())

    def to_payload(self) -> dict[str, Any]:
        arm, _registrations, terminal = _settled_receipt_components(
            self.journal_canonical_bytes,
            process_settlement=self.process_settlement,
        )
        terminal_fingerprint = canonical_fingerprint(terminal.to_payload())
        return {
            "schema_version": DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA,
            "journal": {
                "schema_version": DOCKER_WORKFLOW_SETTLEMENT_SCHEMA,
                "canonical_json": self.journal_canonical_bytes.decode(
                    "utf-8", errors="strict"
                ),
                "sha256": self.journal_sha256,
            },
            "process_settlement": {
                "record_digest": self.process_settlement_record_digest,
                "evidence": dict(self.process_settlement.evidence),
            },
            "binding": {
                "arm_fingerprint": arm.fingerprint,
                "terminal_fingerprint": terminal_fingerprint,
                "process_settlement_record_digest": (
                    self.process_settlement_record_digest
                ),
            },
            "authority": {
                "scope": "provider-free-docker-settlement-evidence-only",
                "verifier_outcome_authority_conferred": False,
                "production_execution_authority_conferred": False,
                "launch_authority_conferred": False,
            },
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())

    @classmethod
    def from_payload(cls, payload: object) -> "DockerWorkflowSettlementReceipt":
        if cls is not DockerWorkflowSettlementReceipt:
            raise ValueError("Docker settlement receipt subclasses are invalid")
        if (
            type(payload) is not dict
            or any(type(key) is not str for key in payload)
            or set(payload)
            != {
                "schema_version",
                "journal",
                "process_settlement",
                "binding",
                "authority",
            }
            or type(payload.get("schema_version")) is not str
            or payload["schema_version"]
            != DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA
        ):
            raise ValueError("Docker settlement receipt fields are invalid")
        journal = payload["journal"]
        process = payload["process_settlement"]
        binding = payload["binding"]
        authority = payload["authority"]
        expected_nested_fields = (
            (journal, {"schema_version", "canonical_json", "sha256"}),
            (process, {"record_digest", "evidence"}),
            (
                binding,
                {
                    "arm_fingerprint",
                    "terminal_fingerprint",
                    "process_settlement_record_digest",
                },
            ),
            (
                authority,
                {
                    "scope",
                    "verifier_outcome_authority_conferred",
                    "production_execution_authority_conferred",
                    "launch_authority_conferred",
                },
            ),
        )
        if any(
            type(value) is not dict
            or any(type(key) is not str for key in value)
            or set(value) != fields
            for value, fields in expected_nested_fields
        ):
            raise ValueError("Docker settlement receipt nested fields are invalid")
        if (
            type(journal["schema_version"]) is not str
            or journal["schema_version"] != DOCKER_WORKFLOW_SETTLEMENT_SCHEMA
            or type(journal["canonical_json"]) is not str
            or type(journal["sha256"]) is not str
            or type(process["record_digest"]) is not str
            or type(process["evidence"]) is not dict
            or type(authority["scope"]) is not str
            or authority["scope"]
            != "provider-free-docker-settlement-evidence-only"
            or authority["verifier_outcome_authority_conferred"] is not False
            or authority["production_execution_authority_conferred"] is not False
            or authority["launch_authority_conferred"] is not False
        ):
            raise ValueError("Docker settlement receipt evidence is invalid")
        try:
            journal_bytes = journal["canonical_json"].encode(
                "utf-8", errors="strict"
            )
        except UnicodeError as exc:
            raise ValueError("Docker settlement receipt journal is not UTF-8") from exc
        if _sha256_digest(journal_bytes) != journal["sha256"]:
            raise ValueError("Docker settlement receipt journal digest changed")
        try:
            process_settlement = ProcessSettlementProof.from_evidence(
                process["evidence"]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Docker settlement receipt process evidence is invalid"
            ) from exc
        if process_settlement.sha256 != process["record_digest"]:
            raise ValueError("Docker settlement receipt process digest changed")
        result = cls(
            journal_canonical_bytes=journal_bytes,
            process_settlement=process_settlement,
        )
        expected_binding = {
            "arm_fingerprint": result.arm.fingerprint,
            "terminal_fingerprint": result.terminal_fingerprint,
            "process_settlement_record_digest": (
                result.process_settlement_record_digest
            ),
        }
        if binding != expected_binding:
            raise ValueError("Docker settlement receipt binding changed")
        try:
            exact = _canonical_json_bytes(result.to_payload()) == _canonical_json_bytes(
                payload
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Docker settlement receipt is not canonical") from exc
        if not exact:
            raise ValueError("Docker settlement receipt is not canonical")
        return result


def _journal_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_receipt_journal_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise DockerWorkflowSettlementError(
            "Docker settlement receipt journal is not a regular file"
        )
    if metadata.st_nlink != 1:
        raise DockerWorkflowSettlementError(
            "Docker settlement receipt journal is not single-link"
        )
    if metadata.st_size <= 0 or metadata.st_size > _MAX_JOURNAL_BYTES:
        raise DockerWorkflowSettlementError(
            "Docker settlement receipt journal exceeds its fixed byte bound"
        )


def _read_durable_settlement_journal(path: Path) -> bytes:
    """Fsync and descriptor-relatively reread one exact settlement leaf."""

    destination = Path(path)
    leaf = destination.name
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if (
        leaf in {"", ".", ".."}
        or "/" in leaf
        or "\x00" in leaf
        or type(no_follow) is not int
        or type(directory_only) is not int
    ):
        raise DockerWorkflowSettlementError(
            "Docker settlement receipt journal path is invalid"
        )
    directory_fd: int | None = None
    rebound_directory_fd: int | None = None
    journal_fd: int | None = None
    rebound_fd: int | None = None
    close_error: OSError | None = None
    try:
        directory_fd = os.open(
            destination.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | directory_only
            | no_follow,
        )
        directory_before = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_before.st_mode):
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt parent is not a directory"
            )
        before = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        _require_receipt_journal_metadata(before)
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | no_follow
        )
        journal_fd = os.open(leaf, file_flags, dir_fd=directory_fd)
        opened = os.fstat(journal_fd)
        _require_receipt_journal_metadata(opened)
        if _journal_file_identity(before) != _journal_file_identity(opened):
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt journal identity changed before fsync"
            )

        # Terminal publication already fsynced this file and directory.  The
        # receipt boundary repeats those barriers on the exact held objects,
        # then obtains its retained bytes from the same descriptor.
        os.fsync(journal_fd)
        os.fsync(directory_fd)
        after_fsync = os.fstat(journal_fd)
        path_after_fsync = os.stat(
            leaf,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            _journal_file_identity(opened) != _journal_file_identity(after_fsync)
            or _journal_file_identity(opened)
            != _journal_file_identity(path_after_fsync)
        ):
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt journal identity changed during fsync"
            )

        chunks: list[bytes] = []
        count = 0
        while True:
            chunk = os.read(
                journal_fd,
                min(1024 * 1024, _MAX_JOURNAL_BYTES + 1 - count),
            )
            if not chunk:
                break
            chunks.append(chunk)
            count += len(chunk)
            if count > _MAX_JOURNAL_BYTES:
                raise DockerWorkflowSettlementError(
                    "Docker settlement receipt journal grew beyond its byte bound"
                )

        after_read = os.fstat(journal_fd)
        path_after_read = os.stat(
            leaf,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        rebound_fd = os.open(leaf, file_flags, dir_fd=directory_fd)
        rebound = os.fstat(rebound_fd)
        _require_receipt_journal_metadata(rebound)
        directory_after = os.fstat(directory_fd)
        rebound_directory_fd = os.open(
            destination.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | directory_only
            | no_follow,
        )
        rebound_directory = os.fstat(rebound_directory_fd)
        rebound_path = os.stat(
            leaf,
            dir_fd=rebound_directory_fd,
            follow_symlinks=False,
        )
        if (
            _journal_file_identity(opened) != _journal_file_identity(after_read)
            or _journal_file_identity(opened)
            != _journal_file_identity(path_after_read)
            or _journal_file_identity(opened) != _journal_file_identity(rebound)
            or (directory_before.st_dev, directory_before.st_ino)
            != (directory_after.st_dev, directory_after.st_ino)
            or (directory_before.st_dev, directory_before.st_ino)
            != (rebound_directory.st_dev, rebound_directory.st_ino)
            or _journal_file_identity(opened)
            != _journal_file_identity(rebound_path)
        ):
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt journal identity changed while reading"
            )
        payload = b"".join(chunks)
        if len(payload) != opened.st_size:
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt journal size changed while reading"
            )
        return payload
    except DockerWorkflowSettlementError:
        raise
    except OSError as exc:
        raise DockerWorkflowSettlementError(
            "Docker settlement receipt journal could not be durably reread"
        ) from exc
    finally:
        for descriptor in (
            rebound_directory_fd,
            rebound_fd,
            journal_fd,
            directory_fd,
        ):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as exc:
                    if close_error is None:
                        close_error = exc
        if close_error is not None:
            raise DockerWorkflowSettlementError(
                "Docker settlement receipt journal descriptors could not be closed"
            ) from close_error


def _admit_recovered_uncertain_prework_journal(
    prior: Mapping[str, Any],
    *,
    process_proof_ref: str | None = None,
) -> tuple[DockerSettlementArmRecord, DockerSettlementTerminal]:
    arm = _uncertain_prework_arm_from_journal(prior)
    return _admit_settled_prework_journal(
        prior,
        arm=arm,
        process_proof_ref=process_proof_ref,
    )


def _admit_recovered_blocked_prework_journal(
    original: Mapping[str, Any],
    recovered: Mapping[str, Any],
    *,
    process_proof_ref: str,
) -> tuple[DockerSettlementArmRecord, DockerSettlementTerminal]:
    original_arm, prior_terminal = _blocked_prework_from_journal(original)
    original_deadline = _admit_prework_journal_header(original)
    recovered_deadline = _admit_prework_journal_header(recovered)
    if recovered_deadline != original_deadline:
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework deadline changed"
        )
    original_container_ids = _blocked_prework_container_ids(
        original_arm,
        prior_terminal,
    )
    if original_arm.status is DockerSettlementArmStatus.UNCERTAIN:
        recovered_arm = _uncertain_prework_arm_from_journal(recovered)
    else:
        recovered_arm = _armed_prework_arm_from_journal(recovered)
    if recovered_arm.to_payload() != original_arm.to_payload():
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework arm changed from its blocker"
        )
    admitted_arm, terminal = _admit_settled_prework_journal(
        recovered,
        arm=recovered_arm,
        process_proof_ref=process_proof_ref,
    )
    derived = set(original_container_ids)
    derived.update(terminal.pre_cleanup_inventory.listed_container_ids)
    derived.update(terminal.pre_cleanup_inventory.absent_container_ids)
    derived.update(
        item.container_id for item in terminal.pre_cleanup_inventory.inspections
    )
    expected_container_ids = tuple(sorted(derived))
    if (
        len(expected_container_ids) > 256
        or terminal.known_container_ids != expected_container_ids
        or set(terminal.removed_container_ids) - set(expected_container_ids)
        or set(terminal.post_cleanup_inventory.absent_container_ids)
        != set(expected_container_ids)
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered prework target inventory differs"
        )
    return admitted_arm, terminal


def _admit_recovered_fork_intent_prework_journal(
    original: Mapping[str, Any],
    recovered: Mapping[str, Any],
    *,
    process_proof_ref: str,
) -> tuple[DockerSettlementArmRecord, DockerSettlementTerminal]:
    """Admit recovery of one clean, zero-work arm after durable fork intent."""

    original_arm = _armed_prework_arm_from_journal(original)
    if (
        original.get("terminal") is not None
        or original.get("terminal_fingerprint") is not None
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker fork-intent prework original terminal differs"
        )
    original_deadline = _admit_prework_journal_header(original)
    recovered_deadline = _admit_prework_journal_header(recovered)
    recovered_arm = _armed_prework_arm_from_journal(recovered)
    original_static = dict(original)
    recovered_static = dict(recovered)
    original_static.pop("terminal")
    original_static.pop("terminal_fingerprint")
    recovered_static.pop("terminal")
    recovered_static.pop("terminal_fingerprint")
    if (
        recovered_deadline != original_deadline
        or recovered_arm.to_payload() != original_arm.to_payload()
        or recovered_static != original_static
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker recovered fork-intent prework journal changed"
        )
    expected_container_ids = _blocked_prework_container_ids(original_arm, None)
    return _admit_settled_prework_journal(
        recovered,
        arm=recovered_arm,
        process_proof_ref=process_proof_ref,
        expected_container_ids=expected_container_ids,
    )


def _admit_prior_journal(path: Path) -> None:
    prior = _read_prior_journal(path)
    if prior is None:
        return
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal cannot be re-read"
        ) from exc
    if _canonical_json_bytes(prior) != raw:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal is not canonical JSON"
        )
    expected_keys = {
        "schema_version",
        "deadline_monotonic",
        "arm_fingerprint",
        "arm",
        "registrations",
        "workflow_issues",
        "terminal_fingerprint",
        "terminal",
    }
    if set(prior) != expected_keys:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal has an unknown shape"
        )
    if prior.get("schema_version") != DOCKER_WORKFLOW_SETTLEMENT_SCHEMA:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal has an unsupported schema"
        )
    deadline = prior.get("deadline_monotonic")
    if not isinstance(deadline, str):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal deadline is invalid"
        )
    try:
        parsed_deadline = float(deadline)
    except ValueError as exc:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal deadline is invalid"
        ) from exc
    if (
        not math.isfinite(parsed_deadline)
        or format(parsed_deadline, ".17g") != deadline
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal deadline is not canonical"
        )
    arm = prior.get("arm")
    if (
        not isinstance(arm, dict)
        or set(arm) != _ARM_PAYLOAD_KEYS
        or arm.get("schema_version") != DOCKER_SETTLEMENT_ARM_SCHEMA
        or arm.get("status") != DockerSettlementArmStatus.ARMED.value
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement journal has an invalid arm record"
        )
    arm_fingerprint = prior.get("arm_fingerprint")
    if (
        not isinstance(arm_fingerprint, str)
        or not _SHA256_RE.fullmatch(arm_fingerprint)
        or canonical_fingerprint(arm) != arm_fingerprint
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement arm fingerprint does not match its bytes"
        )
    arm_labels = arm.get("generation_labels")
    if not isinstance(arm_labels, dict) or not _SETTLEMENT_LABEL_KEYS.issubset(
        arm_labels
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement arm labels are incomplete"
        )
    registrations = prior.get("registrations")
    if not isinstance(registrations, list):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement registrations are invalid"
        )
    seen: set[str] = set()
    for item in registrations:
        if not isinstance(item, dict) or set(item) != {"container_id", "labels"}:
            raise DockerWorkflowRecoveryRequired(
                "Docker settlement registration is not a closed inspection"
            )
        container_id = item.get("container_id")
        labels = item.get("labels")
        if (
            not isinstance(container_id, str)
            or not _CONTAINER_ID_RE.fullmatch(container_id)
            or container_id in seen
            or not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in arm_labels.items())
        ):
            raise DockerWorkflowRecoveryRequired(
                "Docker settlement registration is invalid or duplicated"
            )
        seen.add(container_id)
    workflow_issues = prior.get("workflow_issues")
    if not isinstance(workflow_issues, list) or any(
        not isinstance(item, str) or not item or len(item) > 1_000
        for item in workflow_issues
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement workflow issues are invalid"
        )
    terminal = prior.get("terminal")
    terminal_fingerprint = prior.get("terminal_fingerprint")
    if terminal is None:
        if terminal_fingerprint is not None:
            raise DockerWorkflowRecoveryRequired(
                "Docker settlement terminal fingerprint has no terminal"
            )
        raise DockerWorkflowRecoveryRequired(
            "an unterminated Docker generation blocks the next arm"
        )
    if (
        not isinstance(terminal, dict)
        or set(terminal) != _TERMINAL_PAYLOAD_KEYS
        or terminal.get("schema_version") != DOCKER_SETTLEMENT_TERMINAL_SCHEMA
        or not isinstance(terminal_fingerprint, str)
        or not _SHA256_RE.fullmatch(terminal_fingerprint)
        or canonical_fingerprint(terminal) != terminal_fingerprint
    ):
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement terminal is invalid or changed"
        )
    if terminal.get("arm_fingerprint") != arm_fingerprint:
        raise DockerWorkflowRecoveryRequired(
            "Docker settlement terminal is bound to a different arm"
        )
    post = terminal.get("post_cleanup_inventory")
    if (
        terminal.get("status") != DockerSettlementStatus.SETTLED.value
        or terminal.get("process_tree_proven") is not True
        or not isinstance(terminal.get("process_proof_ref"), str)
        or not terminal.get("process_proof_ref")
        or terminal.get("issues") != []
        or workflow_issues
        or not isinstance(post, dict)
        or post.get("complete") is not True
        or post.get("listed_container_ids") != []
        or post.get("inspections") != []
    ):
        raise DockerWorkflowRecoveryRequired(
            "an uncertain Docker settlement blocks the next arm"
        )


class DockerWorkflowSettlement:
    """One durably armed, parent-owned Docker generation."""

    def __init__(
        self,
        *,
        state_path: Path,
        session: DockerSettlementSession,
        arm: DockerSettlementArmRecord,
        deadline_monotonic: float,
        monotonic: Callable[[], float],
    ) -> None:
        self._state_path = Path(state_path)
        self._session = session
        self._arm = arm
        self._deadline_monotonic = _validate_absolute_deadline(
            deadline_monotonic,
            monotonic=monotonic,
            require_future=False,
        )
        self._monotonic = monotonic
        self._registrations: dict[str, DockerContainerInspection] = {}
        self._workflow_issues: list[str] = []
        self._terminal: DockerSettlementTerminal | None = None
        self._terminal_persisted = False
        self._lock = threading.RLock()

    @classmethod
    def arm_new(
        cls,
        *,
        state_path: Path,
        run_generation: str,
        arm_generation: str,
        control_plane: DockerSettlementControlPlane | None = None,
        deadline_monotonic: float | None = None,
        operation_timeout_s: float = 60.0,
        max_container_ids: int = 256,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "DockerWorkflowSettlement":
        """Admit prior state, arm, and fsync before child creation authority."""

        state_path = Path(state_path)
        _admit_prior_journal(state_path)
        if deadline_monotonic is None:
            if (
                isinstance(operation_timeout_s, bool)
                or not isinstance(operation_timeout_s, (int, float))
                or not math.isfinite(operation_timeout_s)
                or operation_timeout_s <= 0
            ):
                raise ValueError(
                    "Docker settlement operation timeout must be finite > 0"
                )
            # Sample exactly once. All later operations retain this same
            # absolute timestamp instead of renewing a relative allowance.
            deadline = monotonic() + float(operation_timeout_s)
        else:
            deadline = _validate_absolute_deadline(
                deadline_monotonic,
                monotonic=monotonic,
                require_future=True,
            )
        plane = control_plane or UnixDockerEngineControlPlane(monotonic=monotonic)
        session = DockerSettlementSession(
            plane,
            max_container_ids=max_container_ids,
            monotonic=monotonic,
        )
        arm = session.arm(
            run_generation=run_generation,
            arm_generation=arm_generation,
            deadline_monotonic=deadline,
        )
        authority = cls(
            state_path=state_path,
            session=session,
            arm=arm,
            deadline_monotonic=deadline,
            monotonic=monotonic,
        )
        authority._persist()
        if arm.status != DockerSettlementArmStatus.ARMED:
            raise DockerWorkflowRecoveryRequired(
                "Docker generation arm is uncertain; no container creation "
                "is authorized"
            )
        return authority

    @classmethod
    def recover_interrupted(
        cls,
        *,
        state_path: Path,
        process_proof_ref: str,
        control_plane: DockerSettlementControlPlane,
        recovery_deadline_monotonic: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "DockerWorkflowSettlement":
        """Settle one exact persisted unterminated arm after outer extinction.

        This is intentionally not a re-arm.  It reconstructs only the closed
        arm and registered canonical IDs already present in the durable
        journal, uses a fresh bounded cleanup deadline, and overwrites that
        same journal with the resulting terminal.  An uncertain result stays
        terminal and blocking.
        """

        if not isinstance(process_proof_ref, str) or not _SHA256_RE.fullmatch(
            process_proof_ref
        ):
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery process proof reference is invalid"
            )
        deadline = _validate_absolute_deadline(
            recovery_deadline_monotonic,
            monotonic=monotonic,
            require_future=True,
        )
        state_path = Path(state_path)
        prior = _read_prior_journal(state_path)
        if prior is None:
            raise DockerWorkflowRecoveryRequired("Docker recovery journal is absent")
        raw = state_path.read_bytes()
        if _canonical_json_bytes(prior) != raw:
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery journal is not canonical JSON"
            )
        expected_keys = {
            "schema_version",
            "deadline_monotonic",
            "arm_fingerprint",
            "arm",
            "registrations",
            "workflow_issues",
            "terminal_fingerprint",
            "terminal",
        }
        if (
            set(prior) != expected_keys
            or prior.get("schema_version") != DOCKER_WORKFLOW_SETTLEMENT_SCHEMA
            or prior.get("workflow_issues") != []
            or prior.get("terminal") is not None
            or prior.get("terminal_fingerprint") is not None
        ):
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery journal is not one clean unterminated arm"
            )
        retained_deadline = prior.get("deadline_monotonic")
        if not isinstance(retained_deadline, str):
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery retained deadline is invalid"
            )
        try:
            parsed_retained_deadline = float(retained_deadline)
        except ValueError as exc:
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery retained deadline is invalid"
            ) from exc
        if (
            not math.isfinite(parsed_retained_deadline)
            or format(parsed_retained_deadline, ".17g") != retained_deadline
        ):
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery retained deadline is not canonical"
            )

        def daemon_from(value: object) -> DockerDaemonIdentity:
            if not isinstance(value, dict) or set(value) != {
                "configuration_fingerprint",
                "endpoint_fingerprint",
                "engine_id",
                "generation",
            }:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery daemon identity differs"
                )
            try:
                result = DockerDaemonIdentity(**value)
            except (TypeError, ValueError) as exc:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery daemon identity differs"
                ) from exc
            if result.to_payload() != value:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery daemon identity changed"
                )
            return result

        def inspection_from(value: object) -> DockerContainerInspection:
            if (
                not isinstance(value, dict)
                or set(value) != {"container_id", "labels"}
                or not isinstance(value.get("container_id"), str)
                or not isinstance(value.get("labels"), dict)
                or any(
                    not isinstance(key, str) or not isinstance(item, str)
                    for key, item in value["labels"].items()
                )
            ):
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inspection differs"
                )
            try:
                result = DockerContainerInspection.from_parts(
                    value["container_id"],
                    value["labels"],
                )
            except ValueError as exc:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inspection differs"
                ) from exc
            if result.to_payload() != value:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inspection changed"
                )
            return result

        def inventory_from(value: object) -> DockerInventory:
            if not isinstance(value, dict) or set(value) != {
                "absent_container_ids",
                "complete",
                "inspections",
                "listed_container_ids",
            }:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inventory differs"
                )
            listed = value["listed_container_ids"]
            absent = value["absent_container_ids"]
            inspections = value["inspections"]
            if (
                not isinstance(listed, list)
                or not isinstance(absent, list)
                or not isinstance(inspections, list)
                or not isinstance(value["complete"], bool)
                or any(
                    not isinstance(item, str) or not _CONTAINER_ID_RE.fullmatch(item)
                    for item in (*listed, *absent)
                )
                or listed != sorted(set(listed))
                or absent != sorted(set(absent))
            ):
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inventory differs"
                )
            result = DockerInventory(
                listed_container_ids=tuple(listed),
                inspections=tuple(inspection_from(item) for item in inspections),
                absent_container_ids=tuple(absent),
                complete=value["complete"],
            )
            if result.to_payload() != value:
                raise DockerWorkflowRecoveryRequired(
                    "Docker recovery inventory changed"
                )
            return result

        arm_value = prior.get("arm")
        if not isinstance(arm_value, dict) or set(arm_value) != _ARM_PAYLOAD_KEYS:
            raise DockerWorkflowRecoveryRequired("Docker recovery arm differs")
        labels = arm_value.get("generation_labels")
        initial = arm_value.get("initial_inventories")
        if (
            arm_value.get("schema_version") != DOCKER_SETTLEMENT_ARM_SCHEMA
            or arm_value.get("status") != DockerSettlementArmStatus.ARMED.value
            or arm_value.get("issues") != []
            or not isinstance(labels, dict)
            or set(labels) != _SETTLEMENT_LABEL_KEYS
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in labels.items()
            )
            or not isinstance(initial, list)
            or len(initial) != 2
            or not isinstance(arm_value.get("run_generation"), str)
            or not isinstance(arm_value.get("arm_generation"), str)
        ):
            raise DockerWorkflowRecoveryRequired("Docker recovery arm differs")
        try:
            arm = DockerSettlementArmRecord(
                status=DockerSettlementArmStatus.ARMED,
                run_generation=arm_value["run_generation"],
                arm_generation=arm_value["arm_generation"],
                generation_labels=tuple(sorted(labels.items())),
                daemon_identity=daemon_from(arm_value.get("daemon_identity")),
                initial_inventories=tuple(inventory_from(item) for item in initial),
                issues=(),
            )
        except (TypeError, ValueError) as exc:
            raise DockerWorkflowRecoveryRequired(
                "Docker recovery arm could not be reconstructed"
            ) from exc
        if (
            arm.to_payload() != arm_value
            or prior.get("arm_fingerprint") != arm.fingerprint
        ):
            raise DockerWorkflowRecoveryRequired("Docker recovery arm changed")
        registration_values = prior.get("registrations")
        if not isinstance(registration_values, list):
            raise DockerWorkflowRecoveryRequired("Docker recovery registrations differ")
        registrations = tuple(inspection_from(item) for item in registration_values)
        if (
            len(registrations) > 256
            or len({item.container_id for item in registrations}) != len(registrations)
            or tuple(sorted(item.container_id for item in registrations))
            != tuple(item.container_id for item in registrations)
            or any(not item.has_labels(arm.labels) for item in registrations)
        ):
            raise DockerWorkflowRecoveryRequired("Docker recovery registrations differ")
        session = DockerSettlementSession(
            control_plane,
            max_container_ids=256,
            monotonic=monotonic,
        )
        # The explicit recovery boundary has already re-admitted the complete
        # persisted arm and registration set.  Install precisely that state;
        # no new generation or caller-selected object is accepted.
        session._active_arm = arm  # noqa: SLF001
        session._known_container_ids = {  # noqa: SLF001
            item.container_id for item in registrations
        }
        terminal = session.settle(
            process_tree_proven=True,
            process_proof_ref=process_proof_ref,
            deadline_monotonic=deadline,
        )
        authority = cls(
            state_path=state_path,
            session=session,
            arm=arm,
            deadline_monotonic=parsed_retained_deadline,
            monotonic=monotonic,
        )
        authority._registrations = {  # noqa: SLF001
            item.container_id: item for item in registrations
        }
        authority._terminal = terminal  # noqa: SLF001
        authority._persist()  # noqa: SLF001
        authority._terminal_persisted = True  # noqa: SLF001
        return authority

    @property
    def arm(self) -> DockerSettlementArmRecord:
        return self._arm

    @property
    def labels(self) -> dict[str, str]:
        return self._arm.labels

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def deadline_monotonic(self) -> float:
        return self._deadline_monotonic

    @property
    def terminal(self) -> DockerSettlementTerminal | None:
        with self._lock:
            return self._terminal

    @property
    def terminal_fingerprint(self) -> str | None:
        """Return the exact terminal digest only after durable persistence."""

        with self._lock:
            if self._terminal is None or not self._terminal_persisted:
                return None
            return canonical_fingerprint(self._terminal.to_payload())

    def settlement_receipt(
        self,
        *,
        process_settlement: ProcessSettlementProof,
    ) -> DockerWorkflowSettlementReceipt:
        """Retain the exact durable journal and its outer process proof."""

        with self._lock:
            terminal = self._terminal
            if (
                terminal is None
                or not self._terminal_persisted
                or terminal.status is not DockerSettlementStatus.SETTLED
                or terminal.process_tree_proven is not True
                or terminal.issues
                or self._workflow_issues
            ):
                raise DockerWorkflowSettlementError(
                    "Docker settlement receipt requires one clean durable terminal"
                )
            if (
                type(process_settlement) is not ProcessSettlementProof
                or not process_settlement.cgroup_contained
            ):
                raise DockerWorkflowSettlementError(
                    "Docker settlement receipt requires an exact delegated-cgroup "
                    "process proof"
                )
            if terminal.process_proof_ref != process_settlement.sha256:
                raise DockerWorkflowSettlementError(
                    "Docker settlement terminal does not bind the supplied process proof"
                )
            expected_journal = _canonical_json_bytes(self._payload())
            journal = _read_durable_settlement_journal(self._state_path)
            if journal != expected_journal:
                raise DockerWorkflowSettlementError(
                    "Docker settlement durable journal differs from its terminal"
                )
            try:
                return DockerWorkflowSettlementReceipt(
                    journal_canonical_bytes=journal,
                    process_settlement=process_settlement,
                )
            except ValueError as exc:
                raise DockerWorkflowSettlementError(
                    "Docker settlement durable journal could not be revalidated"
                ) from exc

    @property
    def workflow_issues(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._workflow_issues)

    def labels_for(self, labels: Mapping[str, str]) -> dict[str, str]:
        """Merge ordinary managed labels with the reserved arm labels."""

        return _merge_arm_labels(labels, self._arm.labels)

    def _payload(self) -> dict[str, Any]:
        terminal_payload = (
            self._terminal.to_payload() if self._terminal is not None else None
        )
        return {
            "schema_version": DOCKER_WORKFLOW_SETTLEMENT_SCHEMA,
            "deadline_monotonic": format(self._deadline_monotonic, ".17g"),
            "arm_fingerprint": self._arm.fingerprint,
            "arm": self._arm.to_payload(),
            "registrations": [
                self._registrations[key].to_payload()
                for key in sorted(self._registrations)
            ],
            "workflow_issues": list(self._workflow_issues),
            "terminal_fingerprint": (
                canonical_fingerprint(terminal_payload)
                if terminal_payload is not None
                else None
            ),
            "terminal": terminal_payload,
        }

    def _persist(self) -> None:
        try:
            _write_strict_atomic_json(self._state_path, self._payload())
        except Exception as exc:
            issue = "persistence_failed: " + _bounded_detail(exc)
            if issue not in self._workflow_issues:
                self._workflow_issues.append(issue)
            raise DockerWorkflowSettlementError(
                "Docker settlement evidence could not be persisted"
            ) from exc

    def latch_uncertain(self, detail: str) -> None:
        """Retain one wrapper/bridge failure that forbids a clean terminal."""

        bounded = str(detail).strip()[-1_000:]
        if not bounded:
            bounded = "unspecified_workflow_settlement_failure"
        with self._lock:
            if bounded not in self._workflow_issues:
                self._workflow_issues.append(bounded)
            self._persist()

    def register_container(self, reference: str) -> DockerContainerInspection:
        """Bind and fsync one create/run reference before replying to child."""

        with self._lock:
            if self._terminal is not None:
                raise DockerWorkflowSettlementError(
                    "Docker generation is already terminal"
                )
            inspected = self._session.register_container(
                reference,
                deadline_monotonic=self._deadline_monotonic,
            )
            self._registrations[inspected.container_id] = inspected
            self._persist()
            return inspected

    def settle(
        self,
        *,
        process_tree_proven: bool,
        process_proof_ref: str | None,
    ) -> DockerSettlementTerminal:
        """Persist the terminal repeated-absence proof for this arm."""

        with self._lock:
            if self._terminal is not None:
                if not self._terminal_persisted:
                    raise DockerWorkflowSettlementError(
                        "Docker settlement terminal was not durably persisted"
                    )
                return self._terminal
            if process_tree_proven and (
                not isinstance(process_proof_ref, str)
                or not _SHA256_RE.fullmatch(process_proof_ref)
            ):
                issue = "process_proof_ref_invalid"
                if issue not in self._workflow_issues:
                    self._workflow_issues.append(issue)
            effective_process_proof = process_tree_proven and not self._workflow_issues
            terminal = self._session.settle(
                process_tree_proven=effective_process_proof,
                process_proof_ref=(
                    process_proof_ref if effective_process_proof else None
                ),
                deadline_monotonic=self._deadline_monotonic,
            )
            self._terminal = terminal
            self._persist()
            self._terminal_persisted = True
            return terminal


def _peer_uid(connection: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        raise DockerSettlementBridgeError(
            "Docker settlement bridge requires Linux SO_PEERCRED"
        )
    try:
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            _PEER_CREDENTIALS.size,
        )
        _pid, uid, _gid = _PEER_CREDENTIALS.unpack(raw)
    except (OSError, struct.error) as exc:
        raise DockerSettlementBridgeError(
            "Docker settlement bridge peer credentials are unavailable"
        ) from exc
    return uid


def _bridge_address(socket_name: str) -> str:
    if not _SOCKET_NAME_RE.fullmatch(socket_name):
        raise ValueError("Docker settlement abstract socket name is invalid")
    # Leading NUL selects Linux's abstract Unix namespace. It is deliberately
    # not placed in an environment value or retained record.
    return "\0" + socket_name


class DockerSettlementRegistrationServer:
    """Parent-owned authenticated bridge around one settlement authority."""

    def __init__(
        self,
        authority: DockerWorkflowSettlement,
        *,
        socket_name: str | None = None,
        nonce: str | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(authority, DockerWorkflowSettlement):
            raise TypeError("authority must be DockerWorkflowSettlement")
        generated_name = (
            f"crucible-docker-{os.getpid()}-{secrets.token_hex(12)}"
            if socket_name is None
            else socket_name
        )
        _bridge_address(generated_name)
        generated_nonce = secrets.token_hex(32) if nonce is None else nonce
        if not isinstance(generated_nonce, str) or not _NONCE_RE.fullmatch(
            generated_nonce
        ):
            raise ValueError("Docker settlement bridge nonce is invalid")
        self._authority = authority
        self._socket_name = generated_name
        self._nonce = generated_nonce
        self._monotonic = monotonic
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._fatal: BaseException | None = None
        self._state_lock = threading.Lock()

    @property
    def environment(self) -> dict[str, str]:
        with self._state_lock:
            if self._listener is None or self._thread is None:
                raise DockerSettlementBridgeError(
                    "Docker settlement bridge is not running"
                )
        return {
            DOCKER_SETTLEMENT_SOCKET_ENV: self._socket_name,
            DOCKER_SETTLEMENT_NONCE_ENV: self._nonce,
            DOCKER_SETTLEMENT_ARM_ENV: self._authority.arm.fingerprint,
            DOCKER_SETTLEMENT_DEADLINE_ENV: format(
                self._authority.deadline_monotonic,
                ".17g",
            ),
        }

    def start(self) -> "DockerSettlementRegistrationServer":
        _remaining(
            self._authority.deadline_monotonic,
            monotonic=self._monotonic,
        )
        with self._state_lock:
            if self._listener is not None or self._thread is not None:
                raise DockerSettlementBridgeError(
                    "Docker settlement bridge was already started"
                )
            listener = socket.socket(
                socket.AF_UNIX,
                socket.SOCK_SEQPACKET | getattr(socket, "SOCK_CLOEXEC", 0),
            )
            try:
                listener.bind(_bridge_address(self._socket_name))
                listener.listen(8)
                listener.settimeout(0.1)
            except BaseException:
                listener.close()
                raise
            self._listener = listener
            self._thread = threading.Thread(
                target=self._serve,
                name="crucible-docker-settlement-bridge",
                daemon=True,
            )
            self._thread.start()
        return self

    def _response(
        self,
        *,
        request_type: str,
        ok: bool,
        inspection: DockerContainerInspection | None = None,
        error: str | None = None,
    ) -> bytes:
        return _canonical_json_bytes(
            {
                "schema_version": DOCKER_SETTLEMENT_BRIDGE_SCHEMA,
                "type": request_type,
                "ok": ok,
                "arm_fingerprint": self._authority.arm.fingerprint,
                "deadline_monotonic": format(
                    self._authority.deadline_monotonic,
                    ".17g",
                ),
                "labels": self._authority.labels,
                "inspection": (
                    inspection.to_payload() if inspection is not None else None
                ),
                "error": error,
            }
        )

    def _handle(self, connection: socket.socket) -> None:
        remaining = _remaining(
            self._authority.deadline_monotonic,
            monotonic=self._monotonic,
        )
        connection.settimeout(min(remaining, 1.0))
        if _peer_uid(connection) != os.getuid():
            raise DockerSettlementBridgeError(
                "Docker settlement bridge rejected a different-UID peer"
            )
        try:
            payload = connection.recv(_MAX_BRIDGE_PACKET_BYTES + 1)
        except (OSError, TimeoutError) as exc:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge request read failed"
            ) from exc
        request = _decode_packet(payload)
        expected = {
            "schema_version",
            "type",
            "nonce",
            "arm_fingerprint",
            "reference",
        }
        if set(request) != expected:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge request has an unknown shape"
            )
        request_type = request.get("type")
        if request_type not in {"describe", "register"}:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge request type is unsupported"
            )
        if (
            request.get("schema_version") != DOCKER_SETTLEMENT_BRIDGE_SCHEMA
            or not secrets.compare_digest(str(request.get("nonce")), self._nonce)
            or request.get("arm_fingerprint") != self._authority.arm.fingerprint
        ):
            raise DockerSettlementBridgeError(
                "Docker settlement bridge authority did not match"
            )
        reference = request.get("reference")
        registered: DockerContainerInspection | None = None
        if request_type == "describe":
            if reference is not None:
                raise DockerSettlementBridgeError(
                    "Docker settlement describe request carried a reference"
                )
            response = self._response(request_type="describe", ok=True)
        else:
            if (
                not isinstance(reference, str)
                or not reference
                or "\x00" in reference
                or len(reference.encode("utf-8")) > _MAX_REFERENCE_BYTES
            ):
                raise DockerSettlementBridgeError(
                    "Docker settlement registration reference is invalid"
                )
            try:
                inspection = self._authority.register_container(reference)
            except BaseException as exc:
                response = self._response(
                    request_type="register",
                    ok=False,
                    error=_bounded_detail(exc),
                )
            else:
                registered = inspection
                response = self._response(
                    request_type="register",
                    ok=True,
                    inspection=inspection,
                )
        try:
            sent = connection.send(response)
        except (OSError, TimeoutError) as exc:
            error_type = (
                _DockerSettlementBridgeAuthorityError
                if registered is not None
                else DockerSettlementBridgeError
            )
            raise error_type("Docker settlement bridge response write failed") from exc
        if sent != len(response):
            error_type = (
                _DockerSettlementBridgeAuthorityError
                if registered is not None
                else DockerSettlementBridgeError
            )
            raise error_type("Docker settlement bridge response was truncated")
        if registered is not None:
            # A successful send proves only that the kernel accepted the
            # packet.  Require the child to attest that it received and
            # validated the exact persisted canonical ID before this bridge
            # can remain a clean authority.
            try:
                receipt_payload = connection.recv(_MAX_BRIDGE_PACKET_BYTES + 1)
                receipt = _decode_packet(receipt_payload)
            except (OSError, TimeoutError, DockerSettlementBridgeError) as exc:
                raise _DockerSettlementBridgeAuthorityError(
                    "Docker settlement registration acknowledgment was lost"
                ) from exc
            if receipt != {
                "schema_version": DOCKER_SETTLEMENT_BRIDGE_SCHEMA,
                "type": "registration_received",
                "nonce": self._nonce,
                "arm_fingerprint": self._authority.arm.fingerprint,
                "container_id": registered.container_id,
            }:
                raise _DockerSettlementBridgeAuthorityError(
                    "Docker settlement registration acknowledgment did not "
                    "bind the persisted container"
                )

    def _serve(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                return
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if self._stop.is_set():
                    return
                self._record_fatal(exc)
                return
            try:
                with connection:
                    self._handle(connection)
            except _DockerSettlementBridgeAuthorityError as exc:
                self._record_fatal(exc)
                return
            except DockerSettlementBridgeError:
                # Bad authentication or malformed client input is rejected by
                # closing this one packet connection. It cannot mutate the
                # parent session and is not itself settlement uncertainty.
                continue
            except BaseException as exc:
                self._record_fatal(exc)
                return

    def _record_fatal(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal is None:
                self._fatal = exc
        try:
            self._authority.latch_uncertain(
                "registration_bridge_failed: " + _bounded_detail(exc)
            )
        except DockerWorkflowSettlementError:
            pass

    def close(self) -> None:
        self._stop.set()
        with self._state_lock:
            listener = self._listener
            thread = self._thread
            self._listener = None
            self._thread = None
        if listener is not None:
            listener.close()
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                self._record_fatal(
                    DockerSettlementBridgeError(
                        "Docker settlement bridge thread did not stop"
                    )
                )
        if self._fatal is not None:
            raise DockerSettlementBridgeError(
                "Docker settlement registration bridge failed"
            ) from self._fatal

    def __enter__(self) -> "DockerSettlementRegistrationServer":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class DockerSettlementRegistrationClient:
    """Child-side synchronous client for one exact parent-owned arm."""

    def __init__(
        self,
        *,
        socket_name: str,
        nonce: str,
        arm_fingerprint: str,
        deadline_monotonic: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        _bridge_address(socket_name)
        if not _NONCE_RE.fullmatch(nonce):
            raise ValueError("Docker settlement bridge nonce is invalid")
        if not _SHA256_RE.fullmatch(arm_fingerprint):
            raise ValueError("Docker settlement arm fingerprint is invalid")
        self._socket_name = socket_name
        self._nonce = nonce
        self._arm_fingerprint = arm_fingerprint
        self._deadline_monotonic = _validate_absolute_deadline(
            deadline_monotonic,
            monotonic=monotonic,
            require_future=True,
        )
        self._monotonic = monotonic
        response = self._request("describe", reference=None)
        labels = response.get("labels")
        if not isinstance(labels, dict) or set(labels) != _SETTLEMENT_LABEL_KEYS:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge returned incomplete arm labels"
            )
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in labels.items()
        ):
            raise DockerSettlementBridgeError(
                "Docker settlement bridge returned invalid arm labels"
            )
        self._labels = dict(labels)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        required: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> "DockerSettlementRegistrationClient | None":
        source = os.environ if environment is None else environment
        present = {name for name in DOCKER_SETTLEMENT_ENVIRONMENT if name in source}
        if not present:
            if required:
                raise DockerSettlementBridgeError(
                    "required Docker settlement bridge environment is absent"
                )
            return None
        if present != DOCKER_SETTLEMENT_ENVIRONMENT:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge environment is incomplete"
            )
        try:
            deadline = float(source[DOCKER_SETTLEMENT_DEADLINE_ENV])
        except (TypeError, ValueError) as exc:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge deadline is invalid"
            ) from exc
        if (
            not math.isfinite(deadline)
            or format(deadline, ".17g") != source[DOCKER_SETTLEMENT_DEADLINE_ENV]
        ):
            raise DockerSettlementBridgeError(
                "Docker settlement bridge deadline is not canonical"
            )
        try:
            return cls(
                socket_name=source[DOCKER_SETTLEMENT_SOCKET_ENV],
                nonce=source[DOCKER_SETTLEMENT_NONCE_ENV],
                arm_fingerprint=source[DOCKER_SETTLEMENT_ARM_ENV],
                deadline_monotonic=deadline,
                monotonic=monotonic,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DockerSettlementBridgeError(
                "Docker settlement bridge environment is invalid"
            ) from exc

    @property
    def arm_fingerprint(self) -> str:
        return self._arm_fingerprint

    @property
    def deadline_monotonic(self) -> float:
        return self._deadline_monotonic

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    def labels_for(self, labels: Mapping[str, str]) -> dict[str, str]:
        return _merge_arm_labels(labels, self._labels)

    def _inspection_from_response(
        self,
        response: Mapping[str, Any],
    ) -> DockerContainerInspection:
        inspection = response.get("inspection")
        if not isinstance(inspection, dict) or set(inspection) != {
            "container_id",
            "labels",
        }:
            raise DockerSettlementBridgeError(
                "Docker settlement registration returned no inspection"
            )
        container_id = inspection.get("container_id")
        labels = inspection.get("labels")
        if (
            not isinstance(container_id, str)
            or not _CONTAINER_ID_RE.fullmatch(container_id)
            or not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in self._labels.items())
        ):
            raise DockerSettlementBridgeError(
                "Docker settlement registration inspection is invalid"
            )
        return DockerContainerInspection.from_parts(container_id, labels)

    def _request(
        self,
        request_type: str,
        *,
        reference: str | None,
    ) -> dict[str, Any]:
        request = _canonical_json_bytes(
            {
                "schema_version": DOCKER_SETTLEMENT_BRIDGE_SCHEMA,
                "type": request_type,
                "nonce": self._nonce,
                "arm_fingerprint": self._arm_fingerprint,
                "reference": reference,
            }
        )
        connection = socket.socket(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET | getattr(socket, "SOCK_CLOEXEC", 0),
        )
        try:
            try:
                connection.settimeout(
                    _remaining(
                        self._deadline_monotonic,
                        monotonic=self._monotonic,
                    )
                )
                connection.connect(_bridge_address(self._socket_name))
                if _peer_uid(connection) != os.getuid():
                    raise DockerSettlementBridgeError(
                        "Docker settlement bridge server UID did not match"
                    )
                sent = connection.send(request)
                if sent != len(request):
                    raise DockerSettlementBridgeError(
                        "Docker settlement bridge request was truncated"
                    )
                response_payload = connection.recv(_MAX_BRIDGE_PACKET_BYTES + 1)
            except DockerSettlementBridgeError:
                raise
            except (OSError, TimeoutError) as exc:
                raise DockerSettlementBridgeError(
                    "Docker settlement bridge transport failed"
                ) from exc
            _remaining(
                self._deadline_monotonic,
                monotonic=self._monotonic,
            )
            response = _decode_packet(response_payload)
            expected = {
                "schema_version",
                "type",
                "ok",
                "arm_fingerprint",
                "deadline_monotonic",
                "labels",
                "inspection",
                "error",
            }
            if (
                set(response) != expected
                or response.get("schema_version") != DOCKER_SETTLEMENT_BRIDGE_SCHEMA
                or response.get("type") != request_type
                or response.get("arm_fingerprint") != self._arm_fingerprint
                or response.get("deadline_monotonic")
                != format(self._deadline_monotonic, ".17g")
                or type(response.get("ok")) is not bool
            ):
                raise DockerSettlementBridgeError(
                    "Docker settlement bridge response authority did not match"
                )
            if response["ok"] is not True:
                detail = response.get("error")
                raise DockerSettlementBridgeError(
                    "Docker settlement registration was rejected"
                    + (f": {detail}" if isinstance(detail, str) else "")
                )
            if response.get("error") is not None:
                raise DockerSettlementBridgeError(
                    "successful Docker settlement response carried an error"
                )
            if request_type == "register":
                inspection = self._inspection_from_response(response)
                receipt = _canonical_json_bytes(
                    {
                        "schema_version": DOCKER_SETTLEMENT_BRIDGE_SCHEMA,
                        "type": "registration_received",
                        "nonce": self._nonce,
                        "arm_fingerprint": self._arm_fingerprint,
                        "container_id": inspection.container_id,
                    }
                )
                try:
                    acknowledged = connection.send(receipt)
                except (OSError, TimeoutError) as exc:
                    raise DockerSettlementBridgeError(
                        "Docker settlement registration acknowledgment failed"
                    ) from exc
                if acknowledged != len(receipt):
                    raise DockerSettlementBridgeError(
                        "Docker settlement registration acknowledgment was truncated"
                    )
            return response
        finally:
            connection.close()

    def register_container(self, reference: str) -> DockerContainerInspection:
        response = self._request("register", reference=reference)
        return self._inspection_from_response(response)


__all__ = [
    "DOCKER_SETTLEMENT_ARM_ENV",
    "DOCKER_SETTLEMENT_BRIDGE_SCHEMA",
    "DOCKER_SETTLEMENT_DEADLINE_ENV",
    "DOCKER_SETTLEMENT_ENVIRONMENT",
    "DOCKER_SETTLEMENT_NONCE_ENV",
    "DOCKER_SETTLEMENT_SOCKET_ENV",
    "DOCKER_WORKFLOW_SETTLEMENT_RECEIPT_SCHEMA",
    "DOCKER_WORKFLOW_SETTLEMENT_SCHEMA",
    "DockerSettlementBridgeError",
    "DockerSettlementRegistrationClient",
    "DockerSettlementRegistrationServer",
    "DockerWorkflowRecoveryRequired",
    "DockerWorkflowSettlement",
    "DockerWorkflowSettlementError",
    "DockerWorkflowSettlementReceipt",
]
