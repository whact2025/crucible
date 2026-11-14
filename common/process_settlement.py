# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Assay-free canonical evidence for supervised process settlement.

The proof is deliberately a pure byte codec.  It performs no process, cgroup,
filesystem, provider, or workflow-state operations, so product executors can
validate retained outer-process evidence without importing the workflow
package.  Historical supervisor-only proofs remain decodable; callers that
require delegated-cgroup authority must enforce :attr:`cgroup_contained` at
their own admission boundary. Current callers that also require an exact
launched process must enforce :attr:`target_identity`; historical v2 proofs
intentionally expose no such identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


CGROUP_SETTLEMENT_SCHEMA = "crucible.cgroup-settlement.v1"
_PROCESS_SUPERVISOR_SCHEMA = "crucible.process-supervisor.v3"
_PROCESS_SUPERVISOR_PREVIOUS_SCHEMA = "crucible.process-supervisor.v2"
_PROCESS_SUPERVISOR_LEGACY_SCHEMA = "crucible.process-supervisor.v1"
_PROCESS_SUPERVISOR_CONTROL_MAX_BYTES = 4096
_PROCESS_SETTLEMENT_SCHEMA = "crucible.process-settlement.v3"
_PROCESS_SETTLEMENT_EVIDENCE_SCHEMA = "crucible.process-settlement-evidence.v1"
_MAX_CGROUP_SETTLEMENT_PROOF_BYTES = 16 * 1024
_MAX_COMBINED_SETTLEMENT_PROOF_BYTES = 4096


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_settlement_record(
    payload: bytes,
    *,
    label: str,
    maximum_bytes: int,
) -> Mapping[str, Any]:
    """Decode one bounded, closed, byte-canonical settlement preimage."""

    if type(payload) is not bytes or not payload or len(payload) > maximum_bytes:
        raise ValueError(f"{label} is missing or exceeds its fixed bound")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for key, value in pairs:
            if key in record:
                raise ValueError(f"{label} repeats a field")
            record[key] = value
        return record

    try:
        record = json.loads(
            payload.decode("ascii", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
        canonical = (
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is malformed") from exc
    if type(record) is not dict or payload != canonical:
        raise ValueError(f"{label} is not one canonical JSON object")
    return record


def _validated_supervisor_settlement(payload: bytes) -> Mapping[str, Any]:
    record = _canonical_settlement_record(
        payload,
        label="process supervisor settlement proof",
        maximum_bytes=_PROCESS_SUPERVISOR_CONTROL_MAX_BYTES,
    )
    returncode = record.get("returncode")
    schema_version = record.get("schema_version")
    expected_fields = {"schema_version", "returncode", "settled", "error"}
    if schema_version in {
        _PROCESS_SUPERVISOR_SCHEMA,
        _PROCESS_SUPERVISOR_PREVIOUS_SCHEMA,
    }:
        expected_fields.update(
            {
                "cgroup_admission_sha256",
                "supervisor_pid",
                "supervisor_start_token",
            }
        )
        if schema_version == _PROCESS_SUPERVISOR_SCHEMA:
            expected_fields.update({"target_pid", "target_start_token"})
    elif schema_version != _PROCESS_SUPERVISOR_LEGACY_SCHEMA:
        raise ValueError("process supervisor settlement schema is unsupported")
    if (
        set(record) != expected_fields
        or type(returncode) is not int
        or not (-64 <= returncode <= 255 or returncode == -124)
        or record.get("settled") is not True
        or record.get("error") not in {None, "background_descendants"}
        or (
            schema_version
            in {
                _PROCESS_SUPERVISOR_SCHEMA,
                _PROCESS_SUPERVISOR_PREVIOUS_SCHEMA,
            }
            and (
                type(record.get("cgroup_admission_sha256")) is not str
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    record["cgroup_admission_sha256"],
                )
                is None
                or type(record.get("supervisor_pid")) is not int
                or int(record["supervisor_pid"]) <= 1
                or type(record.get("supervisor_start_token")) is not int
                or int(record["supervisor_start_token"]) <= 0
            )
        )
        or (
            schema_version == _PROCESS_SUPERVISOR_SCHEMA
            and not (
                (
                    record.get("target_pid") is None
                    and record.get("target_start_token") is None
                    and returncode in {None, -124}
                )
                or (
                    type(record.get("target_pid")) is int
                    and int(record["target_pid"]) > 1
                    and type(record.get("target_start_token")) is int
                    and int(record["target_start_token"]) > 0
                )
            )
        )
    ):
        raise ValueError("process supervisor settlement proof is not closed")
    return record


def _validated_cgroup_settlement(payload: bytes) -> Mapping[str, Any]:
    record = _canonical_settlement_record(
        payload,
        label="cgroup settlement proof",
        maximum_bytes=_MAX_CGROUP_SETTLEMENT_PROOF_BYTES,
    )
    expected_fields = {
        "controller_cgroup",
        "controller_identity",
        "controller_pid",
        "controller_start_token",
        "admission_outcome",
        "admission_sha256",
        "kill_requested",
        "populated",
        "removed",
        "schema_version",
        "supervisor_pid",
        "supervisor_start_token",
        "target_identity",
    }
    controller_cgroup = record.get("controller_cgroup")
    if type(controller_cgroup) is str:
        controller_path = PurePosixPath(controller_cgroup)
        controller_path_valid = (
            0 < len(controller_cgroup) <= 4096
            and controller_cgroup.isascii()
            and all(0x21 <= ord(character) <= 0x7E for character in controller_cgroup)
            and controller_path.is_absolute()
            and controller_path != PurePosixPath("/")
            and "." not in controller_path.parts
            and ".." not in controller_path.parts
            and str(controller_path) == controller_cgroup
        )
    else:
        controller_path_valid = False
    if (
        set(record) != expected_fields
        or record.get("schema_version") != CGROUP_SETTLEMENT_SCHEMA
        or record.get("admission_outcome") != "committed"
        or type(record.get("admission_sha256")) is not str
        or re.fullmatch(r"sha256:[0-9a-f]{64}", record["admission_sha256"]) is None
        or record.get("kill_requested") is not True
        or type(record.get("populated")) is not int
        or record.get("populated") != 0
        or record.get("removed") is not True
        or not controller_path_valid
        or any(
            type(record.get(field)) is not int or int(record[field]) <= 1
            for field in ("controller_pid", "supervisor_pid")
        )
        or any(
            type(record.get(field)) is not int or int(record[field]) <= 0
            for field in ("controller_start_token", "supervisor_start_token")
        )
        or any(
            type(record.get(field)) is not str
            or re.fullmatch(r"[0-9]+:[0-9]+:[1-9][0-9]*", record[field]) is None
            for field in (
                "controller_identity",
                "target_identity",
            )
        )
    ):
        raise ValueError("cgroup settlement proof is not a committed closed proof")
    return record


@dataclass(frozen=True, slots=True)
class ProcessSettlementProof:
    """Exact canonical preimages for one supervisor/cgroup settlement."""

    supervisor_canonical_bytes: bytes
    cgroup_canonical_bytes: bytes | None
    combined_canonical_bytes: bytes

    def __post_init__(self) -> None:
        supervisor = _validated_supervisor_settlement(self.supervisor_canonical_bytes)
        cgroup_digest: str | None = None
        if self.cgroup_canonical_bytes is not None:
            cgroup = _validated_cgroup_settlement(self.cgroup_canonical_bytes)
            if (
                supervisor.get("schema_version")
                not in {
                    _PROCESS_SUPERVISOR_SCHEMA,
                    _PROCESS_SUPERVISOR_PREVIOUS_SCHEMA,
                }
                or supervisor.get("cgroup_admission_sha256")
                != cgroup.get("admission_sha256")
                or supervisor.get("supervisor_pid") != cgroup.get("supervisor_pid")
                or supervisor.get("supervisor_start_token")
                != cgroup.get("supervisor_start_token")
            ):
                raise ValueError(
                    "supervisor and cgroup settlement identities do not match"
                )
            cgroup_digest = _sha256_bytes(self.cgroup_canonical_bytes)
        elif supervisor.get("schema_version") != _PROCESS_SUPERVISOR_LEGACY_SCHEMA:
            raise ValueError(
                "cgroup-bound supervisor settlement omitted its cgroup preimage"
            )
        combined = _canonical_settlement_record(
            self.combined_canonical_bytes,
            label="combined process settlement proof",
            maximum_bytes=_MAX_COMBINED_SETTLEMENT_PROOF_BYTES,
        )
        expected_containment = (
            "supervisor_and_delegated_cgroup_v2"
            if cgroup_digest is not None
            else "supervisor_only"
        )
        supervisor_returncode = int(supervisor["returncode"])
        expected_returncode = (
            2
            if supervisor.get("error") == "background_descendants"
            and supervisor_returncode == 0
            else supervisor_returncode
        )
        effective_returncode = combined.get("effective_returncode")
        if (
            type(effective_returncode) is not int
            or effective_returncode not in {expected_returncode, -125}
            or combined
            != {
                "cgroup_settlement_sha256": cgroup_digest,
                "containment": expected_containment,
                "effective_returncode": effective_returncode,
                "schema_version": _PROCESS_SETTLEMENT_SCHEMA,
                "supervisor_settlement_sha256": _sha256_bytes(
                    self.supervisor_canonical_bytes
                ),
            }
        ):
            raise ValueError(
                "combined process settlement proof does not bind its preimages"
            )

    @classmethod
    def compose(
        cls,
        *,
        supervisor_canonical_bytes: bytes,
        cgroup_canonical_bytes: bytes | None,
        effective_returncode: int,
    ) -> ProcessSettlementProof:
        supervisor_sha256 = _sha256_bytes(supervisor_canonical_bytes)
        cgroup_sha256 = (
            None
            if cgroup_canonical_bytes is None
            else _sha256_bytes(cgroup_canonical_bytes)
        )
        combined = (
            json.dumps(
                {
                    "cgroup_settlement_sha256": cgroup_sha256,
                    "containment": (
                        "supervisor_and_delegated_cgroup_v2"
                        if cgroup_sha256 is not None
                        else "supervisor_only"
                    ),
                    "effective_returncode": effective_returncode,
                    "schema_version": _PROCESS_SETTLEMENT_SCHEMA,
                    "supervisor_settlement_sha256": supervisor_sha256,
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
        return cls(
            supervisor_canonical_bytes=supervisor_canonical_bytes,
            cgroup_canonical_bytes=cgroup_canonical_bytes,
            combined_canonical_bytes=combined,
        )

    @classmethod
    def from_evidence(cls, value: Mapping[str, Any]) -> ProcessSettlementProof:
        if type(value) is not dict or set(value) != {
            "schema_version",
            "supervisor",
            "cgroup",
            "combined",
        }:
            raise ValueError("process settlement evidence has an open schema")
        if value.get("schema_version") != _PROCESS_SETTLEMENT_EVIDENCE_SCHEMA:
            raise ValueError("process settlement evidence schema is unsupported")

        def preimage(
            raw: Any,
            *,
            label: str,
            maximum_bytes: int,
        ) -> tuple[bytes, str]:
            if type(raw) is not dict or set(raw) != {
                "canonical_json",
                "sha256",
            }:
                raise ValueError(f"{label} evidence has an open schema")
            canonical_json = raw.get("canonical_json")
            claimed_sha256 = raw.get("sha256")
            if type(canonical_json) is not str or type(claimed_sha256) is not str:
                raise ValueError(f"{label} evidence is malformed")
            try:
                payload = canonical_json.encode("ascii", errors="strict")
            except UnicodeEncodeError as exc:
                raise ValueError(f"{label} evidence is not ASCII") from exc
            if len(payload) > maximum_bytes or _sha256_bytes(payload) != claimed_sha256:
                raise ValueError(f"{label} evidence digest does not match")
            return payload, claimed_sha256

        supervisor_bytes, _supervisor_sha256 = preimage(
            value.get("supervisor"),
            label="process supervisor settlement",
            maximum_bytes=_PROCESS_SUPERVISOR_CONTROL_MAX_BYTES,
        )
        raw_cgroup = value.get("cgroup")
        cgroup_bytes: bytes | None
        if raw_cgroup is None:
            cgroup_bytes = None
        else:
            cgroup_bytes, _cgroup_sha256 = preimage(
                raw_cgroup,
                label="cgroup settlement",
                maximum_bytes=_MAX_CGROUP_SETTLEMENT_PROOF_BYTES,
            )
        combined_bytes, _combined_sha256 = preimage(
            value.get("combined"),
            label="combined process settlement",
            maximum_bytes=_MAX_COMBINED_SETTLEMENT_PROOF_BYTES,
        )
        return cls(
            supervisor_canonical_bytes=supervisor_bytes,
            cgroup_canonical_bytes=cgroup_bytes,
            combined_canonical_bytes=combined_bytes,
        )

    @property
    def cgroup_contained(self) -> bool:
        return self.cgroup_canonical_bytes is not None

    @property
    def target_identity(self) -> tuple[int, int] | None:
        """Return the exact launched target bound by a current terminal proof."""

        supervisor = _validated_supervisor_settlement(
            self.supervisor_canonical_bytes
        )
        if supervisor.get("schema_version") not in {
            _PROCESS_SUPERVISOR_SCHEMA,
        }:
            return None
        if (
            supervisor.get("target_pid") is None
            or supervisor.get("target_start_token") is None
        ):
            return None
        return int(supervisor["target_pid"]), int(
            supervisor["target_start_token"]
        )

    @property
    def effective_returncode(self) -> int:
        combined = _canonical_settlement_record(
            self.combined_canonical_bytes,
            label="combined process settlement proof",
            maximum_bytes=_MAX_COMBINED_SETTLEMENT_PROOF_BYTES,
        )
        value = combined.get("effective_returncode")
        if type(value) is not int:
            raise AssertionError("validated process settlement outcome disappeared")
        return value

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.combined_canonical_bytes)

    @property
    def evidence(self) -> Mapping[str, Any]:
        def preimage(payload: bytes) -> dict[str, str]:
            return {
                "canonical_json": payload.decode("ascii", errors="strict"),
                "sha256": _sha256_bytes(payload),
            }

        return {
            "schema_version": _PROCESS_SETTLEMENT_EVIDENCE_SCHEMA,
            "supervisor": preimage(self.supervisor_canonical_bytes),
            "cgroup": (
                None
                if self.cgroup_canonical_bytes is None
                else preimage(self.cgroup_canonical_bytes)
            ),
            "combined": preimage(self.combined_canonical_bytes),
        }


__all__ = [
    "CGROUP_SETTLEMENT_SCHEMA",
    "ProcessSettlementProof",
]
