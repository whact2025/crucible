# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Durable, fingerprint-bound conversation continuity for workflows.

This module owns the provider-neutral SES-01 state machine.  Adapters retain
responsibility for translating a non-authorizing provider conversation ID to
and from their SDK, but they may reuse it only after this controller admits the
exact route, role, prompt, schema, tool authority, feature profile, policy,
runtime, and trusted workflow context.

The persisted representation is intentionally secret-free: semantic values
are reduced to SHA-256 digests before a handle is built, while the codec
accepts only a closed handle schema.  It never serializes prompt text, route
configuration, response schemas, tool catalogues, policies, runtime
configuration, credentials, or trusted-context bodies.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assay import (
    BackendFeatureSet,
    ConversationHandle,
    ConversationIdentity,
    ConversationMode,
    ConversationRecoveryResolution,
    ConversationResumeDecision,
    ConversationResumeScope,
    ConversationTransportRequest,
    PendingConversationOperation,
    Usage,
    advance_conversation_handle,
    decide_conversation_resume,
    mark_conversation_state,
    new_conversation_handle,
)

from crucible.backends.features import feature_profile_digest

from ._durable_state import AtomicStateStore, DurableStateError


CONVERSATION_STORE_SCHEMA_VERSION = "crucible.conversation-store.v3"
CONVERSATION_OPERATION_SCHEMA_VERSION = "crucible.conversation-operation.v1"
_MAX_RECORD_BYTES = 1024 * 1024
_SHA256_PREFIX = "sha256:"
_HANDLE_FIELDS = frozenset({
    "schema_version",
    "mode",
    "identity_fingerprint",
    "opaque_id",
    "revision",
    "turns_used",
    "context_bytes",
    "usage",
    "state",
    "operation_digests",
    "pending_operation",
    "recovery_resolutions",
})
_LEGACY_USAGE_FIELDS = frozenset({
    "input_tokens",
    "output_tokens",
    "cost_usd",
})
_USAGE_FIELDS = _LEGACY_USAGE_FIELDS | frozenset({
    "cache_creation_tokens",
    "cache_read_tokens",
})
_STORE_FIELDS = frozenset({
    "schema_version",
    "key_fingerprint",
    "handle_digest",
    "handle",
})
_RECOVERY_RESOLUTION_FIELDS = frozenset({
    "operation_digest",
    "identity_fingerprint",
    "evidence_digest",
    "resolved_at",
    "actor",
    "disposition",
})


class ConversationStateError(RuntimeError):
    """A durable conversation record cannot be used safely."""


class ConversationContinuityUnsupported(ConversationStateError):
    """The selected route cannot enforce the requested continuity mode."""


class ConversationReuseRejected(ConversationStateError):
    """A stored handle failed identity, scope, mode, or lifecycle admission."""


class ConversationStoreError(ConversationStateError):
    """A handle record is corrupt, unsafe, or cannot be published durably."""


class ConversationBudgetExceeded(ConversationStateError):
    """A completed operation crossed one or more cumulative ceilings."""

    def __init__(
        self,
        handle: ConversationHandle,
        reasons: tuple[str, ...],
    ) -> None:
        self.handle = handle
        self.reasons = reasons
        super().__init__(
            "conversation cumulative budget exceeded: " + "; ".join(reasons)
        )


class ConversationOperationRecoveryRequired(ConversationStateError):
    """A provider operation may have happened and requires reconciliation."""

    def __init__(
        self,
        handle: ConversationHandle,
        pending: PendingConversationOperation,
    ) -> None:
        self.handle = handle
        self.pending = pending
        super().__init__(
            "conversation has an unresolved "
            f"{pending.state} provider operation "
            f"{pending.operation_digest}"
        )


class ConversationOperationConflict(ConversationStateError):
    """A logical operation conflicts with durable operation state."""


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


def _canonical_json(value: Any, *, field: str) -> Any:
    """Return a detached, finite JSON value for deterministic hashing."""

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
        detached: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            detached[key] = _canonical_json(
                child,
                field=f"{field}.{key}",
            )
        return detached
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


def conversation_value_digest(value: Any, *, domain: str) -> str:
    """Hash one semantic input with an explicit collision-separation domain."""

    if not isinstance(domain, str) or not domain.strip():
        raise ValueError("digest domain must be a non-empty string")
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


def build_conversation_identity(
    *,
    role: str,
    backend: str,
    model: str,
    prompt_abi: str,
    conversation_mode: ConversationMode,
    route_identity: Any,
    prompt_content: str,
    response_schema: Mapping[str, Any] | None,
    tool_authority_digest: str,
    feature_profile: BackendFeatureSet,
    policy_identity: Any,
    runtime_identity: Any,
    trusted_context_identity: Any,
) -> ConversationIdentity:
    """Reduce the complete continuity contract to a secret-free identity.

    ``route_identity`` and the other structured inputs may contain operational
    detail because only their domain-separated digests are retained.  Callers
    should nevertheless provide the smallest semantic projection: resolved
    backend/model/endpoint identity, policy controls, SDK/CLI revisions, and
    run/plan/tier/slice/base/workspace identities.  ``prompt_content`` is the
    stable task/system prompt contract; per-correction content belongs in the
    operation digest rather than changing the conversation identity.
    """

    if not isinstance(prompt_content, str):
        raise TypeError("prompt_content must be a string")
    return ConversationIdentity(
        role=role,
        backend=backend,
        model=model,
        prompt_abi=prompt_abi,
        conversation_mode=conversation_mode,
        route_digest=conversation_value_digest(
            route_identity,
            domain="conversation.route",
        ),
        prompt_digest=conversation_value_digest(
            prompt_content,
            domain="conversation.prompt",
        ),
        schema_digest=conversation_value_digest(
            response_schema,
            domain="conversation.response_schema",
        ),
        tool_authority_digest=_require_digest(
            tool_authority_digest,
            field="tool_authority_digest",
        ),
        feature_profile_digest=feature_profile_digest(feature_profile),
        policy_digest=conversation_value_digest(
            policy_identity,
            domain="conversation.policy",
        ),
        runtime_digest=conversation_value_digest(
            runtime_identity,
            domain="conversation.runtime",
        ),
        trusted_context_digest=conversation_value_digest(
            trusted_context_identity,
            domain="conversation.trusted_context",
        ),
    )


def conversation_operation_digest(
    identity: ConversationIdentity,
    *,
    operation_id: str,
    content_identity: Any,
) -> str:
    """Bind one idempotent logical operation to its exact content.

    A caller must reuse ``operation_id`` only for the same logical correction.
    Including the content identity prevents an accidental same-ID/different-
    prompt replay from being silently treated as already charged.
    """

    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError("operation_id must be a non-empty string")
    return conversation_value_digest(
        {
            "schema_version": CONVERSATION_OPERATION_SCHEMA_VERSION,
            "conversation_fingerprint": identity.fingerprint,
            "operation_id": operation_id,
            "content": content_identity,
        },
        domain="conversation.operation",
    )


@dataclass(frozen=True, slots=True)
class ConversationStoreKey:
    """Logical workflow lane whose filename is always a one-way digest."""

    run_identity_digest: str
    workflow: str
    role: str
    cycle: str

    def __post_init__(self) -> None:
        _require_digest(
            self.run_identity_digest,
            field="ConversationStoreKey.run_identity_digest",
        )
        for field_name in ("workflow", "role", "cycle"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value.encode("utf-8")) > 1024
                or any(ord(character) < 0x20 for character in value)
            ):
                raise ValueError(
                    f"ConversationStoreKey.{field_name} must be a non-empty "
                    "control-free UTF-8 string no larger than 1024 bytes"
                )

    @property
    def fingerprint(self) -> str:
        return conversation_value_digest(
            {
                "run_identity_digest": self.run_identity_digest,
                "workflow": self.workflow,
                "role": self.role,
                "cycle": self.cycle,
            },
            domain="conversation.store_key",
        )


class ConversationHandleCodec:
    """Strict canonical JSON codec for the public Assay handle."""

    @staticmethod
    def to_mapping(handle: ConversationHandle) -> dict[str, Any]:
        if not isinstance(handle, ConversationHandle):
            raise TypeError("handle must be an assay ConversationHandle")
        usage: dict[str, int | float] = {
            "input_tokens": handle.usage.input_tokens,
            "output_tokens": handle.usage.output_tokens,
            "cost_usd": handle.usage.cost_usd,
        }
        if handle.usage.cache_creation_tokens:
            usage["cache_creation_tokens"] = (
                handle.usage.cache_creation_tokens
            )
        if handle.usage.cache_read_tokens:
            usage["cache_read_tokens"] = handle.usage.cache_read_tokens
        return {
            "schema_version": handle.schema_version,
            "mode": handle.mode,
            "identity_fingerprint": handle.identity_fingerprint,
            "opaque_id": handle.opaque_id,
            "revision": handle.revision,
            "turns_used": handle.turns_used,
            "context_bytes": handle.context_bytes,
            "usage": usage,
            "state": handle.state,
            "operation_digests": list(handle.operation_digests),
            "pending_operation": (
                None
                if handle.pending_operation is None
                else {
                    "operation_digest": (
                        handle.pending_operation.operation_digest
                    ),
                    "identity_fingerprint": (
                        handle.pending_operation.identity_fingerprint
                    ),
                    "started_at": handle.pending_operation.started_at,
                    "state": handle.pending_operation.state,
                }
            ),
            "recovery_resolutions": [
                {
                    "operation_digest": resolution.operation_digest,
                    "identity_fingerprint": (
                        resolution.identity_fingerprint
                    ),
                    "evidence_digest": resolution.evidence_digest,
                    "resolved_at": resolution.resolved_at,
                    "actor": resolution.actor,
                    "disposition": resolution.disposition,
                }
                for resolution in handle.recovery_resolutions
            ],
        }

    @classmethod
    def encode(cls, handle: ConversationHandle) -> bytes:
        return json.dumps(
            cls.to_mapping(handle),
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
                raise ConversationStoreError(
                    f"conversation JSON contains duplicate field {key!r}"
                )
            result[key] = value
        return result

    @classmethod
    def parse_json(cls, payload: bytes) -> Any:
        if not isinstance(payload, bytes):
            raise TypeError("conversation payload must be bytes")
        if len(payload) > _MAX_RECORD_BYTES:
            raise ConversationStoreError(
                f"conversation record exceeds {_MAX_RECORD_BYTES} bytes"
            )
        try:
            return json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=cls._reject_duplicate_keys,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ConversationStoreError(
                        f"conversation JSON contains invalid constant {value}"
                    )
                ),
            )
        except ConversationStoreError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConversationStoreError(
                "conversation record is not strict UTF-8 JSON"
            ) from exc

    @classmethod
    def from_mapping(cls, raw: Any) -> ConversationHandle:
        if not isinstance(raw, Mapping):
            raise ConversationStoreError(
                "conversation handle must be a JSON object"
            )
        if any(not isinstance(key, str) for key in raw):
            raise ConversationStoreError(
                "conversation handle contains a non-string field"
            )
        fields = set(raw)
        if fields != _HANDLE_FIELDS:
            raise ConversationStoreError(
                "conversation handle fields do not match the closed schema: "
                f"missing={sorted(_HANDLE_FIELDS - fields)}, "
                f"unknown={sorted(fields - _HANDLE_FIELDS)}"
            )
        usage = raw["usage"]
        if not isinstance(usage, Mapping):
            raise ConversationStoreError(
                "conversation usage fields do not match the closed schema"
            )
        usage_fields = set(usage)
        if (
            not _LEGACY_USAGE_FIELDS.issubset(usage_fields)
            or not usage_fields.issubset(_USAGE_FIELDS)
        ):
            raise ConversationStoreError(
                "conversation usage fields do not match the closed schema"
            )
        operation_digests = raw["operation_digests"]
        if not isinstance(operation_digests, list):
            raise ConversationStoreError(
                "conversation operation_digests must be an array"
            )
        resolutions_raw = raw["recovery_resolutions"]
        if not isinstance(resolutions_raw, list):
            raise ConversationStoreError(
                "conversation recovery_resolutions must be an array"
            )
        resolutions: list[ConversationRecoveryResolution] = []
        for index, resolution_raw in enumerate(resolutions_raw):
            if (
                not isinstance(resolution_raw, Mapping)
                or set(resolution_raw) != _RECOVERY_RESOLUTION_FIELDS
            ):
                raise ConversationStoreError(
                    "conversation recovery resolution fields do not match "
                    f"the closed schema at index {index}"
                )
            try:
                resolutions.append(
                    ConversationRecoveryResolution(
                        operation_digest=resolution_raw[
                            "operation_digest"
                        ],
                        identity_fingerprint=resolution_raw[
                            "identity_fingerprint"
                        ],
                        evidence_digest=resolution_raw["evidence_digest"],
                        resolved_at=resolution_raw["resolved_at"],
                        actor=resolution_raw["actor"],
                        disposition=resolution_raw["disposition"],
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ConversationStoreError(
                    "invalid conversation recovery resolution at index "
                    f"{index}: {exc}"
                ) from exc
        pending_raw = raw["pending_operation"]
        pending: PendingConversationOperation | None
        if pending_raw is None:
            pending = None
        else:
            pending_fields = frozenset({
                "operation_digest",
                "identity_fingerprint",
                "started_at",
                "state",
            })
            if (
                not isinstance(pending_raw, Mapping)
                or set(pending_raw) != pending_fields
            ):
                raise ConversationStoreError(
                    "conversation pending_operation fields do not match "
                    "the closed schema"
                )
            try:
                pending = PendingConversationOperation(
                    operation_digest=pending_raw["operation_digest"],
                    identity_fingerprint=(
                        pending_raw["identity_fingerprint"]
                    ),
                    started_at=pending_raw["started_at"],
                    state=pending_raw["state"],
                )
            except (TypeError, ValueError) as exc:
                raise ConversationStoreError(
                    f"invalid pending conversation operation: {exc}"
                ) from exc
        try:
            return ConversationHandle(
                schema_version=raw["schema_version"],
                mode=raw["mode"],
                identity_fingerprint=raw["identity_fingerprint"],
                opaque_id=raw["opaque_id"],
                revision=raw["revision"],
                turns_used=raw["turns_used"],
                context_bytes=raw["context_bytes"],
                usage=Usage(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    cost_usd=usage["cost_usd"],
                    cache_creation_tokens=usage.get(
                        "cache_creation_tokens",
                        0,
                    ),
                    cache_read_tokens=usage.get("cache_read_tokens", 0),
                ),
                state=raw["state"],
                operation_digests=tuple(operation_digests),
                pending_operation=pending,
                recovery_resolutions=tuple(resolutions),
            )
        except (TypeError, ValueError) as exc:
            raise ConversationStoreError(
                f"invalid conversation handle: {exc}"
            ) from exc

    @classmethod
    def decode(cls, payload: bytes) -> ConversationHandle:
        return cls.from_mapping(cls.parse_json(payload))


class ConversationHandleStore:
    """Per-key locked, atomic, crash-recoverable handle persistence."""

    def __init__(self, root: Path | str) -> None:
        try:
            self._atomic = AtomicStateStore(
                root,
                max_record_bytes=_MAX_RECORD_BYTES,
            )
        except DurableStateError as exc:
            raise ConversationStoreError(
                f"conversation store is unavailable: {exc}"
            ) from exc
        self.root = self._atomic.root

    @property
    def closed(self) -> bool:
        return self._atomic.closed

    def close(self) -> None:
        """Release the retained conversation-state root descriptor."""

        try:
            self._atomic.close()
        except DurableStateError as exc:
            raise ConversationStoreError(str(exc)) from exc

    def __enter__(self) -> ConversationHandleStore:
        self._atomic.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _paths(self, key: ConversationStoreKey) -> tuple[Path, Path, Path]:
        return self._atomic.paths(key.fingerprint)

    @staticmethod
    def _record_bytes(
        key: ConversationStoreKey,
        handle: ConversationHandle,
    ) -> bytes:
        handle_payload = ConversationHandleCodec.encode(handle)
        envelope = {
            "schema_version": CONVERSATION_STORE_SCHEMA_VERSION,
            "key_fingerprint": key.fingerprint,
            "handle_digest": (
                _SHA256_PREFIX + hashlib.sha256(handle_payload).hexdigest()
            ),
            "handle": ConversationHandleCodec.to_mapping(handle),
        }
        return json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _decode_record(
        key: ConversationStoreKey,
        payload: bytes,
    ) -> ConversationHandle:
        raw = ConversationHandleCodec.parse_json(payload)
        if not isinstance(raw, Mapping) or set(raw) != _STORE_FIELDS:
            raise ConversationStoreError(
                "conversation store envelope fields do not match the "
                "closed schema"
            )
        if raw["schema_version"] != CONVERSATION_STORE_SCHEMA_VERSION:
            raise ConversationStoreError(
                "unsupported conversation store schema version"
            )
        if raw["key_fingerprint"] != key.fingerprint:
            raise ConversationStoreError(
                "conversation record belongs to a different store key"
            )
        handle = ConversationHandleCodec.from_mapping(raw["handle"])
        actual_digest = (
            _SHA256_PREFIX
            + hashlib.sha256(ConversationHandleCodec.encode(handle)).hexdigest()
        )
        if raw["handle_digest"] != actual_digest:
            raise ConversationStoreError(
                "conversation handle checksum mismatch"
            )
        return handle

    def load(self, key: ConversationStoreKey) -> ConversationHandle | None:
        try:
            payload = self._atomic.load(key.fingerprint)
        except DurableStateError as exc:
            raise ConversationStoreError(str(exc)) from exc
        if payload is None:
            return None
        return self._decode_record(key, payload)

    def update(
        self,
        key: ConversationStoreKey,
        transform: Callable[
            [ConversationHandle | None],
            ConversationHandle,
        ],
    ) -> ConversationHandle:
        """Lock, recover, transform, and atomically publish one handle."""

        if not callable(transform):
            raise TypeError("conversation transform must be callable")
        result: ConversationHandle | None = None

        def _transform(payload: bytes | None) -> bytes:
            nonlocal result
            current = (
                None
                if payload is None
                else self._decode_record(key, payload)
            )
            updated = transform(current)
            if not isinstance(updated, ConversationHandle):
                raise TypeError(
                    "conversation transform must return ConversationHandle"
                )
            result = updated
            return self._record_bytes(key, updated)

        try:
            self._atomic.update(key.fingerprint, _transform)
        except DurableStateError as exc:
            raise ConversationStoreError(str(exc)) from exc
        assert result is not None
        return result


@dataclass(frozen=True, slots=True)
class ConversationLimits:
    """Cumulative logical-conversation ceilings."""

    max_turns: int | None = None
    max_context_bytes: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_turns",
            "max_context_bytes",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                type(value) is not int or value < 0
            ):
                raise ValueError(
                    f"ConversationLimits.{field_name} must be an exact "
                    "integer >= 0 or None"
                )
        cost = self.max_cost_usd
        if cost is not None and (
            type(cost) not in (int, float)
            or not math.isfinite(cost)
            or cost < 0
        ):
            raise ValueError(
                "ConversationLimits.max_cost_usd must be finite and "
                "non-negative or None"
            )


@dataclass(frozen=True, slots=True)
class RemainingConversationBudget:
    turns: int | None
    context_bytes: int | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None


def _remaining(limit: int | float | None, used: int | float):
    if limit is None:
        return None
    return max(type(limit)(0), limit - used)


def remaining_conversation_budget(
    handle: ConversationHandle,
    limits: ConversationLimits,
) -> RemainingConversationBudget:
    return RemainingConversationBudget(
        turns=_remaining(limits.max_turns, handle.turns_used),
        context_bytes=_remaining(
            limits.max_context_bytes,
            handle.context_bytes,
        ),
        input_tokens=_remaining(
            limits.max_input_tokens,
            handle.usage.input_tokens,
        ),
        output_tokens=_remaining(
            limits.max_output_tokens,
            handle.usage.output_tokens,
        ),
        total_tokens=_remaining(
            limits.max_total_tokens,
            handle.usage.input_tokens + handle.usage.output_tokens,
        ),
        cost_usd=_remaining(
            limits.max_cost_usd,
            handle.usage.cost_usd,
        ),
    )


def _budget_reasons(
    handle: ConversationHandle,
    limits: ConversationLimits,
    *,
    exhausted: bool,
) -> tuple[str, ...]:
    comparison = (lambda used, limit: used >= limit) if exhausted else (
        lambda used, limit: used > limit
    )
    values = (
        ("turns", handle.turns_used, limits.max_turns),
        ("context_bytes", handle.context_bytes, limits.max_context_bytes),
        ("input_tokens", handle.usage.input_tokens, limits.max_input_tokens),
        (
            "output_tokens",
            handle.usage.output_tokens,
            limits.max_output_tokens,
        ),
        (
            "total_tokens",
            handle.usage.input_tokens + handle.usage.output_tokens,
            limits.max_total_tokens,
        ),
        ("cost_usd", handle.usage.cost_usd, limits.max_cost_usd),
    )
    return tuple(
        f"{name}={used} {'reached' if exhausted else 'exceeded'} limit={limit}"
        for name, used, limit in values
        if limit is not None and comparison(used, limit)
    )


def _admit_continuity(
    profile: BackendFeatureSet,
    mode: ConversationMode,
) -> None:
    if mode not in {"none", "retry_only", "workflow_bound"}:
        raise ValueError(f"unsupported conversation mode {mode!r}")
    if mode == "none":
        return
    supported = profile.session_continuity
    if mode == "retry_only" and supported in {
        "retry_only",
        "workflow_bound",
    }:
        return
    if mode == "workflow_bound" and supported == "workflow_bound":
        return
    raise ConversationContinuityUnsupported(
        f"route declares session_continuity={supported!r}, which cannot "
        f"satisfy requested mode {mode!r}"
    )


class ConversationController:
    """Atomic workflow-facing handle admission and cumulative accounting."""

    def __init__(
        self,
        *,
        store: ConversationHandleStore,
        key: ConversationStoreKey,
        identity: ConversationIdentity,
        mode: ConversationMode,
        feature_profile: BackendFeatureSet,
        limits: ConversationLimits = ConversationLimits(),
    ) -> None:
        _admit_continuity(feature_profile, mode)
        if key.role != identity.role:
            raise ValueError(
                "conversation store-key role must match identity role"
            )
        expected_profile_digest = feature_profile_digest(feature_profile)
        if identity.feature_profile_digest != expected_profile_digest:
            raise ValueError(
                "conversation identity feature-profile digest does not "
                "match the admitted route profile"
            )
        self.store = store
        self.key = key
        self.identity = identity
        self.mode = mode
        self.feature_profile = feature_profile
        self.limits = limits

    def close_store(self) -> None:
        """Release storage authority without closing the logical conversation."""

        self.store.close()

    def __enter__(self) -> ConversationController:
        self.store.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close_store()

    def _validate_current(
        self,
        handle: ConversationHandle,
    ) -> ConversationHandle:
        if handle.identity_fingerprint != self.identity.fingerprint:
            raise ConversationReuseRejected(
                "conversation identity fingerprint mismatch"
            )
        if handle.mode != self.mode:
            raise ConversationReuseRejected(
                f"stored conversation mode {handle.mode!r} does not match "
                f"requested mode {self.mode!r}"
            )
        return handle

    def resume(
        self,
        *,
        scope: ConversationResumeScope,
        provider_reports_stale: bool = False,
        allow_stale_fallback: bool = False,
    ) -> ConversationResumeDecision:
        decision: ConversationResumeDecision | None = None

        def _decide(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            nonlocal decision
            if current is None:
                handle = new_conversation_handle(
                    self.identity,
                    mode=self.mode,
                    opaque_id=None,
                )
                decision = ConversationResumeDecision(
                    action="fresh",
                    reason="no durable conversation handle exists",
                    handle=handle,
                )
                return handle
            self._validate_current(current)
            if current.pending_operation is not None:
                raise ConversationOperationRecoveryRequired(
                    current,
                    current.pending_operation,
                )
            decision = decide_conversation_resume(
                current,
                self.identity,
                expected_mode=self.mode,
                scope=scope,
                provider_reports_stale=provider_reports_stale,
                allow_stale_fallback=allow_stale_fallback,
            )
            return decision.handle

        handle = self.store.update(self.key, _decide)
        assert decision is not None
        if decision.action == "reject":
            raise ConversationReuseRejected(decision.reason)
        exhausted = _budget_reasons(
            handle,
            self.limits,
            exhausted=True,
        )
        if exhausted:
            raise ConversationBudgetExceeded(handle, exhausted)
        return ConversationResumeDecision(
            action=decision.action,
            reason=decision.reason,
            handle=handle,
        )

    def load(self) -> ConversationHandle | None:
        handle = self.store.load(self.key)
        if handle is None:
            return None
        return self._validate_current(handle)

    def remaining_budget(self) -> RemainingConversationBudget:
        handle = self.load()
        if handle is None:
            handle = new_conversation_handle(
                self.identity,
                mode=self.mode,
                opaque_id=None,
            )
        return remaining_conversation_budget(handle, self.limits)

    def begin_operation(
        self,
        *,
        operation_digest: str,
        started_at: str,
    ) -> ConversationHandle:
        """Durably mark an operation before any provider side effect.

        A process crash after this write and before ``commit_operation`` leaves
        an explicit pending marker.  Restarted workers fail closed instead of
        silently replaying a request whose provider outcome is unknown.
        """

        _require_digest(operation_digest, field="operation_digest")

        def _begin(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation must be resumed before an operation begins"
                )
            self._validate_current(current)
            if operation_digest in current.operation_digests:
                return current
            pending = current.pending_operation
            if pending is not None:
                if (
                    pending.operation_digest == operation_digest
                    and pending.state == "pending"
                ):
                    return current
                raise ConversationOperationRecoveryRequired(current, pending)
            if current.state == "closed":
                raise ConversationOperationConflict(
                    "cannot begin an operation on a closed conversation"
                )
            return dataclasses.replace(
                current,
                revision=current.revision + 1,
                pending_operation=PendingConversationOperation(
                    operation_digest=operation_digest,
                    identity_fingerprint=self.identity.fingerprint,
                    started_at=started_at,
                ),
            )

        return self.store.update(self.key, _begin)

    def claim_operation(
        self,
        *,
        operation_digest: str,
        started_at: str,
    ) -> ConversationHandle:
        """Atomically claim one provider side effect for this caller.

        Unlike :meth:`begin_operation`, this operation is intentionally not
        idempotent when the same pending digest already exists.  Workflow
        callers use it immediately before transport so two processes that
        race on the same logical operation cannot both interpret the durable
        marker as their own and duplicate a provider request.
        """

        _require_digest(operation_digest, field="operation_digest")

        def _claim(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation must be resumed before an operation claim"
                )
            self._validate_current(current)
            if operation_digest in current.operation_digests:
                raise ConversationOperationConflict(
                    "cannot claim an already committed operation"
                )
            pending = current.pending_operation
            if pending is not None:
                raise ConversationOperationRecoveryRequired(current, pending)
            if current.state == "closed":
                raise ConversationOperationConflict(
                    "cannot claim an operation on a closed conversation"
                )
            return dataclasses.replace(
                current,
                revision=current.revision + 1,
                pending_operation=PendingConversationOperation(
                    operation_digest=operation_digest,
                    identity_fingerprint=self.identity.fingerprint,
                    started_at=started_at,
                ),
            )

        return self.store.update(self.key, _claim)

    def transport_request(
        self,
        *,
        operation_digest: str,
    ) -> ConversationTransportRequest:
        """Build the provider request only for the exact durable begin marker."""

        _require_digest(operation_digest, field="operation_digest")
        handle = self.load()
        if handle is None:
            raise ConversationReuseRejected(
                "conversation must be resumed before transport"
            )
        pending = handle.pending_operation
        if pending is None or pending.operation_digest != operation_digest:
            raise ConversationOperationConflict(
                "transport requires the exact pending operation"
            )
        if pending.state != "pending":
            raise ConversationOperationRecoveryRequired(handle, pending)
        reuse = (
            self.mode != "none"
            and handle.state == "active"
            and handle.opaque_id is not None
        )
        return ConversationTransportRequest(
            mode=self.mode,
            action="reuse" if reuse else "fresh",
            opaque_id=handle.opaque_id if reuse else None,
            identity_fingerprint=self.identity.fingerprint,
            operation_digest=operation_digest,
        )

    def commit_operation(
        self,
        *,
        operation_digest: str,
        opaque_id: str | None = None,
        turns: int = 0,
        context_bytes: int = 0,
        usage: Usage = Usage(),
    ) -> ConversationHandle:
        """Atomically clear the begin marker and charge the result once."""

        _require_digest(operation_digest, field="operation_digest")

        def _commit(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation handle does not exist"
                )
            self._validate_current(current)
            if operation_digest in current.operation_digests:
                return current
            pending = current.pending_operation
            if pending is None or pending.operation_digest != operation_digest:
                raise ConversationOperationConflict(
                    "commit requires the exact pending operation"
                )
            if pending.state != "pending":
                raise ConversationOperationRecoveryRequired(current, pending)
            cleared = dataclasses.replace(
                current,
                pending_operation=None,
            )
            return advance_conversation_handle(
                cleared,
                self.identity,
                operation_digest=operation_digest,
                opaque_id=opaque_id,
                turns=turns,
                context_bytes=context_bytes,
                usage=usage,
            )

        handle = self.store.update(self.key, _commit)
        overages = _budget_reasons(
            handle,
            self.limits,
            exhausted=False,
        )
        if overages:
            raise ConversationBudgetExceeded(handle, overages)
        return handle

    def mark_pending_indeterminate(
        self,
        *,
        operation_digest: str,
    ) -> ConversationHandle:
        """Record that provider completion cannot currently be proven."""

        _require_digest(operation_digest, field="operation_digest")

        def _mark(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation handle does not exist"
                )
            self._validate_current(current)
            pending = current.pending_operation
            if pending is None or pending.operation_digest != operation_digest:
                raise ConversationOperationConflict(
                    "indeterminate marker requires the exact pending operation"
                )
            if pending.state == "indeterminate":
                return current
            return dataclasses.replace(
                current,
                revision=current.revision + 1,
                pending_operation=dataclasses.replace(
                    pending,
                    state="indeterminate",
                ),
            )

        return self.store.update(self.key, _mark)

    def resolve_pending_not_started(
        self,
        *,
        operation_digest: str,
        resolution_evidence_digest: str,
        resolved_at: str,
        actor: str,
    ) -> ConversationHandle:
        """Record external not-started proof, then clear the exact marker."""

        _require_digest(operation_digest, field="operation_digest")
        _require_digest(
            resolution_evidence_digest,
            field="resolution_evidence_digest",
        )
        resolution = ConversationRecoveryResolution(
            operation_digest=operation_digest,
            identity_fingerprint=self.identity.fingerprint,
            evidence_digest=resolution_evidence_digest,
            resolved_at=resolved_at,
            actor=actor,
        )

        def _resolve(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation handle does not exist"
                )
            self._validate_current(current)
            pending = current.pending_operation
            if pending is None or pending.operation_digest != operation_digest:
                raise ConversationOperationConflict(
                    "resolution requires the exact pending operation"
                )
            prior = next(
                (
                    item
                    for item in current.recovery_resolutions
                    if item.operation_digest == operation_digest
                ),
                None,
            )
            if prior is not None:
                qualifier = "replayed" if prior == resolution else "conflicting"
                raise ConversationOperationConflict(
                    f"{qualifier} recovery resolution for operation "
                    f"{operation_digest}"
                )
            return dataclasses.replace(
                current,
                revision=current.revision + 1,
                pending_operation=None,
                recovery_resolutions=(
                    *current.recovery_resolutions,
                    resolution,
                ),
            )

        return self.store.update(self.key, _resolve)

    def charge(
        self,
        *,
        operation_digest: str,
        opaque_id: str | None = None,
        turns: int = 0,
        context_bytes: int = 0,
        usage: Usage = Usage(),
    ) -> ConversationHandle:
        """Persist one physical-result charge exactly once.

        The updated ledger is committed before an over-budget exception is
        raised.  A provider response can therefore never disappear merely
        because it crossed a post-response ceiling, and a retry of the same
        operation digest cannot double-charge it.
        """

        _require_digest(operation_digest, field="operation_digest")

        def _advance(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation must be resumed before it can be charged"
                )
            self._validate_current(current)
            pending = current.pending_operation
            if pending is not None:
                if pending.operation_digest != operation_digest:
                    raise ConversationOperationRecoveryRequired(
                        current,
                        pending,
                    )
                if pending.state != "pending":
                    raise ConversationOperationRecoveryRequired(
                        current,
                        pending,
                    )
                current = dataclasses.replace(
                    current,
                    pending_operation=None,
                )
            return advance_conversation_handle(
                current,
                self.identity,
                operation_digest=operation_digest,
                opaque_id=opaque_id,
                turns=turns,
                context_bytes=context_bytes,
                usage=usage,
            )

        handle = self.store.update(self.key, _advance)
        overages = _budget_reasons(
            handle,
            self.limits,
            exhausted=False,
        )
        if overages:
            raise ConversationBudgetExceeded(handle, overages)
        return handle

    def mark_stale(self) -> ConversationHandle:
        return self._mark_state("stale")

    def close(self) -> ConversationHandle:
        return self._mark_state("closed")

    def _mark_state(self, state: str) -> ConversationHandle:
        def _mark(
            current: ConversationHandle | None,
        ) -> ConversationHandle:
            if current is None:
                raise ConversationReuseRejected(
                    "conversation handle does not exist"
                )
            self._validate_current(current)
            return mark_conversation_state(current, state=state)

        return self.store.update(self.key, _mark)


__all__ = [
    "CONVERSATION_OPERATION_SCHEMA_VERSION",
    "CONVERSATION_STORE_SCHEMA_VERSION",
    "ConversationBudgetExceeded",
    "ConversationContinuityUnsupported",
    "ConversationController",
    "ConversationHandleCodec",
    "ConversationHandleStore",
    "ConversationLimits",
    "ConversationOperationConflict",
    "ConversationOperationRecoveryRequired",
    "ConversationReuseRejected",
    "ConversationStateError",
    "ConversationStoreError",
    "ConversationStoreKey",
    "RemainingConversationBudget",
    "build_conversation_identity",
    "conversation_operation_digest",
    "conversation_value_digest",
    "remaining_conversation_budget",
]
