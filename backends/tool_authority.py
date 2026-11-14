# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Canonical, secret-free identity for one attached tool authority.

The authority granted to an agent is more than a list of tool names.  It also
includes the exact schemas the model sees, each tool's read/write
classification, the invocation-local path scopes carried by its
``ToolContext``, the approval semantics, and the bridge contract interpreting
those fields.  This module binds those inputs into a deterministic digest
without retaining connection details, credentials, handler identities, or
absolute workspace paths.

The helper deliberately consumes only the public descriptor/classification
surface plus ``BuiltToolSet``/``CompositeToolSet`` contexts.  In particular it
does *not* inspect ``mcp_config()``, callable handlers, ``workspace_root``,
``repo_root``, endpoint URLs, or authentication material.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .capabilities import normalize_capability, validate_toolset_capability


TOOL_AUTHORITY_SCHEMA_VERSION = "crucible.tool-authority.v1"
TOOL_AUTHORITY_BRIDGE_PROTOCOL = "crucible.tool-authority"
TOOL_AUTHORITY_BRIDGE_VERSION = 1
APPROVAL_POLICY_PER_CALL_EXACT = "per_call_exact"

_DESCRIPTOR_KEYS = frozenset({"name", "description", "input_schema"})
_SCOPE_FIELDS = (
    "write_scope",
    "write_exact_scope",
    "write_deny_scope",
)
_IDENTITY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+~-]*\Z")
_ScopeBinding = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]


@dataclass(frozen=True)
class ToolAuthoritySnapshot:
    """Immutable, secret-free snapshot of one exposed tool authority.

    The canonical descriptors are retained as JSON rather than mutable nested
    dictionaries.  Kimi uses the same snapshot both for the authority digest
    and for construction of its isolated mutation specification, so a mutable
    ``ToolContext`` cannot give those two consumers different path scopes.
    """

    descriptors_json: str
    read_only_names: frozenset[str]
    write_names: frozenset[str]
    scope_bindings: tuple[tuple[str, _ScopeBinding], ...]

    @property
    def tool_names(self) -> frozenset[str]:
        return self.read_only_names | self.write_names

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        value = json.loads(self.descriptors_json)
        if not isinstance(value, list):  # pragma: no cover - constructor invariant
            raise AssertionError("tool authority snapshot descriptors are not a list")
        return value

    def binding_for(
        self,
        name: str,
    ) -> _ScopeBinding:
        for bound_name, scopes in self.scope_bindings:
            if bound_name == name:
                return scopes
        raise KeyError(name)


def _require_identity(value: Any, *, field: str) -> str:
    """Return one canonical non-secret protocol/version identity."""

    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field} must be a non-empty string or integer")
    rendered = str(value)
    if not rendered:
        raise ValueError(f"{field} must be a non-empty string or integer")
    if ":" in rendered or "\x00" in rendered:
        raise ValueError(f"{field} must be an identity, not a URL or host/port")
    if _IDENTITY_RE.fullmatch(rendered) is None:
        raise ValueError(f"{field} contains invalid identity characters")
    return rendered


def _canonical_json_value(value: Any, *, field: str) -> Any:
    """Validate and detach one JSON value without stringifying objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            out[key] = _canonical_json_value(
                child,
                field=f"{field}.{key}",
            )
        return out
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(child, field=f"{field}[{index}]")
            for index, child in enumerate(value)
        ]
    raise ValueError(
        f"{field} contains non-JSON value {type(value).__name__}"
    )


def _canonical_descriptors(toolset: Any) -> list[dict[str, Any]]:
    """Validate and canonically order the Anthropic tool descriptors."""

    renderer = getattr(toolset, "to_anthropic_tools", None)
    if not callable(renderer):
        raise ValueError("toolset must provide to_anthropic_tools()")
    raw_descriptors = renderer()
    if not isinstance(raw_descriptors, (list, tuple)):
        raise ValueError("to_anthropic_tools() must return a list or tuple")

    descriptors: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_descriptors):
        if not isinstance(raw, Mapping):
            raise ValueError(f"tool descriptor {index} must be a mapping")
        if any(not isinstance(key, str) for key in raw):
            raise ValueError(
                f"tool descriptor {index} contains a non-string field name"
            )
        keys = set(raw)
        if keys != _DESCRIPTOR_KEYS:
            missing = sorted(_DESCRIPTOR_KEYS - keys)
            unknown = sorted(keys - _DESCRIPTOR_KEYS)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if unknown:
                details.append(f"unknown={unknown}")
            raise ValueError(
                f"tool descriptor {index} has invalid fields: "
                + ", ".join(details)
            )

        name = raw["name"]
        description = raw["description"]
        schema = raw["input_schema"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"tool descriptor {index} has invalid name")
        if "\x00" in name:
            raise ValueError(f"tool descriptor {index} name contains NUL")
        if name in names:
            raise ValueError(f"duplicate tool descriptor name {name!r}")
        names.add(name)
        if not isinstance(description, str):
            raise ValueError(
                f"tool descriptor {name!r} description must be a string"
            )
        if not isinstance(schema, Mapping):
            raise ValueError(
                f"tool descriptor {name!r} input_schema must be a mapping"
            )
        descriptors.append(
            {
                "name": name,
                "description": description,
                "input_schema": _canonical_json_value(
                    schema,
                    field=f"tool descriptor {name!r} input_schema",
                ),
            }
        )
    descriptors.sort(key=lambda item: item["name"])
    return descriptors


def _metadata_names(toolset: Any, *, field: str) -> set[str]:
    """Return explicitly classified names, rejecting lossy metadata."""

    raw = getattr(toolset, field, None)
    if raw is None:
        raise ValueError(f"toolset has no {field} capability metadata")
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Iterable):
        raise ValueError(f"toolset {field} metadata must be an iterable of names")

    out: set[str] = set()
    for name in raw:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"toolset {field} metadata contains an invalid name"
            )
        if name in out:
            raise ValueError(
                f"toolset {field} metadata contains duplicate {name!r}"
            )
        out.add(name)
    return out


def _classifications(
    toolset: Any,
    descriptor_names: set[str],
) -> tuple[set[str], set[str]]:
    read_only = _metadata_names(toolset, field="read_only_names")
    write = _metadata_names(toolset, field="write_names")
    overlap = read_only & write
    if overlap:
        raise ValueError(
            "tool capability metadata classifies tools as both read-only "
            f"and mutating: {sorted(overlap)}"
        )
    classified = read_only | write
    if classified != descriptor_names:
        missing = descriptor_names - classified
        unknown = classified - descriptor_names
        details: list[str] = []
        if missing:
            details.append(f"unclassified={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ValueError(
            "tool capability metadata does not match descriptors: "
            + ", ".join(details)
        )
    return read_only, write


def _canonical_relative_scope(value: Any, *, field: str) -> str:
    """Canonicalize one repository-relative scope and reject escape aliases."""

    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} entries must be non-empty relative paths")
    if "://" in value:
        raise ValueError(f"{field} entries must not be URLs")

    rendered = value.replace("\\", "/")
    # PurePosixPath intentionally does not recognize Windows drive roots.
    if (
        len(rendered) >= 3
        and rendered[0].isalpha()
        and rendered[1:3] == ":/"
    ):
        raise ValueError(f"{field} entries must be repository-relative")
    candidate = PurePosixPath(rendered)
    if candidate.is_absolute() or candidate.as_posix() in {"", "."}:
        raise ValueError(f"{field} entries must be repository-relative")
    if ".." in candidate.parts:
        raise ValueError(f"{field} entries must not contain traversal")
    return candidate.as_posix()


def _context_scopes(context: Any) -> _ScopeBinding:
    """Return the three canonical scope sets without retaining context roots."""

    if context is None:
        return ((), (), ())
    out: list[tuple[str, ...]] = []
    for field in _SCOPE_FIELDS:
        raw = getattr(context, field, ())
        if isinstance(raw, (str, bytes, Mapping)) or not isinstance(
            raw,
            Iterable,
        ):
            raise ValueError(f"ToolContext {field} must be an iterable of paths")
        normalized: set[str] = set()
        for value in raw:
            path = _canonical_relative_scope(value, field=field)
            if path in normalized:
                raise ValueError(
                    f"ToolContext {field} contains duplicate scope {path!r}"
                )
            normalized.add(path)
        out.append(tuple(sorted(normalized)))
    return (out[0], out[1], out[2])


def _scope_bindings(
    toolset: Any,
    *,
    known_names: set[str] | None = None,
) -> dict[str, _ScopeBinding]:
    """Bind every descriptor name to its effective leaf ToolContext scopes."""

    if isinstance(toolset, ToolAuthoritySnapshot):
        snapshot_bindings: dict[str, _ScopeBinding] = dict(
            toolset.scope_bindings
        )
        if known_names is None:
            return snapshot_bindings
        return {
            name: snapshot_bindings[name]
            for name in known_names
            if name in snapshot_bindings
        }

    # Import lazily so this leaf utility does not expand the backend package's
    # import graph unless an authority digest is actually requested.
    from crucible.tools import BuiltToolSet, CompositeToolSet  # noqa: PLC0415
    from crucible.tools._dispatch_budget import (  # noqa: PLC0415
        ProgressBoundedMutatingToolSet,
    )

    if isinstance(toolset, ProgressBoundedMutatingToolSet):
        return _scope_bindings(
            toolset.isolated_mutation_inner_toolset(),
            known_names=known_names,
        )

    if isinstance(toolset, CompositeToolSet):
        bindings: dict[str, _ScopeBinding] = {}
        for member in toolset.members:
            raw_member_names = getattr(member, "tool_names", None)
            if raw_member_names is None:
                member_names = {
                    descriptor["name"]
                    for descriptor in _canonical_descriptors(member)
                }
            else:
                member_names = _metadata_names(member, field="tool_names")
            for name, scopes in _scope_bindings(
                member,
                known_names=member_names,
            ).items():
                if name in bindings:
                    raise ValueError(
                        f"duplicate tool descriptor name {name!r} "
                        "across composite members"
                    )
                bindings[name] = scopes
        return bindings

    if known_names is None:
        known_names = {
            descriptor["name"] for descriptor in _canonical_descriptors(toolset)
        }
    context = toolset.context if isinstance(toolset, BuiltToolSet) else None
    scopes = _context_scopes(context)
    return {name: scopes for name in known_names}


def snapshot_tool_authority(toolset: Any) -> ToolAuthoritySnapshot:
    """Detach descriptors, classifications, and path scopes in one read.

    ``_canonical_descriptors`` renders the exposed catalogue once.  Scope
    traversal then uses leaf ``tool_names`` metadata rather than rendering the
    catalogue again.  The result is suitable for both hashing and construction
    of a confined transport that must enforce exactly the hashed authority.
    """

    descriptors = _canonical_descriptors(toolset)
    descriptor_names = {item["name"] for item in descriptors}
    read_only, write = _classifications(toolset, descriptor_names)
    bindings = _scope_bindings(toolset, known_names=descriptor_names)
    if set(bindings) != descriptor_names:
        missing = descriptor_names - set(bindings)
        unknown = set(bindings) - descriptor_names
        details: list[str] = []
        if missing:
            details.append(f"unbound={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ValueError(
            "tool context bindings do not match descriptors: "
            + ", ".join(details)
        )
    return ToolAuthoritySnapshot(
        descriptors_json=json.dumps(
            descriptors,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        read_only_names=frozenset(read_only),
        write_names=frozenset(write),
        scope_bindings=tuple(
            (name, bindings[name])
            for name in sorted(bindings)
        ),
    )


def canonical_tool_authority(
    toolset: Any,
    *,
    capability: str,
    approval_policy: str,
    bridge_protocol: str,
    bridge_version: str | int,
    context_budget: int | None,
) -> dict[str, Any]:
    """Return the canonical secret-free authority payload for one invocation.

    ``approval_policy`` is intentionally a closed semantic vocabulary.  The
    only currently qualified behavior, ``per_call_exact``, means that each
    approval is accepted only when its fully qualified tool name is an exact
    member of this payload's advertised set.
    """

    normalized_capability = normalize_capability(capability)
    if approval_policy != APPROVAL_POLICY_PER_CALL_EXACT:
        raise ValueError(
            "approval_policy must be the qualified semantic policy "
            f"{APPROVAL_POLICY_PER_CALL_EXACT!r}"
        )
    if context_budget is not None and (
        isinstance(context_budget, bool)
        or not isinstance(context_budget, int)
        or context_budget <= 0
    ):
        raise ValueError("context_budget must be a positive integer or None")

    descriptors = _canonical_descriptors(toolset)
    descriptor_names = {item["name"] for item in descriptors}
    read_only, write = _classifications(toolset, descriptor_names)
    validate_toolset_capability(normalized_capability, toolset)

    bindings = _scope_bindings(
        toolset,
        known_names=descriptor_names,
    )
    if set(bindings) != descriptor_names:
        missing = descriptor_names - set(bindings)
        unknown = set(bindings) - descriptor_names
        details: list[str] = []
        if missing:
            details.append(f"unbound={sorted(missing)}")
        if unknown:
            details.append(f"unknown={sorted(unknown)}")
        raise ValueError(
            "tool context bindings do not match descriptors: "
            + ", ".join(details)
        )

    tools: list[dict[str, Any]] = []
    for descriptor in descriptors:
        name = descriptor["name"]
        scope, exact_scope, deny_scope = bindings[name]
        tools.append(
            {
                **descriptor,
                "classification": (
                    "write" if name in write else "read_only"
                ),
                "write_scope": list(scope),
                "write_exact_scope": list(exact_scope),
                "write_deny_scope": list(deny_scope),
            }
        )

    return {
        "schema_version": TOOL_AUTHORITY_SCHEMA_VERSION,
        "bridge": {
            "protocol": _require_identity(
                bridge_protocol,
                field="bridge_protocol",
            ),
            "version": _require_identity(
                bridge_version,
                field="bridge_version",
            ),
        },
        "capability": normalized_capability,
        "approval_policy": approval_policy,
        "context_budget": context_budget,
        "classification": {
            "read_only": sorted(read_only),
            "write": sorted(write),
        },
        "tools": tools,
    }


def tool_authority_digest(
    toolset: Any,
    *,
    capability: str,
    approval_policy: str,
    bridge_protocol: str,
    bridge_version: str | int,
    context_budget: int | None,
) -> str:
    """Return ``sha256:<hex>`` over :func:`canonical_tool_authority`."""

    payload = canonical_tool_authority(
        toolset,
        capability=capability,
        approval_policy=approval_policy,
        bridge_protocol=bridge_protocol,
        bridge_version=bridge_version,
        context_budget=context_budget,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "APPROVAL_POLICY_PER_CALL_EXACT",
    "TOOL_AUTHORITY_BRIDGE_PROTOCOL",
    "TOOL_AUTHORITY_BRIDGE_VERSION",
    "TOOL_AUTHORITY_SCHEMA_VERSION",
    "ToolAuthoritySnapshot",
    "canonical_tool_authority",
    "snapshot_tool_authority",
    "tool_authority_digest",
]
