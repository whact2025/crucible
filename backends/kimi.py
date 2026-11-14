# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Dedicated Kimi Agent SDK backend for explicit Moonshot K2.7 and K3 routes.

This adapter deliberately does not reuse :mod:`backends.openai_compat`.
``kimi-agent-sdk`` embeds a Kimi CLI runtime whose low-level ``Session`` API
owns the model turn and Wire stream.  Its dependency on ``openai<2.15`` is
incompatible with Crucible's root runtime, and the released runtime changes
process cwd, consults process environment variables, and journals session
context.  Every real invocation therefore runs in the separately locked
``backends/kimi_worker`` environment through a short-lived subprocess.

The qualified surface keeps provider-native capabilities disabled and supports
either prompt-only evaluation or an explicitly attached Crucible ``ToolSet``:

* exact model ``kimi-k2.7-code-highspeed`` or ``kimi-k3`` at
  ``https://api.moonshot.ai/v1``;
* endpoint and credential read at invocation time from ``KIMI_BASE_URL`` and
  ``MOONSHOT_API_KEY``;
* in-memory native ``kimi`` provider configuration;
* K2.7's always-on reasoning preserved as ``reasoning_content`` while the
  pinned CLI's incompatible ``reasoning_effort`` shim is disabled;
* K3's explicit ``max`` reasoning, ``max_completion_tokens``, and implicit
  five-minute caching, with disjoint cache-read/write accounting;
* a non-inheriting agent with no native tools or subagents;
* an optional capability-filtered ToolSet exposed through one authenticated,
  ephemeral loopback MCP server;
* ``yolo=False`` and exact per-call, fail-closed approval handling;
* response schemas expressed only as prompt text;
* normalized visible text and usage events, never raw ``ThinkPart`` content.

The parent never imports the Kimi SDK.  The worker verifies the exact released
SDK/CLI tuple that this implementation was reviewed against before accepting a
request.  Inference and ephemeral MCP credentials cross the process boundary
only in JSON on stdin; they are never placed in argv, the child environment,
the MCP URL, or a persistent file.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime
import functools
import hashlib
import importlib.metadata
import json
import logging
import math
import multiprocessing
import os
import re
import secrets
import selectors
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import types
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ContentRejected,
    ExecutionCompleteness,
    ExternalCancellation,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
    UsageCostBounds,
    UsageCostBoundSegment,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from ._classify import RETRIABLE_HTTP_STATUSES
from ._physical_telemetry import (
    PhysicalUsageEvidence,
    PhysicalUsageResponse,
    capture_physical_usage,
    emit_physical_usage_attempt,
)
from ._sdk_debug import SdkDebugSink
from .pricing import compute_cost

logger = logging.getLogger(__name__)

_MODEL = "kimi-k2.7-code-highspeed"
_K3_MODEL = "kimi-k3"
_MODEL_CONTEXT_SIZES = {_MODEL: 262_144, _K3_MODEL: 1_048_576}
_BASE_URL = "https://api.moonshot.ai/v1"
_SDK_VERSION = "0.0.5"
_CLI_VERSION = "1.12.0"
_KOSONG_VERSION = "0.42.0"
_OPENAI_VERSION = "2.14.0"
_PROVIDER_NAME = "crucible-moonshot"
_MAX_CONTEXT_SIZE = 262_144
_DEFAULT_TIMEOUT_SECONDS = 10 * 60.0
_WORKER_DRAIN_SECONDS = 2.0
_PROCESS_STOP_SECONDS = 2.0
_WORKER_PROTOCOL = "crucible.kimi-worker.v6"
_K3_BOUND_POLICY = "kimi-k3-cache-read-upper-bound-v1"
_WORKER_PROJECT = Path(__file__).with_name("kimi_worker")
_DEFAULT_WORKER_PYTHON = _WORKER_PROJECT / ".venv" / "bin" / "python"
KIMI_PROJECT_LOCKED_WORKER_SOURCE = "project_locked"
_WORKER_SOURCES = frozenset({
    "environment",
    KIMI_PROJECT_LOCKED_WORKER_SOURCE,
})
_MAX_WORKER_LINE_BYTES = 1_048_576
_MAX_WORKER_STDERR_BYTES = 65_536
_WORKER_PROBE_TIMEOUT_SECONDS = 15.0
_MAX_TOOL_STEPS = 100
_MAX_TOOL_EVENTS = 2_048
_MAX_AUTH_TOKEN_CHARS = 4_096
_MAX_USAGE_MESSAGE_ID_CHARS = 256
_DEFAULT_MAX_RESPONSE_TOKENS = 32_768
_TOOL_DISPATCH_BUDGET_ERROR = "ToolDispatchBudgetExceeded"
_TOOL_MUTATION_INDETERMINATE_ERROR = "ToolMutationIndeterminate"
_MCP_TOKEN_BYTES = 48
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_TOOL_AUTHORITY_PROTOCOL = "crucible.tool-authority"
_TOOL_AUTHORITY_VERSION = 1
_MCP_TOOL_TIMEOUT_MS = 1_800_000
_INVALID_TOOL_TRACE_ERROR = (
    "Kimi worker emitted an invalid per-attempt tool event sequence"
)
KIMI_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA = (
    "crucible.kimi-project-locked-runtime-identity.v1"
)


@dataclass(frozen=True, slots=True)
class KimiWorkerPackageVersions:
    """Exact credential-free package tuple reported by the worker probe."""

    kimi_agent_sdk: str
    kimi_cli: str
    kosong: str
    openai: str

    def __post_init__(self) -> None:
        if (
            self.kimi_agent_sdk,
            self.kimi_cli,
            self.kosong,
            self.openai,
        ) != (
            _SDK_VERSION,
            _CLI_VERSION,
            _KOSONG_VERSION,
            _OPENAI_VERSION,
        ):
            raise ValueError("Kimi worker package tuple is not exact")

    def to_mapping(self) -> dict[str, str]:
        return {
            "kimi-agent-sdk": self.kimi_agent_sdk,
            "kimi-cli": self.kimi_cli,
            "kosong": self.kosong,
            "openai": self.openai,
        }


@dataclass(frozen=True, slots=True)
class KimiProjectLockedRuntimeIdentity:
    """Closed manifest projection for the project-locked Kimi worker."""

    logical_worker_python: str
    resolved_worker_python: str
    resolved_worker_python_sha256: str
    lock_source_installed_tree_fingerprint: str
    runtime_fingerprint: str
    package_versions: KimiWorkerPackageVersions

    def __post_init__(self) -> None:
        if (
            not Path(self.logical_worker_python).is_absolute()
            or not Path(self.resolved_worker_python).is_absolute()
        ):
            raise ValueError("Kimi worker identity paths must be absolute")
        for label, value in (
            ("resolved worker SHA-256", self.resolved_worker_python_sha256),
            (
                "lock/source/installed-tree fingerprint",
                self.lock_source_installed_tree_fingerprint,
            ),
            ("runtime fingerprint", self.runtime_fingerprint),
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
                raise ValueError(f"Kimi {label} is malformed")

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema_version": KIMI_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
            "worker_source": KIMI_PROJECT_LOCKED_WORKER_SOURCE,
            "worker_protocol": _WORKER_PROTOCOL,
            "logical_worker_python": self.logical_worker_python,
            "resolved_worker_python": self.resolved_worker_python,
            "resolved_worker_python_sha256": (
                self.resolved_worker_python_sha256
            ),
            "lock_source_installed_tree_fingerprint": (
                self.lock_source_installed_tree_fingerprint
            ),
            "runtime_fingerprint": self.runtime_fingerprint,
            "package_versions": self.package_versions.to_mapping(),
        }


def _hash_file_into(digest: Any, path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise KimiSDKQualificationError(
            f"locked Kimi worker file is unavailable: {label}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise KimiSDKQualificationError(
            f"locked Kimi worker file is not regular: {label}"
        )
    digest.update(b"F\0")
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
    digest.update(b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")


def _hash_tree_into(digest: Any, root: Path, *, label: str) -> None:
    if not root.is_dir():
        raise KimiSDKQualificationError(
            f"locked Kimi worker tree is unavailable: {label}"
        )
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        entry_label = f"{label}/{relative}"
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(b"D\0")
            digest.update(entry_label.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
            digest.update(b"\0")
        elif stat.S_ISREG(metadata.st_mode):
            _hash_file_into(digest, path, label=entry_label)
        else:
            raise KimiSDKQualificationError(
                "locked Kimi worker tree contains an unsupported entry: "
                f"{entry_label}"
            )


def _worker_project_for_interpreter(path: Path) -> Path:
    """Return the project that owns one locked ``.venv/bin/python`` path."""

    candidate = Path(os.path.abspath(path))
    if (
        candidate.name != "python"
        or candidate.parent.name != "bin"
        or candidate.parent.parent.name != ".venv"
    ):
        raise KimiSDKQualificationError(
            "locked Kimi worker interpreter must be <project>/.venv/bin/python"
        )
    return candidate.parent.parent.parent


def _worker_runtime_fingerprint(worker_project: Path | None = None) -> str:
    """Hash the lock, worker source, venv metadata, and installed runtime."""

    project = _WORKER_PROJECT if worker_project is None else worker_project
    digest = hashlib.sha256()
    for relative in (
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        ".venv/pyvenv.cfg",
    ):
        _hash_file_into(
            digest,
            project / relative,
            label=relative,
        )
    _hash_tree_into(
        digest,
        project / "src" / "crucible_kimi_worker",
        label="src/crucible_kimi_worker",
    )
    site_packages = tuple(
        (project / ".venv" / "lib").glob(
            "python*/site-packages"
        )
    )
    if len(site_packages) != 1:
        raise KimiSDKQualificationError(
            "locked Kimi worker must contain exactly one site-packages tree"
        )
    _hash_tree_into(
        digest,
        site_packages[0],
        label="site-packages",
    )
    return "sha256:" + digest.hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _installed_worker_identity_fingerprint(
    *,
    invocation_path: Path,
    invocation_metadata: os.stat_result,
    invocation_link_target: str | None,
    resolved_path: Path,
    resolved_metadata: os.stat_result,
    resolved_sha256: str,
    installed_tree_fingerprint: str,
) -> str:
    """Bind one host-specific interpreter and installed worker tree."""

    payload = {
        "invocation_path": str(invocation_path),
        "invocation_mode": invocation_metadata.st_mode,
        "invocation_link_target": invocation_link_target,
        "resolved_path": str(resolved_path),
        "resolved_mode": resolved_metadata.st_mode,
        "resolved_sha256": resolved_sha256,
        "installed_tree_fingerprint": installed_tree_fingerprint,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class _PinnedKimiWorkerRuntime:
    """Executable and installed-tree identity for frozen Moonshot Arm C."""

    invocation_path: str
    invocation_device: int
    invocation_inode: int
    invocation_mode: int
    invocation_size: int
    invocation_mtime_ns: int
    invocation_ctime_ns: int
    invocation_link_target: str | None
    resolved_path: str
    resolved_device: int
    resolved_inode: int
    resolved_mode: int
    resolved_size: int
    resolved_mtime_ns: int
    resolved_ctime_ns: int
    resolved_sha256: str
    installed_tree_fingerprint: str
    runtime_fingerprint: str

    @classmethod
    def capture(cls, path: Path) -> _PinnedKimiWorkerRuntime:
        candidate = Path(os.path.abspath(path))
        try:
            invocation = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            executable = resolved.stat()
        except OSError as exc:
            raise KimiSDKUnavailable(
                "locked Kimi worker interpreter is unavailable"
            ) from exc
        if (
            not stat.S_ISREG(executable.st_mode)
            or not os.access(candidate, os.X_OK)
        ):
            raise KimiSDKUnavailable(
                "locked Kimi worker interpreter is not executable"
            )
        link_target = (
            os.readlink(candidate)
            if stat.S_ISLNK(invocation.st_mode)
            else None
        )
        resolved_sha256 = _file_sha256(resolved)
        worker_project = _worker_project_for_interpreter(candidate)
        installed_tree_fingerprint = _worker_runtime_fingerprint(worker_project)
        return cls(
            invocation_path=str(candidate),
            invocation_device=invocation.st_dev,
            invocation_inode=invocation.st_ino,
            invocation_mode=invocation.st_mode,
            invocation_size=invocation.st_size,
            invocation_mtime_ns=invocation.st_mtime_ns,
            invocation_ctime_ns=invocation.st_ctime_ns,
            invocation_link_target=link_target,
            resolved_path=str(resolved),
            resolved_device=executable.st_dev,
            resolved_inode=executable.st_ino,
            resolved_mode=executable.st_mode,
            resolved_size=executable.st_size,
            resolved_mtime_ns=executable.st_mtime_ns,
            resolved_ctime_ns=executable.st_ctime_ns,
            resolved_sha256=resolved_sha256,
            installed_tree_fingerprint=installed_tree_fingerprint,
            runtime_fingerprint=_installed_worker_identity_fingerprint(
                invocation_path=candidate,
                invocation_metadata=invocation,
                invocation_link_target=link_target,
                resolved_path=resolved,
                resolved_metadata=executable,
                resolved_sha256=resolved_sha256,
                installed_tree_fingerprint=installed_tree_fingerprint,
            ),
        )

    def revalidate(self) -> None:
        try:
            candidate = Path(self.invocation_path)
            invocation = candidate.lstat()
            resolved = candidate.resolve(strict=True)
            executable = resolved.stat()
            link_target = (
                os.readlink(candidate)
                if stat.S_ISLNK(invocation.st_mode)
                else None
            )
        except OSError as exc:
            raise KimiSDKQualificationError(
                "locked Kimi worker identity disappeared before transport"
            ) from exc
        observed_invocation = (
            invocation.st_dev,
            invocation.st_ino,
            invocation.st_mode,
            invocation.st_size,
            invocation.st_mtime_ns,
            invocation.st_ctime_ns,
            link_target,
        )
        expected_invocation = (
            self.invocation_device,
            self.invocation_inode,
            self.invocation_mode,
            self.invocation_size,
            self.invocation_mtime_ns,
            self.invocation_ctime_ns,
            self.invocation_link_target,
        )
        observed_resolved = (
            str(resolved),
            executable.st_dev,
            executable.st_ino,
            executable.st_mode,
            executable.st_size,
            executable.st_mtime_ns,
            executable.st_ctime_ns,
        )
        expected_resolved = (
            self.resolved_path,
            self.resolved_device,
            self.resolved_inode,
            self.resolved_mode,
            self.resolved_size,
            self.resolved_mtime_ns,
            self.resolved_ctime_ns,
        )
        observed_sha256 = _file_sha256(resolved)
        worker_project = _worker_project_for_interpreter(candidate)
        observed_installed_tree_fingerprint = _worker_runtime_fingerprint(
            worker_project
        )
        observed_runtime_fingerprint = (
            _installed_worker_identity_fingerprint(
                invocation_path=candidate,
                invocation_metadata=invocation,
                invocation_link_target=link_target,
                resolved_path=resolved,
                resolved_metadata=executable,
                resolved_sha256=observed_sha256,
                installed_tree_fingerprint=(
                    observed_installed_tree_fingerprint
                ),
            )
        )
        if (
            observed_invocation != expected_invocation
            or observed_resolved != expected_resolved
            or not stat.S_ISREG(executable.st_mode)
            or not os.access(candidate, os.X_OK)
            or observed_sha256 != self.resolved_sha256
            or observed_installed_tree_fingerprint
            != self.installed_tree_fingerprint
            or observed_runtime_fingerprint != self.runtime_fingerprint
        ):
            raise KimiSDKQualificationError(
                "locked Kimi worker identity drifted before transport"
            )
_WORKER_TERMINAL_FIELDS = frozenset(
    {
        "type",
        "final_message",
        "input_tokens",
        "output_tokens",
        "cache_creation_tokens",
        "cache_read_tokens",
        "usage_seen",
        "usage_valid",
        "terminal_seen",
        "configured_model",
        "served_model",
        "stop_reason",
        "error_kind",
        "error_message",
        "http_status",
        "retriable",
        "steps_used",
        "cost_bound_segments",
        "tool_events",
    }
)
_WORKER_ERROR_KINDS = frozenset(
    {
        "unavailable",
        "qualification",
        "cancellation",
        "max_turns",
        "transport",
        "content_rejected",
        "worker_internal",
        "max_tokens",
        "tool_budget",
        "tool_mutation",
        "tool_terminal",
        "cost_budget",
    }
)
_THINK_MARKER_LIKE_RE = re.compile(
    r"<\s*/?\s*think(?:\s*/?\s*>|(?=\s|$))",
    re.IGNORECASE,
)

_DEFAULT_POLICY = BackendPolicy(
    retry_budget=3,
    timeout_seconds=None,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|"
    r"passwd|secret|token)\b(\s*[:=]\s*)([^\s,;\]\}\"']+|"
    r"\"[^\"]*\"|'[^']*')"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")

_ToolEvent = tuple[Literal["tool_call", "tool_result"], str, str]


class KimiSDKUnavailable(RuntimeError):
    """The optional reviewed Kimi Agent SDK distribution is not installed."""


class KimiSDKQualificationError(RuntimeError):
    """The installed SDK/runtime or observed session violates qualification."""


class KimiWorkerLifecycleError(KimiSDKQualificationError):
    """A credential-bearing Kimi worker lease could not be settled."""


class KimiToolDispatchBudgetTerminal(RuntimeError):
    """The workflow-owned source budget terminated the SDK tool loop."""


class KimiToolMutationIndeterminateTerminal(RuntimeError):
    """An indeterminate mutation terminally stopped the SDK tool loop."""


class KimiToolDispatchTerminal(RuntimeError):
    """A workflow-owned terminal dispatch stopped the SDK tool loop."""


def _extract_final_text(text: str) -> str:
    """Return K2.7 visible text without guessing at raw reasoning envelopes.

    Moonshot K2.7 streams private reasoning in ``reasoning_content``.  The
    qualified provider converts that field to SDK ``ThinkPart`` objects, which
    are never copied into the final message.  A reasoning marker arriving in
    visible ``TextPart`` content is therefore an unqualified wire shape and
    fails closed instead of reusing the NVIDIA K2.6 envelope heuristic.
    """

    if _THINK_MARKER_LIKE_RE.search(text):
        raise KimiSDKQualificationError(
            "Kimi emitted a reasoning marker in visible content"
        )
    return text


@dataclass(frozen=True)
class _SdkBindings:
    Session: Any
    Config: Any
    KaosPath: Any
    ApprovalRequest: Any
    StatusUpdate: Any
    TextPart: Any
    ThinkPart: Any
    ToolCall: Any
    ToolResult: Any
    StepBegin: Any
    TurnEnd: Any
    StepInterrupted: Any
    RunCancelled: Any
    MaxStepsReached: Any
    APIStatusError: Any
    APIConnectionError: Any
    APITimeoutError: Any
    APIEmptyResponseError: Any
    ChatProviderError: Any


@dataclass(frozen=True)
class _WorkerRequest:
    system: str = field(repr=False)
    user: str = field(repr=False)
    model: str = _MODEL
    reasoning_effort: str | None = None
    diagnostic_cost_upper_bound: bool = False
    diagnostic_budget_usd: float | None = None
    schema_json: str | None = field(default=None, repr=False)
    base_url: str = ""
    auth_token: str = field(default="", repr=False)
    max_steps: int = 1
    max_response_tokens: int = _DEFAULT_MAX_RESPONSE_TOKENS
    mcp_url: str | None = None
    mcp_auth_token: str | None = field(default=None, repr=False)
    mcp_tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _WorkerPaths:
    root: Path
    home: Path
    share: Path
    work: Path
    skills: Path
    config: Path
    scratch: Path


@dataclass(frozen=True)
class _WorkerOutcome:
    final_message: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    usage_seen: bool = False
    usage_valid: bool = False
    terminal_seen: bool = False
    configured_model: str | None = None
    served_model: str | None = None
    stop_reason: str = "error"
    error_kind: str | None = None
    error_message: str = ""
    http_status: int | None = None
    retriable: bool = False
    tool_events: tuple[_ToolEvent, ...] = ()
    steps_used: int = 0
    cost_bound_segments: tuple[dict[str, Any], ...] = ()
    # Parent-only evidence derived from protocol ordering.  The locked worker
    # emits exactly one cumulative usage record after each provider step has
    # returned and before it can begin the next step.  Tracking how many begun
    # steps reached that checkpoint lets a nonterminal outcome retain a
    # truthful lower bound without claiming that an in-flight provider step
    # was accounted.  This field is deliberately absent from the v5 wire
    # envelope; the parent derives it rather than trusting the child to assert
    # settlement.
    settled_usage_steps: int = 0
    # Parent-only ordered pricing units derived by subtracting each admitted
    # cumulative usage checkpoint from its predecessor. The worker never
    # asserts this ledger and the field is deliberately absent from the wire
    # envelope.
    pricing_segments: tuple[UsagePricingSegment, ...] = ()
    # Parent-only raw-source projection, appended at the same admitted worker
    # usage checkpoint as ``pricing_segments``.  Unlike Assay's neutral
    # segment type, this preserves the exact source and field semantics needed
    # by a standalone frontier canary.
    physical_usage_responses: tuple[PhysicalUsageResponse, ...] = ()
    # Parent-only evidence. The SDK worker cannot attest execution inside the
    # separately isolated mutation host, so this field is intentionally absent
    # from the worker wire envelope and is bound after each physical attempt.
    host_mutation_trace_complete: bool = True

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "terminal",
            "final_message": self.final_message,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "usage_seen": self.usage_seen,
            "usage_valid": self.usage_valid,
            "terminal_seen": self.terminal_seen,
            "configured_model": self.configured_model,
            "served_model": self.served_model,
            "stop_reason": self.stop_reason,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "http_status": self.http_status,
            "retriable": self.retriable,
            "steps_used": self.steps_used,
            "cost_bound_segments": list(self.cost_bound_segments),
            "tool_events": [
                {"kind": kind, "name": name, "call_id": call_id}
                for kind, name, call_id in self.tool_events
            ],
        }

    @classmethod
    def from_wire(
        cls, payload: Mapping[str, Any], *, model: str = _MODEL,
        diagnostic_cost_upper_bound: bool = False,
    ) -> _WorkerOutcome:
        expected_fields = _WORKER_TERMINAL_FIELDS | (
            {"protocol"} if "protocol" in payload else set()
        )
        if (
            set(payload) != expected_fields
            or payload.get("type") != "terminal"
            or (
                "protocol" in payload
                and payload.get("protocol") != _WORKER_PROTOCOL
            )
        ):
            raise ValueError("Kimi worker emitted an invalid terminal schema")
        counter_names = (
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
        )
        counters = tuple(payload.get(name) for name in counter_names)
        if not _usage_counters_valid(*counters):
            raise ValueError("Kimi worker emitted invalid usage counters")
        boolean_names = (
            "usage_seen",
            "usage_valid",
            "terminal_seen",
            "retriable",
        )
        if any(type(payload.get(name)) is not bool for name in boolean_names):
            raise ValueError("Kimi worker emitted invalid terminal booleans")
        for name in ("final_message", "stop_reason", "error_message"):
            if type(payload.get(name)) is not str:
                raise ValueError("Kimi worker emitted invalid terminal text")
        for name in ("configured_model", "served_model", "error_kind"):
            if payload.get(name) is not None and type(payload.get(name)) is not str:
                raise ValueError("Kimi worker emitted invalid terminal identity")
        http_status = payload.get("http_status")
        if http_status is not None and (
            type(http_status) is not int or not 100 <= http_status <= 599
        ):
            raise ValueError("Kimi worker emitted an invalid HTTP status")
        steps_used = payload.get("steps_used")
        if (
            type(steps_used) is not int
            or not 0 <= steps_used <= _MAX_TOOL_STEPS
            or (payload["terminal_seen"] and steps_used == 0)
        ):
            raise ValueError("Kimi worker emitted invalid logical-step evidence")
        bound_records = _parse_cost_bound_records(payload["cost_bound_segments"])
        if bound_records and (
            not diagnostic_cost_upper_bound or model != _K3_MODEL
            or len(bound_records) > steps_used
        ):
            raise ValueError("Kimi worker emitted unauthorized cost bounds")
        usage_seen = cast(bool, payload["usage_seen"])
        usage_valid = cast(bool, payload["usage_valid"])
        if (
            (usage_valid and not usage_seen)
            or (
                not usage_seen
                and (usage_valid or any(cast(int, value) != 0 for value in counters))
            )
        ):
            raise ValueError("Kimi worker emitted inconsistent usage evidence")
        terminal_seen = cast(bool, payload["terminal_seen"])
        stop_reason = cast(str, payload["stop_reason"])
        error_kind = cast(str | None, payload["error_kind"])
        error_message = cast(str, payload["error_message"])
        final_message = cast(str, payload["final_message"])
        configured_model = cast(str | None, payload["configured_model"])
        served_model = cast(str | None, payload["served_model"])
        retriable = cast(bool, payload["retriable"])
        tool_events = _parse_tool_events(payload["tool_events"])
        if error_kind is not None and error_kind not in _WORKER_ERROR_KINDS:
            raise ValueError("Kimi worker emitted an unknown error kind")
        success_valid = (
            error_kind is None
            and terminal_seen
            and stop_reason == "end_turn"
            and error_message == ""
            and http_status is None
            and not retriable
        )
        bounded_terminal_error_valid = (
            terminal_seen
            and http_status is None
            and not retriable
            and bool(error_message)
            and (
                (stop_reason, error_kind)
                in {
                    ("max_tokens", "max_tokens"),
                    ("content_filter", "content_rejected"),
                }
            )
            and (stop_reason != "content_filter" or final_message == "")
        )
        nonterminal_error_valid = (
            error_kind is not None
            and not terminal_seen
            and stop_reason == "error"
            and final_message == ""
            and configured_model is None
            and served_model is None
            and bool(error_message)
            and (http_status is None or error_kind == "transport")
            and (not retriable or error_kind in {"transport", "cancellation"})
            and error_kind != "max_tokens"
            and not usage_seen
            and not usage_valid
            and not any(cast(int, value) for value in counters)
            and not tool_events
        )
        max_steps_error_valid = (
            error_kind == "max_turns"
            and not terminal_seen
            and stop_reason == "max_turns"
            and final_message == ""
            and configured_model == model
            and served_model == model
            and bool(error_message)
            and http_status is None
            and not retriable
            and usage_seen
            and usage_valid
            and steps_used > 0
        )
        cost_budget_error_valid = (
            diagnostic_cost_upper_bound and bool(bound_records)
            and error_kind == "cost_budget" and stop_reason == "cost_budget"
            and not terminal_seen and configured_model == model
            and served_model == model and usage_seen and usage_valid
            and bool(error_message) and not retriable and http_status is None
            and final_message == ""
        )
        if not (
            success_valid
            or bounded_terminal_error_valid
            or nonterminal_error_valid
            or max_steps_error_valid
            or cost_budget_error_valid
        ):
            raise ValueError("Kimi worker emitted an invalid terminal state")
        if terminal_seen and (
            not configured_model
            or not served_model
        ):
            raise ValueError("Kimi worker omitted terminal model identity")
        return cls(
            final_message=final_message,
            input_tokens=cast(int, counters[0]),
            output_tokens=cast(int, counters[1]),
            cache_creation_tokens=cast(int, counters[2]),
            cache_read_tokens=cast(int, counters[3]),
            usage_seen=usage_seen,
            usage_valid=usage_valid,
            terminal_seen=terminal_seen,
            configured_model=configured_model,
            served_model=served_model,
            stop_reason=stop_reason,
            error_kind=error_kind,
            error_message=error_message,
            http_status=cast(int | None, http_status),
            retriable=retriable,
            tool_events=tool_events,
            steps_used=steps_used,
            cost_bound_segments=bound_records,
        )


@dataclass
class _StepLedger:
    max_steps: int
    emit: Callable[[dict[str, Any]], None]
    steps_used: int = 0

    def begin(self) -> None:
        if self.steps_used >= self.max_steps:
            raise KimiSDKQualificationError(
                "Kimi emitted more StepBegin events than its physical-attempt bound"
            )
        self.steps_used += 1
        self.emit(
            {
                "type": "step_progress",
                "steps_used": self.steps_used,
            }
        )


class _FrozenMcpToolset:
    """Immutable authority view over a live read-only dispatcher."""

    def __init__(
        self,
        toolset: Any,
        *,
        names: tuple[str, ...],
        write_names: frozenset[str],
        authority_snapshot: Any | None = None,
    ) -> None:
        from .tool_authority import snapshot_tool_authority

        snapshot = (
            snapshot_tool_authority(toolset)
            if authority_snapshot is None
            else authority_snapshot
        )
        descriptors = snapshot.to_anthropic_tools()
        frozen_names = tuple(
            sorted(str(descriptor.get("name", "")) for descriptor in descriptors)
        )
        if frozen_names != names:
            raise ValueError("Kimi ToolSet catalogue changed after attachment")
        if snapshot.write_names != write_names:
            raise ValueError("Kimi ToolSet classification changed after attachment")
        self._toolset = toolset
        self._authority_snapshot = snapshot
        self.write_names = set(snapshot.write_names)
        self.read_only_names = set(names) - self.write_names

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return self._authority_snapshot.to_anthropic_tools()

    @property
    def authority_snapshot(self) -> Any:
        return self._authority_snapshot

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        return self._toolset.dispatch(name, arguments)


class _UsageAccumulator:
    """Admit one immutable usage sample per SDK step.

    The pinned SDK may repeat a cumulative ``StatusUpdate``. Exact repeats are
    idempotent, while any ambiguity within the step makes completeness
    sticky-invalid without replacing or double-counting the first counters.
    """

    def __init__(self) -> None:
        self._by_step: dict[
            int,
            tuple[str | None, tuple[int, int, int, int]],
        ] = {}
        self.valid = True

    def observe(
        self,
        usage: Any,
        message_id: Any,
        *,
        step: int,
    ) -> None:
        raw = (
            getattr(usage, "input", None),
            getattr(usage, "output", None),
            getattr(usage, "input_cache_creation", None),
            getattr(usage, "input_cache_read", None),
        )
        sample: tuple[int, int, int, int] = (
            _usage_counter(raw[0]),
            _usage_counter(raw[1]),
            _usage_counter(raw[2]),
            _usage_counter(raw[3]),
        )
        step_valid = type(step) is int and step > 0
        message_id_valid = _usage_message_id_valid(message_id)
        self.valid = (
            self.valid
            and step_valid
            and message_id_valid
            and _usage_counters_valid(*raw)
        )
        step_key = step if type(step) is int else 0
        normalized_id = cast(str, message_id) if message_id_valid else None
        existing = self._by_step.get(step_key)
        if existing is None:
            self._by_step[step_key] = (normalized_id, sample)
        elif existing != (normalized_id, sample):
            # Retain the first numeric evidence. Replacing it or adding a
            # second sample could silently under- or over-count billed usage.
            self.valid = False

    def totals(self) -> tuple[int, int, int, int]:
        samples = [entry[1] for entry in self._by_step.values()]
        return (
            sum(sample[0] for sample in samples),
            sum(sample[1] for sample in samples),
            sum(sample[2] for sample in samples),
            sum(sample[3] for sample in samples),
        )

    @property
    def seen(self) -> bool:
        return bool(self._by_step)


def _tool_dispatch_terminal_kind(
    wire_message: Any,
) -> Literal["budget", "mutation", "other"] | None:
    """Parse the bridge's closed terminal-dispatch error envelope.

    The pinned MCP client converts ``CallToolResult(isError=True)`` into a
    ``ToolResult`` whose model-visible output contains the bridge error text.
    Successful repository content containing the same marker is not terminal.
    """

    return_value = getattr(wire_message, "return_value", None)
    if getattr(return_value, "is_error", None) is not True:
        return None
    output = getattr(return_value, "output", "")
    if isinstance(output, str):
        texts = (output,)
    elif isinstance(output, (list, tuple)):
        texts = tuple(
            text
            for part in output
            if isinstance((text := getattr(part, "text", None)), str)
        )
    else:
        return None
    for text in texts:
        prefix = "Error: TerminalToolDispatchError: "
        if text.startswith(prefix):
            error_name = text[len(prefix):].partition(":")[0]
            if error_name == _TOOL_DISPATCH_BUDGET_ERROR:
                return "budget"
            if error_name == _TOOL_MUTATION_INDETERMINATE_ERROR:
                return "mutation"
            return "other"
        if text.startswith(f"Error: {_TOOL_DISPATCH_BUDGET_ERROR}:"):
            return "budget"
        if text.startswith(f"Error: {_TOOL_MUTATION_INDETERMINATE_ERROR}:"):
            return "mutation"
    return None


def _tool_dispatch_budget_exhausted(wire_message: Any) -> bool:
    """Recognize the bridge's typed dispatch-budget denial."""

    return _tool_dispatch_terminal_kind(wire_message) == "budget"


def _tool_mutation_indeterminate(wire_message: Any) -> bool:
    """Recognize the bridge's typed indeterminate-mutation terminal."""

    return _tool_dispatch_terminal_kind(wire_message) == "mutation"


def _usage_sum(left: Usage, right: Usage) -> Usage:
    """Add usage already priced for two physical worker attempts."""

    if left == Usage():
        return right
    if right == Usage():
        return left
    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
        cache_creation_tokens=(
            left.cache_creation_tokens + right.cache_creation_tokens
        ),
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
    )


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _usage_counters_valid(
    input_tokens: Any,
    output_tokens: Any,
    cache_creation_tokens: Any,
    cache_read_tokens: Any,
) -> bool:
    return (
        all(
            _is_nonnegative_int(value)
            for value in (
                input_tokens,
                output_tokens,
                cache_creation_tokens,
                cache_read_tokens,
            )
        )
        and cache_creation_tokens + cache_read_tokens <= input_tokens
    )


def _usage_checkpoint_valid(
    latest: _WorkerOutcome,
    counters: tuple[Any, Any, Any, Any],
    usage_valid: Any,
) -> bool:
    """Validate one ordered provider-step usage checkpoint.

    The SDK may repeat an identical StatusUpdate for the current step.  Such a
    frame is idempotent; changed counters or validity at the same step are
    ambiguous and fail closed.  A new frame must settle exactly the next begun
    step, so a missing earlier usage envelope can never be hidden by a later
    cumulative total.
    """

    if latest.steps_used == 0:
        return False
    previous = (
        latest.input_tokens,
        latest.output_tokens,
        latest.cache_creation_tokens,
        latest.cache_read_tokens,
    )
    if latest.settled_usage_steps == latest.steps_used:
        return (
            latest.usage_seen
            and counters == previous
            and usage_valid is latest.usage_valid
        )
    if latest.settled_usage_steps + 1 != latest.steps_used:
        return False
    if not latest.usage_seen:
        return True
    return (
        not any(
            cast(int, current) < prior
            for current, prior in zip(counters, previous, strict=True)
        )
        # An exact repeat is idempotent only while the already-settled step
        # remains current (the branch above).  Once a new StepBegin has been
        # observed, at least one cumulative counter must advance; otherwise a
        # delayed duplicate for the preceding step could falsely settle the
        # newly begun provider call.
        and any(
            cast(int, current) > prior
            for current, prior in zip(counters, previous, strict=True)
        )
        and not (not latest.usage_valid and usage_valid is True)
    )


def _settle_usage_checkpoint(
    latest: _WorkerOutcome,
    counters: tuple[int, int, int, int],
    *,
    usage_valid: bool,
    model: str = _MODEL,
    bound_record: dict[str, Any] | None = None,
) -> _WorkerOutcome:
    """Bind one newly settled provider step to an inclusive-input segment."""

    if latest.settled_usage_steps == latest.steps_used:
        # An identical repeat for the current settled step is idempotent. The
        # caller has already checked exact counter and validity equality.
        return latest
    previous = (
        latest.input_tokens,
        latest.output_tokens,
        latest.cache_creation_tokens,
        latest.cache_read_tokens,
    )
    delta = tuple(
        current - prior
        for current, prior in zip(counters, previous, strict=True)
    )
    segment = UsagePricingSegment(
        input_tokens=delta[0],
        output_tokens=delta[1],
        cost_usd=(_cost_bound_amounts(bound_record)[0] if bound_record is not None else compute_cost(
            model,
            input_tokens=delta[0],
            output_tokens=delta[1],
            cache_creation_tokens=delta[2],
            cache_read_tokens=delta[3],
        )),
        cache_creation_tokens=delta[2],
        cache_read_tokens=delta[3],
    )
    physical_response = PhysicalUsageResponse(
        response_ordinal=len(latest.physical_usage_responses),
        input_tokens=delta[0],
        output_tokens=delta[1],
        cache_read_tokens=delta[3],
        reported_cache_write_tokens=delta[2],
        cache_write_field_state="present",
        cache_write_interpretation="worker_checkpoint",
    )
    return replace(
        latest,
        input_tokens=counters[0],
        output_tokens=counters[1],
        cache_creation_tokens=counters[2],
        cache_read_tokens=counters[3],
        usage_seen=True,
        usage_valid=usage_valid,
        settled_usage_steps=latest.steps_used,
        pricing_segments=(*latest.pricing_segments, segment),
        physical_usage_responses=(
            *latest.physical_usage_responses,
            *((physical_response,) if bound_record is None or bound_record["cache_read_tokens"] is not None else ()),
        ),
    )


def _admit_cost_bound_checkpoint(
    latest: _WorkerOutcome, payload: Mapping[str, Any], *,
    diagnostic_cost_upper_bound: bool, model: str,
) -> _WorkerOutcome:
    if (
        not diagnostic_cost_upper_bound or model != _K3_MODEL
        or set(payload) != {"type", "ordinal", "segment"}
        or type(payload["ordinal"]) is not int
        or payload["ordinal"] != len(latest.cost_bound_segments)
        or payload["ordinal"] + 1 != latest.steps_used
        or latest.settled_usage_steps + 1 != latest.steps_used
    ):
        raise ValueError("Kimi worker cost-bound checkpoint is not authorized or ordered")
    record = _parse_cost_bound_records([payload["segment"]])[0]
    counters = (
        latest.input_tokens + record["input_tokens"],
        latest.output_tokens + record["output_tokens"],
        latest.cache_creation_tokens + record["cache_creation_tokens"],
        latest.cache_read_tokens + (record["cache_read_tokens"] or 0),
    )
    settled = _settle_usage_checkpoint(
        latest, counters, usage_valid=True, model=model, bound_record=record,
    )
    return replace(
        settled, cost_bound_segments=(*latest.cost_bound_segments, record),
    )


def _usage_message_id_valid(value: Any) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= _MAX_USAGE_MESSAGE_ID_CHARS
        and not _contains_control_character(value)
    )


def _usage_counter(value: Any) -> int:
    """Preserve an exact provider counter or replace invalid evidence with zero."""

    return cast(int, value) if _is_nonnegative_int(value) else 0


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _parse_tool_events(value: Any) -> tuple[_ToolEvent, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_TOOL_EVENTS:
        raise ValueError("Kimi worker tool_events is not a bounded sequence")
    parsed: list[_ToolEvent] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("Kimi worker emitted a malformed tool event")
        kind = item.get("kind")
        name = item.get("name")
        call_id = item.get("call_id")
        if (
            kind not in {"tool_call", "tool_result"}
            or not isinstance(name, str)
            or _TOOL_NAME_RE.fullmatch(name) is None
            or not isinstance(call_id, str)
            or _TOOL_CALL_ID_RE.fullmatch(call_id) is None
        ):
            raise ValueError("Kimi worker emitted a malformed tool event")
        parsed.append((kind, name, call_id))
    return tuple(parsed)


def _tool_trace_complete(events: Sequence[BackendExecutionEvent]) -> bool:
    """Pair tool results exactly within their physical attempt."""

    attempt_ids = {
        event.child_id
        for event in events
        if event.kind == "request" and event.child_id is not None
    }
    pending: dict[tuple[str, str], str] = {}
    for event in events:
        if event.kind not in {"tool_call", "tool_result"}:
            continue
        if (
            event.parent_id is None
            or event.parent_id not in attempt_ids
            or event.child_id is None
            or event.name is None
        ):
            return False
        key = (event.parent_id, event.child_id)
        if event.kind == "tool_call":
            if key in pending:
                return False
            pending[key] = event.name
            continue
        expected_name = pending.pop(key, None)
        if expected_name is None or expected_name != event.name:
            return False
    return not pending


def _qualify_worker_tool_events(
    outcome: _WorkerOutcome,
    *,
    allowed_names: Sequence[str],
) -> _WorkerOutcome:
    """Fail closed when one physical worker attempt has an invalid tool trace.

    This validation is deliberately scoped to the worker outcome for a single
    physical attempt.  Aggregate traces may contain several independently
    balanced attempts after a read-only transport retry; that remains valid
    evidence whose aggregate completeness is conservatively reported by
    ``_result_from_outcome``.
    """

    allowed = frozenset(allowed_names)
    seen_call_ids: set[str] = set()
    pending: dict[str, str] = {}
    invalid = False
    for kind, name, call_id in outcome.tool_events:
        if name not in allowed:
            invalid = True
            break
        if kind == "tool_call":
            if call_id in seen_call_ids:
                invalid = True
                break
            seen_call_ids.add(call_id)
            pending[call_id] = name
            continue
        expected_name = pending.pop(call_id, None)
        if expected_name is None or expected_name != name:
            invalid = True
            break
    if pending:
        invalid = True
    if not invalid:
        return outcome
    return replace(
        outcome,
        final_message="",
        terminal_seen=False,
        configured_model=None,
        served_model=None,
        stop_reason="error",
        error_kind="qualification",
        error_message=_INVALID_TOOL_TRACE_ERROR,
        retriable=False,
    )


def _is_header_safe_token(value: str) -> bool:
    return all(0x21 <= ord(character) <= 0x7E for character in value)


def _qualify_worker_steps(
    outcome: _WorkerOutcome,
    *,
    physical_limit: int,
) -> _WorkerOutcome:
    max_steps_valid = (
        outcome.error_kind != "max_turns"
        or (
            outcome.stop_reason == "max_turns"
            and outcome.steps_used == physical_limit
            and outcome.settled_usage_steps == outcome.steps_used
            and outcome.usage_seen
            and outcome.usage_valid
        )
    )
    if (
        type(outcome.steps_used) is int
        and 0 <= outcome.steps_used <= physical_limit
        and type(outcome.settled_usage_steps) is int
        and 0 <= outcome.settled_usage_steps <= outcome.steps_used
        and (not outcome.terminal_seen or outcome.steps_used > 0)
        and (
            outcome.error_kind is not None
            or not outcome.final_message
            or (outcome.terminal_seen and outcome.steps_used > 0)
        )
        and max_steps_valid
    ):
        return outcome
    return replace(
        outcome,
        final_message="",
        terminal_seen=False,
        stop_reason="error",
        error_kind="qualification",
        error_message="Kimi worker emitted invalid logical-step evidence",
        retriable=False,
        steps_used=0,
        settled_usage_steps=0,
        usage_valid=False,
    )


def _qualify_worker_model_identity(
    outcome: _WorkerOutcome,
    *, model: str = _MODEL,
) -> _WorkerOutcome:
    """Require exact configured and served model proof on worker results.

    The child independently validates ``Session.model_name``, but its JSON
    envelope is still a process trust boundary. Requiring both that configured
    identity and the captured response ``model`` again in the parent prevents
    a malformed, drifted, or substituted terminal from being admitted as the
    requested Moonshot route. Membership in the model allowlist is not proof
    that the worker served this invocation's requested model.
    """

    if (
        outcome.configured_model == model
        and outcome.served_model == model
    ) or (
        outcome.configured_model is None
        and outcome.served_model is None
        and outcome.error_kind is not None
        and not outcome.terminal_seen
        and not outcome.final_message
    ):
        return outcome
    return replace(
        outcome,
        final_message="",
        terminal_seen=False,
        configured_model=None,
        served_model=None,
        stop_reason="error",
        error_kind="qualification",
        error_message=(
            "Kimi worker did not prove exact configured model identity "
            "and served model identity"
        ),
        usage_valid=False,
        retriable=False,
    )


def _merge_incremental_outcome(
    latest: _WorkerOutcome,
    terminal: _WorkerOutcome,
) -> _WorkerOutcome:
    if terminal.steps_used != latest.steps_used:
        return replace(
            latest,
            final_message="",
            terminal_seen=False,
            configured_model=None,
            served_model=None,
            stop_reason="error",
            error_kind="qualification",
            error_message=(
                "Kimi worker terminal logical-step evidence disagreed with "
                "its incremental progress"
            ),
            retriable=False,
            steps_used=latest.steps_used,
        )
    incremental_usage = (
        latest.input_tokens,
        latest.output_tokens,
        latest.cache_creation_tokens,
        latest.cache_read_tokens,
        latest.usage_seen,
        latest.usage_valid,
    )
    terminal_usage = (
        terminal.input_tokens,
        terminal.output_tokens,
        terminal.cache_creation_tokens,
        terminal.cache_read_tokens,
        terminal.usage_seen,
        terminal.usage_valid,
    )
    if not terminal.terminal_seen:
        canonical_empty_terminal = (
            terminal_usage == (0, 0, 0, 0, False, False)
            and not terminal.tool_events
        )
        if canonical_empty_terminal:
            return replace(
                terminal,
                input_tokens=latest.input_tokens,
                output_tokens=latest.output_tokens,
                cache_creation_tokens=latest.cache_creation_tokens,
                cache_read_tokens=latest.cache_read_tokens,
                usage_seen=latest.usage_seen,
                usage_valid=latest.usage_valid,
                tool_events=latest.tool_events,
                settled_usage_steps=latest.settled_usage_steps,
                pricing_segments=latest.pricing_segments,
                physical_usage_responses=latest.physical_usage_responses,
                cost_bound_segments=latest.cost_bound_segments,
            )
    if (
        incremental_usage != terminal_usage
        or latest.cost_bound_segments != terminal.cost_bound_segments
    ):
        evidence = latest if latest.usage_seen else terminal
        return replace(
            evidence,
            final_message="",
            usage_valid=False,
            terminal_seen=False,
            configured_model=None,
            served_model=None,
            stop_reason="error",
            error_kind="qualification",
            error_message=(
                "Kimi worker terminal usage evidence disagreed with its "
                "incremental accounting"
            ),
            retriable=False,
            steps_used=latest.steps_used,
            tool_events=latest.tool_events,
        )
    if terminal.tool_events != latest.tool_events:
        return replace(
            latest,
            final_message="",
            usage_valid=False,
            terminal_seen=False,
            configured_model=None,
            served_model=None,
            stop_reason="error",
            error_kind="qualification",
            error_message=(
                "Kimi worker terminal tool trace disagreed with its "
                "incremental events"
            ),
            retriable=False,
        )
    return replace(
        terminal,
        settled_usage_steps=latest.settled_usage_steps,
        pricing_segments=latest.pricing_segments,
        physical_usage_responses=latest.physical_usage_responses,
    )


def _tool_inventory(toolset: Any) -> tuple[tuple[str, ...], frozenset[str]]:
    descriptors = list(toolset.to_anthropic_tools())
    names: list[str] = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping):
            raise ValueError("Kimi ToolSet descriptor must be a mapping")
        name = descriptor.get("name")
        if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "Kimi ToolSet names must be 1-128 portable function-name characters"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Kimi ToolSet contains duplicate tool descriptors")

    write_metadata = getattr(toolset, "write_names", None)
    read_metadata = getattr(toolset, "read_only_names", None)
    if write_metadata is None or read_metadata is None:
        raise ValueError(
            "Kimi ToolSet requires explicit read_only_names and write_names "
            "classification"
        )
    writes = frozenset(str(name) for name in write_metadata)
    reads = frozenset(str(name) for name in read_metadata)
    overlap = reads & writes
    if overlap:
        raise ValueError(
            "Kimi ToolSet classifies tools as both read-only and mutating: "
            + ", ".join(sorted(overlap))
        )
    classified = reads | writes
    inventory = frozenset(names)
    if classified != inventory:
        missing = sorted(inventory - classified)
        unknown = sorted(classified - inventory)
        details = []
        if missing:
            details.append(f"unclassified={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError(
            "Kimi ToolSet capability metadata does not match descriptors: "
            + ", ".join(details)
        )
    return tuple(sorted(names)), writes


def _tool_authority_digest(
    frozen_toolset: _FrozenMcpToolset,
    *,
    capability: str,
    context_budget: int | None,
) -> str:
    """Bind authority to the exact frozen catalogue exposed over MCP."""

    from .tool_authority import (
        APPROVAL_POLICY_PER_CALL_EXACT,
        tool_authority_digest,
    )

    return tool_authority_digest(
        frozen_toolset.authority_snapshot,
        capability=capability,
        approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
        bridge_protocol=_TOOL_AUTHORITY_PROTOCOL,
        bridge_version=_TOOL_AUTHORITY_VERSION,
        context_budget=context_budget,
    )


def _distribution_version(name: str) -> str:
    return importlib.metadata.version(name)


def _assert_supported_versions() -> None:
    try:
        sdk_version = _distribution_version("kimi-agent-sdk")
        cli_version = _distribution_version("kimi-cli")
    except importlib.metadata.PackageNotFoundError as exc:
        raise KimiSDKUnavailable(
            "KimiBackend requires optional dependency "
            f"`kimi-agent-sdk=={_SDK_VERSION}` with "
            f"`kimi-cli=={_CLI_VERSION}`"
        ) from exc
    if sdk_version != _SDK_VERSION or cli_version != _CLI_VERSION:
        raise KimiSDKQualificationError(
            "KimiBackend is qualified only for "
            f"kimi-agent-sdk=={_SDK_VERSION} / kimi-cli=={_CLI_VERSION}; "
            f"found {sdk_version} / {cli_version}"
        )


def _load_sdk_bindings() -> _SdkBindings:
    _assert_supported_versions()
    try:
        import kimi_agent_sdk as sdk  # type: ignore[import-not-found]
        from kaos.path import KaosPath  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KimiSDKUnavailable(
            "the reviewed Kimi SDK distributions are present but their "
            "Python modules cannot be imported"
        ) from exc
    return _SdkBindings(
        Session=sdk.Session,
        Config=sdk.Config,
        KaosPath=KaosPath,
        ApprovalRequest=sdk.ApprovalRequest,
        StatusUpdate=sdk.StatusUpdate,
        TextPart=sdk.TextPart,
        ThinkPart=sdk.ThinkPart,
        ToolCall=sdk.ToolCall,
        ToolResult=sdk.ToolResult,
        StepBegin=sdk.StepBegin,
        TurnEnd=sdk.TurnEnd,
        StepInterrupted=sdk.StepInterrupted,
        RunCancelled=sdk.RunCancelled,
        MaxStepsReached=sdk.MaxStepsReached,
        APIStatusError=sdk.APIStatusError,
        APIConnectionError=sdk.APIConnectionError,
        APITimeoutError=sdk.APITimeoutError,
        APIEmptyResponseError=sdk.APIEmptyResponseError,
        ChatProviderError=sdk.ChatProviderError,
    )


def _parse_inference_base_url(value: str, *, setting: str) -> Any:
    """Parse only qualified encrypted routes or an exact IPv4 loopback test route."""

    if (
        not isinstance(value, str)
        or not value
        or _contains_control_character(value)
        or any(character.isspace() for character in value)
    ):
        raise KimiSDKQualificationError(
            f"{setting} must be a qualified inference base URL"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise KimiSDKQualificationError(
            f"{setting} must be a qualified inference base URL"
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65_535)
    ):
        raise KimiSDKQualificationError(
            f"{setting} must be an HTTPS base URL without embedded "
            "credentials, query parameters, or fragments"
        )
    if parsed.scheme == "http" and (
        port is None or parsed.netloc != f"127.0.0.1:{port}"
    ):
        raise KimiSDKQualificationError(
            f"{setting} requires HTTPS except for an exact "
            "http://127.0.0.1:<port> test endpoint"
        )
    return parsed


def _validate_runtime_credentials() -> tuple[str, str]:
    base_url = os.environ.get("KIMI_BASE_URL", "")
    token = os.environ.get("MOONSHOT_API_KEY", "")
    if not base_url:
        raise KimiSDKQualificationError(
            "KimiBackend requires KIMI_BASE_URL at invocation time"
        )
    _parse_inference_base_url(base_url, setting="KIMI_BASE_URL")
    if (
        not token
        or len(token) > _MAX_AUTH_TOKEN_CHARS
        or not _is_header_safe_token(token)
    ):
        raise KimiSDKQualificationError(
            "KimiBackend requires a valid bounded MOONSHOT_API_KEY "
            "containing only header-safe printable ASCII at invocation time"
        )
    return base_url, token


def _normalize_base_url(value: str, *, setting: str) -> str:
    """Validate and normalize an endpoint for route-identity comparison."""

    parsed = _parse_inference_base_url(value, setting=setting)
    path = parsed.path.rstrip("/")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            "",
            "",
        )
    )


def _resolve_worker_python(
    explicit: Path | str | None = None,
) -> Path:
    """Resolve the qualified worker interpreter without a PATH lookup.

    An explicit constructor value wins, followed by
    ``CRUCIBLE_KIMI_WORKER_PYTHON``.  The deterministic project-local worker
    environment is the final fallback.  Relative overrides are rejected so a
    cwd change cannot silently select a different executable.
    """

    configured = explicit
    if configured is None:
        configured = os.environ.get("CRUCIBLE_KIMI_WORKER_PYTHON")
    candidate = Path(configured).expanduser() if configured else _DEFAULT_WORKER_PYTHON
    if not candidate.is_absolute():
        raise KimiSDKQualificationError("Kimi worker Python must be an absolute path")
    # Keep the final ``.venv/bin/python`` symlink intact: executing the
    # resolved base interpreter would lose the virtual environment's prefix
    # and therefore its isolated site-packages.
    return Path(os.path.abspath(candidate))


def _worker_environment() -> dict[str, str]:
    """Return a minimal child environment with no ambient credentials."""

    # TLS trust paths are required in some enterprise installations. Locale
    # and timezone affect only presentation. Do not inherit PATH, proxies,
    # cloud credentials, CI tokens, dynamic-loader controls, or provider vars.
    allowed = (
        "LANG",
        "LC_ALL",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    )
    child = {name: os.environ[name] for name in allowed if name in os.environ}
    child["PYTHONNOUSERSITE"] = "1"
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


@functools.lru_cache(maxsize=8)
def _qualify_worker_runtime(
    worker_python_text: str,
    runtime_fingerprint: str | None = None,
) -> KimiWorkerPackageVersions:
    """Probe the separately installed worker and enforce its reviewed tuple."""

    # ``runtime_fingerprint`` is intentionally part of the qualification
    # cache key.  A newly captured project-locked runtime can never inherit a
    # successful probe for different installed bytes at the same path.
    del runtime_fingerprint
    worker_python = Path(worker_python_text)
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise KimiSDKUnavailable(
            "Kimi worker runtime is not installed; run "
            f"`uv sync --project {_WORKER_PROJECT}` or set "
            "CRUCIBLE_KIMI_WORKER_PYTHON to its absolute interpreter path"
        )
    try:
        completed = subprocess.run(
            [
                str(worker_python),
                "-B",
                "-I",
                "-m",
                "crucible_kimi_worker",
                "--probe",
            ],
            check=False,
            capture_output=True,
            env=_worker_environment(),
            timeout=_WORKER_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise KimiSDKUnavailable(
            f"failed to probe the isolated Kimi worker runtime ({type(exc).__name__})"
        ) from exc
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(stdout)
    except (TypeError, ValueError) as exc:
        raise KimiSDKQualificationError(
            "Kimi worker probe did not return its qualified JSON identity"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(payload, Mapping)
        or payload.get("protocol") != _WORKER_PROTOCOL
        or payload.get("type") != "probe"
        or payload.get("sdk_version") != _SDK_VERSION
        or payload.get("cli_version") != _CLI_VERSION
        or payload.get("kosong_version") != _KOSONG_VERSION
        or payload.get("openai_version") != _OPENAI_VERSION
    ):
        raise KimiSDKQualificationError(
            "Kimi worker does not match the qualified "
            "kimi-agent-sdk==0.0.5 / kimi-cli==1.12.0 / "
            "kosong==0.42.0 / openai==2.14.0 "
            f"runtime (exit code {completed.returncode})"
        )
    return KimiWorkerPackageVersions(
        kimi_agent_sdk=_SDK_VERSION,
        kimi_cli=_CLI_VERSION,
        kosong=_KOSONG_VERSION,
        openai=_OPENAI_VERSION,
    )


def capture_project_locked_kimi_runtime_identity(
) -> KimiProjectLockedRuntimeIdentity:
    """Capture and probe the exact project-owned worker without credentials.

    The function intentionally rejects the ambient worker override just like
    ``KimiBackend(worker_source="project_locked")``.  It executes only the
    worker's local ``--probe`` path, which imports installed packages and
    returns version metadata; it does not read provider credentials or make a
    provider request.
    """

    if os.environ.get("CRUCIBLE_KIMI_WORKER_PYTHON", "").strip():
        raise KimiSDKQualificationError(
            "project-locked Kimi runtime capture rejects ambient "
            "CRUCIBLE_KIMI_WORKER_PYTHON"
        )
    logical_worker = _resolve_worker_python(_DEFAULT_WORKER_PYTHON)
    pin = _PinnedKimiWorkerRuntime.capture(logical_worker)
    package_versions = _qualify_worker_runtime(
        str(logical_worker),
        pin.runtime_fingerprint,
    )
    # Close the capture window after both the probe and explicit tree
    # projection. Any symlink, interpreter, lock, source, or installed-tree
    # drift causes the existing strict pin revalidation to fail.
    pin.revalidate()
    return KimiProjectLockedRuntimeIdentity(
        logical_worker_python=pin.invocation_path,
        resolved_worker_python=pin.resolved_path,
        resolved_worker_python_sha256="sha256:" + pin.resolved_sha256,
        lock_source_installed_tree_fingerprint=(
            pin.installed_tree_fingerprint
        ),
        runtime_fingerprint=pin.runtime_fingerprint,
        package_versions=package_versions,
    )


def _schema_json(schema: Mapping[str, Any] | None) -> str | None:
    if schema is None:
        return None
    try:
        return json.dumps(
            schema,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "KimiBackend response_schema must be JSON serializable"
        ) from exc


def _write_private(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        payload = text.encode("utf-8")
        while payload:
            written = os.write(fd, payload)
            payload = payload[written:]
    finally:
        os.close(fd)


def _isolate_worker_environment(root: Path) -> _WorkerPaths:
    os.umask(0o077)
    root = root.resolve(strict=True)
    root.chmod(0o700)
    paths = _WorkerPaths(
        root=root,
        home=root / "home",
        share=root / "share",
        work=root / "work",
        skills=root / "skills",
        config=root / "config",
        scratch=root / "tmp",
    )
    for path in (
        paths.home,
        paths.share,
        paths.work,
        paths.skills,
        paths.config,
        paths.scratch,
    ):
        path.mkdir(mode=0o700)
        path.chmod(0o700)

    allowed = (
        "LANG",
        "LC_ALL",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    )
    retained = {name: os.environ[name] for name in allowed if name in os.environ}
    os.environ.clear()
    retained.update(
        {
            "HOME": str(paths.home),
            "XDG_CONFIG_HOME": str(paths.config),
            "XDG_DATA_HOME": str(paths.root / "data"),
            "XDG_CACHE_HOME": str(paths.root / "cache"),
            "KIMI_SHARE_DIR": str(paths.share),
            "TMPDIR": str(paths.scratch),
            "TMP": str(paths.scratch),
            "TEMP": str(paths.scratch),
            "PYTHONNOUSERSITE": "1",
        }
    )
    os.environ.update(retained)
    tempfile.tempdir = str(paths.scratch)
    os.chdir(paths.work)
    return paths


def _prompt_system_text(request: _WorkerRequest) -> str:
    if request.schema_json is None:
        return request.system
    prompt_abi_marker = "<!-- assay.prompt.v2 -->"
    if prompt_abi_marker in request.system:
        canonical_block = f"```json\n{request.schema_json}\n```"
        if (
            not request.system.startswith(prompt_abi_marker + "\n")
            or request.system.count(prompt_abi_marker) != 1
            or request.system.count("## Required output contract") != 1
            or request.system.count(canonical_block) != 1
            or request.system.count(request.schema_json) != 1
            or "CRUCIBLE OUTPUT CONTRACT:" in request.system
        ):
            raise KimiSDKQualificationError(
                "Kimi prompt ABI v2 output contract does not contain exactly "
                "one matching canonical response schema"
            )
        return request.system
    return (
        request.system.rstrip()
        + "\n\nCRUCIBLE OUTPUT CONTRACT:\n"
        + "Return only one JSON value matching the following JSON Schema. "
        + "Do not wrap it in Markdown fences or add prose outside the value.\n"
        + request.schema_json
    )


def _write_agent_definition(
    request: _WorkerRequest,
    paths: _WorkerPaths,
) -> tuple[Path, Path]:
    system_file = paths.config / "system.md"
    agent_file = paths.config / "agent.yaml"
    _write_private(system_file, "${CRUCIBLE_SYSTEM}\n")
    agent_doc = {
        "version": "1",
        "agent": {
            "name": (
                "crucible-kimi-prompt-only"
                if request.mcp_url is None
                else "crucible-kimi-toolset"
            ),
            "system_prompt_path": "./system.md",
            "system_prompt_args": {"CRUCIBLE_SYSTEM": _prompt_system_text(request)},
            "tools": [],
            "exclude_tools": [],
            "subagents": {},
        },
    }
    # JSON is a YAML subset and avoids quoting ambiguities in user prompts.
    _write_private(
        agent_file,
        json.dumps(agent_doc, ensure_ascii=False, allow_nan=False),
    )
    return agent_file, system_file


def _build_sdk_config(bindings: _SdkBindings, request: _WorkerRequest) -> Any:
    return bindings.Config.model_validate(
        {
            "default_model": request.model,
            # K2.7 is always-thinking.  Keep the SDK session in thinking mode
            # for its state machine, but intentionally omit the model
            # capability below: CLI 1.12 otherwise translates this flag to
            # reasoning_effort="high", which Moonshot rejects.
            "default_thinking": True,
            "default_yolo": False,
            "providers": {
                _PROVIDER_NAME: {
                    "type": "kimi",
                    "base_url": request.base_url,
                    "api_key": request.auth_token,
                }
            },
            "models": {
                request.model: {
                    "provider": _PROVIDER_NAME,
                    "model": request.model,
                    "max_context_size": _MODEL_CONTEXT_SIZES[request.model],
                    "capabilities": [],
                }
            },
            "loop_control": {
                "max_steps_per_turn": request.max_steps,
                # Crucible owns physical-attempt accounting.  Do not multiply
                # it with the CLI's internal retry loop.
                "max_retries_per_step": 1,
                "max_ralph_iterations": 0,
            },
            "services": {},
            "mcp": {
                "client": {
                    "tool_call_timeout_ms": _MCP_TOOL_TIMEOUT_MS,
                }
            },
        }
    )


def _k3_usage_failure(code: str, *, field: str = "none", state: str = "none") -> KimiSDKQualificationError:
    # All arguments are adapter-owned enum labels, never provider strings.
    return KimiSDKQualificationError(
        "Kimi K3 usage has invalid raw billing counters; "
        f"code={code}; field={field}; state={state}"
    )


def _k3_raw_type(value: Any) -> str:
    return {
        type(None): "null", bool: "boolean", int: "integer", float: "float",
        str: "string", list: "array", dict: "object",
    }.get(type(value), "other")



class _K3CostBoundBudget(RuntimeError):
    """A completed response exhausted the explicitly bound diagnostic budget."""


def _parse_cost_bound_records(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_TOOL_STEPS:
        raise ValueError("invalid Kimi cost-bound sequence")
    records = []
    for record in value:
        if not isinstance(record, dict) or set(record) != {
            "input_tokens", "output_tokens", "cache_creation_tokens",
            "cache_read_tokens", "policy",
        }:
            raise ValueError("invalid Kimi cost-bound fields")
        p, o, w, r = (
            record["input_tokens"], record["output_tokens"],
            record["cache_creation_tokens"], record["cache_read_tokens"],
        )
        if (
            any(type(n) is not int or n < 0 for n in (p, o, w))
            or (r is not None and (type(r) is not int or r < 0))
            or w + (r or 0) > p or record["policy"] != _K3_BOUND_POLICY
        ):
            raise ValueError("invalid Kimi cost-bound counters")
        try:
            finite = all(math.isfinite(amount) for amount in _cost_bound_amounts(record))
        except OverflowError as exc:
            raise ValueError("Kimi cost-bound counters exceed finite accounting") from exc
        if not finite:
            raise ValueError("Kimi cost-bound counters exceed finite accounting")
        records.append(dict(record))
    return tuple(records)


def _cost_bound_amounts(record: Mapping[str, Any]) -> tuple[float, float]:
    # K3's request-bound implicit 5m policy charges input and writes $3/M,
    # reads $0.30/M, and output $15/M. P includes both cache categories.
    p, o, w, r = (
        record["input_tokens"], record["output_tokens"],
        record["cache_creation_tokens"], record["cache_read_tokens"],
    )
    upper = (3.0 * p + 15.0 * o) / 1_000_000
    if r is None:
        lower = (0.30 * (p - w) + 3.0 * w + 15.0 * o) / 1_000_000
        return lower, upper
    exact = (3.0 * (p - r) + 0.30 * r + 15.0 * o) / 1_000_000
    return exact, exact


class _K3CostBoundLedger:
    """Invocation-local, request-bound response accounting; never invoice truth."""

    def __init__(
        self, enabled: bool, budget: float | None,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        self.enabled = enabled
        self.budget = budget
        self.emit = emit
        self.records: list[dict[str, Any]] = []

    @property
    def upper(self) -> float:
        return math.fsum(_cost_bound_amounts(record)[1] for record in self.records)

    def admit_next(self) -> None:
        if self.enabled and self.budget is not None and self.upper >= self.budget:
            raise _K3CostBoundBudget("Kimi diagnostic upper-bound budget exhausted")

    def settle(self, counters: tuple[int, int, int, int | None, int]) -> None:
        if not self.enabled:
            return
        p, o, _total, r, w = counters
        record = _parse_cost_bound_records([{
            "input_tokens": p, "output_tokens": o, "cache_creation_tokens": w,
            "cache_read_tokens": r, "policy": _K3_BOUND_POLICY,
        }])[0]
        if len(self.records) >= _MAX_TOOL_STEPS:
            raise ValueError("Kimi cost-bound response limit exceeded")
        self.records.append(record)
        self.emit({
            "type": "cost_bound", "ordinal": len(self.records) - 1,
            "segment": record,
        })
        if self.budget is None or self.upper > self.budget:
            raise _K3CostBoundBudget("Kimi diagnostic upper-bound budget exceeded")


def _validate_k3_raw_usage(
    value: Any, *, diagnostic_cost_upper_bound: bool = False,
) -> tuple[int, int, int, int | None, int]:
    """Reject coerced or ambiguous billing counters at the raw JSON boundary."""

    if not isinstance(value, dict):
        raise _k3_usage_failure("usage_type", state=_k3_raw_type(value))
    details = value.get("prompt_tokens_details")
    if not isinstance(details, dict):
        state = _k3_raw_type(details) if "prompt_tokens_details" in value else "missing"
        raise _k3_usage_failure("details_type", field="details", state=state)
    prompt = value.get("prompt_tokens")
    output = value.get("completion_tokens")
    read = details.get("cached_tokens")
    write = details.get("cache_write_tokens")
    total = value.get("total_tokens")
    for container, key, label in (
        (value, "prompt_tokens", "P"), (value, "completion_tokens", "O"),
        (value, "total_tokens", "T"), (details, "cached_tokens", "R"),
        (details, "cache_write_tokens", "W"),
    ):
        if (
            label == "R" and diagnostic_cost_upper_bound
            and "cached_tokens" not in details and "cached_tokens" not in value
        ):
            continue
        if key not in container:
            raise _k3_usage_failure("counter_missing", field=label, state="missing")
        counter = container[key]
        if type(counter) is not int:
            raise _k3_usage_failure("counter_type", field=label, state=_k3_raw_type(counter))
        if counter < 0:
            raise _k3_usage_failure("counter_negative", field=label, state="integer")
    if (0 if read is None else read) + write > prompt:
        raise _k3_usage_failure("cache_overlap")
    if total != prompt + output:
        raise _k3_usage_failure("total_mismatch")
    if "cached_tokens" in value:
        alias = value["cached_tokens"]
        if type(alias) is not int:
            raise _k3_usage_failure("alias_type", field="alias", state=_k3_raw_type(alias))
        if alias != read:
            raise _k3_usage_failure("alias_conflict", field="alias", state="integer")
    return prompt, output, total, read, write


@dataclass
class _K3StreamUsageState:
    """Only terminal, raw-validated counters can become pricing authority."""

    finish_seen: bool = False
    final_stats_seen: bool = False
    done: bool = False
    counters: tuple[int, int, int, int | None, int] | None = None
    diagnostic_cost_upper_bound: bool = False

    def observe(self, raw: Any) -> None:
        if not isinstance(raw, dict):
            raise _k3_usage_failure("frame_type", state=_k3_raw_type(raw))
        if raw.get("error"):
            # Preserve the native API-error conversion; never price it.
            return
        if self.final_stats_seen:
            raise _k3_usage_failure("data_after_final_stats")
        choices = raw.get("choices")
        if not isinstance(choices, list):
            raise _k3_usage_failure("choices_type", state=_k3_raw_type(choices))
        is_finish = (
            len(choices) == 1 and isinstance(choices[0], dict)
            and type(choices[0].get("index")) is int
            and choices[0]["index"] == 0
            and choices[0].get("finish_reason") is not None
        )
        if choices and self.finish_seen:
            raise _k3_usage_failure("choice_after_finish")
        if is_finish:
            self.finish_seen = True
        if not choices:
            if not self.finish_seen:
                raise _k3_usage_failure("statistics_before_finish")
            counters = _validate_k3_raw_usage(
                raw.get("usage"),
                diagnostic_cost_upper_bound=self.diagnostic_cost_upper_bound,
            )
            self.final_stats_seen = True
        elif is_finish and raw.get("usage") is not None:
            value = raw["usage"]
            details = value.get("prompt_tokens_details") if isinstance(value, dict) else None
            complete_candidate = (
                isinstance(details, dict)
                and all(value.get(key) is not None for key in (
                    "prompt_tokens", "completion_tokens", "total_tokens",
                ))
                and details.get("cache_write_tokens") is not None
                and (
                    details.get("cached_tokens") is not None
                    or (
                        self.diagnostic_cost_upper_bound
                        and "cached_tokens" not in details
                        and "cached_tokens" not in value
                    )
                )
            )
            if not complete_candidate:
                # The finish chunk can still carry partial progress counters.
                # Require a later complete statistics packet before [DONE].
                return
            counters = _validate_k3_raw_usage(
                value, diagnostic_cost_upper_bound=self.diagnostic_cost_upper_bound,
            )
        else:
            # Choice-local usage (including on finish) and top-level progress
            # usage are not billing authority. Do not price, normalize or
            # project them into SDK StatusUpdate evidence.
            return
        if self.counters is not None and counters != self.counters:
            raise _k3_usage_failure("contradictory_terminal_usage")
        self.counters = counters

    def canonical_usage(self) -> dict[str, Any]:
        if not self.done:
            raise _k3_usage_failure("missing_done")
        if not self.finish_seen or self.counters is None:
            raise _k3_usage_failure("missing_terminal_usage")
        prompt, output, total, read, write = self.counters
        return {
            "prompt_tokens": prompt, "completion_tokens": output,
            "total_tokens": total,
            "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write},
        }


def _guard_k3_raw_stream(
    response: Any, *, diagnostic_cost_upper_bound: bool = False,
) -> _K3StreamUsageState:
    """Validate SSE usage before OpenAI 2.14 can coerce JSON scalar types.

    This invocation-local hook leaves the native stream's parsing, error
    handling and response-closing finally block intact. It runs before its
    Pydantic Optional[CompletionUsage] conversion, which otherwise turns
    numeric strings and integral floats into apparently exact integers.
    """

    state = _K3StreamUsageState(
        diagnostic_cost_upper_bound=diagnostic_cost_upper_bound,
    )
    original = getattr(response, "_iter_events", None)
    if (
        type(response).__name__ != "AsyncStream"
        or type(response).__module__ != "openai"
        or not isinstance(original, types.MethodType)
        or original.__func__.__module__ != "openai._streaming"
        or original.__func__.__qualname__ != "AsyncStream._iter_events"
    ):
        raise KimiSDKQualificationError("Kimi K3 raw stream boundary is unqualified")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise KimiSDKQualificationError("Kimi K3 stream contains duplicate JSON fields")
            result[key] = value
        return result

    async def validated_events(_response: Any) -> Any:
        async for event in original():
            if event.data == "[DONE]":
                state.done = True
                state.canonical_usage()
            elif event.data.startswith("[DONE]"):
                raise _k3_usage_failure("invalid_done")
            else:
                try:
                    raw = json.loads(event.data, object_pairs_hook=strict_object)
                except (ValueError, TypeError) as exc:
                    raise KimiSDKQualificationError("Kimi K3 stream contains invalid JSON") from exc
                state.observe(raw)
            yield event
        if not state.done:
            raise _k3_usage_failure("missing_done")

    response._iter_events = types.MethodType(validated_events, response)
    return state


def _install_moonshot_stream_compat(
    provider: Any, *, model: str = _MODEL,
    cost_bounds: _K3CostBoundLedger | None = None,
) -> None:
    """Install the version-pinned, request-bound Moonshot stream parser.

    Kosong 0.42.0 already handles Moonshot reasoning and cached-token usage,
    but its streaming tool parser drops ``tool_call.index`` and its public
    stream discards ``finish_reason`` and response model identity.  This
    invocation-scoped generator keeps the reviewed native message/tool
    conversion while aggregating parallel argument deltas by index and
    retaining both terminal fields for admission.
    """

    try:
        from kosong.chat_provider import kimi as kimi_provider  # type: ignore[import-not-found]
    except ImportError as exc:
        raise KimiSDKQualificationError(
            "Kimi native stream compatibility module is unavailable"
        ) from exc

    native_stream_type = getattr(kimi_provider, "KimiStreamedMessage", None)
    if (
        not isinstance(native_stream_type, type)
        or native_stream_type.__module__ != "kosong.chat_provider.kimi"
        or native_stream_type.__name__ != "KimiStreamedMessage"
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime resolved an unqualified native stream type"
        )

    class _QualifiedKimiStream(native_stream_type):
        def __init__(
            self, response: Any, *, k3_state: _K3StreamUsageState | None = None,
        ) -> None:
            if isinstance(response, kimi_provider.ChatCompletion):
                raise KimiSDKQualificationError(
                    "Kimi runtime unexpectedly disabled response streaming"
                )
            self._iter = self._convert_qualified_stream(response)
            self._id: str | None = None
            self._usage: Any = None
            self.finish_reason: str | None = None
            self.served_model: str | None = None
            self._k3_state = k3_state

        @property
        def usage(self) -> Any:
            if model != _K3_MODEL:
                return super().usage
            if self._usage is None:
                return None
            # CLI/Kosong 0.42 does not know K3 cache writes. Override its
            # projection at the physical stream boundary, before StatusUpdate
            # is produced, so SDK totals and parent pricing share one source.
            raw = self._usage
            details = getattr(raw, "prompt_tokens_details", None)
            prompt = getattr(raw, "prompt_tokens", None)
            output = getattr(raw, "completion_tokens", None)
            read = getattr(details, "cached_tokens", None)
            write = getattr(details, "cache_write_tokens", None)
            if (
                read is None and cost_bounds is not None and cost_bounds.enabled
                and self._k3_state is not None
                and self._k3_state.counters is not None
                and self._k3_state.counters[3] is None
            ):
                # SDK TokenUsage cannot express unknown R. This is only the
                # known-counter projection; the nullable raw record travels
                # separately and revokes exact token/cost completeness.
                read = 0
            if (
                not _usage_counters_valid(prompt, output, write, read)
                or type(getattr(raw, "total_tokens", None)) is not int
                or raw.total_tokens != prompt + output
                or (
                    getattr(raw, "cached_tokens", None) is not None
                    and (
                        type(raw.cached_tokens) is not int
                        or raw.cached_tokens != read
                    )
                )
            ):
                raise KimiSDKQualificationError(
                    "Kimi K3 final usage lacks exact disjoint cache counters"
                )
            return kimi_provider.TokenUsage(
                input_other=prompt - read - write,
                output=output,
                input_cache_read=read,
                input_cache_creation=write,
            )

        async def _convert_qualified_stream(self, response: Any) -> Any:
            tool_calls: dict[int, dict[str, str | None]] = {}
            message_part_seen = False
            try:
                async for chunk in response:
                    if chunk.id:
                        self._id = chunk.id
                    chunk_model = str(getattr(chunk, "model", "") or "")
                    if chunk_model:
                        if (
                            self.served_model is not None
                            and self.served_model != chunk_model
                        ):
                            raise KimiSDKQualificationError(
                                "Kimi stream changed served model identity"
                            )
                        self.served_model = chunk_model
                    if model != _K3_MODEL:
                        if usage := kimi_provider.extract_usage_from_chunk(chunk):
                            self._usage = usage
                    if not chunk.choices:
                        continue
                    if (
                        len(chunk.choices) != 1
                        or getattr(chunk.choices[0], "index", None) != 0
                    ):
                        raise KimiSDKQualificationError(
                            "Kimi stream violated the qualified n=1 choice shape"
                        )
                    choice = chunk.choices[0]
                    raw_finish = getattr(choice, "finish_reason", None)
                    if raw_finish is not None:
                        finish = str(raw_finish)
                        if (
                            self.finish_reason is not None
                            and self.finish_reason != finish
                        ):
                            raise KimiSDKQualificationError(
                                "Kimi stream emitted conflicting finish reasons"
                            )
                        self.finish_reason = finish
                    delta = choice.delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not isinstance(reasoning, str):
                            raise KimiSDKQualificationError(
                                "Kimi stream emitted non-text reasoning content"
                            )
                        message_part_seen = True
                        yield kimi_provider.ThinkPart(think=reasoning)
                    if delta.content:
                        message_part_seen = True
                        yield kimi_provider.TextPart(text=delta.content)
                    for tool_delta in delta.tool_calls or ():
                        index = getattr(tool_delta, "index", None)
                        if (
                            type(index) is not int
                            or not 0 <= index < _MAX_TOOL_EVENTS
                        ):
                            raise KimiSDKQualificationError(
                                "Kimi stream emitted an invalid tool-call index"
                            )
                        state = tool_calls.setdefault(
                            index,
                            {"id": None, "name": None, "arguments": ""},
                        )
                        raw_id = getattr(tool_delta, "id", None)
                        if raw_id:
                            tool_id = str(raw_id)
                            if state["id"] not in {None, tool_id}:
                                raise KimiSDKQualificationError(
                                    "Kimi stream changed a tool-call id"
                                )
                            state["id"] = tool_id
                        function = getattr(tool_delta, "function", None)
                        if function is None:
                            continue
                        raw_name = getattr(function, "name", None)
                        if raw_name:
                            tool_name = str(raw_name)
                            if state["name"] not in {None, tool_name}:
                                raise KimiSDKQualificationError(
                                    "Kimi stream changed a tool-call name"
                                )
                            state["name"] = tool_name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            if not isinstance(arguments, str):
                                raise KimiSDKQualificationError(
                                    "Kimi stream emitted non-text tool arguments"
                                )
                            state["arguments"] = (
                                str(state["arguments"] or "") + arguments
                            )
            except (
                kimi_provider.OpenAIError,
                kimi_provider.httpx.HTTPError,
            ) as exc:
                raise kimi_provider.convert_error(exc) from exc

            if model == _K3_MODEL:
                if self._k3_state is None:
                    raise _k3_usage_failure("missing_raw_stream_state")
                # Only a raw-validated final packet followed by exact [DONE]
                # reaches SDK StatusUpdate or the tool-call yield below.
                self._usage = kimi_provider.CompletionUsage.model_validate(
                    self._k3_state.canonical_usage()
                )
            if self.served_model != model:
                raise KimiSDKQualificationError(
                    "Kimi stream did not prove the exact served model"
                )
            if self.finish_reason is None:
                raise KimiSDKQualificationError(
                    "Kimi stream ended without a finish reason"
                )
            if cost_bounds is not None and cost_bounds.enabled:
                if self._k3_state is None or self._k3_state.counters is None:
                    raise _k3_usage_failure("missing_raw_stream_state")
                cost_bounds.settle(self._k3_state.counters)
            if model == _K3_MODEL and self._usage is None:
                raise KimiSDKQualificationError(
                    "Kimi K3 stream ended without final usage evidence"
                )
            if self.finish_reason in {"length", "content_filter"}:
                # Kosong raises APIEmptyResponseError when the provider emits
                # only terminal metadata. Preserve the qualified terminal
                # model, finish reason, and usage through its generation
                # layer with an inert content part. Partial tool calls are
                # deliberately discarded: neither a truncated nor rejected
                # call is safe to dispatch.
                if not message_part_seen:
                    yield kimi_provider.TextPart(text="")
                return
            if tool_calls and self.finish_reason != "tool_calls":
                raise KimiSDKQualificationError(
                    "Kimi stream emitted tool calls under an invalid finish reason"
                )
            if self.finish_reason == "tool_calls":
                indexes = sorted(tool_calls)
                if indexes != list(range(len(indexes))):
                    raise KimiSDKQualificationError(
                        "Kimi stream emitted a sparse tool-call index set"
                    )
                for index in indexes:
                    state = tool_calls[index]
                    if (
                        not state["id"]
                        or not state["name"]
                        or state["arguments"] is None
                    ):
                        raise KimiSDKQualificationError(
                            "Kimi stream ended with an incomplete tool call"
                        )
                    yield kimi_provider.ToolCall(
                        id=str(state["id"]),
                        function=kimi_provider.ToolCall.FunctionBody(
                            name=str(state["name"]),
                            arguments=str(state["arguments"]),
                        ),
                    )

    original_generate = getattr(provider, "generate", None)
    if (
        not isinstance(original_generate, types.MethodType)
        or original_generate.__func__.__module__ != "kosong.chat_provider.kimi"
        or original_generate.__func__.__qualname__ != "Kimi.generate"
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime resolved an unqualified native generator"
        )

    qualified_generation_kwargs = copy.deepcopy(provider._generation_kwargs)

    async def _qualified_generate(
        native_provider: Any,
        system_prompt: str,
        tools: Sequence[Any],
        history: Sequence[Any],
    ) -> Any:
        if cost_bounds is not None:
            cost_bounds.admit_next()
        if native_provider.model != model:
            raise KimiSDKQualificationError(
                "Kimi provider changed requested model identity"
            )
        if native_provider._generation_kwargs != qualified_generation_kwargs:
            raise KimiSDKQualificationError(
                "Kimi provider changed qualified generation options"
            )
        messages: list[Any] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(
            kimi_provider._convert_message(message)  # noqa: SLF001
            for message in history
        )
        generation_kwargs: dict[str, Any] = (
            {"max_tokens": 32_000} if model == _MODEL else {}
        )
        generation_kwargs.update(native_provider._generation_kwargs)
        try:
            response = await native_provider.client.chat.completions.create(
                model=native_provider.model,
                messages=messages,
                tools=(
                    kimi_provider._convert_tool(tool)  # noqa: SLF001
                    for tool in tools
                ),
                stream=True,
                stream_options={"include_usage": True},
                **generation_kwargs,
            )
        except (
            kimi_provider.OpenAIError,
            kimi_provider.httpx.HTTPError,
        ) as exc:
            raise kimi_provider.convert_error(exc) from exc
        k3_state = None
        if model == _K3_MODEL:
            try:
                k3_state = _guard_k3_raw_stream(
                    response, diagnostic_cost_upper_bound=(
                        cost_bounds is not None and cost_bounds.enabled
                    ),
                )
            except BaseException:
                with contextlib.suppress(Exception):
                    await response.close()
                raise
        stream = _QualifiedKimiStream(response, k3_state=k3_state)
        native_provider._crucible_last_stream = stream
        return stream

    provider.generate = types.MethodType(_qualified_generate, provider)


def _moonshot_terminal_identity(session: Any) -> tuple[str, str]:
    """Return finish reason and response model from the final provider step."""

    try:
        provider = session._cli.soul._runtime.llm.chat_provider
        stream = provider._crucible_last_stream
    except AttributeError as exc:
        raise KimiSDKQualificationError(
            "Kimi runtime did not retain qualified terminal response evidence"
        ) from exc
    finish_reason = getattr(stream, "finish_reason", None)
    served_model = getattr(stream, "served_model", None)
    if not isinstance(finish_reason, str) or not isinstance(served_model, str):
        raise KimiSDKQualificationError(
            "Kimi runtime emitted invalid terminal response evidence"
        )
    return finish_reason, served_model


def _disable_cli_auto_compaction(session: Any) -> None:
    """Prevent CLI 1.12 from making an unaccounted compaction inference."""

    try:
        soul = session._cli.soul
        compact_context = soul.compact_context
    except AttributeError as exc:
        raise KimiSDKQualificationError(
            "Kimi runtime no longer exposes the qualified compaction boundary"
        ) from exc
    soul_type = type(soul)
    if (
        soul_type.__name__ != "KimiSoul"
        or soul_type.__module__ != "kimi_cli.soul.kimisoul"
        or not isinstance(compact_context, types.MethodType)
        or compact_context.__func__.__module__ != "kimi_cli.soul.kimisoul"
        or compact_context.__func__.__qualname__ != "KimiSoul.compact_context"
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime resolved an unqualified compaction boundary"
        )

    async def _reject_hidden_compaction(_soul: Any) -> None:
        raise KimiSDKQualificationError(
            "Kimi automatic context compaction is disabled because it would "
            "perform an unaccounted Moonshot generation"
        )

    soul.compact_context = types.MethodType(_reject_hidden_compaction, soul)
    installed = getattr(soul, "compact_context", None)
    if (
        not isinstance(installed, types.MethodType)
        or installed.__func__ is not _reject_hidden_compaction
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime could not disable automatic context compaction"
        )


def _qualify_moonshot_provider(
    session: Any,
    *,
    max_response_tokens: int,
    model: str = _MODEL,
    reasoning_effort: str | None = None,
    cost_bounds: _K3CostBoundLedger | None = None,
) -> None:
    """Apply explicit K2.7/K3 request policy to CLI 1.12.0's native provider.

    Moonshot K2.7 always thinks, rejects ``reasoning_effort``, and requires
    every historical assistant message to retain ``reasoning_content``.
    CLI 1.12's native provider already preserves the reasoning field and
    Moonshot's top-level cached-token usage, but its thinking capability path
    sends ``reasoning_effort="high"``. The model configuration deliberately
    avoids that path. This version-pinned qualification verifies the exact
    private provider shape, disables hidden HTTP retries and inference-backed
    CLI compaction, and installs the bounded response limit. The native
    factory owns a stable per-session ``prompt_cache_key``.

    K3 also bypasses the CLI effort translation, but explicitly requests
    ``max`` at the HTTP boundary, uses ``max_completion_tokens``, and binds
    implicit caching to five minutes. Neither effort nor cache TTL is inferred
    as served: Moonshot does not echo those settings in the response.
    """

    try:
        provider = session._cli.soul._runtime.llm.chat_provider
    except AttributeError as exc:
        raise KimiSDKQualificationError(
            "Kimi runtime no longer exposes the qualified LLM provider shape"
        ) from exc
    provider_type = type(provider)
    if (
        provider_type.__name__ != "Kimi"
        or provider_type.__module__ != "kosong.chat_provider.kimi"
        or getattr(provider, "stream", None) is not True
        or not hasattr(provider, "client")
        or not hasattr(provider, "_generation_kwargs")
        or not isinstance(provider._generation_kwargs, dict)
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime resolved an unqualified chat provider"
        )
    session_id = getattr(session, "id", None)
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > 256
        or _contains_control_character(session_id)
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime did not expose a bounded session cache identity"
        )
    if provider._generation_kwargs != {"prompt_cache_key": session_id}:
        raise KimiSDKQualificationError(
            "Kimi runtime resolved unqualified Moonshot generation options"
        )
    if getattr(provider, "thinking_effort", object()) is not None:
        raise KimiSDKQualificationError(
            "Kimi runtime attempted to send unsupported reasoning effort"
        )
    client = provider.client
    client_type = type(client)
    if (
        client_type.__name__ != "AsyncOpenAI"
        or client_type.__module__ != "openai"
        or getattr(client, "max_retries", None) != 2
        or not callable(getattr(client, "with_options", None))
    ):
        raise KimiSDKQualificationError(
            "Kimi runtime resolved an unqualified Moonshot HTTP client"
        )
    # Crucible owns the only physical-attempt retry ledger. The OpenAI
    # client's default two hidden HTTP retries could bill accepted-then-lost
    # responses without a trace event, so disable them on the invocation-
    # scoped native provider.
    provider.client = client.with_options(max_retries=0)
    if getattr(provider.client, "max_retries", None) != 0:
        raise KimiSDKQualificationError(
            "Kimi runtime could not disable hidden Moonshot HTTP retries"
        )
    _disable_cli_auto_compaction(session)
    if model not in _MODEL_CONTEXT_SIZES or (
        (model == _MODEL and reasoning_effort is not None)
        or (model == _K3_MODEL and reasoning_effort != "max")
    ):
        raise KimiSDKQualificationError("unqualified Kimi model/effort")
    if model == _K3_MODEL:
        provider._generation_kwargs.update({
            "max_completion_tokens": max_response_tokens,
            "reasoning_effort": reasoning_effort,
            "extra_body": {
                "prompt_cache_options": {"mode": "implicit", "ttl": "5m"},
            },
        })
    else:
        provider._generation_kwargs["max_tokens"] = max_response_tokens
    if cost_bounds is not None and cost_bounds.enabled and model != _K3_MODEL:
        raise KimiSDKQualificationError("Kimi diagnostic bounds require K3")
    _install_moonshot_stream_compat(provider, model=model, cost_bounds=cost_bounds)


def _usage_wire(usage: _UsageAccumulator) -> dict[str, Any]:
    input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
    return {
        "type": "usage",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "usage_valid": usage.valid,
    }


async def _run_sdk_session(
    request: _WorkerRequest,
    paths: _WorkerPaths,
    bindings: _SdkBindings,
    emit: Callable[[dict[str, Any]], None],
    step_ledger: _StepLedger | None = None,
) -> _WorkerOutcome:
    agent_file, _ = _write_agent_definition(request, paths)
    config = _build_sdk_config(bindings, request)
    session: Any = None
    usage = _UsageAccumulator()
    cost_bounds = _K3CostBoundLedger(
        request.diagnostic_cost_upper_bound, request.diagnostic_budget_usd,
        emit,
    )
    text_parts: list[str] = []
    pending_tool_calls: dict[str, str] = {}
    seen_tool_call_ids: set[str] = set()
    approved_tool_calls: set[str] = set()
    tool_events: list[_ToolEvent] = []
    tool_budget_terminal_pending = False
    tool_mutation_terminal_pending = False
    tool_dispatch_terminal_pending = False
    terminal_seen = False
    step_interrupted_seen = False
    ledger = step_ledger or _StepLedger(request.max_steps, emit)
    configured_model: str | None = None
    try:
        session = await bindings.Session.create(
            work_dir=bindings.KaosPath(str(paths.work)),
            config=config,
            model=request.model,
            thinking=True,
            yolo=False,
            agent_file=agent_file,
            mcp_configs=(
                []
                if request.mcp_url is None
                else [
                    {
                        "mcpServers": {
                            "crucible": {
                                "url": request.mcp_url,
                                "transport": "streamable-http",
                                "auth": request.mcp_auth_token,
                            }
                        }
                    }
                ]
            ),
            skills_dir=bindings.KaosPath(str(paths.skills)),
            max_steps_per_turn=request.max_steps,
            max_retries_per_step=1,
            max_ralph_iterations=0,
        )
        configured_model = str(getattr(session, "model_name", "") or "")
        if configured_model != request.model:
            raise KimiSDKQualificationError("Kimi Session resolved an unexpected model")
        status = getattr(session, "status", None)
        if bool(getattr(status, "yolo_enabled", True)):
            raise KimiSDKQualificationError(
                "Kimi Session unexpectedly enabled automatic approval"
            )
        _qualify_moonshot_provider(
            session,
            max_response_tokens=request.max_response_tokens,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            cost_bounds=cost_bounds,
        )

        # Prefixing prevents a user task beginning with "/" from invoking a
        # Kimi CLI slash command instead of reaching the configured model.
        prompt = "Complete the following Crucible task:\n\n" + request.user
        async for wire_message in session.prompt(
            prompt,
            merge_wire_messages=False,
        ):
            if (
                (
                    tool_budget_terminal_pending
                    or tool_mutation_terminal_pending
                    or tool_dispatch_terminal_pending
                )
                and not isinstance(wire_message, bindings.ToolResult)
            ):
                session.cancel()
                if tool_mutation_terminal_pending:
                    raise KimiToolMutationIndeterminateTerminal(
                        "Kimi tool loop stopped after an indeterminate "
                        "source mutation"
                    )
                if tool_dispatch_terminal_pending:
                    raise KimiToolDispatchTerminal(
                        "Kimi tool loop stopped at a workflow-owned terminal "
                        "dispatch boundary"
                    )
                raise KimiToolDispatchBudgetTerminal(
                    "Kimi tool loop stopped because the workflow-owned "
                    "tool dispatch budget is exhausted"
                )
            if isinstance(wire_message, bindings.StepBegin):
                cost_bounds.admit_next()
                # Intermediate tool-calling steps may contain visible prose.
                # Only the final provider step belongs in final_message.
                try:
                    ledger.begin()
                except KimiSDKQualificationError:
                    session.cancel()
                    raise
                text_parts.clear()
            elif isinstance(wire_message, bindings.TextPart):
                text_parts.append(str(wire_message.text))
            elif isinstance(wire_message, bindings.ThinkPart):
                # Reasoning is neither normalized nor emitted over IPC.  The
                # worker's entire Kimi session root is deleted by the parent.
                continue
            elif isinstance(wire_message, bindings.StatusUpdate):
                token_usage = getattr(wire_message, "token_usage", None)
                if token_usage is not None:
                    usage.observe(
                        token_usage,
                        getattr(wire_message, "message_id", None),
                        step=ledger.steps_used,
                    )
                    emit(_usage_wire(usage))
            elif isinstance(wire_message, bindings.ToolCall):
                raw_tool_call_id = getattr(wire_message, "id", None)
                tool_call_id = (
                    raw_tool_call_id if isinstance(raw_tool_call_id, str) else ""
                )
                function = getattr(wire_message, "function", None)
                tool_name = str(getattr(function, "name", "") or "")
                if (
                    _TOOL_CALL_ID_RE.fullmatch(tool_call_id) is None
                    or tool_call_id in seen_tool_call_ids
                    or tool_name not in request.mcp_tool_names
                    or len(tool_events) >= _MAX_TOOL_EVENTS
                ):
                    session.cancel()
                    raise KimiSDKQualificationError(
                        "Kimi requested a tool outside the attached ToolSet"
                    )
                seen_tool_call_ids.add(tool_call_id)
                pending_tool_calls[tool_call_id] = tool_name
                tool_events.append(("tool_call", tool_name, tool_call_id))
                emit(
                    {
                        "type": "tool_event",
                        "kind": "tool_call",
                        "name": tool_name,
                        "call_id": tool_call_id,
                    }
                )
            elif isinstance(wire_message, bindings.ApprovalRequest):
                raw_tool_call_id = getattr(
                    wire_message,
                    "tool_call_id",
                    None,
                )
                tool_call_id = (
                    raw_tool_call_id if isinstance(raw_tool_call_id, str) else ""
                )
                expected_name = pending_tool_calls.get(tool_call_id)
                sender = str(getattr(wire_message, "sender", "") or "")
                action = str(getattr(wire_message, "action", "") or "")
                if (
                    expected_name is not None
                    and tool_call_id not in approved_tool_calls
                    and sender == expected_name
                    and action == f"mcp:{expected_name}"
                ):
                    approved_tool_calls.add(tool_call_id)
                    wire_message.resolve("approve")
                    continue
                wire_message.resolve("reject")
                session.cancel()
                raise KimiSDKQualificationError(
                    "Kimi requested an unknown or mismatched tool approval"
                )
            elif isinstance(wire_message, bindings.ToolResult):
                raw_tool_call_id = getattr(
                    wire_message,
                    "tool_call_id",
                    None,
                )
                tool_call_id = (
                    raw_tool_call_id if isinstance(raw_tool_call_id, str) else ""
                )
                result_tool_name = pending_tool_calls.get(tool_call_id)
                if (
                    result_tool_name is None
                    or tool_call_id not in approved_tool_calls
                    or len(tool_events) >= _MAX_TOOL_EVENTS
                ):
                    session.cancel()
                    raise KimiSDKQualificationError(
                        "Kimi emitted an unmatched tool result"
                    )
                pending_tool_calls.pop(tool_call_id)
                approved_tool_calls.remove(tool_call_id)
                tool_events.append(("tool_result", result_tool_name, tool_call_id))
                emit(
                    {
                        "type": "tool_event",
                        "kind": "tool_result",
                        "name": result_tool_name,
                        "call_id": tool_call_id,
                    }
                )
                terminal_kind = _tool_dispatch_terminal_kind(wire_message)
                if terminal_kind == "budget":
                    tool_budget_terminal_pending = True
                if terminal_kind == "mutation":
                    tool_mutation_terminal_pending = True
                if terminal_kind == "other":
                    tool_dispatch_terminal_pending = True
                if (
                    (
                        tool_budget_terminal_pending
                        or tool_mutation_terminal_pending
                        or tool_dispatch_terminal_pending
                    )
                    and not pending_tool_calls
                ):
                    session.cancel()
                    if tool_mutation_terminal_pending:
                        raise KimiToolMutationIndeterminateTerminal(
                            "Kimi tool loop stopped after an indeterminate "
                            "source mutation"
                        )
                    if tool_dispatch_terminal_pending:
                        raise KimiToolDispatchTerminal(
                            "Kimi tool loop stopped at a workflow-owned "
                            "terminal dispatch boundary"
                        )
                    raise KimiToolDispatchBudgetTerminal(
                        "Kimi tool loop stopped because the workflow-owned "
                        "tool dispatch budget is exhausted"
                    )
            elif type(wire_message).__name__ == "ToolCallRequest":
                session.cancel()
                raise KimiSDKQualificationError(
                    "Kimi requested an unqualified external tool call"
                )
            elif isinstance(wire_message, bindings.StepInterrupted):
                # The pinned CLI emits this marker in an ``except`` block and
                # then re-raises the underlying provider exception. Do not
                # mask that exception before the worker classifier sees it.
                step_interrupted_seen = True
                continue
            elif isinstance(wire_message, bindings.TurnEnd):
                terminal_seen = True

        if not terminal_seen:
            detail = " after StepInterrupted" if step_interrupted_seen else ""
            raise KimiSDKQualificationError(
                f"Kimi session stream ended without TurnEnd{detail}"
            )
        if pending_tool_calls:
            raise KimiSDKQualificationError(
                "Kimi session ended with incomplete tool calls"
            )
        if ledger.steps_used == 0:
            raise KimiSDKQualificationError(
                "Kimi session reached TurnEnd without StepBegin evidence"
            )
        finish_reason, served_model = _moonshot_terminal_identity(session)
        input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
        final_message = _extract_final_text("".join(text_parts))
        if finish_reason == "length":
            return _WorkerOutcome(
                final_message=final_message,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                usage_seen=usage.seen,
                usage_valid=usage.valid,
                terminal_seen=True,
                configured_model=configured_model,
                served_model=served_model,
                stop_reason="max_tokens",
                error_kind="max_tokens",
                error_message="Moonshot reached its response-token limit",
                retriable=False,
                tool_events=tuple(tool_events),
                steps_used=ledger.steps_used,
                cost_bound_segments=tuple(cost_bounds.records),
            )
        if finish_reason == "content_filter":
            return _WorkerOutcome(
                final_message="",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                usage_seen=usage.seen,
                usage_valid=usage.valid,
                terminal_seen=True,
                configured_model=configured_model,
                served_model=served_model,
                stop_reason="content_filter",
                error_kind="content_rejected",
                error_message="Moonshot rejected the generated content",
                retriable=False,
                tool_events=tuple(tool_events),
                steps_used=ledger.steps_used,
                cost_bound_segments=tuple(cost_bounds.records),
            )
        if finish_reason != "stop":
            raise KimiSDKQualificationError(
                "Kimi session ended under an unqualified finish reason"
            )
        return _WorkerOutcome(
            final_message=final_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            usage_seen=usage.seen,
            usage_valid=usage.valid,
            terminal_seen=True,
            configured_model=configured_model,
            served_model=served_model,
            stop_reason="end_turn",
            tool_events=tuple(tool_events),
            steps_used=ledger.steps_used,
            cost_bound_segments=tuple(cost_bounds.records),
        )
    except _K3CostBoundBudget:
        records = tuple(cost_bounds.records)
        return _WorkerOutcome(
            input_tokens=sum(r["input_tokens"] for r in records),
            output_tokens=sum(r["output_tokens"] for r in records),
            cache_creation_tokens=sum(r["cache_creation_tokens"] for r in records),
            cache_read_tokens=sum(r["cache_read_tokens"] or 0 for r in records),
            usage_seen=bool(records), usage_valid=bool(records),
            terminal_seen=False, configured_model=configured_model,
            served_model=request.model, stop_reason="cost_budget",
            error_kind="cost_budget",
            error_message="Kimi diagnostic upper-bound budget exhausted",
            tool_events=tuple(tool_events), steps_used=ledger.steps_used,
            cost_bound_segments=records,
        )
    except bindings.MaxStepsReached as exc:
        # Kimi raises MaxStepsReached on the next loop boundary after the
        # configured number of provider steps have completed.  The generic
        # process-boundary classifier cannot retain the final stream's served
        # model identity, which would make otherwise exact per-step pricing
        # evidence incomplete.  Preserve that identity while the qualified
        # session and its final stream are still available, but only for the
        # exact exhausted-tool-loop shape.
        if (
            session is None
            or configured_model != request.model
            or ledger.steps_used != request.max_steps
            or ledger.steps_used <= 0
            or terminal_seen
            or step_interrupted_seen
            or pending_tool_calls
            or approved_tool_calls
            or tool_budget_terminal_pending
            or tool_mutation_terminal_pending
            or tool_dispatch_terminal_pending
            or not usage.seen
            or not usage.valid
        ):
            raise KimiSDKQualificationError(
                "Kimi reached its step limit without an exact settled loop boundary"
            ) from exc
        finish_reason, served_model = _moonshot_terminal_identity(session)
        if finish_reason != "tool_calls" or served_model != request.model:
            raise KimiSDKQualificationError(
                "Kimi step-limit terminal lacked exact served-model tool-loop evidence"
            ) from exc
        input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
        return _WorkerOutcome(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation,
            cache_read_tokens=cache_read,
            usage_seen=usage.seen,
            usage_valid=usage.valid,
            terminal_seen=False,
            configured_model=configured_model,
            served_model=served_model,
            stop_reason="max_turns",
            error_kind="max_turns",
            error_message="Kimi reached its configured logical step limit",
            retriable=False,
            tool_events=tuple(tool_events),
            steps_used=ledger.steps_used,
            cost_bound_segments=tuple(cost_bounds.records),
        )
    finally:
        if session is not None:
            await session.close()


def _redact_text(message: str, *tokens: str | None) -> str:
    for token in tokens:
        if token:
            message = message.replace(token, "[REDACTED]")
    message = _BEARER_RE.sub("Bearer [REDACTED]", message)
    message = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        message,
    )
    if len(message) > 1_000:
        message = message[:1_000] + "...[truncated]"
    return message


def _redact_error_message(
    exc: BaseException,
    *tokens: str | None,
) -> str:
    return _redact_text(str(exc), *tokens)


def _classify_worker_exception(
    exc: BaseException,
    request: _WorkerRequest,
    bindings: _SdkBindings | None,
    *,
    steps_used: int = 0,
) -> _WorkerOutcome:
    name = type(exc).__name__
    message = _redact_error_message(
        exc,
        request.auth_token,
        request.mcp_auth_token,
    )
    if isinstance(exc, KimiSDKUnavailable):
        kind, retriable = "unavailable", False
    elif isinstance(exc, KimiToolMutationIndeterminateTerminal):
        kind, retriable = "tool_mutation", False
    elif isinstance(exc, KimiToolDispatchTerminal):
        kind, retriable = "tool_terminal", False
    elif isinstance(exc, KimiToolDispatchBudgetTerminal):
        kind, retriable = "tool_budget", False
    elif isinstance(exc, KimiSDKQualificationError):
        kind, retriable = "qualification", False
    elif isinstance(exc, asyncio.CancelledError) or (
        bindings is not None and isinstance(exc, bindings.RunCancelled)
    ):
        kind, retriable = "cancellation", True
    elif bindings is not None and isinstance(exc, bindings.MaxStepsReached):
        kind, retriable = "max_turns", False
    elif bindings is not None and isinstance(exc, bindings.APIStatusError):
        status = getattr(exc, "status_code", None)
        status = int(status) if status is not None else None
        return _WorkerOutcome(
            stop_reason="error",
            error_kind="transport",
            error_message=f"{name}: {message}",
            http_status=status,
            retriable=status in RETRIABLE_HTTP_STATUSES,
            steps_used=steps_used,
        )
    elif bindings is not None and isinstance(
        exc,
        (
            bindings.APIConnectionError,
            bindings.APITimeoutError,
            bindings.APIEmptyResponseError,
            bindings.ChatProviderError,
        ),
    ):
        kind, retriable = "transport", True
    elif "refusal" in message.lower():
        kind, retriable = "content_rejected", False
    else:
        # Do not serialize a traceback from the credential-bearing worker.
        # Unknown SDK/programming failures are terminal qualification faults.
        kind, retriable = "worker_internal", False
        message = f"{name}: worker failed without a qualified error shape"
    return _WorkerOutcome(
        stop_reason="error",
        error_kind=kind,
        error_message=f"{name}: {message}",
        retriable=retriable,
        steps_used=steps_used,
    )


def _worker_entry(
    send_connection: Any,
    request: _WorkerRequest,
    root_text: str,
) -> None:
    bindings: _SdkBindings | None = None

    def emit(payload: dict[str, Any]) -> None:
        send_connection.send(payload)

    step_ledger = _StepLedger(request.max_steps, emit)

    try:
        paths = _isolate_worker_environment(Path(root_text))
        bindings = _load_sdk_bindings()
        outcome = asyncio.run(
            _run_sdk_session(
                request,
                paths,
                bindings,
                emit,
                step_ledger,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - process boundary
        outcome = _classify_worker_exception(
            exc,
            request,
            bindings,
            steps_used=step_ledger.steps_used,
        )
    try:
        emit(outcome.to_wire())
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        send_connection.close()


def _make_private_temp_root() -> Path:
    candidates = [Path("/dev/shm"), Path(tempfile.gettempdir())]
    for parent in candidates:
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            continue
        try:
            root = Path(tempfile.mkdtemp(prefix=".crucible-kimi-", dir=parent))
        except OSError:
            continue
        root.chmod(0o700)
        return root
    raise KimiSDKQualificationError(
        "no writable private temporary directory is available for Kimi"
    )


def _stop_process(process: Any) -> None:
    stop_lock = getattr(process, "_crucible_stop_lock", None)
    if stop_lock is None:
        stop_lock = threading.RLock()
        with contextlib.suppress(AttributeError):
            process._crucible_stop_lock = stop_lock
    with stop_lock:
        if getattr(process, "_crucible_worker_settled", False):
            return
        try:
            _stop_process_locked(process)
        except KimiWorkerLifecycleError:
            raise
        except OSError as exc:
            raise KimiWorkerLifecycleError(
                "isolated Kimi worker settlement hit an OS lifecycle error"
            ) from exc
        retained_root = getattr(process, "_crucible_private_root", None)
        if retained_root is not None:
            try:
                _remove_private_root(Path(retained_root))
            except BaseException as exc:  # noqa: BLE001 - credential cleanup
                raise KimiWorkerLifecycleError(
                    "isolated Kimi worker credential root cleanup failed"
                ) from exc
            with contextlib.suppress(AttributeError):
                del process._crucible_private_root
        with contextlib.suppress(AttributeError):
            process._crucible_worker_settled = True


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        raise KimiWorkerLifecycleError(
            "could not inspect the isolated Kimi worker process group"
        ) from exc
    return True


def _signal_process_group(pgid: int, signal_number: int) -> bool:
    try:
        os.killpg(pgid, signal_number)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise KimiWorkerLifecycleError(
            "could not signal the isolated Kimi worker process group"
        ) from exc
    return True


def _stop_process_locked(process: Any) -> None:
    if hasattr(process, "poll"):
        pgid = int(process.pid)
        if _process_group_exists(pgid):
            if not _signal_process_group(pgid, signal.SIGTERM):
                if process.poll() is None:
                    try:
                        process.terminate()
                    except OSError as exc:
                        raise KimiWorkerLifecycleError(
                            "could not terminate the isolated Kimi worker"
                        ) from exc
        elif process.poll() is None:
            try:
                process.terminate()
            except OSError as exc:
                raise KimiWorkerLifecycleError(
                    "could not terminate the isolated Kimi worker"
                ) from exc
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=_PROCESS_STOP_SECONDS)
        if _process_group_exists(pgid):
            _signal_process_group(pgid, signal.SIGKILL)
        elif process.poll() is None:
            try:
                process.kill()
            except OSError as exc:
                raise KimiWorkerLifecycleError(
                    "could not kill the isolated Kimi worker"
                ) from exc
        try:
            process.wait(timeout=_PROCESS_STOP_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise KimiWorkerLifecycleError(
                "isolated Kimi worker could not be reaped after SIGKILL"
            ) from exc
        group_deadline = time.monotonic() + _PROCESS_STOP_SECONDS
        while _process_group_exists(pgid):
            if time.monotonic() >= group_deadline:
                raise KimiWorkerLifecycleError(
                    "isolated Kimi worker process group survived SIGKILL"
                )
            time.sleep(0.01)
        return
    try:
        alive = process.is_alive()
    except (AssertionError, ValueError):
        return
    if not alive:
        with contextlib.suppress(AssertionError, ValueError):
            process.join(timeout=0)
        return
    process.terminate()
    process.join(timeout=_PROCESS_STOP_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(timeout=_PROCESS_STOP_SECONDS)
    if process.is_alive():
        raise KimiWorkerLifecycleError(
            "isolated Kimi worker could not be reaped after kill"
        )


def _remove_private_root(root: Path) -> None:
    try:
        shutil.rmtree(root)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise KimiSDKQualificationError(
            f"failed to remove private Kimi worker root {root}"
        ) from exc
    if root.exists():
        raise KimiSDKQualificationError(
            f"private Kimi worker root survived cleanup: {root}"
        )


def _run_worker_process(
    request: _WorkerRequest,
    timeout_seconds: float,
    *,
    process_context: Any = None,
    worker_target: Callable[..., None] = _worker_entry,
    on_process_start: Callable[[Any], None] | None = None,
    on_process_stop: Callable[[Any], None] | None = None,
) -> _WorkerOutcome:
    context = process_context or multiprocessing.get_context("spawn")
    root = _make_private_temp_root()
    receive_connection: Any = None
    send_connection: Any = None
    process: Any = None
    latest_usage = _WorkerOutcome()
    outcome: _WorkerOutcome | None = None
    cleanup_error: BaseException | None = None
    try:
        receive_connection, send_connection = context.Pipe(duplex=False)
        process = context.Process(
            target=worker_target,
            args=(send_connection, request, str(root)),
            daemon=True,
            name="crucible-kimi-sdk",
        )
        # Bind the credential root to the exact process owner before start.
        # If physical settlement fails, KimiBackend retains this process and
        # its retry can remove the same root only after the worker stops.
        with contextlib.suppress(AttributeError):
            process._crucible_private_root = root
        try:
            process.start()
        except (OSError, RuntimeError) as exc:
            return _WorkerOutcome(
                stop_reason="error",
                error_kind="worker_start",
                error_message=(
                    f"failed to start isolated Kimi SDK worker ({type(exc).__name__})"
                ),
                retriable=False,
            )
        send_connection.close()
        send_connection = None
        if on_process_start is not None:
            on_process_start(process)

        deadline = time.monotonic() + timeout_seconds
        pipe_eof = False
        while outcome is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_for = min(0.2, remaining)
            if pipe_eof:
                # EOF can precede process exit. Reap within the same deadline
                # without repeatedly polling a permanently readable pipe.
                process.join(timeout=wait_for)
            elif receive_connection.poll(wait_for):
                try:
                    payload = receive_connection.recv()
                except EOFError:
                    pipe_eof = True
                    continue
                if not isinstance(payload, Mapping):
                    outcome = replace(
                        latest_usage,
                        usage_valid=False,
                        stop_reason="error",
                        error_kind="qualification",
                        error_message=(
                            "Kimi worker emitted a non-object protocol record"
                        ),
                        retriable=False,
                    )
                    break
                if request.diagnostic_cost_upper_bound:
                    latest_usage, outcome = _consume_worker_line(
                        json.dumps({"protocol": _WORKER_PROTOCOL, **payload}).encode(),
                        latest_usage=latest_usage, max_steps=request.max_steps,
                        model=request.model, diagnostic_cost_upper_bound=True,
                        diagnostic_budget_usd=request.diagnostic_budget_usd,
                    )
                    continue
                record_type = payload.get("type")
                if record_type == "cost_bound":
                    try:
                        latest_usage = _admit_cost_bound_checkpoint(
                            latest_usage, payload, model=request.model,
                            diagnostic_cost_upper_bound=request.diagnostic_cost_upper_bound,
                        )
                    except (ValueError, TypeError):
                        outcome = replace(
                            latest_usage, usage_valid=False, stop_reason="error",
                            error_kind="qualification",
                            error_message="Kimi worker emitted malformed cost-bound evidence",
                            retriable=False,
                        )
                elif record_type == "usage":
                    counters = (
                        payload.get("input_tokens"),
                        payload.get("output_tokens"),
                        payload.get("cache_creation_tokens"),
                        payload.get("cache_read_tokens"),
                    )
                    if (
                        set(payload)
                        != {
                            "type",
                            "input_tokens",
                            "output_tokens",
                            "cache_creation_tokens",
                            "cache_read_tokens",
                            "usage_valid",
                        }
                        or type(payload.get("usage_valid")) is not bool
                        or not _usage_counters_valid(*counters)
                        or not _usage_checkpoint_valid(
                            latest_usage,
                            counters,
                            payload.get("usage_valid"),
                        )
                    ):
                        outcome = replace(
                            latest_usage,
                            usage_valid=False,
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted malformed usage evidence"
                            ),
                            retriable=False,
                        )
                        break
                    latest_usage = _settle_usage_checkpoint(
                        latest_usage,
                        cast(tuple[int, int, int, int], counters),
                        usage_valid=cast(bool, payload["usage_valid"]),
                        model=request.model,
                    )
                elif record_type == "step_progress":
                    steps_used = payload.get("steps_used")
                    if (
                        set(payload) != {"type", "steps_used"}
                        or type(steps_used) is not int
                        or steps_used != latest_usage.steps_used + 1
                        or steps_used > request.max_steps
                    ):
                        outcome = replace(
                            latest_usage,
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted malformed logical-step progress"
                            ),
                            retriable=False,
                        )
                        break
                    latest_usage = replace(
                        latest_usage,
                        steps_used=steps_used,
                    )
                elif record_type == "tool_event":
                    if (
                        set(payload) != {"type", "kind", "name", "call_id"}
                        or latest_usage.steps_used == 0
                    ):
                        outcome = replace(
                            latest_usage,
                            usage_valid=False,
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted a malformed tool event"
                            ),
                            retriable=False,
                        )
                        break
                    try:
                        event = _parse_tool_events([payload])[0]
                    except (TypeError, ValueError):
                        outcome = _WorkerOutcome(
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted a malformed tool event"
                            ),
                            retriable=False,
                            tool_events=latest_usage.tool_events,
                        )
                        break
                    if len(latest_usage.tool_events) >= _MAX_TOOL_EVENTS:
                        outcome = _WorkerOutcome(
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=("Kimi worker exceeded the tool event limit"),
                            retriable=False,
                            tool_events=latest_usage.tool_events,
                        )
                        break
                    latest_usage = replace(
                        latest_usage,
                        tool_events=latest_usage.tool_events + (event,),
                    )
                elif record_type == "terminal":
                    try:
                        outcome = _WorkerOutcome.from_wire(
                            payload, model=request.model,
                            diagnostic_cost_upper_bound=request.diagnostic_cost_upper_bound,
                        )
                    except (TypeError, ValueError):
                        outcome = replace(
                            latest_usage,
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted a malformed terminal payload"
                            ),
                            retriable=False,
                        )
                    break
                else:
                    outcome = replace(
                        latest_usage,
                        usage_valid=False,
                        stop_reason="error",
                        error_kind="qualification",
                        error_message=(
                            "Kimi worker emitted an unknown protocol record type"
                        ),
                        retriable=False,
                    )
                    break
            if not process.is_alive():
                # Drain a terminal message already buffered at process exit.
                if not pipe_eof and receive_connection.poll(0):
                    continue
                break

        if outcome is None and process.is_alive():
            _stop_process(process)
            outcome = replace(
                latest_usage,
                stop_reason="timeout",
                error_kind="timeout",
                error_message=(
                    f"Kimi worker exceeded {timeout_seconds:g}s wall deadline"
                ),
                retriable=False,
            )
        elif outcome is None:
            outcome = replace(
                latest_usage,
                stop_reason="error",
                error_kind="worker_exit",
                error_message=(
                    "Kimi worker exited without a terminal message "
                    f"(exit code {process.exitcode})"
                ),
                retriable=False,
            )
        else:
            outcome = _merge_incremental_outcome(latest_usage, outcome)
            process.join(timeout=_WORKER_DRAIN_SECONDS)
            if not process.is_alive() and receive_connection.poll(0):
                outcome = replace(
                    outcome,
                    final_message="",
                    usage_valid=False,
                    terminal_seen=False,
                    configured_model=None,
                    served_model=None,
                    stop_reason="error",
                    error_kind="qualification",
                    error_message=(
                        "Kimi worker emitted protocol data after its terminal record"
                    ),
                    retriable=False,
                )
            if process.is_alive():
                _stop_process(process)
                outcome = replace(
                    outcome,
                    stop_reason="error",
                    error_kind="lifecycle",
                    error_message=(
                        "Kimi worker did not exit after closing its Session"
                    ),
                    retriable=False,
                )
        return outcome
    finally:
        if process is not None:
            try:
                _stop_process(process)
            except BaseException as exc:  # noqa: BLE001 - retain exact owner
                cleanup_error = exc
            else:
                if on_process_stop is not None:
                    try:
                        on_process_stop(process)
                    except BaseException as exc:  # noqa: BLE001 - retain owner
                        cleanup_error = exc
                if cleanup_error is None:
                    with contextlib.suppress(AttributeError, ValueError):
                        process.close()
        if receive_connection is not None:
            receive_connection.close()
        if send_connection is not None:
            send_connection.close()
        if process is None:
            try:
                _remove_private_root(root)
            except BaseException as exc:  # noqa: BLE001 - security cleanup
                cleanup_error = exc
        if cleanup_error is not None:
            raise KimiWorkerLifecycleError(
                "isolated Kimi worker lifecycle did not settle"
            ) from cleanup_error


def _subprocess_request_wire(
    request: _WorkerRequest,
    root: Path,
) -> dict[str, Any]:
    return {
        "protocol": _WORKER_PROTOCOL,
        "type": "invoke",
        "root": str(root),
        "system": request.system,
        "user": request.user,
        "model": request.model,
        "reasoning_effort": request.reasoning_effort,
        "diagnostic_cost_upper_bound": request.diagnostic_cost_upper_bound,
        "diagnostic_budget_usd": request.diagnostic_budget_usd,
        "schema_json": request.schema_json,
        "base_url": request.base_url,
        "auth_token": request.auth_token,
        "max_steps": request.max_steps,
        "max_response_tokens": request.max_response_tokens,
        "mcp_url": request.mcp_url,
        "mcp_auth_token": request.mcp_auth_token,
        "mcp_tool_names": list(request.mcp_tool_names),
    }


def _worker_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, value in pairs:
        if name in payload:
            raise ValueError("duplicate worker protocol field")
        payload[name] = value
    return payload


def _consume_worker_line(
    raw_line: bytes,
    *,
    latest_usage: _WorkerOutcome,
    max_steps: int,
    model: str = _MODEL,
    diagnostic_cost_upper_bound: bool = False,
    diagnostic_budget_usd: float | None = None,
) -> tuple[_WorkerOutcome, _WorkerOutcome | None]:
    """Parse one bounded protocol line from the isolated worker."""

    try:
        payload = json.loads(
            raw_line.decode("utf-8"),
            object_pairs_hook=_worker_json_object,
        )
    except (UnicodeDecodeError, ValueError):
        return latest_usage, _WorkerOutcome(
            stop_reason="error",
            error_kind="qualification",
            error_message="Kimi worker emitted a non-JSON protocol line",
            retriable=False,
        )
    if not isinstance(payload, Mapping):
        return latest_usage, replace(
            latest_usage,
            usage_valid=False,
            stop_reason="error",
            error_kind="qualification",
            error_message="Kimi worker emitted a non-object protocol record",
            retriable=False,
        )
    if payload.get("protocol") != _WORKER_PROTOCOL:
        return latest_usage, _WorkerOutcome(
            stop_reason="error",
            error_kind="qualification",
            error_message="Kimi worker emitted an unexpected protocol version",
            retriable=False,
        )
    record_type = payload.get("type")
    if record_type == "cost_bound":
        try:
            return _admit_cost_bound_checkpoint(
                latest_usage,
                {key: value for key, value in payload.items() if key != "protocol"},
                model=model, diagnostic_cost_upper_bound=diagnostic_cost_upper_bound,
            ), None
        except (ValueError, TypeError):
            return latest_usage, replace(
                latest_usage, usage_valid=False, stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker emitted malformed cost-bound evidence",
                retriable=False,
            )
    if diagnostic_cost_upper_bound and record_type in {"tool_event", "step_progress"}:
        has_prior_step = latest_usage.steps_used > 0
        exhausted = diagnostic_budget_usd is None or math.fsum(
            _cost_bound_amounts(record)[1] for record in latest_usage.cost_bound_segments
        ) > diagnostic_budget_usd
        if has_prior_step and (
            len(latest_usage.cost_bound_segments) != latest_usage.steps_used
            or exhausted
        ):
            return latest_usage, replace(
                latest_usage, usage_valid=False, stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker crossed an unsettled diagnostic cost boundary",
                retriable=False,
            )
    if record_type == "usage":
        expected_fields = {
            "protocol",
            "type",
            "input_tokens",
            "output_tokens",
            "cache_creation_tokens",
            "cache_read_tokens",
            "usage_valid",
        }
        counters = (
            payload.get("input_tokens"),
            payload.get("output_tokens"),
            payload.get("cache_creation_tokens"),
            payload.get("cache_read_tokens"),
        )
        if (
            set(payload) != expected_fields
            or type(payload.get("usage_valid")) is not bool
            or not _usage_counters_valid(*counters)
            or not _usage_checkpoint_valid(
                latest_usage,
                counters,
                payload.get("usage_valid"),
            )
        ):
            return latest_usage, replace(
                latest_usage,
                usage_valid=False,
                stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker emitted malformed usage evidence",
                retriable=False,
            )
        return (
            _settle_usage_checkpoint(
                latest_usage,
                cast(tuple[int, int, int, int], counters),
                usage_valid=cast(bool, payload["usage_valid"]),
                model=model,
            ),
            None,
        )
    if record_type == "step_progress":
        steps_used = payload.get("steps_used")
        if (
            set(payload) != {"protocol", "type", "steps_used"}
            or
            type(steps_used) is not int
            or steps_used != latest_usage.steps_used + 1
            or steps_used > max_steps
        ):
            return latest_usage, replace(
                latest_usage,
                stop_reason="error",
                error_kind="qualification",
                error_message=("Kimi worker emitted malformed logical-step progress"),
                retriable=False,
            )
        return replace(latest_usage, steps_used=steps_used), None
    if record_type == "tool_event":
        if set(payload) != {
            "protocol",
            "type",
            "kind",
            "name",
            "call_id",
        } or latest_usage.steps_used == 0:
            return latest_usage, replace(
                latest_usage,
                usage_valid=False,
                stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker emitted a malformed tool event",
                retriable=False,
            )
        try:
            event = _parse_tool_events([payload])[0]
        except (TypeError, ValueError):
            return latest_usage, _WorkerOutcome(
                stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker emitted a malformed tool event",
                retriable=False,
                tool_events=latest_usage.tool_events,
            )
        if len(latest_usage.tool_events) >= _MAX_TOOL_EVENTS:
            return latest_usage, _WorkerOutcome(
                stop_reason="error",
                error_kind="qualification",
                error_message="Kimi worker exceeded the tool event limit",
                retriable=False,
                tool_events=latest_usage.tool_events,
            )
        return (
            replace(
                latest_usage,
                tool_events=latest_usage.tool_events + (event,),
            ),
            None,
        )
    if record_type == "terminal":
        try:
            return latest_usage, _WorkerOutcome.from_wire(
                payload, model=model,
                diagnostic_cost_upper_bound=diagnostic_cost_upper_bound,
            )
        except (TypeError, ValueError):
            return latest_usage, replace(
                latest_usage,
                stop_reason="error",
                error_kind="qualification",
                error_message=("Kimi worker emitted a malformed terminal payload"),
                retriable=False,
            )
    return latest_usage, replace(
        latest_usage,
        usage_valid=False,
        stop_reason="error",
        error_kind="qualification",
        error_message="Kimi worker emitted an unknown protocol record type",
        retriable=False,
    )


def _drain_protocol_after_terminal(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    stdout_buffer: bytearray,
    stderr_tail: bytearray,
) -> tuple[bool, bool]:
    """Drain through child settlement and reject every post-terminal byte."""

    trailing_protocol = bool(stdout_buffer)
    stdout_buffer.clear()
    deadline = time.monotonic() + _WORKER_DRAIN_SECONDS
    stdout_open = any(key.data == "stdout" for key in selector.get_map().values())
    while (
        time.monotonic() < deadline
        and (process.poll() is None or stdout_open)
    ):
        ready = selector.select(min(0.05, max(0.0, deadline - time.monotonic())))
        if not ready:
            continue
        for key, _mask in ready:
            try:
                chunk = os.read(key.fd, 65_536)
            except OSError:
                chunk = b""
            if not chunk:
                with contextlib.suppress(KeyError):
                    selector.unregister(key.fileobj)
                if key.data == "stdout":
                    stdout_open = False
                continue
            if key.data == "stdout":
                trailing_protocol = True
            else:
                stderr_tail.extend(chunk)
                if len(stderr_tail) > _MAX_WORKER_STDERR_BYTES:
                    del stderr_tail[:-_MAX_WORKER_STDERR_BYTES]
    return trailing_protocol, process.poll() is not None and not stdout_open


def _run_isolated_worker_subprocess(
    request: _WorkerRequest,
    timeout_seconds: float,
    *,
    worker_python: Path,
    worker_pin: _PinnedKimiWorkerRuntime | None = None,
    process_context: Any = None,
    on_process_start: Callable[[Any], None] | None = None,
    on_process_stop: Callable[[Any], None] | None = None,
) -> _WorkerOutcome:
    """Run one SDK turn in the independently locked worker environment."""

    del process_context
    try:
        if worker_pin is not None:
            worker_pin.revalidate()
            _qualify_worker_runtime(
                str(worker_python),
                worker_pin.runtime_fingerprint,
            )
        else:
            _qualify_worker_runtime(str(worker_python))
    except (KimiSDKUnavailable, KimiSDKQualificationError) as exc:
        return _WorkerOutcome(
            stop_reason="error",
            error_kind=(
                "unavailable"
                if isinstance(exc, KimiSDKUnavailable)
                else "qualification"
            ),
            error_message=str(exc),
            retriable=False,
        )

    root = _make_private_temp_root()
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    latest_usage = _WorkerOutcome()
    outcome: _WorkerOutcome | None = None
    stdout_buffer = bytearray()
    stderr_tail = bytearray()
    cleanup_error: BaseException | None = None
    try:
        try:
            if worker_pin is not None:
                # This is the final identity check before the kernel resolves
                # the venv interpreter for the paid transport.
                worker_pin.revalidate()
            process = subprocess.Popen(
                [
                    str(worker_python),
                    "-B",
                    "-I",
                    "-m",
                    "crucible_kimi_worker",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_worker_environment(),
                start_new_session=True,
                bufsize=0,
            )
            process._crucible_stop_lock = threading.RLock()  # type: ignore[attr-defined]
            process._crucible_private_root = root  # type: ignore[attr-defined]
            process._crucible_worker_settled = False  # type: ignore[attr-defined]
        except OSError as exc:
            return _WorkerOutcome(
                stop_reason="error",
                error_kind="worker_start",
                error_message=(
                    f"failed to start isolated Kimi SDK worker ({type(exc).__name__})"
                ),
                retriable=False,
            )
        if on_process_start is not None:
            on_process_start(process)

        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        request_bytes = (
            json.dumps(
                _subprocess_request_wire(request, root),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        try:
            process.stdin.write(request_bytes)
            process.stdin.close()
        except (BrokenPipeError, OSError):
            with contextlib.suppress(OSError):
                process.stdin.close()

        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        open_streams = 2
        deadline = time.monotonic() + timeout_seconds
        while outcome is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            for key, _mask in selector.select(min(0.2, remaining)):
                try:
                    chunk = os.read(key.fd, 65_536)
                except OSError:
                    chunk = b""
                if not chunk:
                    with contextlib.suppress(KeyError):
                        selector.unregister(key.fileobj)
                    open_streams -= 1
                    continue
                if key.data == "stderr":
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > _MAX_WORKER_STDERR_BYTES:
                        del stderr_tail[:-_MAX_WORKER_STDERR_BYTES]
                    continue
                stdout_buffer.extend(chunk)
                if len(stdout_buffer) > _MAX_WORKER_LINE_BYTES:
                    outcome = _WorkerOutcome(
                        stop_reason="error",
                        error_kind="qualification",
                        error_message=("Kimi worker exceeded the protocol line limit"),
                        retriable=False,
                    )
                    break
                while b"\n" in stdout_buffer:
                    raw_line, _, remainder = stdout_buffer.partition(b"\n")
                    stdout_buffer[:] = remainder
                    if not raw_line.strip():
                        outcome = replace(
                            latest_usage,
                            usage_valid=False,
                            stop_reason="error",
                            error_kind="qualification",
                            error_message=(
                                "Kimi worker emitted a blank protocol record"
                            ),
                            retriable=False,
                        )
                        break
                    latest_usage, terminal = _consume_worker_line(
                        bytes(raw_line),
                        latest_usage=latest_usage,
                        max_steps=request.max_steps,
                        model=request.model,
                        diagnostic_cost_upper_bound=request.diagnostic_cost_upper_bound,
                        diagnostic_budget_usd=request.diagnostic_budget_usd,
                    )
                    if terminal is not None:
                        outcome = terminal
                        break
                if outcome is not None:
                    break
            if process.poll() is not None and open_streams <= 0:
                break

        if outcome is None and process.poll() is None:
            _stop_process(process)
            outcome = replace(
                latest_usage,
                stop_reason="timeout",
                error_kind="timeout",
                error_message=(
                    f"Kimi worker exceeded {timeout_seconds:g}s wall deadline"
                ),
                retriable=False,
            )
        elif outcome is None:
            outcome = replace(
                latest_usage,
                stop_reason="error",
                error_kind="worker_exit",
                error_message=(
                    "Kimi worker exited without a terminal message "
                    f"(exit code {process.returncode}); private stderr "
                    f"captured {len(stderr_tail)} bytes"
                ),
                retriable=False,
            )
        else:
            outcome = _merge_incremental_outcome(latest_usage, outcome)
            trailing_protocol, worker_settled = _drain_protocol_after_terminal(
                process,
                selector,
                stdout_buffer,
                stderr_tail,
            )
            if not worker_settled:
                _stop_process(process)
            if trailing_protocol:
                outcome = replace(
                    outcome,
                    final_message="",
                    usage_valid=False,
                    terminal_seen=False,
                    configured_model=None,
                    served_model=None,
                    stop_reason="error",
                    error_kind="qualification",
                    error_message=(
                        "Kimi worker emitted protocol data after its terminal record"
                    ),
                    retriable=False,
                )
            elif not worker_settled:
                outcome = replace(
                    outcome,
                    final_message="",
                    terminal_seen=False,
                    configured_model=None,
                    served_model=None,
                    stop_reason="error",
                    error_kind="lifecycle",
                    error_message=(
                        "Kimi worker did not exit after closing its Session"
                    ),
                    retriable=False,
                )
            elif process.returncode != 0:
                outcome = replace(
                    outcome,
                    final_message="",
                    terminal_seen=False,
                    configured_model=None,
                    served_model=None,
                    stop_reason="error",
                    error_kind="lifecycle",
                    error_message=(
                        "Kimi worker returned a terminal payload but "
                        f"exited with code {process.returncode}"
                    ),
                    retriable=False,
                )
        return outcome
    finally:
        if selector is not None:
            selector.close()
        if process is not None:
            try:
                _stop_process(process)
            except BaseException as exc:  # noqa: BLE001 - security cleanup
                cleanup_error = exc
            else:
                if on_process_stop is not None:
                    on_process_stop(process)
                for stream in (
                    process.stdin,
                    process.stdout,
                    process.stderr,
                ):
                    if stream is not None:
                        with contextlib.suppress(OSError):
                            stream.close()
        stderr_tail.clear()
        if process is None:
            try:
                _remove_private_root(root)
            except BaseException as exc:  # noqa: BLE001 - security cleanup
                cleanup_error = exc
        if cleanup_error is not None:
            raise KimiWorkerLifecycleError(
                "isolated Kimi worker lifecycle did not settle"
            ) from cleanup_error


class KimiBackend:
    """Confined Assay backend over the official Kimi Agent SDK Session."""

    name = "kimi"

    @property
    def features(self) -> Any:
        """Return the observed profile for this exact dedicated SDK route."""

        from .features import feature_profile_for_route

        return feature_profile_for_route(
            self.name,
            model=self.model,
            base_url=self._expected_base_url,
        )

    def __init__(
        self,
        *,
        model: str = _MODEL,
        effort: str | None = None,
        cwd: str | None = None,
        log_dir: Path | str | None = None,
        capability: str = "text",
        expected_base_url: str | None = None,
        max_response_tokens: int = _DEFAULT_MAX_RESPONSE_TOKENS,
        diagnostic_cost_upper_bound: bool = False,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        worker_python: Path | str | None = None,
        worker_source: str = "environment",
        expected_worker_runtime_fingerprint: str | None = None,
        _worker_runner: Callable[..., _WorkerOutcome] | None = None,
        _process_context: Any = None,
    ) -> None:
        from .capabilities import normalize_capability

        self.capability = normalize_capability(capability)
        if model not in _MODEL_CONTEXT_SIZES:
            raise ValueError(
                "KimiBackend supports only the qualified models "
                f"{tuple(_MODEL_CONTEXT_SIZES)!r}"
            )
        if (model == _MODEL and effort is not None) or (
            model == _K3_MODEL and effort not in {None, "max"}
        ):
            raise ValueError("KimiBackend received an unqualified model/effort")
        self.model = model
        self.effort = "max" if model == _K3_MODEL else None
        if type(diagnostic_cost_upper_bound) is not bool or (
            diagnostic_cost_upper_bound and model != _K3_MODEL
        ):
            raise ValueError("diagnostic_cost_upper_bound requires exact K3/max")
        self.diagnostic_cost_upper_bound = diagnostic_cost_upper_bound
        self._max_context_size = _MODEL_CONTEXT_SIZES[model]
        if (
            type(max_response_tokens) is not int
            or not 1 <= max_response_tokens <= self._max_context_size
        ):
            raise ValueError(
                "max_response_tokens must be an integer from 1 through "
                f"{self._max_context_size}"
            )
        # Route-owned and fixed for ordinary invocations. A caller cannot
        # silently widen this bound through the Invocation payload.
        self._max_response_tokens = max_response_tokens
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        # Routing may give a slow tool-using role a larger closed default.
        # Per-invocation policy may tighten, but never widen, this route-owned
        # bound.
        self._timeout_seconds = float(timeout_seconds)
        normalized_default = _normalize_base_url(
            _BASE_URL,
            setting="KimiBackend default_base_url",
        )
        self._expected_base_url = _normalize_base_url(
            expected_base_url or _BASE_URL,
            setting="KimiBackend expected_base_url",
        )
        expected_host = urlsplit(self._expected_base_url).hostname
        if (
            self._expected_base_url != normalized_default
            and expected_host != "127.0.0.1"
        ):
            raise KimiSDKQualificationError(
                "KimiBackend accepts only the qualified Moonshot endpoint"
            )
        if type(worker_source) is not str or worker_source not in _WORKER_SOURCES:
            raise ValueError(
                "worker_source must be one of "
                f"{sorted(_WORKER_SOURCES)}"
            )
        if expected_worker_runtime_fingerprint is not None and (
            type(expected_worker_runtime_fingerprint) is not str
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                expected_worker_runtime_fingerprint,
            )
            is None
        ):
            raise ValueError(
                "expected_worker_runtime_fingerprint must be "
                "sha256:<64 lowercase hex>"
            )
        self._worker_source = worker_source
        self._worker_pin: _PinnedKimiWorkerRuntime | None
        if worker_source == KIMI_PROJECT_LOCKED_WORKER_SOURCE:
            if worker_python is not None:
                raise KimiSDKQualificationError(
                    "project-locked Kimi worker rejects an explicit "
                    "worker_python override"
                )
            if os.environ.get("CRUCIBLE_KIMI_WORKER_PYTHON", "").strip():
                raise KimiSDKQualificationError(
                    "project-locked Kimi worker rejects ambient "
                    "CRUCIBLE_KIMI_WORKER_PYTHON"
                )
            self._worker_python = _resolve_worker_python(_DEFAULT_WORKER_PYTHON)
            self._worker_pin = _PinnedKimiWorkerRuntime.capture(
                self._worker_python
            )
            if (
                expected_worker_runtime_fingerprint is not None
                and self._worker_pin.runtime_fingerprint
                != expected_worker_runtime_fingerprint
            ):
                raise KimiSDKQualificationError(
                    "project-locked Kimi worker runtime does not match the "
                    "configured expected fingerprint"
                )
        else:
            if expected_worker_runtime_fingerprint is not None:
                raise KimiSDKQualificationError(
                    "expected_worker_runtime_fingerprint requires "
                    "worker_source='project_locked'"
                )
            self._worker_python = _resolve_worker_python(worker_python)
            self._worker_pin = None
        # Retained for route diagnostics only.  The SDK worker intentionally
        # receives an empty work directory so ambient project instructions
        # and provider-native files cannot expand its authority.
        self._cwd = cwd or os.getcwd()
        self._sdk_sink = SdkDebugSink(
            Path(log_dir) if log_dir is not None else None,
            backend_name=self.name,
        )
        self._worker_runner = _worker_runner or functools.partial(
            _run_isolated_worker_subprocess,
            worker_python=self._worker_python,
            worker_pin=self._worker_pin,
        )
        self._process_context = _process_context
        self._state_lock = threading.RLock()
        self._invoke_lock = threading.RLock()
        self._active_processes: set[Any] = set()
        self._active_mcp_servers: set[Any] = set()
        self._active_mutation_hosts: set[Any] = set()
        self._mcp_toolset: Any = None
        self._mcp_tool_names: tuple[str, ...] = ()
        self._mcp_write_names: frozenset[str] = frozenset()
        self._mcp_context_budget: int | None = None
        self._close_event = threading.Event()
        self._closed = False

    @property
    def worker_runtime_fingerprint(self) -> str | None:
        """Return the secret-free project-locked execution identity."""

        if self._worker_pin is None:
            return None
        return self._worker_pin.runtime_fingerprint

    @property
    def max_response_tokens(self) -> int:
        """Return the immutable route-owned response-token bound."""

        return self._max_response_tokens

    def attach_toolset(
        self,
        toolset: Any,
        *,
        context_budget: int | None = None,
    ) -> None:
        """Attach one capability-classified ToolSet for the next turns."""

        from .capabilities import validate_toolset_capability

        if context_budget is not None and (
            isinstance(context_budget, bool)
            or not isinstance(context_budget, int)
            or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        validate_toolset_capability(self.capability, toolset)
        names: tuple[str, ...] = ()
        writes: frozenset[str] = frozenset()
        if toolset is not None:
            names, writes = _tool_inventory(toolset)

        # Reattachment is serialized with invocation so a worker can never
        # observe a URL from one ToolSet and an allowlist from another.
        with self._invoke_lock, self._state_lock:
            if self._closed:
                raise RuntimeError("KimiBackend is closed")
            self._mcp_toolset = toolset
            self._mcp_tool_names = names
            self._mcp_write_names = writes
            self._mcp_context_budget = context_budget if toolset is not None else None

    def chat_with_tools(
        self,
        invocation: Invocation,
        toolset: Any,
        *,
        cache: Any = None,
        max_rounds: int = 10,
        max_response_tokens: int | None = None,
        retry_budget: int = 3,
        strict: bool = False,
    ) -> InvocationResult:
        """Run Kimi's SDK-owned loop over the confined MCP ToolSet."""

        if invocation.conversation is not None:
            raise NotImplementedError(
                "KimiBackend does not support provider conversation "
                "continuity"
            )
        if toolset is None:
            if strict:
                raise NotImplementedError(
                    "KimiBackend.chat_with_tools called with no toolset"
                )
            return self.invoke(invocation)
        if cache is not None:
            raise NotImplementedError("KimiBackend does not support ToolCallCache")
        for name, value in (
            ("max_rounds", max_rounds),
            ("retry_budget", retry_budget),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        selected_max_response_tokens = (
            self._max_response_tokens
            if max_response_tokens is None
            else max_response_tokens
        )
        if (
            type(selected_max_response_tokens) is not int
            or not 1 <= selected_max_response_tokens <= self._max_context_size
        ):
            raise ValueError(
                "max_response_tokens must be an integer from 1 through "
                f"{self._max_context_size}"
            )
        if selected_max_response_tokens > self._max_response_tokens:
            raise ValueError(
                "max_response_tokens cannot exceed the route-owned bound "
                f"{self._max_response_tokens}"
            )
        self.attach_toolset(toolset)
        policy = invocation.policy or _DEFAULT_POLICY
        bounded = replace(
            invocation,
            max_turns=min(invocation.max_turns, max_rounds),
            policy=replace(
                policy,
                retry_budget=min(policy.retry_budget, retry_budget),
            ),
        )
        from .pricing import pricing_for_model

        pricing_for_model(self.model)
        with self._invoke_lock:
            return self._invoke_serialized(
                bounded,
                max_response_tokens=selected_max_response_tokens,
            )

    def invoke(self, invocation: Invocation) -> InvocationResult:
        if self.diagnostic_cost_upper_bound and (
            type(invocation.budget_usd) not in {int, float}
            or not math.isfinite(invocation.budget_usd)
            or invocation.budget_usd <= 0
        ):
            raise ValueError("Kimi diagnostic cost bounds require a finite positive budget")
        if invocation.conversation is not None:
            raise NotImplementedError(
                "KimiBackend does not support provider conversation "
                "continuity"
            )
        from .pricing import pricing_for_model

        pricing_for_model(self.model)
        with self._invoke_lock:
            return self._invoke_serialized(
                invocation,
                max_response_tokens=self._max_response_tokens,
            )

    def invoke_with_physical_telemetry(
        self,
        invocation: Invocation,
    ) -> tuple[InvocationResult, PhysicalUsageEvidence]:
        """Invoke and return usage copied from admitted worker checkpoints."""

        with capture_physical_usage("kimi") as collector:
            result = self.invoke(invocation)
        return result, collector.freeze()

    def _invoke_serialized(
        self,
        invocation: Invocation,
        *,
        max_response_tokens: int,
    ) -> InvocationResult:
        if invocation.tools:
            raise ValueError("KimiBackend accepts tools only through attach_toolset")
        if invocation.max_turns < 1:
            raise ValueError("KimiBackend invocation.max_turns must be >= 1")
        with self._state_lock:
            if self._closed:
                raise RuntimeError("KimiBackend is closed")
            toolset = self._mcp_toolset
            tool_names = self._mcp_tool_names
            write_names = self._mcp_write_names
            context_budget = self._mcp_context_budget
        max_steps = invocation.max_turns if toolset is not None else 1
        if max_steps > _MAX_TOOL_STEPS:
            raise ValueError(f"KimiBackend max_turns cannot exceed {_MAX_TOOL_STEPS}")
        mcp_toolset: _FrozenMcpToolset | None = None
        mutation_spec: Any = None
        tool_authority: str | None = None
        if toolset is not None:
            from .tool_authority import snapshot_tool_authority

            authority_snapshot = snapshot_tool_authority(toolset)
            current_names = tuple(sorted(authority_snapshot.tool_names))
            current_writes = authority_snapshot.write_names
            if current_names != tool_names or current_writes != write_names:
                raise ValueError(
                    "Kimi ToolSet catalogue or classification changed after attachment"
                )
            mcp_toolset = _FrozenMcpToolset(
                toolset,
                names=tool_names,
                write_names=write_names,
                authority_snapshot=authority_snapshot,
            )
            tool_authority = _tool_authority_digest(
                mcp_toolset,
                capability=self.capability,
                context_budget=context_budget,
            )
            if write_names:
                from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
                    MutationHostQualificationError,
                    build_isolated_toolset_spec,
                )

                try:
                    mutation_spec = build_isolated_toolset_spec(
                        toolset,
                        expected_write_names=write_names,
                        authority_snapshot=authority_snapshot,
                    )
                except MutationHostQualificationError as exc:
                    raise KimiSDKQualificationError(str(exc)) from exc

        base_url, token = _validate_runtime_credentials()
        normalized_base_url = _normalize_base_url(
            base_url,
            setting="KIMI_BASE_URL",
        )
        if normalized_base_url != self._expected_base_url:
            raise KimiSDKQualificationError(
                "runtime KIMI_BASE_URL does not match the configured "
                "Kimi route endpoint"
            )
        schema_json = _schema_json(invocation.response_schema)
        policy = invocation.policy or _DEFAULT_POLICY
        timeout_seconds = (
            self._timeout_seconds
            if policy.timeout_seconds is None
            else min(self._timeout_seconds, float(policy.timeout_seconds))
        )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("KimiBackend timeout_seconds must be finite and positive")
        attempts = max(1, policy.retry_budget)
        events: list[BackendExecutionEvent] = []
        logical_deadline = time.monotonic() + timeout_seconds
        logical_steps_remaining = invocation.max_turns
        previous_outcome: _WorkerOutcome | None = None
        prior_usage = Usage()
        prior_pricing_segments: tuple[UsagePricingSegment, ...] = ()
        prior_pricing_complete = True
        if mcp_toolset is not None and context_budget is not None:
            from crucible.tools._mcp_bridge import (  # noqa: PLC0415
                ContextBudgetTracker,
            )

            context_budget_tracker = ContextBudgetTracker(context_budget)
        else:
            context_budget_tracker = None

        def deadline_result() -> InvocationResult:
            if not events or events[-1].kind != "limit":
                events.append(
                    BackendExecutionEvent(
                        sequence=len(events),
                        kind="limit",
                        name="deadline",
                    )
                )
            timed_out = replace(
                previous_outcome or _WorkerOutcome(),
                final_message="",
                terminal_seen=False,
                stop_reason="timeout",
                error_kind="timeout",
                error_message=(
                    "Kimi invocation exhausted its cumulative "
                    f"{timeout_seconds:g}s wall deadline"
                ),
                retriable=False,
            )
            return replace(
                self._result_from_outcome(
                    invocation,
                    policy,
                    timed_out,
                    base_url=base_url,
                    events=events,
                    tool_authority_digest=tool_authority,
                ),
                # ``timed_out`` retains the last outcome's identity/evidence,
                # but ``prior_usage`` already contains every completed
                # physical attempt. Never add the last attempt twice.
                usage=prior_usage,
                pricing_evidence=UsagePricingEvidence(
                    segments=prior_pricing_segments,
                    complete=False,
                ),
            )

        def logical_result(result: InvocationResult) -> InvocationResult:
            usage = _usage_sum(prior_usage, result.usage)
            result_pricing = result.pricing_evidence or UsagePricingEvidence(
                segments=(),
                complete=False,
            )
            error = result.error
            if (
                invocation.budget_usd is not None
                and usage.budget_cost_usd > invocation.budget_usd
                and (
                    error is None
                    or isinstance(error, BudgetExhausted)
                )
            ):
                error = BudgetExhausted(
                    spent_usd=usage.budget_cost_usd,
                    limit_usd=invocation.budget_usd,
                    message=(
                        f"Logical invocation budget charge ${usage.budget_cost_usd:.4f} "
                        f"exceeding budget ${invocation.budget_usd:.4f}"
                    ),
                )
            return replace(
                result,
                usage=usage,
                error=error,
                pricing_evidence=UsagePricingEvidence(
                    segments=(
                        *prior_pricing_segments,
                        *result_pricing.segments,
                    ),
                    complete=(
                        prior_pricing_complete
                        and result_pricing.complete
                    ),
                ),
            )

        for attempt in range(attempts):
            remaining = logical_deadline - time.monotonic()
            if remaining <= 0:
                return deadline_result()
            if logical_steps_remaining <= 0:
                raise AssertionError(
                    "Kimi physical retry started without a logical step budget"
                )
            attempt_step_limit = min(max_steps, logical_steps_remaining)
            attempt_id = f"kimi-attempt-{attempt + 1}"
            events.append(
                BackendExecutionEvent(
                    sequence=len(events),
                    kind="request",
                    child_id=attempt_id,
                )
            )
            started_iso = datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            )
            start_mono = time.monotonic()
            mcp_server: Any = None
            mcp_auth_token: str | None = None
            mutation_host: Any = None
            attempt_mutation_started = False
            attempt_started_writes: tuple[str, ...] = ()
            cleanup_error: BaseException | None = None
            late_terminal_error: BaseException | None = None
            provider_error: BaseException | None = None
            try:
                mcp_url: str | None = None
                if mcp_toolset is not None:
                    attempt_mcp_toolset: Any = mcp_toolset
                    if mutation_spec is not None:
                        from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
                            IsolatedMutationHost,
                            ProcessHostedToolSet,
                        )

                        mutation_host = IsolatedMutationHost(
                            mutation_spec,
                            invocation_deadline=logical_deadline,
                        )
                        self._register_mutation_host(mutation_host)
                        try:
                            mutation_host.start()
                        except BaseException:
                            try:
                                mutation_host.stop()
                            except BaseException:
                                # Keep a failed-to-settle host registered so
                                # close() can retry the exact process group.
                                raise
                            else:
                                self._unregister_mutation_host(mutation_host)
                                mutation_host = None
                            raise
                        attempt_mcp_toolset = ProcessHostedToolSet(
                            mcp_toolset,
                            mutation_host,
                        )
                    from crucible.tools._mcp_http_server import (  # noqa: PLC0415
                        CrucibleHttpMcpServer,
                    )

                    mcp_auth_token = secrets.token_urlsafe(_MCP_TOKEN_BYTES)
                    mcp_server = CrucibleHttpMcpServer(
                        attempt_mcp_toolset,
                        auth_token=mcp_auth_token,
                        context_budget_tracker=context_budget_tracker,
                        invocation_deadline=logical_deadline,
                        start_timeout=min(10.0, remaining),
                    )
                    self._register_mcp_server(mcp_server)
                    try:
                        mcp_url = mcp_server.start()
                    except BaseException:
                        try:
                            mcp_server.stop()
                        except BaseException:
                            # Retain ownership so close() can retry a listener
                            # whose physical thread did not settle.
                            raise
                        else:
                            self._unregister_mcp_server(mcp_server)
                            mcp_server = None
                        raise
                request = _WorkerRequest(
                    system=invocation.system,
                    user=invocation.user,
                    model=self.model,
                    reasoning_effort=self.effort,
                    diagnostic_cost_upper_bound=self.diagnostic_cost_upper_bound,
                    diagnostic_budget_usd=(
                        invocation.budget_usd if self.diagnostic_cost_upper_bound else None
                    ),
                    schema_json=schema_json,
                    base_url=base_url,
                    auth_token=token,
                    max_steps=attempt_step_limit,
                    max_response_tokens=max_response_tokens,
                    mcp_url=mcp_url,
                    mcp_auth_token=mcp_auth_token,
                    mcp_tool_names=tool_names,
                )
                try:
                    remaining = logical_deadline - time.monotonic()
                    if remaining <= 0:
                        outcome = _WorkerOutcome(
                            stop_reason="timeout",
                            error_kind="timeout",
                            error_message=(
                                "Kimi invocation deadline expired during "
                                "confined MCP bridge startup"
                            ),
                            retriable=False,
                        )
                    else:
                        outcome = self._worker_runner(
                            request,
                            remaining,
                            process_context=self._process_context,
                            on_process_start=self._register_process,
                            on_process_stop=self._unregister_process,
                        )
                except KimiWorkerLifecycleError:
                    raise
                except KimiSDKQualificationError as exc:
                    outcome = _WorkerOutcome(
                        stop_reason="error",
                        error_kind="qualification",
                        error_message=_redact_error_message(
                            exc,
                            token,
                            mcp_auth_token,
                        ),
                        retriable=False,
                    )
            except KimiWorkerLifecycleError as exc:
                provider_error = exc
            except KimiSDKQualificationError as exc:
                outcome = _WorkerOutcome(
                    stop_reason="error",
                    error_kind="qualification",
                    error_message=_redact_error_message(
                        exc,
                        token,
                        mcp_auth_token,
                    ),
                    retriable=False,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                outcome = _WorkerOutcome(
                    stop_reason="error",
                    error_kind="qualification",
                    error_message=(
                        "failed to start the confined Kimi MCP bridge "
                        f"({type(exc).__name__})"
                    ),
                    retriable=False,
                )
            except BaseException as exc:
                provider_error = exc
            finally:
                server_settled = True
                if mcp_server is not None:
                    try:
                        mcp_server.stop()
                    except BaseException as exc:  # fail closed on listener leak
                        cleanup_error = cleanup_error or exc
                        server_settled = False
                    else:
                        self._unregister_mcp_server(mcp_server)
                    observed_terminal = getattr(
                        mcp_server,
                        "terminal_dispatch_error",
                        None,
                    )
                    if isinstance(observed_terminal, BaseException):
                        late_terminal_error = observed_terminal
                if mutation_host is not None and server_settled:
                    attempt_mutation_started = (
                        mutation_host.ever_started_write
                    )
                    attempt_started_writes = (
                        mutation_host.started_write_names
                    )
                    try:
                        mutation_host.stop()
                    except BaseException as exc:  # fail closed after reap
                        cleanup_error = cleanup_error or exc
                    else:
                        self._unregister_mutation_host(mutation_host)
            if late_terminal_error is not None:
                from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                    ToolMutationIndeterminate,
                )

                if isinstance(
                    late_terminal_error,
                    ToolMutationIndeterminate,
                ):
                    if (
                        provider_error is not None
                        and provider_error is not late_terminal_error
                    ):
                        late_terminal_error.add_note(
                            "Kimi provider execution also raised "
                            f"{type(provider_error).__name__}; the "
                            "indeterminate mutation remains primary."
                        )
                    if cleanup_error is not None:
                        late_terminal_error.add_note(
                            "Kimi retained invocation-scoped cleanup "
                            "authority; KimiBackend.close() must settle the "
                            "exact registered server and mutation host before "
                            "journal rollback."
                        )
                        if cleanup_error is late_terminal_error:
                            raise late_terminal_error
                        raise late_terminal_error from cleanup_error
                    if provider_error is not None:
                        if provider_error is late_terminal_error:
                            raise late_terminal_error
                        raise late_terminal_error from provider_error
            if cleanup_error is not None:
                if provider_error is not None:
                    cleanup_error.add_note(
                        "Kimi provider execution also raised "
                        f"{type(provider_error).__name__}."
                    )
                raise KimiSDKQualificationError(
                    "confined Kimi mutation/MCP lifecycle did not settle "
                    f"({type(cleanup_error).__name__})"
                ) from cleanup_error
            if provider_error is not None:
                raise provider_error
            if late_terminal_error is not None:
                from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                    ToolDispatchBudgetExceeded,
                    ToolMutationIndeterminate,
                )

                mutation_indeterminate = isinstance(
                    late_terminal_error,
                    ToolMutationIndeterminate,
                )
                budget_exhausted = isinstance(
                    late_terminal_error,
                    ToolDispatchBudgetExceeded,
                )
                outcome = replace(
                    outcome,
                    final_message="",
                    stop_reason=(
                        "tool_mutation_indeterminate"
                        if mutation_indeterminate
                        else (
                            "tool_budget_exhausted"
                            if budget_exhausted
                            else "tool_dispatch_terminal"
                        )
                    ),
                    error_kind=(
                        "tool_mutation"
                        if mutation_indeterminate
                        else (
                            "tool_budget"
                            if budget_exhausted
                            else "tool_terminal"
                        )
                    ),
                    error_message=_redact_text(
                        (
                            "Kimi tool loop terminated after an "
                            "indeterminate source mutation: "
                            f"{late_terminal_error}"
                            if mutation_indeterminate
                            else (
                                "Kimi tool loop terminated after a "
                                "workflow-owned tool-dispatch budget "
                                f"denial: {late_terminal_error}"
                                if budget_exhausted
                                else (
                                    "Kimi tool loop terminated at a "
                                    "workflow-owned dispatch boundary: "
                                    f"{late_terminal_error}"
                                )
                            )
                        ),
                        token,
                        mcp_auth_token,
                    ),
                    retriable=False,
                )
            outcome = _qualify_worker_model_identity(outcome, model=self.model)
            outcome = _qualify_worker_tool_events(
                outcome,
                allowed_names=tool_names,
            )
            traced_writes = tuple(
                name
                for kind, name, _call_id in outcome.tool_events
                if kind == "tool_call" and name in write_names
            )
            host_write_counts = Counter(attempt_started_writes)
            trace_write_counts = Counter(traced_writes)
            missing_host_write_trace = any(
                host_write_counts[name] > trace_write_counts[name]
                for name in host_write_counts
            )
            if (
                missing_host_write_trace
                and outcome.error_kind is None
                and outcome.terminal_seen
            ):
                outcome = replace(
                    outcome,
                    final_message="",
                    terminal_seen=False,
                    stop_reason="error",
                    error_kind="qualification",
                    error_message=(
                        "Kimi worker mutation trace does not match isolated "
                        "host execution evidence"
                    ),
                    retriable=False,
                    host_mutation_trace_complete=False,
                )
            elif missing_host_write_trace:
                outcome = replace(
                    outcome,
                    host_mutation_trace_complete=False,
                )
            outcome = _qualify_worker_steps(
                outcome,
                physical_limit=attempt_step_limit,
            )
            emit_physical_usage_attempt(
                backend="kimi",
                source="kimi_worker.status_update.usage_checkpoint_delta.v1",
                complete=(
                    outcome.usage_seen
                    and outcome.usage_valid
                    and outcome.steps_used > 0
                    and outcome.settled_usage_steps == outcome.steps_used
                    and len(outcome.physical_usage_responses)
                    == outcome.settled_usage_steps
                ),
                responses=outcome.physical_usage_responses,
            )
            logical_steps_remaining -= outcome.steps_used
            outcome = replace(
                outcome,
                error_message=_redact_text(
                    outcome.error_message,
                    token,
                    mcp_auth_token,
                ),
            )
            previous_outcome = outcome
            for kind, name, call_id in outcome.tool_events:
                events.append(
                    BackendExecutionEvent(
                        sequence=len(events),
                        kind=kind,
                        name=name,
                        parent_id=attempt_id,
                        child_id=call_id,
                    )
                )
            with self._state_lock:
                was_closed = self._closed
            if was_closed:
                outcome = replace(
                    outcome,
                    stop_reason="cancelled",
                    error_kind="cancellation",
                    error_message="Kimi worker cancelled by backend close",
                    retriable=True,
                )

            result = self._result_from_outcome(
                invocation,
                policy,
                outcome,
                base_url=base_url,
                events=events,
                tool_authority_digest=tool_authority,
            )
            mutation_started = attempt_mutation_started
            result_error = result.error
            suppressed_error: TransportError | ExternalCancellation | None = None
            if (
                mutation_started
                and isinstance(result_error, TransportError)
                and result_error.retriable
            ):
                suppressed_error = replace(
                    result_error,
                    message=(
                        f"{result_error.message}; automatic retry "
                        "suppressed after a mutating tool call"
                    ),
                    retriable=False,
                )
            elif (
                mutation_started
                and isinstance(result_error, ExternalCancellation)
                and result_error.retriable
            ):
                suppressed_error = replace(
                    result_error,
                    message=(
                        f"{result_error.message}; automatic retry "
                        "suppressed after a mutating tool call"
                    ),
                    retriable=False,
                )
            if suppressed_error is not None:
                result = replace(
                    result,
                    error=cast(Any, suppressed_error),
                )
            if (
                result.error is not None
                and getattr(result.error, "retriable", False)
                and logical_steps_remaining <= 0
            ):
                result = replace(
                    result,
                    error=cast(
                        Any,
                        replace(
                            result.error,
                            message=(
                                f"{result.error.message}; automatic retry "
                                "suppressed because the logical step budget "
                                "is exhausted"
                            ),
                            retriable=False,
                        ),
                    ),
                )
            if (
                result.error is not None
                and getattr(result.error, "retriable", False)
                and attempt + 1 < attempts
                and not self._close_event.is_set()
            ):
                # The pinned SDK reports one aggregate envelope per worker
                # attempt, while repeated physical requests have no qualified
                # attempt-level billing/trace identity. Starting a second paid
                # request would make the invocation's accounting ambiguous.
                # Preserve the first attempt's known lower bound and fail
                # closed before backoff or transport.
                result = replace(
                    result,
                    error=cast(
                        Any,
                        replace(
                            result.error,
                            message=(
                                f"{result.error.message}; automatic retry "
                                "suppressed because aggregate accounting "
                                "across Kimi physical attempts is not "
                                "qualified"
                            ),
                            retriable=False,
                        ),
                    ),
                )
            self._sdk_sink.dump(
                invocation=invocation,
                result=result,
                started_iso=started_iso,
                duration_s=time.monotonic() - start_mono,
                attempt=attempt,
            )
            if (
                result.error is None
                or not getattr(result.error, "retriable", False)
                or attempt + 1 >= attempts
                or self._close_event.is_set()
            ):
                return logical_result(result)
            if outcome.usage_seen and outcome.usage_valid:
                events.append(
                    BackendExecutionEvent(
                        sequence=len(events),
                        kind="usage",
                        parent_id=attempt_id,
                    )
                )
            prior_usage = _usage_sum(prior_usage, result.usage)
            result_pricing = result.pricing_evidence or UsagePricingEvidence(
                segments=(),
                complete=False,
            )
            prior_pricing_segments = (
                *prior_pricing_segments,
                *result_pricing.segments,
            )
            prior_pricing_complete = (
                prior_pricing_complete
                and result_pricing.complete
            )
            events.append(
                BackendExecutionEvent(
                    sequence=len(events),
                    kind="retry",
                    name=getattr(result.error, "kind", "transport"),
                )
            )
            delay = min(2**attempt, 16) + (time.monotonic() % 1.0)
            remaining = logical_deadline - time.monotonic()
            if remaining <= 0:
                return deadline_result()
            logger.info(
                "KimiBackend transient error on attempt %d; retrying in %.1fs",
                attempt + 1,
                min(delay, remaining),
            )
            self._close_event.wait(min(delay, remaining))

        raise AssertionError("KimiBackend retry loop exited without a result")

    def _result_from_outcome(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        outcome: _WorkerOutcome,
        *,
        base_url: str,
        events: list[BackendExecutionEvent],
        tool_authority_digest: str | None,
    ) -> InvocationResult:
        bounds = None
        if self.diagnostic_cost_upper_bound:
            bounds = UsageCostBounds(
                segments=tuple(
                    UsageCostBoundSegment(
                        **record,
                        lower_bound_usd=_cost_bound_amounts(record)[0],
                        upper_bound_usd=_cost_bound_amounts(record)[1],
                    )
                    for record in _parse_cost_bound_records(outcome.cost_bound_segments)
                ),
                coverage_complete=(
                    outcome.usage_seen and outcome.usage_valid
                    and outcome.steps_used > 0
                    and len(outcome.cost_bound_segments) == outcome.steps_used
                    and outcome.settled_usage_steps == outcome.steps_used
                    and outcome.configured_model == self.model
                    and outcome.served_model == self.model
                    and not any(event.kind == "retry" for event in events)
                ),
            )
        usage = Usage(
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            cost_usd=compute_cost(
                self.model,
                input_tokens=outcome.input_tokens,
                output_tokens=outcome.output_tokens,
                cache_creation_tokens=outcome.cache_creation_tokens,
                cache_read_tokens=outcome.cache_read_tokens,
            ),
            cache_creation_tokens=outcome.cache_creation_tokens,
            cache_read_tokens=outcome.cache_read_tokens,
            cost_bounds=bounds,
        )
        if bounds is not None:
            usage = replace(usage, cost_usd=bounds.lower_bound_usd)
            if bounds.coverage_complete and not bounds.covers(usage):
                bounds = replace(bounds, coverage_complete=False)
                usage = replace(usage, cost_bounds=bounds)
        transcript: tuple[Message, ...] = (
            Message(role="user", content=invocation.user),
        )
        if outcome.final_message:
            transcript += (Message(role="assistant", content=outcome.final_message),)

        error: Any = None
        stop_reason = outcome.stop_reason
        if outcome.error_kind == "cost_budget" and bounds is not None:
            stop_reason = "budget_exhausted"
            error = BudgetExhausted(
                spent_usd=bounds.upper_bound_usd,
                limit_usd=invocation.budget_usd or 0.0,
                message="Kimi diagnostic conservative cost bound exhausted the budget",
            )
        elif outcome.error_kind == "cancellation":
            if policy.on_cancellation == "raise":
                raise asyncio.CancelledError(outcome.error_message)
            error = ExternalCancellation(
                cause="KimiWorker",
                message=outcome.error_message,
                retriable=True,
            )
        elif outcome.error_kind == "content_rejected":
            error = ContentRejected(
                message=outcome.error_message,
                retriable=False,
            )
        elif outcome.error_kind == "max_turns":
            stop_reason = "max_turns"
            if policy.on_max_turns == "raise":
                error = MaxTurnsExceeded(
                    turns_used=invocation.max_turns,
                    turns_limit=invocation.max_turns,
                    message=outcome.error_message,
                )
        elif outcome.error_kind == "tool_mutation":
            stop_reason = "tool_mutation_indeterminate"
            error = TransportError(
                http_status=None,
                message=outcome.error_message,
                retriable=False,
            )
        elif outcome.error_kind == "tool_terminal":
            stop_reason = "tool_dispatch_terminal"
            error = TransportError(
                http_status=None,
                message=outcome.error_message,
                retriable=False,
            )
        elif outcome.error_kind is not None:
            error = TransportError(
                http_status=outcome.http_status,
                message=outcome.error_message,
                retriable=outcome.retriable,
            )
        elif not outcome.final_message:
            error = TransportError(
                message=(
                    "Kimi SDK returned a terminal but content-free visible completion"
                ),
                retriable=True,
            )
        elif (
            invocation.budget_usd is not None and usage.budget_cost_usd > invocation.budget_usd
        ):
            error = BudgetExhausted(
                spent_usd=usage.budget_cost_usd,
                limit_usd=invocation.budget_usd,
                message=(
                    f"Invocation budget charge ${usage.budget_cost_usd:.4f} exceeding "
                    f"budget ${invocation.budget_usd:.4f}"
                ),
            )

        trace_events = list(events)
        if outcome.usage_seen and outcome.usage_valid:
            trace_events.append(
                BackendExecutionEvent(
                    sequence=len(trace_events),
                    kind="usage",
                )
            )
        trace_events.append(
            BackendExecutionEvent(
                sequence=len(trace_events),
                kind="terminal",
                name=stop_reason,
            )
        )
        from .features import feature_profile_digest

        retry_seen = any(event.kind == "retry" for event in events)
        # Terminal state independently closes only the conversation lifecycle;
        # its usage totals come from the same StatusUpdate accumulator as the
        # ordered checkpoints.  Token/cost evidence is exact only when every
        # begun provider step reached the parent's usage checkpoint.  If a
        # worker terminates with one begun-but-unsettled step, retain the
        # numeric lower bound while revoking token/cost completeness.
        usage_complete = (
            outcome.usage_seen
            and outcome.usage_valid
            and outcome.steps_used > 0
            and outcome.settled_usage_steps == outcome.steps_used
            and not retry_seen
            and (
                bounds is None
                or (
                    bounds.covers(usage)
                    and all(segment.cache_read_tokens is not None for segment in bounds.segments)
                )
            )
        )
        pricing_complete = (
            usage_complete
            and outcome.configured_model == self.model
            and outcome.served_model == self.model
            and sum(
                segment.input_tokens for segment in outcome.pricing_segments
            )
            == usage.input_tokens
            and sum(
                segment.output_tokens for segment in outcome.pricing_segments
            )
            == usage.output_tokens
            and sum(
                segment.cache_creation_tokens
                for segment in outcome.pricing_segments
            )
            == usage.cache_creation_tokens
            and sum(
                segment.cache_read_tokens
                for segment in outcome.pricing_segments
            )
            == usage.cache_read_tokens
            and math.isclose(
                math.fsum(
                    segment.cost_usd
                    for segment in outcome.pricing_segments
                ),
                usage.cost_usd,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )

        return InvocationResult(
            final_message=outcome.final_message,
            transcript=transcript,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            completeness=ExecutionCompleteness(
                # A failed physical attempt may have been accepted/billed
                # before its usage envelope was lost. The successful retry
                # cannot make aggregate usage/cost evidence complete.
                token_usage_complete=usage_complete,
                cost_complete=usage_complete,
                terminal_complete=outcome.terminal_seen,
                trace_complete=(
                    _tool_trace_complete(trace_events)
                    and not retry_seen
                    and outcome.error_message != _INVALID_TOOL_TRACE_ERROR
                    and outcome.host_mutation_trace_complete
                ),
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=outcome.served_model,
                # Record requested effort, never invent an unreturned echo.
                requested_effort=self.effort,
                served_effort=None,
                endpoint=base_url,
                feature_profile_digest=feature_profile_digest(self.features),
                tool_authority_digest=tool_authority_digest,
                events=tuple(trace_events),
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=outcome.pricing_segments,
                complete=pricing_complete,
            ),
        )

    def _register_process(self, process: Any) -> None:
        with self._state_lock:
            closed = self._closed
            self._active_processes.add(process)
        if closed:
            try:
                _stop_process(process)
            except BaseException:
                # Preserve the exact process/root lease for a later close()
                # retry rather than forgetting an unconfirmed child.
                raise
            else:
                self._unregister_process(process)
            raise RuntimeError("KimiBackend is closed")

    def _unregister_process(self, process: Any) -> None:
        with self._state_lock:
            self._active_processes.discard(process)

    def _register_mcp_server(self, server: Any) -> None:
        with self._state_lock:
            closed = self._closed
            self._active_mcp_servers.add(server)
        if closed:
            try:
                server.stop()
            except BaseException:
                raise
            else:
                self._unregister_mcp_server(server)
            raise RuntimeError("KimiBackend is closed")

    def _unregister_mcp_server(self, server: Any) -> None:
        with self._state_lock:
            self._active_mcp_servers.discard(server)

    def _register_mutation_host(self, host: Any) -> None:
        with self._state_lock:
            closed = self._closed
            self._active_mutation_hosts.add(host)
        if closed:
            try:
                host.stop()
            except BaseException:
                raise
            else:
                self._unregister_mutation_host(host)
            raise RuntimeError("KimiBackend is closed")

    def _unregister_mutation_host(self, host: Any) -> None:
        with self._state_lock:
            self._active_mutation_hosts.discard(host)

    def close(self) -> None:
        """Cancel and positively settle every invocation-owned resource."""

        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._close_event.set()

        def settle_current() -> list[BaseException]:
            with self._state_lock:
                processes = tuple(self._active_processes)
                mutation_hosts = tuple(self._active_mutation_hosts)
                servers = tuple(self._active_mcp_servers)
            errors: list[BaseException] = []
            for process in processes:
                try:
                    _stop_process(process)
                except BaseException as exc:  # noqa: BLE001 - security cleanup
                    errors.append(exc)
                    logger.warning(
                        "KimiBackend.close: worker process settlement raised",
                        exc_info=True,
                    )
                else:
                    self._unregister_process(process)
            servers_settled = True
            for server in servers:
                try:
                    server.stop()
                except BaseException as exc:  # noqa: BLE001 - security cleanup
                    errors.append(exc)
                    servers_settled = False
                    logger.warning(
                        "KimiBackend.close: MCP server stop() raised",
                        exc_info=True,
                    )
                else:
                    self._unregister_mcp_server(server)
            if servers_settled:
                for host in mutation_hosts:
                    try:
                        host.stop()
                    except BaseException as exc:  # noqa: BLE001 - security cleanup
                        errors.append(exc)
                        logger.warning(
                            "KimiBackend.close: mutation host stop() raised",
                            exc_info=True,
                        )
                    else:
                        self._unregister_mutation_host(host)
            return errors

        initial_errors = settle_current()
        # A process/listener may have been between construction and registry
        # attachment when close() took its first snapshot.  Once the invoke
        # lock is acquired no normal invocation can attach another resource;
        # settle the final registry contents before declaring close complete.
        acquired = self._invoke_lock.acquire(timeout=_PROCESS_STOP_SECONDS)
        if acquired:
            try:
                final_errors = settle_current()
            finally:
                self._invoke_lock.release()
        else:
            final_errors = [
                KimiSDKQualificationError(
                    "KimiBackend.close could not drain the active invocation"
                )
            ]
        with self._state_lock:
            resources_remain = bool(
                self._active_processes
                or self._active_mutation_hosts
                or self._active_mcp_servers
            )
        for cleanup_error in (*initial_errors, *final_errors):
            if isinstance(cleanup_error, KeyboardInterrupt):
                # A first pass may have positively settled the resource and
                # replayed cancellation while the final pass only unregisters
                # it. Preserve that cancellation beyond the final positive
                # registry proof instead of silently consuming Ctrl-C.
                raise cleanup_error
        if final_errors or resources_remain:
            causes = final_errors or initial_errors
            cause = (
                causes[0]
                if causes
                else KimiSDKQualificationError(
                    "a confined resource appeared during final close"
                )
            )
            raise KimiSDKQualificationError(
                "KimiBackend.close could not confirm all confined resources settled"
            ) from cause


__all__ = [
    "KimiBackend",
    "KimiSDKQualificationError",
    "KimiSDKUnavailable",
    "KimiWorkerLifecycleError",
]
