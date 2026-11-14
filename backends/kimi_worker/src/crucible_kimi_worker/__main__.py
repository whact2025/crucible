# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Bounded JSON-lines worker around the reviewed Kimi Agent SDK release."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import os
import re
import stat
import sys
import tempfile
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

_PROTOCOL = "crucible.kimi-worker.v6"
_K3_BOUND_POLICY = "kimi-k3-cache-read-upper-bound-v1"
_MODEL = "kimi-k2.7-code-highspeed"
_K3_MODEL = "kimi-k3"
_MODEL_CONTEXT_SIZES = {_MODEL: 262_144, _K3_MODEL: 1_048_576}
_SDK_VERSION = "0.0.5"
_CLI_VERSION = "1.12.0"
_KOSONG_VERSION = "0.42.0"
_OPENAI_VERSION = "2.14.0"
_PROVIDER_NAME = "crucible-moonshot"
_MAX_CONTEXT_SIZE = 262_144
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_RETRIABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_MAX_TOOL_STEPS = 100
_MAX_TOOL_EVENTS = 2_048
_MAX_AUTH_TOKEN_CHARS = 4_096
_MAX_USAGE_MESSAGE_ID_CHARS = 256
_DEFAULT_MAX_RESPONSE_TOKENS = 32_768
_TOOL_DISPATCH_BUDGET_ERROR = "ToolDispatchBudgetExceeded"
_TOOL_MUTATION_INDETERMINATE_ERROR = "ToolMutationIndeterminate"
_MCP_TOOL_TIMEOUT_MS = 1_800_000
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_TOOL_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_THINK_MARKER_LIKE_RE = re.compile(
    r"<\s*/?\s*think(?:\s*/?\s*>|(?=\s|$))",
    re.IGNORECASE,
)

_ToolEvent = tuple[Literal["tool_call", "tool_result"], str, str]


class WorkerQualificationError(RuntimeError):
    """The worker request or observed SDK behavior is not qualified."""


class ToolDispatchBudgetTerminal(RuntimeError):
    """The workflow-owned source budget terminated the SDK tool loop."""


class ToolMutationIndeterminateTerminal(RuntimeError):
    """An indeterminate mutation terminally stopped the SDK tool loop."""


class ToolDispatchTerminal(RuntimeError):
    """A workflow-owned terminal dispatch stopped the SDK tool loop."""


def _extract_final_text(text: str) -> str:
    """Return K2.7 visible text and fail closed on leaked reasoning markers."""

    if _THINK_MARKER_LIKE_RE.search(text):
        raise WorkerQualificationError(
            "Kimi emitted a reasoning marker in visible content"
        )
    return text


@dataclass(frozen=True)
class _Bindings:
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
class _Request:
    root: Path
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

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> _Request:
        expected_fields = {
            "protocol", "type", "root", "system", "user", "model",
            "reasoning_effort", "schema_json", "base_url", "auth_token",
            "max_steps", "max_response_tokens", "mcp_url", "mcp_auth_token",
            "mcp_tool_names", "diagnostic_cost_upper_bound", "diagnostic_budget_usd",
        }
        if set(payload) != expected_fields:
            raise WorkerQualificationError("request uses an invalid worker schema")
        if payload.get("protocol") != _PROTOCOL or payload.get("type") != "invoke":
            raise WorkerQualificationError("request uses an unexpected worker protocol")
        model = payload.get("model")
        effort = payload.get("reasoning_effort")
        if (
            not isinstance(model, str)
            or model not in _MODEL_CONTEXT_SIZES
            or "reasoning_effort" not in payload
            or (model == _MODEL and effort is not None)
            or (model == _K3_MODEL and effort != "max")
        ):
            raise WorkerQualificationError("request uses an unqualified model/effort")
        bounded = payload["diagnostic_cost_upper_bound"]
        budget = payload["diagnostic_budget_usd"]
        if type(bounded) is not bool or (
            bounded and (
                model != _K3_MODEL or effort != "max"
                or type(budget) not in {int, float}
                or not math.isfinite(budget) or budget <= 0
            )
        ) or (not bounded and budget is not None):
            raise WorkerQualificationError("request uses an invalid diagnostic cost policy")
        required_strings = (
            "root",
            "system",
            "user",
            "base_url",
            "auth_token",
        )
        if any(not isinstance(payload.get(name), str) for name in required_strings):
            raise WorkerQualificationError("request contains an invalid required field")
        schema_json = payload.get("schema_json")
        if schema_json is not None and not isinstance(schema_json, str):
            raise WorkerQualificationError(
                "request schema_json must be a string or null"
            )
        mcp_url = payload.get("mcp_url")
        mcp_auth_token = payload.get("mcp_auth_token")
        raw_tool_names = payload.get("mcp_tool_names", [])
        if not isinstance(raw_tool_names, list):
            raise WorkerQualificationError("request mcp_tool_names must be a list")
        if (
            len(raw_tool_names) > _MAX_TOOL_EVENTS
            or any(
                not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None
                for name in raw_tool_names
            )
            or raw_tool_names != sorted(set(raw_tool_names))
        ):
            raise WorkerQualificationError(
                "request mcp_tool_names must be sorted, unique, and portable"
            )
        if mcp_url is not None:
            if not isinstance(mcp_url, str):
                raise WorkerQualificationError(
                    "request mcp_url must be a string or null"
                )
            parsed_mcp = urlsplit(mcp_url)
            try:
                mcp_port = parsed_mcp.port
            except ValueError as exc:
                raise WorkerQualificationError(
                    "request mcp_url has an invalid port"
                ) from exc
            if (
                parsed_mcp.scheme != "http"
                or parsed_mcp.hostname != "127.0.0.1"
                or mcp_port is None
                or not 1 <= mcp_port <= 65_535
                or parsed_mcp.username is not None
                or parsed_mcp.password is not None
                or parsed_mcp.query
                or parsed_mcp.fragment
                or parsed_mcp.path != "/mcp"
                or any(character in mcp_url for character in ("\x00", "\r", "\n"))
            ):
                raise WorkerQualificationError(
                    "request mcp_url is not the qualified loopback endpoint"
                )
            if (
                not isinstance(mcp_auth_token, str)
                or len(mcp_auth_token) < 32
                or len(mcp_auth_token) > 1024
                or _contains_control_character(mcp_auth_token)
            ):
                raise WorkerQualificationError(
                    "request contains an invalid MCP authorization token"
                )
        elif mcp_auth_token is not None or raw_tool_names:
            raise WorkerQualificationError(
                "plain Kimi requests cannot carry MCP authority"
            )
        max_steps = payload.get("max_steps")
        if not isinstance(max_steps, int) or isinstance(max_steps, bool):
            raise WorkerQualificationError("request max_steps must be an integer")
        if (
            max_steps < 1
            or max_steps > _MAX_TOOL_STEPS
            or (mcp_url is None and max_steps != 1)
        ):
            raise WorkerQualificationError(
                "request max_steps is outside the qualified Kimi bound"
            )
        token = payload["auth_token"]
        if (
            not token
            or len(token) > _MAX_AUTH_TOKEN_CHARS
            or not _is_header_safe_token(token)
        ):
            raise WorkerQualificationError(
                "request contains an invalid authorization token"
            )
        base_url = payload["base_url"]
        _validate_inference_base_url(base_url)
        max_response_tokens = payload.get("max_response_tokens")
        if (
            type(max_response_tokens) is not int
            or not 1 <= max_response_tokens <= _MODEL_CONTEXT_SIZES[model]
        ):
            raise WorkerQualificationError(
                "request max_response_tokens is outside the qualified Kimi bound"
            )
        return cls(
            root=Path(str(payload["root"])),
            system=str(payload["system"]),
            user=str(payload["user"]),
            model=model,
            reasoning_effort=effort,
            diagnostic_cost_upper_bound=bounded,
            diagnostic_budget_usd=budget,
            schema_json=schema_json,
            base_url=base_url,
            auth_token=token,
            max_steps=max_steps,
            max_response_tokens=max_response_tokens,
            mcp_url=mcp_url,
            mcp_auth_token=mcp_auth_token,
            mcp_tool_names=tuple(raw_tool_names),
        )


@dataclass(frozen=True)
class _Paths:
    root: Path
    home: Path
    share: Path
    work: Path
    skills: Path
    config: Path
    scratch: Path


@dataclass(frozen=True)
class _Outcome:
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

    def to_wire(self) -> dict[str, Any]:
        return {
            "protocol": _PROTOCOL,
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


@dataclass
class _StepLedger:
    max_steps: int
    emit: Callable[[dict[str, Any]], None]
    steps_used: int = 0

    def begin(self) -> None:
        if self.steps_used >= self.max_steps:
            raise WorkerQualificationError(
                "Kimi emitted more StepBegin events than its physical-attempt bound"
            )
        self.steps_used += 1
        self.emit(
            {
                "protocol": _PROTOCOL,
                "type": "step_progress",
                "steps_used": self.steps_used,
            }
        )


class _UsageAccumulator:
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
    """Parse the bridge's closed terminal-dispatch error envelope."""

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


def _usage_message_id_valid(value: Any) -> bool:
    return (
        type(value) is str
        and 0 < len(value) <= _MAX_USAGE_MESSAGE_ID_CHARS
        and not _contains_control_character(value)
    )


def _usage_counter(value: Any) -> int:
    return cast(int, value) if _is_nonnegative_int(value) else 0


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _is_header_safe_token(value: str) -> bool:
    return all(0x21 <= ord(character) <= 0x7E for character in value)


def _validate_inference_base_url(value: str) -> None:
    if (
        not value
        or _contains_control_character(value)
        or any(character.isspace() for character in value)
    ):
        raise WorkerQualificationError("request contains an invalid inference endpoint")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise WorkerQualificationError(
            "request contains an invalid inference endpoint"
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
        or (
            parsed.scheme == "http"
            and (port is None or parsed.netloc != f"127.0.0.1:{port}")
        )
    ):
        raise WorkerQualificationError("request contains an invalid inference endpoint")


def _distribution_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _runtime_identity() -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "type": "probe",
        "sdk_version": _distribution_version("kimi-agent-sdk"),
        "cli_version": _distribution_version("kimi-cli"),
        "kosong_version": _distribution_version("kosong"),
        "openai_version": _distribution_version("openai"),
    }


def _assert_runtime_identity() -> None:
    identity = _runtime_identity()
    expected = {
        "sdk_version": _SDK_VERSION,
        "cli_version": _CLI_VERSION,
        "kosong_version": _KOSONG_VERSION,
        "openai_version": _OPENAI_VERSION,
    }
    if any(identity[name] != version for name, version in expected.items()):
        raise WorkerQualificationError(
            "installed SDK tuple does not match the worker lock"
        )


def _load_bindings() -> _Bindings:
    _assert_runtime_identity()
    try:
        import kimi_agent_sdk as sdk  # type: ignore[import-not-found]
        from kaos.path import KaosPath  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WorkerQualificationError(
            "qualified distributions could not be imported"
        ) from exc
    return _Bindings(
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


def _read_request() -> _Request:
    raw = sys.stdin.buffer.readline(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        raise WorkerQualificationError("request exceeds the worker size limit")
    if not raw or sys.stdin.buffer.read(1):
        raise WorkerQualificationError("worker requires exactly one JSON request line")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkerQualificationError("request is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise WorkerQualificationError("request JSON must be an object")
    return _Request.from_wire(payload)


def _isolate_environment(root: Path) -> _Paths:
    os.umask(0o077)
    root = root.resolve(strict=True)
    root_stat = root.stat()
    if (
        not root.is_dir()
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) & 0o077
    ):
        raise WorkerQualificationError(
            "worker root must be a private directory owned by this user"
        )
    paths = _Paths(
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


def _write_private(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        payload = text.encode("utf-8")
        while payload:
            written = os.write(descriptor, payload)
            payload = payload[written:]
    finally:
        os.close(descriptor)


def _system_text(request: _Request) -> str:
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
            raise WorkerQualificationError(
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


def _write_agent(request: _Request, paths: _Paths) -> Path:
    system_file = paths.config / "system.md"
    agent_file = paths.config / "agent.yaml"
    _write_private(system_file, "${CRUCIBLE_SYSTEM}\n")
    document = {
        "version": "1",
        "agent": {
            "name": (
                "crucible-kimi-prompt-only"
                if request.mcp_url is None
                else "crucible-kimi-toolset"
            ),
            "system_prompt_path": "./system.md",
            "system_prompt_args": {"CRUCIBLE_SYSTEM": _system_text(request)},
            "tools": [],
            "exclude_tools": [],
            "subagents": {},
        },
    }
    _write_private(
        agent_file,
        json.dumps(document, ensure_ascii=False, allow_nan=False),
    )
    return agent_file


def _build_config(bindings: _Bindings, request: _Request) -> Any:
    return bindings.Config.model_validate(
        {
            "default_model": request.model,
            # K2.7 always thinks, but advertising the thinking capability to
            # CLI 1.12 would emit unsupported reasoning_effort="high".
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


def _k3_usage_failure(code: str, *, field: str = "none", state: str = "none") -> WorkerQualificationError:
    # All arguments are adapter-owned enum labels, never provider strings.
    return WorkerQualificationError(
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
        raise WorkerQualificationError("Kimi K3 raw stream boundary is unqualified")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerQualificationError("Kimi K3 stream contains duplicate JSON fields")
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
                    raise WorkerQualificationError("Kimi K3 stream contains invalid JSON") from exc
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
    """Install request-bound tool streaming and exact K2.7/K3 usage evidence."""

    try:
        from kosong.chat_provider import kimi as kimi_provider  # type: ignore[import-not-found]
    except ImportError as exc:
        raise WorkerQualificationError(
            "Kimi native stream compatibility module is unavailable"
        ) from exc

    native_stream_type = getattr(kimi_provider, "KimiStreamedMessage", None)
    if (
        not isinstance(native_stream_type, type)
        or native_stream_type.__module__ != "kosong.chat_provider.kimi"
        or native_stream_type.__name__ != "KimiStreamedMessage"
    ):
        raise WorkerQualificationError(
            "Kimi runtime resolved an unqualified native stream type"
        )

    class _QualifiedKimiStream(native_stream_type):
        def __init__(
            self, response: Any, *, k3_state: _K3StreamUsageState | None = None,
        ) -> None:
            if isinstance(response, kimi_provider.ChatCompletion):
                raise WorkerQualificationError(
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
                raise WorkerQualificationError(
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
                            raise WorkerQualificationError(
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
                        raise WorkerQualificationError(
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
                            raise WorkerQualificationError(
                                "Kimi stream emitted conflicting finish reasons"
                            )
                        self.finish_reason = finish
                    delta = choice.delta
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not isinstance(reasoning, str):
                            raise WorkerQualificationError(
                                "Kimi stream emitted non-text reasoning content"
                            )
                        message_part_seen = True
                        yield kimi_provider.ThinkPart(think=reasoning)
                    if delta.content:
                        message_part_seen = True
                        yield kimi_provider.TextPart(text=delta.content)
                    for tool_delta in delta.tool_calls or ():
                        index = getattr(tool_delta, "index", None)
                        if type(index) is not int or not 0 <= index < _MAX_TOOL_EVENTS:
                            raise WorkerQualificationError(
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
                                raise WorkerQualificationError(
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
                                raise WorkerQualificationError(
                                    "Kimi stream changed a tool-call name"
                                )
                            state["name"] = tool_name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            if not isinstance(arguments, str):
                                raise WorkerQualificationError(
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
                raise WorkerQualificationError(
                    "Kimi stream did not prove the exact served model"
                )
            if self.finish_reason is None:
                raise WorkerQualificationError(
                    "Kimi stream ended without a finish reason"
                )
            if cost_bounds is not None and cost_bounds.enabled:
                if self._k3_state is None or self._k3_state.counters is None:
                    raise _k3_usage_failure("missing_raw_stream_state")
                cost_bounds.settle(self._k3_state.counters)
            if model == _K3_MODEL and self._usage is None:
                raise WorkerQualificationError(
                    "Kimi K3 stream ended without final usage evidence"
                )
            if self.finish_reason in {"length", "content_filter"}:
                # Keep terminal-only truncation/rejection responses alive
                # through Kosong's empty-response check so the worker can
                # retain their model, finish reason, and billed usage.
                # Never dispatch tool calls cut short by either condition.
                if not message_part_seen:
                    yield kimi_provider.TextPart(text="")
                return
            if tool_calls and self.finish_reason != "tool_calls":
                raise WorkerQualificationError(
                    "Kimi stream emitted tool calls under an invalid finish reason"
                )
            if self.finish_reason == "tool_calls":
                indexes = sorted(tool_calls)
                if indexes != list(range(len(indexes))):
                    raise WorkerQualificationError(
                        "Kimi stream emitted a sparse tool-call index set"
                    )
                for index in indexes:
                    state = tool_calls[index]
                    if (
                        not state["id"]
                        or not state["name"]
                        or state["arguments"] is None
                    ):
                        raise WorkerQualificationError(
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
        raise WorkerQualificationError(
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
            raise WorkerQualificationError(
                "Kimi provider changed requested model identity"
            )
        if native_provider._generation_kwargs != qualified_generation_kwargs:
            raise WorkerQualificationError(
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
    try:
        provider = session._cli.soul._runtime.llm.chat_provider
        stream = provider._crucible_last_stream
    except AttributeError as exc:
        raise WorkerQualificationError(
            "Kimi runtime did not retain qualified terminal response evidence"
        ) from exc
    finish_reason = getattr(stream, "finish_reason", None)
    served_model = getattr(stream, "served_model", None)
    if not isinstance(finish_reason, str) or not isinstance(served_model, str):
        raise WorkerQualificationError(
            "Kimi runtime emitted invalid terminal response evidence"
        )
    return finish_reason, served_model


def _disable_cli_auto_compaction(session: Any) -> None:
    """Prevent CLI 1.12 from making an unaccounted compaction inference."""

    try:
        soul = session._cli.soul
        compact_context = soul.compact_context
    except AttributeError as exc:
        raise WorkerQualificationError(
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
        raise WorkerQualificationError(
            "Kimi runtime resolved an unqualified compaction boundary"
        )

    async def _reject_hidden_compaction(_soul: Any) -> None:
        raise WorkerQualificationError(
            "Kimi automatic context compaction is disabled because it would "
            "perform an unaccounted Moonshot generation"
        )

    soul.compact_context = types.MethodType(_reject_hidden_compaction, soul)
    installed = getattr(soul, "compact_context", None)
    if (
        not isinstance(installed, types.MethodType)
        or installed.__func__ is not _reject_hidden_compaction
    ):
        raise WorkerQualificationError(
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
    """Bind K2.7/K3 generation and cache policy; disable hidden inference."""

    try:
        provider = session._cli.soul._runtime.llm.chat_provider
    except AttributeError as exc:
        raise WorkerQualificationError(
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
        raise WorkerQualificationError(
            "Kimi runtime resolved an unqualified chat provider"
        )
    session_id = getattr(session, "id", None)
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > 256
        or _contains_control_character(session_id)
    ):
        raise WorkerQualificationError(
            "Kimi runtime did not expose a bounded session cache identity"
        )
    if provider._generation_kwargs != {"prompt_cache_key": session_id}:
        raise WorkerQualificationError(
            "Kimi runtime resolved unqualified Moonshot generation options"
        )
    if getattr(provider, "thinking_effort", object()) is not None:
        raise WorkerQualificationError(
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
        raise WorkerQualificationError(
            "Kimi runtime resolved an unqualified Moonshot HTTP client"
        )
    provider.client = client.with_options(max_retries=0)
    if getattr(provider.client, "max_retries", None) != 0:
        raise WorkerQualificationError(
            "Kimi runtime could not disable hidden Moonshot HTTP retries"
        )
    _disable_cli_auto_compaction(session)
    if model not in _MODEL_CONTEXT_SIZES or (
        (model == _MODEL and reasoning_effort is not None)
        or (model == _K3_MODEL and reasoning_effort != "max")
    ):
        raise WorkerQualificationError("unqualified Kimi model/effort")
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
        raise WorkerQualificationError("Kimi diagnostic bounds require K3")
    _install_moonshot_stream_compat(provider, model=model, cost_bounds=cost_bounds)


def _usage_wire(usage: _UsageAccumulator) -> dict[str, Any]:
    input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
    return {
        "protocol": _PROTOCOL,
        "type": "usage",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation,
        "cache_read_tokens": cache_read,
        "usage_valid": usage.valid,
    }


async def _run_session(
    request: _Request,
    paths: _Paths,
    bindings: _Bindings,
    emit: Callable[[dict[str, Any]], None],
    step_ledger: _StepLedger | None = None,
) -> _Outcome:
    agent_file = _write_agent(request, paths)
    config = _build_config(bindings, request)
    session: Any = None
    usage = _UsageAccumulator()
    cost_bounds = _K3CostBoundLedger(
        request.diagnostic_cost_upper_bound, request.diagnostic_budget_usd,
        lambda record: emit({"protocol": _PROTOCOL, **record}),
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
            raise WorkerQualificationError("Kimi Session resolved an unexpected model")
        status = getattr(session, "status", None)
        if bool(getattr(status, "yolo_enabled", True)):
            raise WorkerQualificationError(
                "Kimi Session unexpectedly enabled automatic approval"
            )
        _qualify_moonshot_provider(
            session,
            max_response_tokens=request.max_response_tokens,
            model=request.model,
            reasoning_effort=request.reasoning_effort,
            cost_bounds=cost_bounds,
        )
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
                    raise ToolMutationIndeterminateTerminal(
                        "Kimi tool loop stopped after an indeterminate "
                        "source mutation"
                    )
                if tool_dispatch_terminal_pending:
                    raise ToolDispatchTerminal(
                        "Kimi tool loop stopped at a workflow-owned terminal "
                        "dispatch boundary"
                    )
                raise ToolDispatchBudgetTerminal(
                    "Kimi tool loop stopped because the workflow-owned "
                    "tool dispatch budget is exhausted"
                )
            if isinstance(wire_message, bindings.StepBegin):
                cost_bounds.admit_next()
                try:
                    ledger.begin()
                except WorkerQualificationError:
                    session.cancel()
                    raise
                text_parts.clear()
            elif isinstance(wire_message, bindings.TextPart):
                text_parts.append(str(wire_message.text))
            elif isinstance(wire_message, bindings.ThinkPart):
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
                    raise WorkerQualificationError(
                        "Kimi requested a tool outside the attached ToolSet"
                    )
                seen_tool_call_ids.add(tool_call_id)
                pending_tool_calls[tool_call_id] = tool_name
                tool_events.append(("tool_call", tool_name, tool_call_id))
                emit(
                    {
                        "protocol": _PROTOCOL,
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
                raise WorkerQualificationError(
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
                    raise WorkerQualificationError(
                        "Kimi emitted an unmatched tool result"
                    )
                pending_tool_calls.pop(tool_call_id)
                approved_tool_calls.remove(tool_call_id)
                tool_events.append(("tool_result", result_tool_name, tool_call_id))
                emit(
                    {
                        "protocol": _PROTOCOL,
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
                        raise ToolMutationIndeterminateTerminal(
                            "Kimi tool loop stopped after an indeterminate "
                            "source mutation"
                        )
                    if tool_dispatch_terminal_pending:
                        raise ToolDispatchTerminal(
                            "Kimi tool loop stopped at a workflow-owned "
                            "terminal dispatch boundary"
                        )
                    raise ToolDispatchBudgetTerminal(
                        "Kimi tool loop stopped because the workflow-owned "
                        "tool dispatch budget is exhausted"
                    )
            elif type(wire_message).__name__ == "ToolCallRequest":
                session.cancel()
                raise WorkerQualificationError(
                    "Kimi requested an unqualified external tool call"
                )
            elif isinstance(wire_message, bindings.StepInterrupted):
                # The pinned CLI yields this marker immediately before
                # re-raising the underlying provider exception. Let that
                # exception reach the worker classifier unchanged.
                step_interrupted_seen = True
                continue
            elif isinstance(wire_message, bindings.TurnEnd):
                terminal_seen = True
        if not terminal_seen:
            detail = " after StepInterrupted" if step_interrupted_seen else ""
            raise WorkerQualificationError(
                f"Kimi session stream ended without TurnEnd{detail}"
            )
        if ledger.steps_used == 0:
            raise WorkerQualificationError(
                "Kimi session reached TurnEnd without StepBegin evidence"
            )
        if pending_tool_calls:
            raise WorkerQualificationError(
                "Kimi session ended with incomplete tool calls"
            )
        finish_reason, served_model = _moonshot_terminal_identity(session)
        input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
        final_message = _extract_final_text("".join(text_parts))
        if finish_reason == "length":
            return _Outcome(
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
            return _Outcome(
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
            raise WorkerQualificationError(
                "Kimi session ended under an unqualified finish reason"
            )
        return _Outcome(
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
        return _Outcome(
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
        # configured number of provider steps have completed.  Preserve the
        # qualified final stream's served-model identity before the isolated
        # session closes so exact per-step pricing remains attributable.
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
            raise WorkerQualificationError(
                "Kimi reached its step limit without an exact settled loop boundary"
            ) from exc
        finish_reason, served_model = _moonshot_terminal_identity(session)
        if finish_reason != "tool_calls" or served_model != request.model:
            raise WorkerQualificationError(
                "Kimi step-limit terminal lacked exact served-model tool-loop evidence"
            ) from exc
        input_tokens, output_tokens, cache_creation, cache_read = usage.totals()
        return _Outcome(
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


def _redact(message: str, *tokens: str | None) -> str:
    for token in tokens:
        if token:
            message = message.replace(token, "[REDACTED]")
    if len(message) > 1_000:
        message = message[:1_000] + "...[truncated]"
    return message


def _classify(
    exc: BaseException,
    request: _Request,
    bindings: _Bindings | None,
    *,
    steps_used: int = 0,
) -> _Outcome:
    name = type(exc).__name__
    message = _redact(
        str(exc),
        request.auth_token,
        request.mcp_auth_token,
    )
    if isinstance(exc, WorkerQualificationError):
        kind, retriable = "qualification", False
    elif isinstance(exc, ToolMutationIndeterminateTerminal):
        kind, retriable = "tool_mutation", False
    elif isinstance(exc, ToolDispatchTerminal):
        kind, retriable = "tool_terminal", False
    elif isinstance(exc, ToolDispatchBudgetTerminal):
        kind, retriable = "tool_budget", False
    elif isinstance(exc, asyncio.CancelledError) or (
        bindings is not None and isinstance(exc, bindings.RunCancelled)
    ):
        kind, retriable = "cancellation", True
    elif bindings is not None and isinstance(exc, bindings.MaxStepsReached):
        kind, retriable = "max_turns", False
    elif bindings is not None and isinstance(exc, bindings.APIStatusError):
        status = getattr(exc, "status_code", None)
        status = int(status) if status is not None else None
        return _Outcome(
            error_kind="transport",
            error_message=f"{name}: {message}",
            http_status=status,
            retriable=status in _RETRIABLE_HTTP_STATUSES,
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
        kind, retriable = "worker_internal", False
        message = f"{name}: worker failed without a qualified error shape"
    return _Outcome(
        error_kind=kind,
        error_message=f"{name}: {message}",
        retriable=retriable,
        steps_used=steps_used,
    )


def _protocol_stream() -> Any:
    """Reserve original stdout for protocol and route all other output away."""

    protocol_fd = os.dup(sys.stdout.fileno())
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)


def _emit(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )
    stream.flush()


def _run_worker() -> int:
    stream = _protocol_stream()
    request: _Request | None = None
    bindings: _Bindings | None = None
    step_ledger: _StepLedger | None = None
    try:
        request = _read_request()
        paths = _isolate_environment(request.root)
        bindings = _load_bindings()
        step_ledger = _StepLedger(
            request.max_steps,
            lambda payload: _emit(stream, payload),
        )
        outcome = asyncio.run(
            _run_session(
                request,
                paths,
                bindings,
                lambda payload: _emit(stream, payload),
                step_ledger,
            )
        )
    except BaseException as exc:  # noqa: BLE001 - process boundary
        if request is None:
            outcome = _Outcome(
                error_kind="qualification",
                error_message=(f"{type(exc).__name__}: invalid worker request"),
                retriable=False,
            )
        else:
            outcome = _classify(
                exc,
                request,
                bindings,
                steps_used=(step_ledger.steps_used if step_ledger is not None else 0),
            )
    with contextlib.suppress(BrokenPipeError, OSError):
        _emit(stream, outcome.to_wire())
    stream.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["--probe"]:
        identity = _runtime_identity()
        print(json.dumps(identity, sort_keys=True))
        expected = (
            identity["sdk_version"] == _SDK_VERSION
            and identity["cli_version"] == _CLI_VERSION
            and identity["openai_version"] == _OPENAI_VERSION
        )
        return 0 if expected else 2
    if arguments:
        return 2
    return _run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
