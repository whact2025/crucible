# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""One-shot, bounded JSON transport over the official OpenAI Codex SDK."""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from . import PROTOCOL_VERSION, QUALIFIED_SDK_VERSION

_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_REDACTIONS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+"),
    re.compile(r"(?i)\b(sk-[A-Za-z0-9_-]{8,})\b"),
)
_REQUEST_KEYS = frozenset(
    {
        "protocol",
        "cwd",
        "model",
        "effort",
        "system",
        "user",
        "output_schema",
        "mcp_url",
        "native_shell",
        "trusted_project_instructions",
        "resume_thread_id",
    }
)
_NON_AUTHORIZING_ITEM_TYPES = frozenset(
    {
        "agentMessage",
        "contextCompaction",
        "enteredReviewMode",
        "exitedReviewMode",
        "plan",
        "reasoning",
        "userMessage",
    }
)


class ProtocolError(ValueError):
    """The parent supplied a malformed or authority-widening request."""


def _exact_string(
    payload: dict[str, Any],
    name: str,
    *,
    optional: bool = False,
) -> str | None:
    value = payload.get(name)
    if optional and value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value


def _validate_loopback_mcp_url(value: str | None) -> str | None:
    if value is None:
        return None
    if any(
        ord(character) < 0x21 or ord(character) == 0x7F
        for character in value
    ):
        raise ProtocolError(
            "mcp_url must be an uncredentialed exact IPv4 loopback HTTP URL"
        )
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ProtocolError(
            "mcp_url must be an uncredentialed exact IPv4 loopback HTTP URL"
        ) from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/mcp"
        or parsed.netloc != f"127.0.0.1:{port}"
    ):
        raise ProtocolError(
            "mcp_url must be an uncredentialed exact IPv4 loopback HTTP URL"
        )
    return value


def validate_request(payload: Any) -> dict[str, Any]:
    """Validate and copy one closed-vocabulary worker request."""

    if not isinstance(payload, dict):
        raise ProtocolError("request must be a JSON object")
    unknown = set(payload) - _REQUEST_KEYS
    missing = _REQUEST_KEYS - set(payload)
    if unknown or missing:
        raise ProtocolError(
            f"request fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise ProtocolError("worker protocol mismatch")
    cwd_text = _exact_string(payload, "cwd")
    assert cwd_text is not None
    cwd = Path(cwd_text)
    if not cwd.is_absolute() or not cwd.is_dir():
        raise ProtocolError("cwd must be an existing absolute directory")
    native_shell = payload.get("native_shell")
    trusted = payload.get("trusted_project_instructions")
    if type(native_shell) is not bool:
        raise ProtocolError("native_shell must be an exact bool")
    if native_shell:
        # The official-SDK pilot is qualified only for Crucible-owned MCP
        # authority.  Its parent adapter rejects native shell at construction;
        # repeat that invariant at the process boundary so a malformed or
        # substituted parent request cannot widen the worker to host-native
        # command/file tools before terminal item validation can observe it.
        raise ProtocolError(
            "native_shell is not qualified for the official SDK worker"
        )
    if type(trusted) is not bool:
        raise ProtocolError(
            "trusted_project_instructions must be an exact bool"
        )
    output_schema = payload.get("output_schema")
    if output_schema is not None and not isinstance(output_schema, dict):
        raise ProtocolError("output_schema must be an object or null")
    effort = _exact_string(payload, "effort")
    if effort == "ultra":
        raise ProtocolError(
            "effort 'ultra' enables automatic task delegation and is not "
            "qualified for the official SDK worker"
        )
    result = {
        "protocol": PROTOCOL_VERSION,
        "cwd": str(cwd),
        "model": _exact_string(payload, "model"),
        "effort": effort,
        "system": _exact_string(payload, "system"),
        "user": _exact_string(payload, "user"),
        "output_schema": deepcopy(output_schema),
        "mcp_url": _validate_loopback_mcp_url(
            _exact_string(payload, "mcp_url", optional=True)
        ),
        "native_shell": native_shell,
        "trusted_project_instructions": trusted,
        "resume_thread_id": _exact_string(
            payload, "resume_thread_id", optional=True
        ),
    }
    return result


def _redacted_error(exc: BaseException) -> str:
    value = str(exc).replace("\r", " ").replace("\n", " ")[:1000]
    for pattern in _REDACTIONS:
        value = pattern.sub(r"\1<redacted>", value)
    return value


def _config_overrides(request: dict[str, Any]) -> tuple[str, ...]:
    overrides = [
        "features.apps=false",
        "features.browser_use=false",
        "features.computer_use=false",
        "features.hooks=false",
        "features.multi_agent=false",
        "features.multi_agent_v2=false",
        "features.plugins=false",
        "features.plugin_sharing=false",
        "features.skill_mcp_dependency_install=false",
        "features.tool_suggest=false",
        "features.workspace_dependencies=false",
        "mcp_servers={}",
    ]
    if not request["trusted_project_instructions"]:
        overrides.append("project_doc_max_bytes=0")
    if not request["native_shell"]:
        overrides.extend(
            ("features.shell_tool=false", "features.unified_exec=false")
        )
    mcp_url = request["mcp_url"]
    if mcp_url is not None:
        # The URL has already been constrained to exact IPv4 loopback and
        # contains no credentials. JSON quoting is valid TOML string quoting.
        overrides.extend(
            (
                f"mcp_servers.crucible.url={json.dumps(mcp_url)}",
                (
                    "mcp_servers.crucible.default_tools_approval_mode="
                    '"approve"'
                ),
            )
        )
    return tuple(overrides)


def _tool_items(items: list[Any]) -> list[dict[str, Any]]:
    """Project public SDK items into secret-free balanced tool evidence."""

    projected: list[dict[str, Any]] = []
    for item in items:
        root = getattr(item, "root", item)
        item_type = getattr(root, "type", None)
        if item_type in _NON_AUTHORIZING_ITEM_TYPES:
            continue
        if item_type != "mcpToolCall":
            # Native shell/file, dynamic, app, browser, image, hook, and
            # delegation items are impossible on this confined transport.
            # Silently discarding one would turn a feature/config regression
            # into apparently complete authority evidence.
            raise ProtocolError(
                "official SDK returned an unqualified or unknown thread item"
            )
        status = getattr(getattr(root, "status", None), "value", None)
        if status not in {"inProgress", "completed", "failed"}:
            raise ProtocolError("official SDK returned an unknown MCP status")
        call_id = getattr(root, "id", None)
        server = getattr(root, "server", None)
        tool = getattr(root, "tool", None)
        if not all(
            isinstance(value, str) and value
            for value in (call_id, server, tool)
        ):
            raise ProtocolError(
                "official SDK returned incomplete MCP tool identity"
            )
        projected.append(
            {
                "call_id": call_id,
                "server": server,
                "tool": tool,
                "status": status,
                "has_result": getattr(root, "result", None) is not None,
                "has_error": getattr(root, "error", None) is not None,
            }
        )
    return projected


def run_request(
    request: dict[str, Any],
    *,
    on_thread_started: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one validated request using only the official public SDK."""

    import openai_codex
    from openai_codex import ApprovalMode, Codex, CodexConfig, Sandbox

    if openai_codex.__version__ != QUALIFIED_SDK_VERSION:
        raise RuntimeError(
            "official Codex SDK version drift: expected "
            f"{QUALIFIED_SDK_VERSION}, observed {openai_codex.__version__}"
        )
    config = CodexConfig(
        cwd=request["cwd"],
        config_overrides=_config_overrides(request),
    )
    sandbox = (
        Sandbox.workspace_write
        if request["native_shell"]
        else Sandbox.read_only
    )
    with Codex(config=config) as codex:
        thread_id = request["resume_thread_id"]
        thread_kwargs = {
            "approval_mode": ApprovalMode.deny_all,
            "base_instructions": request["system"],
            "cwd": request["cwd"],
            "model": request["model"],
            "sandbox": sandbox,
        }
        if thread_id is None:
            thread = codex.thread_start(**thread_kwargs)
        else:
            thread = codex.thread_resume(thread_id, **thread_kwargs)
        if on_thread_started is not None:
            on_thread_started(thread.id)
        turn = thread.run(
            request["user"],
            approval_mode=ApprovalMode.deny_all,
            cwd=request["cwd"],
            effort=request["effort"],
            output_schema=request["output_schema"],
            sandbox=sandbox,
        )
        usage = None
        if turn.usage is not None:
            last = turn.usage.last
            usage = {
                "input_tokens": last.input_tokens,
                "cached_input_tokens": last.cached_input_tokens,
                "output_tokens": last.output_tokens,
                "reasoning_output_tokens": last.reasoning_output_tokens,
                "total_tokens": last.total_tokens,
            }
        status = getattr(turn.status, "value", None)
        if status not in {"completed", "interrupted", "failed", "inProgress"}:
            raise ProtocolError("official SDK returned an unknown turn status")
        turn_error = turn.error
        return {
            "protocol": PROTOCOL_VERSION,
            "type": "terminal",
            "ok": True,
            "sdk_version": openai_codex.__version__,
            "thread_id": thread.id,
            "turn_id": turn.id,
            "status": status,
            "error": (
                None
                if turn_error is None
                else {
                    "message": _redacted_error(
                        RuntimeError(turn_error.message)
                    ),
                    "has_additional_details": bool(
                        turn_error.additional_details
                    ),
                    "has_codex_error_info": (
                        turn_error.codex_error_info is not None
                    ),
                }
            ),
            "final_response": turn.final_response or "",
            "usage": usage,
            "tool_items": _tool_items(turn.items),
        }


def _read_request() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        raise ProtocolError("request exceeds worker byte limit")
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("request is not valid UTF-8 JSON") from exc
    return validate_request(decoded)


def _write_response(
    payload: dict[str, Any],
    *,
    newline: bool = False,
) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_RESPONSE_BYTES:
        raise ProtocolError("response exceeds worker byte limit")
    sys.stdout.buffer.write(encoded + (b"\n" if newline else b""))
    sys.stdout.buffer.flush()


def main() -> int:
    if sys.argv[1:]:
        if sys.argv[1:] != ["--probe"]:
            _write_response(
                {
                    "protocol": PROTOCOL_VERSION,
                    "ok": False,
                    "error_type": "ProtocolError",
                    "error": "unsupported worker argument",
                }
            )
            return 2
        try:
            sdk_version = version("openai-codex")
            cli_version = version("openai-codex-cli-bin")
        except PackageNotFoundError as exc:
            _write_response(
                {
                    "protocol": PROTOCOL_VERSION,
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": "qualified SDK runtime package is missing",
                }
            )
            return 2
        qualified = (
            sdk_version == QUALIFIED_SDK_VERSION
            and cli_version == QUALIFIED_SDK_VERSION
        )
        _write_response(
            {
                "protocol": PROTOCOL_VERSION,
                "ok": qualified,
                "type": "probe",
                "sdk_version": sdk_version,
                "cli_version": cli_version,
            }
        )
        return 0 if qualified else 2
    try:
        request = _read_request()
        response = run_request(
            request,
            on_thread_started=lambda thread_id: _write_response(
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "thread_started",
                    "thread_id": thread_id,
                },
                newline=True,
            ),
        )
    except BaseException as exc:  # noqa: BLE001 - typed process boundary
        response = {
            "protocol": PROTOCOL_VERSION,
            "type": "terminal",
            "ok": False,
            "error_type": type(exc).__name__,
            "error": _redacted_error(exc),
        }
    _write_response(response, newline=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
