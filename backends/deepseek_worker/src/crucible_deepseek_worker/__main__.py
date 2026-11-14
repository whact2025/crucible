# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""One-shot, closed-wire DeepSeek Responses SDK worker."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import openai


_PROTOCOL = "crucible.deepseek-worker.v1"
_MODEL = "deepseek-v4-flash"
_BASE_URL = "https://api.deepseek.com"
_RESPONSES_ENDPOINT = "https://api.deepseek.com/responses"
_OPENAI_VERSION = "2.53.0"
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_OUTPUT_TOKENS = 32_000
_MAX_EVENTS = 65_536
_TERMINAL_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)
_REQUEST_FIELDS = frozenset(
    {
        "protocol",
        "api_key",
        "input",
        "max_output_tokens",
        "model",
        "reasoning",
        "stream",
        "tools",
    }
)
_TOOL_FIELDS = frozenset({"type", "name", "description", "parameters"})
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class WorkerContractError(ValueError):
    """The parent request or provider result left the frozen contract."""


def _canonical_json(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (-float("inf") < value < float("inf")):
            raise WorkerContractError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _canonical_json(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkerContractError(f"{field} has a non-string key")
            out[key] = _canonical_json(item, field=f"{field}.{key}")
        return out
    raise WorkerContractError(f"{field} contains unsupported {type(value).__name__}")


def _require_nonempty(value: Any, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise WorkerContractError(f"{field} is invalid")
    return value


def _require_tool_name(value: Any, *, field: str) -> str:
    name = _require_nonempty(value, field=field, maximum=128)
    if _TOOL_NAME_RE.fullmatch(name) is None:
        raise WorkerContractError(f"{field} does not match the provider grammar")
    return name


def _validate_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 256:
        raise WorkerContractError("tools must be a bounded list")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _TOOL_FIELDS:
            raise WorkerContractError(
                f"tools[{index}] must contain only client function fields"
            )
        if raw["type"] != "function":
            raise WorkerContractError(f"tools[{index}] is not a client function tool")
        name = _require_tool_name(raw["name"], field=f"tools[{index}].name")
        if name in names:
            raise WorkerContractError(f"duplicate function tool {name!r}")
        names.add(name)
        description = raw["description"]
        if not isinstance(description, str) or len(description) > 16_384:
            raise WorkerContractError(f"tools[{index}].description is invalid")
        parameters = raw["parameters"]
        if not isinstance(parameters, Mapping):
            raise WorkerContractError(f"tools[{index}].parameters must be an object")
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": _canonical_json(
                    parameters,
                    field=f"tools[{index}].parameters",
                ),
            }
        )
    return tools


def _validate_request(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _REQUEST_FIELDS:
        unknown = sorted(set(raw) - _REQUEST_FIELDS) if isinstance(raw, Mapping) else []
        missing = sorted(_REQUEST_FIELDS - set(raw)) if isinstance(raw, Mapping) else []
        raise WorkerContractError(
            "request fields differ from the frozen contract: "
            f"missing={missing}, unknown={unknown}"
        )
    if raw["protocol"] != _PROTOCOL:
        raise WorkerContractError("worker protocol mismatch")
    if raw["model"] != _MODEL:
        raise WorkerContractError("request model differs from frozen Flash alias")
    if raw["stream"] is not True:
        raise WorkerContractError("DeepSeek Responses must stream")
    if raw["reasoning"] != {"effort": "high"}:
        raise WorkerContractError("DeepSeek reasoning must be exact high effort")
    api_key = _require_nonempty(raw["api_key"], field="api_key", maximum=4096)
    max_output_tokens = raw["max_output_tokens"]
    if (
        type(max_output_tokens) is not int
        or not 1 <= max_output_tokens <= _MAX_OUTPUT_TOKENS
    ):
        raise WorkerContractError(
            f"max_output_tokens must be from 1 through {_MAX_OUTPUT_TOKENS}"
        )
    input_value = _canonical_json(raw["input"], field="input")
    if not isinstance(input_value, list) or len(input_value) < 2:
        raise WorkerContractError("input must be a non-empty typed input list")
    for index, role in enumerate(("system", "user")):
        item = input_value[index]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"role", "content"}
            or item["role"] != role
            or not isinstance(item["content"], str)
        ):
            raise WorkerContractError(
                "initial input must contain system and user messages"
            )
    pending_calls: set[str] = set()
    seen_calls: set[str] = set()
    for index, item in enumerate(input_value[2:], start=2):
        if not isinstance(item, Mapping):
            raise WorkerContractError(f"input[{index}] is not an object")
        item_type = item.get("type")
        if item_type in {"reasoning", "message"}:
            expected = (
                {"type", "content"}
                if item_type == "reasoning"
                else {"type", "role", "content"}
            )
            expected_part_type = (
                "reasoning_text" if item_type == "reasoning" else "output_text"
            )
            if set(item) != expected or (
                item_type == "message" and item.get("role") != "assistant"
            ):
                raise WorkerContractError(
                    f"input[{index}] is not an exact prior {item_type} item"
                )
            content = item.get("content")
            if not isinstance(content, list):
                raise WorkerContractError(
                    f"input[{index}].content must be a list"
                )
            for part_index, part in enumerate(content):
                if (
                    not isinstance(part, Mapping)
                    or set(part) != {"type", "text"}
                    or part.get("type") != expected_part_type
                    or not isinstance(part.get("text"), str)
                ):
                    raise WorkerContractError(
                        f"input[{index}].content[{part_index}] is invalid"
                    )
            continue
        if item_type == "function_call":
            if set(item) != {"type", "call_id", "name", "arguments"}:
                raise WorkerContractError(
                    f"input[{index}] is not an exact prior function_call"
                )
            call_id = _require_nonempty(
                item.get("call_id"),
                field=f"input[{index}].call_id",
                maximum=512,
            )
            if call_id in seen_calls:
                raise WorkerContractError("prior function call IDs must be unique")
            seen_calls.add(call_id)
            pending_calls.add(call_id)
            _require_tool_name(item.get("name"), field=f"input[{index}].name")
            _require_nonempty(
                item.get("arguments"),
                field=f"input[{index}].arguments",
                maximum=2 * 1024 * 1024,
            )
            continue
        if item_type == "function_call_output":
            if (
                set(item) != {"type", "call_id", "output"}
                or not isinstance(item.get("output"), str)
            ):
                raise WorkerContractError(
                    f"input[{index}] is not an exact function_call_output"
                )
            call_id = _require_nonempty(
                item.get("call_id"),
                field=f"input[{index}].call_id",
                maximum=512,
            )
            if call_id not in pending_calls:
                raise WorkerContractError(
                    "function_call_output has no unmatched prior function call"
                )
            pending_calls.remove(call_id)
            continue
        raise WorkerContractError(
            f"input[{index}] has unsupported history type {item_type!r}"
        )
    if pending_calls:
        raise WorkerContractError("prior function calls lack outputs")
    return {
        "protocol": _PROTOCOL,
        "api_key": api_key,
        "input": input_value,
        "max_output_tokens": max_output_tokens,
        "model": _MODEL,
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": _validate_tools(raw["tools"]),
    }


def _record_digest(distribution: importlib.metadata.Distribution) -> str:
    record = distribution.read_text("RECORD")
    if record is None:
        return "sha256:" + hashlib.sha256(b"").hexdigest()
    return "sha256:" + hashlib.sha256(record.encode("utf-8")).hexdigest()


def _sbom() -> list[dict[str, str]]:
    components: dict[str, dict[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name:
            continue
        canonical = name.lower().replace("_", "-")
        components[canonical] = {
            "name": canonical,
            "version": distribution.version,
            "record_sha256": _record_digest(distribution),
        }
    return [components[name] for name in sorted(components)]


def _probe() -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "type": "probe",
        "ok": True,
        "openai_version": openai.__version__,
        "client_policy": {
            "base_url": _BASE_URL,
            "responses_endpoint": _RESPONSES_ENDPOINT,
            "max_retries": 0,
            "trust_env": False,
            "follow_redirects": False,
        },
        "sbom": _sbom(),
    }


def _event_mapping(event: Any) -> dict[str, Any]:
    dumper = getattr(event, "model_dump", None)
    if callable(dumper):
        # ``exclude_unset`` is accounting-critical. OpenAI's generated
        # response types may expose numeric defaults for fields absent from
        # the provider JSON; retaining only fields actually deserialized from
        # the SSE prevents an absent DeepSeek cache counter from becoming a
        # synthetic zero.
        value = dumper(mode="json", exclude_unset=True)
    elif isinstance(event, Mapping):
        value = dict(event)
    else:
        raise WorkerContractError("SDK stream event is not serializable")
    if not isinstance(value, dict):
        raise WorkerContractError("SDK stream event did not serialize to an object")
    return value


def _exact_count(value: Any, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise WorkerContractError(f"{field} is not an exact bounded count")
    return value


def _optional_count(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _exact_count(value, field=field)


def _usage(response: Mapping[str, Any]) -> dict[str, int | None] | None:
    raw = response.get("usage")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise WorkerContractError("terminal usage is not an object")
    allowed_usage = {
        "input_tokens",
        "input_tokens_details",
        "output_tokens",
        "output_tokens_details",
        "total_tokens",
    }
    required_usage = {"input_tokens", "output_tokens", "total_tokens"}
    if set(raw) - allowed_usage or not required_usage <= set(raw):
        raise WorkerContractError("terminal usage fields changed")
    input_tokens = _exact_count(raw.get("input_tokens"), field="input_tokens")
    output_tokens = _exact_count(raw.get("output_tokens"), field="output_tokens")
    total_tokens = _exact_count(raw.get("total_tokens"), field="total_tokens")
    if total_tokens != input_tokens + output_tokens:
        raise WorkerContractError("terminal total_tokens is contradictory")
    input_details = raw.get("input_tokens_details")
    cached_tokens: int | None = None
    if input_details is not None:
        if not isinstance(input_details, Mapping):
            raise WorkerContractError("input_tokens_details is not an object")
        if set(input_details) - {"cached_tokens", "cache_write_tokens"}:
            raise WorkerContractError("input token detail fields changed")
        cache_write_tokens = _optional_count(
            input_details.get("cache_write_tokens"),
            field="cache_write_tokens",
        )
        cached_tokens = _optional_count(
            input_details.get("cached_tokens"), field="cached_tokens"
        )
        if cached_tokens is not None and cached_tokens > input_tokens:
            raise WorkerContractError("cached_tokens exceeds input_tokens")
        if (
            cache_write_tokens is not None
            and cached_tokens is not None
            and cache_write_tokens + cached_tokens > input_tokens
        ):
            raise WorkerContractError(
                "cache write plus cached input exceeds input_tokens"
            )
    else:
        cache_write_tokens = None
    output_details = raw.get("output_tokens_details")
    reasoning_tokens: int | None = None
    if output_details is not None:
        if not isinstance(output_details, Mapping):
            raise WorkerContractError("output_tokens_details is not an object")
        if set(output_details) - {"reasoning_tokens"}:
            raise WorkerContractError("output token detail fields changed")
        reasoning_tokens = _optional_count(
            output_details.get("reasoning_tokens"), field="reasoning_tokens"
        )
        if reasoning_tokens is not None and reasoning_tokens > output_tokens:
            raise WorkerContractError("reasoning_tokens exceeds output_tokens")
    return {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_tokens,
        "cache_write_tokens": cache_write_tokens,
        "output_tokens": output_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def _terminal_output(
    response: Mapping[str, Any],
) -> tuple[str, list[dict[str, str]], list[dict[str, Any]]]:
    output = response.get("output")
    if not isinstance(output, list):
        raise WorkerContractError("terminal output is not a list")
    text_parts: list[str] = []
    calls: list[dict[str, str]] = []
    continuation: list[dict[str, Any]] = []
    call_ids: set[str] = set()
    for index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise WorkerContractError(f"output[{index}] is not an object")
        item_type = item.get("type")
        if item_type == "reasoning":
            content = item.get("content")
            if not isinstance(content, list):
                raise WorkerContractError(f"output[{index}].content is invalid")
            reasoning_content: list[dict[str, str]] = []
            for content_index, part in enumerate(content):
                if not isinstance(part, Mapping) or part.get("type") != "reasoning_text":
                    raise WorkerContractError(
                        f"output[{index}].content[{content_index}] is unsupported"
                    )
                text = part.get("text")
                if not isinstance(text, str):
                    raise WorkerContractError("reasoning text is not text")
                reasoning_content.append({"type": "reasoning_text", "text": text})
            continuation.append(
                {"type": "reasoning", "content": reasoning_content}
            )
            continue
        if item_type == "message":
            content = item.get("content")
            if not isinstance(content, list):
                raise WorkerContractError(f"output[{index}].content is invalid")
            for content_index, part in enumerate(content):
                if not isinstance(part, Mapping) or part.get("type") != "output_text":
                    raise WorkerContractError(
                        f"output[{index}].content[{content_index}] is unsupported"
                    )
                text = part.get("text")
                if not isinstance(text, str):
                    raise WorkerContractError("output text is not text")
                text_parts.append(text)
            continuation.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": part.get("text")}
                        for part in content
                    ],
                }
            )
            continue
        if item_type == "function_call":
            call_id = _require_nonempty(
                item.get("call_id"), field="function_call.call_id", maximum=512
            )
            if call_id in call_ids:
                raise WorkerContractError("duplicate function call id")
            call_ids.add(call_id)
            call = {
                "call_id": call_id,
                "name": _require_tool_name(
                    item.get("name"), field="function_call.name"
                ),
                "arguments": _require_nonempty(
                    item.get("arguments"),
                    field="function_call.arguments",
                    maximum=2 * 1024 * 1024,
                ),
            }
            calls.append(call)
            continuation.append({"type": "function_call", **call})
            continue
        raise WorkerContractError(
            f"provider returned unsupported output item type {item_type!r}"
        )
    return "".join(text_parts), calls, continuation


def _terminal(event_type: str, event: Mapping[str, Any]) -> dict[str, Any]:
    response = event.get("response")
    if not isinstance(response, Mapping):
        raise WorkerContractError("terminal event has no response object")
    response_id = _require_nonempty(
        response.get("id"), field="response.id", maximum=512
    )
    model = _require_nonempty(
        response.get("model"), field="response.model", maximum=256
    )
    status = response.get("status")
    expected_status = event_type.removeprefix("response.")
    if status != expected_status:
        raise WorkerContractError("terminal event and response status disagree")
    system_fingerprint = response.get("system_fingerprint")
    if system_fingerprint is not None:
        system_fingerprint = _require_nonempty(
            system_fingerprint,
            field="response.system_fingerprint",
            maximum=512,
        )
    output_text, tool_calls, continuation = _terminal_output(response)
    return {
        "protocol": _PROTOCOL,
        "type": "terminal",
        "terminal_type": event_type,
        "response_id": response_id,
        "model": model,
        "system_fingerprint": system_fingerprint,
        "output_text": output_text,
        "tool_calls": tool_calls,
        # DeepSeek's Responses API is stateless. A tool round must replay the
        # complete prior output, including reasoning. This field lives only in
        # the parent process's bounded in-memory transport and is never copied
        # into traces, debug artifacts, or public evidence.
        "continuation_items": continuation if tool_calls else [],
        "usage": _usage(response),
    }


def _emit(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def _invoke(request: Mapping[str, Any]) -> int:
    api_key = request["api_key"]
    http_client = httpx.Client(trust_env=False, follow_redirects=False)
    client: Any = None
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=_BASE_URL,
            max_retries=0,
            http_client=http_client,
        )
        kwargs: dict[str, Any] = {
            "model": _MODEL,
            "input": request["input"],
            "max_output_tokens": request["max_output_tokens"],
            "reasoning": {"effort": "high"},
            "stream": True,
            "tools": request["tools"],
        }
        sequence = 0
        last_provider_sequence = -1
        terminal_seen = False
        stream = client.responses.create(**kwargs)
        for raw_event in stream:
            if sequence >= _MAX_EVENTS:
                raise WorkerContractError("provider stream exceeded event bound")
            event = _event_mapping(raw_event)
            event_type = event.get("type")
            if not isinstance(event_type, str) or not event_type:
                raise WorkerContractError("provider event has no type")
            provider_sequence = event.get("sequence_number")
            if (
                type(provider_sequence) is not int
                or provider_sequence < 0
                or provider_sequence <= last_provider_sequence
            ):
                raise WorkerContractError(
                    "provider SSE sequence is not strictly monotonic"
                )
            last_provider_sequence = provider_sequence
            if terminal_seen:
                raise WorkerContractError("provider emitted data after terminal")
            if event_type in _TERMINAL_TYPES:
                payload = _terminal(event_type, event)
                terminal_seen = True
            else:
                # Stream order and identity cross the boundary, but raw deltas
                # (especially reasoning) do not.
                payload = {
                    "protocol": _PROTOCOL,
                    "type": "event",
                    "event_type": event_type,
                }
            _emit(
                {
                    **payload,
                    "sequence": sequence,
                    "provider_sequence": provider_sequence,
                }
            )
            sequence += 1
        if not terminal_seen:
            # EOF is intentionally not promoted into a synthetic terminal.
            return 3
        return 0
    finally:
        try:
            if client is not None:
                client.close()
        finally:
            http_client.close()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["--probe"]:
        _emit(_probe())
        return 0 if openai.__version__ == _OPENAI_VERSION else 4
    if args:
        return 64
    raw = sys.stdin.buffer.read(_MAX_REQUEST_BYTES + 1)
    if len(raw) > _MAX_REQUEST_BYTES:
        return 65
    try:
        if openai.__version__ != _OPENAI_VERSION:
            raise WorkerContractError("OpenAI SDK version differs from the lock")
        request = _validate_request(json.loads(raw))
        return _invoke(request)
    except BaseException as exc:  # noqa: BLE001
        # Error details are deliberately type-only. Provider bodies and the
        # request (which carries the credential) never cross stderr/stdout.
        _emit(
            {
                "protocol": _PROTOCOL,
                "type": "error",
                "sequence": 0,
                "error_type": type(exc).__name__,
            }
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
