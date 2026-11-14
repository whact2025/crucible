# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free conformance check executed by the isolated worker Python."""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai


EXPECTED_VERSION = "2.53.0"
EXPECTED_URL = "https://api.deepseek.com/responses"
MODEL = "deepseek-v4-flash"


def _completed_event() -> bytes:
    event = {
        "type": "response.completed",
        "sequence_number": 0,
        "response": {
            "id": "resp_provider_free_ci",
            "model": MODEL,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 1,
                    "cache_write_tokens": 0,
                },
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 1},
                "total_tokens": 12,
            },
        },
    }
    return (
        "data:" + json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n\n"
    ).encode("utf-8")


def main() -> int:
    if openai.__version__ != EXPECTED_VERSION:
        raise AssertionError("isolated worker did not import exact openai==2.53.0")
    full_history: list[dict[str, Any]] = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "private"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"life"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42",
        },
    ]
    expected_body = {
        "input": full_history,
        "max_output_tokens": 32_000,
        "model": MODEL,
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ],
    }
    attempts: list[httpx.Request] = []

    def success(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if str(request.url) != EXPECTED_URL:
            raise AssertionError("SDK changed the exact DeepSeek Responses URL")
        if json.loads(request.content) != expected_body:
            raise AssertionError("SDK changed the frozen stateless request body")
        if request.headers.get("authorization") != "Bearer sk-provider-free-test":
            raise AssertionError("SDK changed bearer credential placement")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_completed_event(),
        )

    http_client = httpx.Client(
        transport=httpx.MockTransport(success),
        trust_env=False,
        follow_redirects=False,
    )
    client = openai.OpenAI(
        api_key="sk-provider-free-test",
        base_url="https://api.deepseek.com",
        max_retries=0,
        http_client=http_client,
    )
    try:
        stream = client.responses.create(**expected_body)
        events = [event.model_dump(mode="json", exclude_unset=True) for event in stream]
    finally:
        client.close()
        http_client.close()
    if len(attempts) != 1 or [event.get("type") for event in events] != [
        "response.completed"
    ]:
        raise AssertionError("SDK changed the one-terminal streaming contract")

    failed_attempts = 0

    def fail_once(_request: httpx.Request) -> httpx.Response:
        nonlocal failed_attempts
        failed_attempts += 1
        return httpx.Response(500, json={"error": {"message": "synthetic"}})

    failing_http_client = httpx.Client(
        transport=httpx.MockTransport(fail_once),
        trust_env=False,
        follow_redirects=False,
    )
    failing_client = openai.OpenAI(
        api_key="sk-provider-free-test",
        base_url="https://api.deepseek.com",
        max_retries=0,
        http_client=failing_http_client,
    )
    try:
        try:
            failing_client.responses.create(**expected_body)
        except openai.InternalServerError:
            pass
        else:
            raise AssertionError("synthetic HTTP 500 unexpectedly succeeded")
    finally:
        failing_client.close()
        failing_http_client.close()
    if failed_attempts != 1:
        raise AssertionError("OpenAI SDK performed a hidden retry")

    print(
        json.dumps(
            {
                "attempts": len(attempts),
                "failed_attempts": failed_attempts,
                "model": MODEL,
                "openai_version": openai.__version__,
                "url": EXPECTED_URL,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
