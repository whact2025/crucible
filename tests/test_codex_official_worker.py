# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib
import sys
import types
from enum import Enum
from pathlib import Path

import pytest


@pytest.fixture
def official_worker(monkeypatch):
    source = (
        Path(__file__).parents[1]
        / "backends"
        / "codex_worker"
        / "src"
    )
    monkeypatch.syspath_prepend(str(source))
    for name in tuple(sys.modules):
        if name == "crucible_codex_worker" or name.startswith(
            "crucible_codex_worker."
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("crucible_codex_worker.__main__")


def _request(tmp_path: Path, **overrides):
    payload = {
        "protocol": "crucible.codex-worker.v1",
        "cwd": str(tmp_path),
        "model": "gpt-5.6-sol",
        "effort": "max",
        "system": "system contract",
        "user": "task",
        "output_schema": {"type": "object"},
        "mcp_url": "http://127.0.0.1:34567/mcp",
        "native_shell": False,
        "trusted_project_instructions": False,
        "resume_thread_id": None,
    }
    payload.update(overrides)
    return payload


def test_validate_request_rejects_authority_widening_fields_and_remote_mcp(
    official_worker, tmp_path
):
    request = _request(tmp_path)
    request["unexpected"] = True
    with pytest.raises(official_worker.ProtocolError, match="fields mismatch"):
        official_worker.validate_request(request)

    request = _request(tmp_path, mcp_url="https://example.test/mcp")
    with pytest.raises(official_worker.ProtocolError, match="IPv4 loopback"):
        official_worker.validate_request(request)


@pytest.mark.parametrize(
    "mcp_url",
    [
        "http://127.0.0.1:34567/other",
        "http://127.0.0.1:34567/mcp/child",
        "http://127.0.0.1:34567/mcp\n",
        "http://127.0.0.1:34567/mcp?server=other",
        "http://127.0.0.1:34567/mcp#fragment",
    ],
)
def test_validate_request_requires_exact_loopback_mcp_endpoint(
    official_worker,
    tmp_path,
    mcp_url,
):
    with pytest.raises(official_worker.ProtocolError, match="IPv4 loopback"):
        official_worker.validate_request(
            _request(tmp_path, mcp_url=mcp_url)
        )


def test_confined_overrides_clear_ambient_surfaces(official_worker, tmp_path):
    request = official_worker.validate_request(_request(tmp_path))
    overrides = official_worker._config_overrides(request)

    assert "mcp_servers={}" in overrides
    assert "project_doc_max_bytes=0" in overrides
    assert "features.shell_tool=false" in overrides
    assert "features.unified_exec=false" in overrides
    assert "features.multi_agent=false" in overrides
    assert "features.multi_agent_v2=false" in overrides
    assert (
        'mcp_servers.crucible.url="http://127.0.0.1:34567/mcp"'
        in overrides
    )
    assert all("AGENTS" not in value for value in overrides)


def test_official_worker_rejects_unqualified_delegating_effort(
    official_worker,
    tmp_path,
):
    with pytest.raises(
        official_worker.ProtocolError,
        match="automatic task delegation",
    ):
        official_worker.validate_request(_request(tmp_path, effort="ultra"))


def test_trusted_context_remains_read_only_and_does_not_disable_project_docs(
    official_worker, tmp_path
):
    request = official_worker.validate_request(
        _request(
            tmp_path,
            trusted_project_instructions=True,
            mcp_url=None,
        )
    )
    overrides = official_worker._config_overrides(request)
    assert "project_doc_max_bytes=0" not in overrides
    assert "features.shell_tool=false" in overrides

    with pytest.raises(official_worker.ProtocolError, match="not qualified"):
        official_worker.validate_request(
            _request(
                tmp_path,
                trusted_project_instructions=True,
                native_shell=True,
            )
        )


def test_official_worker_rejects_native_shell_for_untrusted_context(
    official_worker,
    tmp_path,
):
    with pytest.raises(official_worker.ProtocolError, match="not qualified"):
        official_worker.validate_request(
            _request(
                tmp_path,
                trusted_project_instructions=False,
                native_shell=True,
            )
        )


def test_official_worker_uses_public_sdk_and_projects_complete_evidence(
    official_worker, monkeypatch, tmp_path
):
    captured = {}

    class ApprovalMode(Enum):
        deny_all = "deny_all"

    class Sandbox(Enum):
        read_only = "read-only"
        workspace_write = "workspace-write"

    class CodexConfig:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

    class Status(Enum):
        completed = "completed"

    class ToolStatus(Enum):
        completed = "completed"

    class Breakdown:
        input_tokens = 11
        cached_input_tokens = 3
        output_tokens = 7
        reasoning_output_tokens = 2
        total_tokens = 18

    class Usage:
        last = Breakdown()

    class ToolRoot:
        type = "mcpToolCall"
        id = "call-1"
        server = "crucible"
        tool = "read_source"
        status = ToolStatus.completed
        result = object()
        error = None

    class Item:
        root = ToolRoot()

    class TurnResult:
        id = "turn-1"
        status = Status.completed
        error = None
        final_response = '{"ok":true}'
        usage = Usage()
        items = [Item()]

    class Thread:
        id = "thread-1"

        def run(self, user, **kwargs):
            captured["run"] = (user, kwargs)
            return TurnResult()

    class Codex:
        def __init__(self, *, config):
            captured["client_config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            captured["closed"] = True

        def thread_start(self, **kwargs):
            captured["thread_start"] = kwargs
            return Thread()

        def thread_resume(self, *_args, **_kwargs):
            raise AssertionError("fresh request must not resume")

    sdk = types.ModuleType("openai_codex")
    sdk.__version__ = "0.144.4"
    sdk.ApprovalMode = ApprovalMode
    sdk.Codex = Codex
    sdk.CodexConfig = CodexConfig
    sdk.Sandbox = Sandbox
    monkeypatch.setitem(sys.modules, "openai_codex", sdk)

    request = official_worker.validate_request(_request(tmp_path))
    result = official_worker.run_request(request)

    assert result == {
        "protocol": "crucible.codex-worker.v1",
        "type": "terminal",
        "ok": True,
        "sdk_version": "0.144.4",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "status": "completed",
        "error": None,
        "final_response": '{"ok":true}',
        "usage": {
            "input_tokens": 11,
            "cached_input_tokens": 3,
            "output_tokens": 7,
            "reasoning_output_tokens": 2,
            "total_tokens": 18,
        },
        "tool_items": [
            {
                "call_id": "call-1",
                "server": "crucible",
                "tool": "read_source",
                "status": "completed",
                "has_result": True,
                "has_error": False,
            }
        ],
    }
    assert captured["thread_start"]["approval_mode"] is ApprovalMode.deny_all
    assert captured["thread_start"]["sandbox"] is Sandbox.read_only
    assert captured["run"][1]["output_schema"] == {"type": "object"}
    assert captured["closed"] is True


@pytest.mark.parametrize(
    "item_type",
    [
        "commandExecution",
        "fileChange",
        "dynamicToolCall",
        "collabAgentToolCall",
        "webSearch",
        "unknownFutureTool",
    ],
)
def test_confined_worker_rejects_unqualified_sdk_items(
    official_worker,
    item_type,
):
    item = types.SimpleNamespace(
        root=types.SimpleNamespace(type=item_type)
    )

    with pytest.raises(
        official_worker.ProtocolError,
        match="unqualified or unknown thread item",
    ):
        official_worker._tool_items([item])


def test_lock_exactly_binds_official_sdk_and_cli_runtime():
    root = Path(__file__).parents[1] / "backends" / "codex_worker"
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = (root / "uv.lock").read_text(encoding="utf-8")

    assert '"openai-codex==0.144.4"' in pyproject
    assert 'name = "openai-codex"' in lock
    assert 'version = "0.144.4"' in lock
    assert 'name = "openai-codex-cli-bin"' in lock
