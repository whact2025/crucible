# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``ClaudeBackend.chat_with_tools`` direct-API tool-use loop.

Mocks ``anthropic.Anthropic`` so no live API calls. Verifies:
- Single-turn path (no tool_use blocks) terminates cleanly.
- Multi-turn path: tool_use → dispatch → tool_result → next round.
- max_rounds hard ceiling sets stop_reason=max_turns.
- budget_usd soft cap aborts mid-loop with stop_reason=budget_exhausted.
- ToolCallCache hit skips dispatch but still records the transcript entry.
- Tool dispatch raising surfaces is_error=True on the tool_result block.
- API exception classification on round 1 surfaces InvocationResult.error.
- Cumulative Usage aggregates across rounds.
- Final InvocationResult shape matches invoke()'s contract.
- Codex / Local stubs fall back to invoke() with a warning.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from assay import Invocation, ToolCallCache


class _FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50,
                 cache_creation: int = 0, cache_read: int = 0) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


class _FakeTextBlock:
    """Mimics anthropic.types.TextBlock for isinstance checks."""
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock."""
    type = "tool_use"

    def __init__(self, id: str, name: str, input: dict) -> None:
        self.id = id
        self.name = name
        self.input = input


class _FakeResponse:
    def __init__(self, content: list, stop_reason: str = "end_turn",
                 usage: _FakeUsage | None = None,
                 model: str | None = "claude-fable-5-1") -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage or _FakeUsage()
        self.model = model


def _mock_toolset(dispatch_fn=None, anthropic_tools=None) -> MagicMock:
    ts = MagicMock()
    ts.write_names = set()
    descriptors = anthropic_tools or [
        {"name": "test_tool", "description": "test", "input_schema": {"type": "object"}},
    ]
    ts.read_only_names = {item["name"] for item in descriptors}
    ts.to_anthropic_tools.return_value = descriptors
    ts.dispatch.side_effect = dispatch_fn or (lambda name, args: "OK")
    return ts


def _patch_anthropic_types():
    """Patch anthropic.types so isinstance(TextBlock, ToolUseBlock) checks
    in chat_with_tools match our fakes."""
    import anthropic.types
    return patch.multiple(
        anthropic.types,
        TextBlock=_FakeTextBlock,
        ToolUseBlock=_FakeToolUseBlock,
    )


def test_single_turn_no_tools_terminates():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(
        system="be brief", user="hello", tools=None, max_turns=10,
        budget_usd=None,
    )
    toolset = _mock_toolset()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock("hi")],
        stop_reason="end_turn",
    )
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset)
    assert res.final_message == "hi"
    assert res.stop_reason == "end_turn"
    assert res.error is None
    assert res.execution_trace is not None
    assert res.execution_trace.tool_authority_digest == (
        backend._tool_authority_digest(  # noqa: SLF001
            toolset,
            context_budget=None,
        )
    )
    # One round only.
    assert mock_client.messages.create.call_count == 1
    # Tool dispatch never fired.
    toolset.dispatch.assert_not_called()


def test_multi_turn_tool_dispatch_loop():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="please read foo.txt", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset(
        dispatch_fn=lambda name, args: f"contents of {args.get('path')}",
    )
    # Round 1: model emits a tool_use; Round 2: model emits final text.
    round1 = _FakeResponse(
        content=[
            _FakeTextBlock("Let me read it."),
            _FakeToolUseBlock(id="tu_1", name="test_tool", input={"path": "foo.txt"}),
        ],
        stop_reason="tool_use",
    )
    round2 = _FakeResponse(
        content=[_FakeTextBlock("Done — foo.txt has X bytes.")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [round1, round2]
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset)
    assert res.stop_reason == "end_turn"
    assert "Done" in res.final_message
    toolset.dispatch.assert_called_once_with("test_tool", {"path": "foo.txt"})
    # Cumulative usage across rounds.
    assert res.usage.input_tokens == 200  # 100 per round × 2
    assert res.usage.output_tokens == 100


def test_direct_writer_dispatches_through_process_host(tmp_path):
    from crucible.backends.claude import ClaudeBackend
    from crucible.tools import ToolContext
    from crucible.tools.profiles import build_autonomous_implementer_profile

    toolset = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    first = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                id="tu_write",
                name="write_source",
                input={
                    "file_path": "target.txt",
                    "content": "isolated\n",
                },
            )
        ],
        stop_reason="tool_use",
    )
    final = _FakeResponse(
        content=[_FakeTextBlock("done")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first, final]

    with _patch_anthropic_types(), patch(
        "anthropic.Anthropic",
        return_value=mock_client,
    ):
        result = ClaudeBackend(
            capability="workspace_write",
            native_shell=False,
        ).chat_with_tools(
            Invocation(
                system="write exactly one file",
                user="write target.txt",
                tools=None,
                max_turns=2,
                budget_usd=None,
            ),
            toolset,
        )

    assert result.error is None
    assert (tmp_path / "target.txt").read_text(encoding="utf-8") == (
        "isolated\n"
    )
    second_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    tool_result = next(
        message["content"][0]
        for message in second_messages
        if (
            message["role"] == "user"
            and isinstance(message["content"], list)
            and message["content"]
            and isinstance(message["content"][0], dict)
            and message["content"][0].get("type") == "tool_result"
        )
    )
    assert tool_result["tool_use_id"] == "tu_write"
    assert "OK: wrote" in tool_result["content"]


def test_direct_writer_rejects_cache_before_inference(tmp_path):
    from crucible.backends.claude import ClaudeBackend
    from crucible.tools import ToolContext
    from crucible.tools.profiles import build_autonomous_implementer_profile

    toolset = build_autonomous_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="cannot safely cache",
    ):
        ClaudeBackend(
            capability="workspace_write",
            native_shell=False,
        ).chat_with_tools(
            Invocation(
                system="s",
                user="u",
                tools=None,
                max_turns=1,
                budget_usd=None,
            ),
            toolset,
            cache=MagicMock(),
        )


@pytest.mark.parametrize("backend_name", ["codex", "claude"])
def test_subprocess_backends_reject_unsupported_mutators_at_attachment(
    backend_name,
    tmp_path,
):
    from crucible.backends.claude import ClaudeBackend
    from crucible.backends.codex import CodexBackend
    from crucible.tools import ToolContext
    from crucible.tools._isolated_mutation_host import (
        MutationHostQualificationError,
    )
    from crucible.tools.profiles import build_implementer_profile

    toolset = build_implementer_profile(
        ToolContext(
            workspace_root=tmp_path,
            repo_root=tmp_path,
            write_scope=("target.txt",),
        )
    )
    backend = (
        CodexBackend(capability="workspace_write", native_shell=False)
        if backend_name == "codex"
        else ClaudeBackend(
            capability="workspace_write",
            native_shell=False,
        )
    )

    with pytest.raises(
        MutationHostQualificationError,
        match="build_project, create_rust_crate, materialize_artifact",
    ):
        backend.attach_toolset(toolset)


def test_direct_tool_trace_binds_exact_ids_and_caps_cumulative_utf8_results():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    invocation = Invocation(
        system="sys",
        user="read twice",
        tools=None,
        max_turns=3,
        budget_usd=None,
    )
    toolset = _mock_toolset(
        dispatch_fn=lambda _name, _args: "é",
    )
    responses = [
        _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id="tu_utf8_1",
                    name="test_tool",
                    input={},
                )
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id="tu_utf8_2",
                    name="test_tool",
                    input={},
                )
            ],
            stop_reason="tool_use",
        ),
    ]
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses

    with _patch_anthropic_types(), patch(
        "anthropic.Anthropic",
        return_value=mock_client,
    ):
        result = backend.chat_with_tools(
            invocation,
            toolset,
            context_budget=3,
        )

    assert result.stop_reason == "context_limit"
    assert result.error is not None
    assert "4 > 3 UTF-8 bytes" in result.error.message
    assert toolset.dispatch.call_count == 2
    assert result.execution_trace is not None
    assert result.execution_trace.terminal_event_count == 1
    assert result.execution_trace.events[-1].kind == "terminal"
    calls = [
        event
        for event in result.execution_trace.events
        if event.kind == "tool_call"
    ]
    results = [
        event
        for event in result.execution_trace.events
        if event.kind == "tool_result"
    ]
    assert [(event.parent_id, event.child_id) for event in calls] == [
        ("messages_round_1_attempt_1", "tu_utf8_1"),
        ("messages_round_2_attempt_1", "tu_utf8_2"),
    ]
    assert [(event.parent_id, event.child_id) for event in results] == [
        ("messages_round_1_attempt_1", "tu_utf8_1"),
    ]
    assert any(
        event.kind == "limit" and event.name == "context_bytes"
        for event in result.execution_trace.events
    )


def test_direct_duplicate_tool_call_id_rejects_before_second_dispatch():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    toolset = _mock_toolset()
    duplicate = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                id="tu_duplicate",
                name="test_tool",
                input={},
            )
        ],
        stop_reason="tool_use",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [duplicate, duplicate]

    with _patch_anthropic_types(), patch(
        "anthropic.Anthropic",
        return_value=mock_client,
    ):
        result = backend.chat_with_tools(
            Invocation(
                system="sys",
                user="loop",
                tools=None,
                max_turns=2,
                budget_usd=None,
            ),
            toolset,
        )

    assert result.stop_reason == "error"
    assert result.error is not None
    assert "duplicate tool-use id" in result.error.message
    assert toolset.dispatch.call_count == 1
    assert result.execution_trace is not None
    assert result.execution_trace.terminal_event_count == 1


def test_hallucinated_tool_name_is_rejected_before_cache_or_dispatch():
    """The digest's per-call-exact policy is an authorization boundary."""
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    invocation = Invocation(
        system="sys",
        user="use the exposed tool",
        tools=None,
        max_turns=2,
        budget_usd=None,
    )
    toolset = _mock_toolset()
    cache = MagicMock()
    cache.get.return_value = "must-not-be-used"
    first = _FakeResponse(
        content=[
            _FakeToolUseBlock(
                id="tu-unknown",
                name="unadvertised_tool",
                input={},
            ),
        ],
        stop_reason="tool_use",
    )
    final = _FakeResponse(
        content=[_FakeTextBlock("done")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [first, final]

    with _patch_anthropic_types(), patch(
        "anthropic.Anthropic",
        return_value=mock_client,
    ):
        result = backend.chat_with_tools(
            invocation,
            toolset,
            cache=cache,
        )

    assert result.final_message == "done"
    toolset.dispatch.assert_not_called()
    cache.get.assert_not_called()
    cache.put.assert_not_called()
    second_messages = mock_client.messages.create.call_args_list[1].kwargs[
        "messages"
    ]
    rejected = next(
        message["content"][0]
        for message in second_messages
        if (
            isinstance(message["content"], list)
            and message["content"]
            and isinstance(message["content"][0], dict)
            and message["content"][0].get("type") == "tool_result"
        )
    )
    assert rejected["tool_use_id"] == "tu-unknown"
    assert rejected["is_error"] is True
    assert "not in the exposed tool catalog" in rejected["content"]


def test_tool_authority_digest_is_identical_across_shared_backends():
    """Equivalent authority must retain one identity across adapter routes."""
    from crucible.backends import kimi as kimi_mod
    from crucible.backends.claude import ClaudeBackend
    from crucible.backends.codex import CodexBackend
    from crucible.backends.openai_compat import OpenAIBackend

    toolset = _mock_toolset()
    context_budget = 4096
    codex = CodexBackend(capability="read_only")
    claude = ClaudeBackend(capability="read_only")
    openai = OpenAIBackend(model="x", capability="read_only")
    codex.attach_toolset(toolset, context_budget=context_budget)
    kimi_names, kimi_writes = kimi_mod._tool_inventory(toolset)  # noqa: SLF001
    kimi_toolset = kimi_mod._FrozenMcpToolset(  # noqa: SLF001
        toolset,
        names=kimi_names,
        write_names=kimi_writes,
    )
    try:
        digests = {
            codex._attached_tool_authority_digest(),  # noqa: SLF001
            claude._tool_authority_digest(  # noqa: SLF001
                toolset,
                context_budget=context_budget,
            ),
            openai._tool_authority_digest(  # noqa: SLF001
                toolset,
                context_budget=context_budget,
            ),
            kimi_mod._tool_authority_digest(  # noqa: SLF001
                kimi_toolset,
                capability="read_only",
                context_budget=context_budget,
            ),
        }
    finally:
        codex.close()

    assert len(digests) == 1
    assert next(iter(digests)).startswith("sha256:")


def test_max_rounds_ceiling_returns_max_turns():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="loop forever", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    # Every response asks for another tool call — loop will only stop
    # when max_rounds is hit.
    mock_client = MagicMock()
    calls = 0

    def another_tool_call(**_kwargs):
        nonlocal calls
        calls += 1
        return _FakeResponse(
            content=[
                _FakeToolUseBlock(
                    id=f"tu_{calls}",
                    name="test_tool",
                    input={},
                )
            ],
            stop_reason="tool_use",
        )

    mock_client.messages.create.side_effect = another_tool_call
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset, max_rounds=3)
    assert res.stop_reason == "max_turns"
    assert mock_client.messages.create.call_count == 3


def test_budget_exhausted_terminates_loop():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(
        system="sys", user="expensive", tools=None, max_turns=10,
        budget_usd=0.0001,  # immediate exhaustion after round 1
    )
    toolset = _mock_toolset()
    # Heavy usage so even one round exceeds the tiny budget.
    big_usage = _FakeUsage(input_tokens=10_000, output_tokens=5_000)
    response = _FakeResponse(
        content=[_FakeToolUseBlock(id="tu_1", name="test_tool", input={})],
        stop_reason="tool_use",
        usage=big_usage,
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = response
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset, max_rounds=10)
    assert res.stop_reason == "budget_exhausted"
    # Exactly one round runs before the cap kicks in.
    assert mock_client.messages.create.call_count == 1


def test_tool_cache_hit_skips_dispatch():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)
    cache = ToolCallCache()
    cache.put("test_tool", {"path": "foo"}, "cached result")
    toolset = _mock_toolset()
    round1 = _FakeResponse(
        content=[_FakeToolUseBlock(id="tu_1", name="test_tool", input={"path": "foo"})],
        stop_reason="tool_use",
    )
    round2 = _FakeResponse(
        content=[_FakeTextBlock("done")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [round1, round2]
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset, cache=cache)
    assert res.stop_reason == "end_turn"
    # Toolset.dispatch should NOT have been called — cache served the result.
    toolset.dispatch.assert_not_called()
    # The transcript still records the cached call.
    tool_msgs = [m for m in res.transcript if m.role == "tool"]
    assert any("cached result" in m.content for m in tool_msgs)


def test_tool_dispatch_error_surfaces_is_error():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)

    def boom(name, args):
        raise RuntimeError("kaboom")

    toolset = _mock_toolset(dispatch_fn=boom)
    round1 = _FakeResponse(
        content=[_FakeToolUseBlock(id="tu_1", name="test_tool", input={})],
        stop_reason="tool_use",
    )
    round2 = _FakeResponse(
        content=[_FakeTextBlock("aborting")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [round1, round2]
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        res = backend.chat_with_tools(inv, toolset)
    assert res.stop_reason == "end_turn"
    # The error result is in the transcript.
    tool_msgs = [m for m in res.transcript if m.role == "tool"]
    assert any("ERROR: RuntimeError: kaboom" in m.content for m in tool_msgs)
    # Round 2's messages.create call must have received the error
    # tool_result. MagicMock holds references to the messages list
    # (which is mutated each round), so find the tool_result message
    # by role + content shape rather than indexing.
    second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
    tool_result_msgs = [
        m for m in second_call_kwargs["messages"]
        if m["role"] == "user"
        and isinstance(m["content"], list)
        and m["content"] and isinstance(m["content"][0], dict)
        and m["content"][0].get("type") == "tool_result"
    ]
    assert len(tool_result_msgs) == 1
    tool_result = tool_result_msgs[0]["content"][0]
    assert tool_result["is_error"] is True
    assert "kaboom" in tool_result["content"]


def test_api_exception_round_1_surfaces_error():
    """A non-retriable transport error breaks the loop immediately."""
    from crucible.backends.claude import ClaudeBackend
    from assay.types.invocation import TransportError

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ConnectionError("timeout")
    # Classify as non-retriable so the per-round retry loop doesn't fire.
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client), \
         patch.object(
             backend, "_classify_exception",
             return_value=TransportError(
                 kind="conn", message="timeout", retriable=False,
             ),
         ):
        res = backend.chat_with_tools(inv, toolset, retry_budget=3)
    assert res.error is not None
    assert "timeout" in res.error.message
    # Non-retriable → one attempt, no retries.
    assert mock_client.messages.create.call_count == 1


def test_transient_without_usage_envelope_does_not_retry():
    """A direct transport failure cannot safely authorize a paid retry.

    The request may have reached the provider even though no response usage
    envelope arrived.  Retrying would make logical token/cost accounting
    permanently ambiguous, so even an otherwise-retriable classification is
    made terminal.
    """
    from crucible.backends.claude import ClaudeBackend
    from assay.types.invocation import TransportError

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()

    # The second response would succeed, but must never be requested after an
    # unaccounted physical attempt.
    success_response = _FakeResponse(
        content=[_FakeTextBlock("ok")],
        stop_reason="end_turn",
    )
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = [
        ConnectionError("flake"),  # round 1 attempt 1: transient
        success_response,           # round 1 attempt 2: success
    ]
    # Force the underlying classification to be retriable; the direct-loop
    # accounting guard must still suppress the retry.
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client), \
         patch("time.sleep") as sleep_mock, \
         patch.object(
             backend, "_classify_exception",
             return_value=TransportError(kind="conn", message="flake", retriable=True),
         ):
        res = backend.chat_with_tools(inv, toolset, retry_budget=3)
    assert res.error is not None
    assert res.error.retriable is False
    assert "automatic retry suppressed" in res.error.message
    assert "token/cost accounting is incomplete" in res.error.message
    assert res.final_message == ""
    assert res.stop_reason == "error"
    assert res.completeness.token_usage_complete is False
    assert res.completeness.cost_complete is False
    assert res.completeness.terminal_complete is False
    assert mock_client.messages.create.call_count == 1
    sleep_mock.assert_not_called()


def test_retry_budget_does_not_override_incomplete_accounting():
    """A larger compatibility retry budget cannot authorize blind retries."""
    from crucible.backends.claude import ClaudeBackend
    from assay.types.invocation import TransportError

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ConnectionError("flake")
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client), \
         patch("time.sleep"), \
         patch.object(
             backend, "_classify_exception",
             return_value=TransportError(kind="conn", message="flake", retriable=True),
    ):
        res = backend.chat_with_tools(inv, toolset, retry_budget=2)
    assert res.error is not None
    assert res.error.retriable is False
    assert "automatic retry suppressed" in res.error.message
    assert mock_client.messages.create.call_count == 1


def test_max_response_tokens_passed_through():
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock("ok")], stop_reason="end_turn",
    )
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        backend.chat_with_tools(inv, toolset, max_response_tokens=32_000)
    kwargs = mock_client.messages.create.call_args_list[0].kwargs
    assert kwargs["max_tokens"] == 32_000


def test_codex_strict_requires_and_uses_the_mcp_toolset():
    """A supplied MCP ToolSet satisfies Codex strict tool-loop admission."""
    from crucible.backends.codex import CodexBackend

    backend = CodexBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=2, budget_usd=None)
    toolset = _mock_toolset()
    with patch.object(backend, "invoke") as mock_invoke:
        mock_invoke.return_value = "fallback-result"
        result = backend.chat_with_tools(inv, toolset, strict=True)
    assert result == "fallback-result"
    mock_invoke.assert_called_once_with(inv)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("cache", object()),
        ("max_rounds", 10),
        ("max_response_tokens", 16_000),
        ("retry_budget", 3),
    ],
)
def test_codex_rejects_unsupported_chat_settings_before_transport(
    setting,
    value,
):
    from crucible.backends.codex import CodexBackend

    backend = CodexBackend(capability="read_only")
    inv = Invocation(
        system="sys",
        user="x",
        tools=None,
        max_turns=2,
        budget_usd=None,
    )
    toolset = _mock_toolset()
    with (
        patch.object(backend, "attach_toolset") as attach,
        patch.object(backend, "invoke") as invoke,
        pytest.raises(NotImplementedError, match=setting),
    ):
        backend.chat_with_tools(inv, toolset, **{setting: value})

    attach.assert_not_called()
    invoke.assert_not_called()


def test_codex_strict_without_toolset_rejects_before_transport():
    from crucible.backends.codex import CodexBackend

    backend = CodexBackend(capability="read_only")
    inv = Invocation(
        system="sys",
        user="x",
        tools=None,
        max_turns=2,
        budget_usd=None,
    )
    with (
        patch.object(backend, "invoke") as invoke,
        pytest.raises(NotImplementedError, match="requires a toolset"),
    ):
        backend.chat_with_tools(inv, None, strict=True)

    invoke.assert_not_called()


def test_openai_compat_strict_no_toolset_raises_not_implemented():
    """OpenAIBackend honors strict only when NO toolset is passed; with a
    toolset it drives a real OpenAI tool loop (no stub fallback)."""
    from crucible.backends.openai_compat import OpenAIBackend

    backend = OpenAIBackend(model="local-test", base_url="http://localhost:9999")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=2, budget_usd=None)
    with pytest.raises(NotImplementedError, match="chat_with_tools"):
        backend.chat_with_tools(inv, None, strict=True)


def test_codex_stub_falls_back_to_invoke():
    from crucible.backends.codex import CodexBackend

    backend = CodexBackend(capability="read_only")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=2, budget_usd=None)
    toolset = _mock_toolset()
    with patch.object(backend, "invoke") as mock_invoke:
        mock_invoke.return_value = "fallback-result"
        result = backend.chat_with_tools(inv, toolset)
    assert result == "fallback-result"
    mock_invoke.assert_called_once_with(inv)
    # Toolset dropped (no dispatch call attempted).
    toolset.dispatch.assert_not_called()


def test_openai_compat_no_toolset_falls_back_to_invoke():
    from crucible.backends.openai_compat import OpenAIBackend

    backend = OpenAIBackend(model="local-test", base_url="http://localhost:9999")
    inv = Invocation(system="sys", user="x", tools=None, max_turns=2, budget_usd=None)
    with patch.object(backend, "invoke") as mock_invoke:
        mock_invoke.return_value = "fallback-result"
        result = backend.chat_with_tools(inv, None)
    assert result == "fallback-result"
    mock_invoke.assert_called_once_with(inv)


# ----------------------------------------------------------------------
# Prompt-cache marker placement (Anthropic ephemeral cache_control)
# ----------------------------------------------------------------------


def _cache_marked_block_count(messages: list[dict]) -> int:
    """Count content blocks in ``messages`` carrying cache_control."""
    n = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "cache_control" in blk:
                    n += 1
    return n


def _last_user_block_has_cache(messages: list[dict]) -> bool:
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            last_user_idx = i
    if last_user_idx < 0:
        return False
    content = messages[last_user_idx].get("content")
    if not isinstance(content, list) or not content:
        return False
    last_block = content[-1]
    return isinstance(last_block, dict) and "cache_control" in last_block


def _snapshotting_create(responses: list) -> tuple[list, callable]:
    """Build a side_effect that deep-copies ``messages`` at call time.

    MagicMock.call_args_list captures live references to mutable
    objects, so by inspection time the chat loop has already appended
    later rounds' assistant content to the list. The returned
    ``snapshots`` list contains a deep-copied messages-list per call,
    matching what the API actually received.
    """
    import copy as _copy
    snapshots: list[list[dict]] = []
    iterator = iter(responses)

    def side_effect(*args, **kwargs):
        snapshots.append(_copy.deepcopy(kwargs.get("messages", [])))
        return next(iterator)

    return snapshots, side_effect


def test_cache_marker_set_on_initial_user_message():
    """First round: the initial user message (bare string) is promoted
    to a block list with cache_control on its text block."""
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="hello", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    snapshots, side_effect = _snapshotting_create([
        _FakeResponse(content=[_FakeTextBlock("hi")], stop_reason="end_turn"),
    ])
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = side_effect
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        backend.chat_with_tools(inv, toolset)
    assert len(snapshots) == 1
    messages = snapshots[0]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    # Promoted from str -> [{"type": "text", "text": "hello", "cache_control": ...}]
    assert isinstance(content, list)
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "hello"
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_cache_marker_moves_to_latest_user_message_each_round():
    """Across rounds, exactly ONE message-level cache_control marker is
    live at request time, on the latest user message — earlier markers
    get stripped so we don't blow past Anthropic's 4-breakpoint cap."""
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="please read foo.txt", tools=None,
                     max_turns=10, budget_usd=None)
    toolset = _mock_toolset(dispatch_fn=lambda name, args: "tool output")

    snapshots, side_effect = _snapshotting_create([
        _FakeResponse(
            content=[_FakeToolUseBlock(id="tu_1", name="test_tool", input={})],
            stop_reason="tool_use",
        ),
        _FakeResponse(
            content=[_FakeToolUseBlock(id="tu_2", name="test_tool", input={})],
            stop_reason="tool_use",
        ),
        _FakeResponse(content=[_FakeTextBlock("done")], stop_reason="end_turn"),
    ])
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = side_effect
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        backend.chat_with_tools(inv, toolset)

    assert len(snapshots) == 3
    # On every round, exactly one message-level cache marker is live,
    # and it sits on the latest user message.
    for r, messages in enumerate(snapshots):
        assert _cache_marked_block_count(messages) == 1, (
            f"round {r+1}: expected exactly one cache_control marker, "
            f"got: {messages}"
        )
        assert _last_user_block_has_cache(messages), (
            f"round {r+1}: latest user message missing cache_control: "
            f"{messages}"
        )


def test_cache_marker_does_not_pollute_assistant_messages():
    """cache_control is only added to user messages; assistant content
    (returned by the API verbatim) is never mutated."""
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys", user="go", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset(dispatch_fn=lambda name, args: "ok")
    snapshots, side_effect = _snapshotting_create([
        _FakeResponse(
            content=[
                _FakeTextBlock("thinking..."),
                _FakeToolUseBlock(id="tu", name="test_tool", input={}),
            ],
            stop_reason="tool_use",
        ),
        _FakeResponse(content=[_FakeTextBlock("done")], stop_reason="end_turn"),
    ])
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = side_effect
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        backend.chat_with_tools(inv, toolset)
    # Round 2 snapshot: [initial_user, assistant_r1, tool_result_user]
    r2 = snapshots[1]
    assert r2[1]["role"] == "assistant"
    # Assistant blocks must not carry cache_control. We deepcopy the
    # snapshot, so attrs survive; check both dict-key and attr forms.
    for blk in r2[1]["content"]:
        if isinstance(blk, dict):
            assert "cache_control" not in blk, (
                "helper must not mutate assistant content"
            )
        else:
            assert not hasattr(blk, "cache_control"), (
                "helper must not mutate assistant content"
            )


def test_system_block_cache_marker_unchanged():
    """The system block's cache_control stays put across rounds."""
    from crucible.backends.claude import ClaudeBackend

    backend = ClaudeBackend(capability="read_only")
    inv = Invocation(system="sys text", user="go", tools=None, max_turns=10, budget_usd=None)
    toolset = _mock_toolset()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _FakeResponse(
        content=[_FakeTextBlock("ok")], stop_reason="end_turn",
    )
    with _patch_anthropic_types(), \
         patch("anthropic.Anthropic", return_value=mock_client):
        backend.chat_with_tools(inv, toolset)
    # system kwarg is a list[dict] — deepcopy-immune, safe to inspect post-hoc.
    system = mock_client.messages.create.call_args_list[0].kwargs["system"]
    assert isinstance(system, list) and len(system) == 1
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "sys text"
