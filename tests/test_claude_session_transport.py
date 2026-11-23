# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Transport-level tests for the Claude organization-session boundary.

The fake executable records the environment it actually receives.  No Claude
provider or network endpoint is contacted.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import anyio
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._errors import CLIConnectionError, ProcessError
from claude_agent_sdk._internal.transport import subprocess_cli as _sdk_cli

from crucible.backends._claude_session_transport import (
    CLAUDE_SESSION_FIXED_PROCESS_ENV,
    CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL,
    CLAUDE_SESSION_REQUIRED_EXTRA_ARGS,
    ClaudeSessionIdentityError,
    FrozenClaudeSessionOptions,
    PinnedClaudeCLI,
    REVIEWED_CLAUDE_AGENT_SDK_VERSION,
    REVIEWED_CLAUDE_CODE_VERSION,
    REVIEWED_CLAUDE_FABLE_CODE_VERSION,
    REVIEWED_CLAUDE_LEGACY_CODE_VERSION,
    REVIEWED_CLAUDE_MYTHOS_CODE_VERSION,
    SanitizedClaudeSessionTransport,
)
from crucible.backends.claude import (
    CLAUDE_NVIDIA_ORG_ID,
    CLAUDE_NVIDIA_ORG_NAME,
)


def _recording_cli(
    tmp_path: Path,
    *,
    logged_in: bool = True,
    auth_method: str = "claude.ai",
    api_provider: str = "firstParty",
    org_id: str = CLAUDE_NVIDIA_ORG_ID,
    org_name: str = CLAUDE_NVIDIA_ORG_NAME,
    auth_sleep: bool = False,
    cli_version: str = REVIEWED_CLAUDE_FABLE_CODE_VERSION,
) -> tuple[Path, Path]:
    capture = tmp_path / "claude-child-env.jsonl"
    cli = tmp_path / "recording-claude"
    auth_delay = "time.sleep(60)" if auth_sleep else ""
    cli.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
import time

capture = {str(capture)!r}
record = {{
    "argv": sys.argv[1:],
    "env": dict(os.environ),
    "pid": os.getpid(),
}}
with open(capture, "a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\\n")

if sys.argv[1:] == ["auth", "status", "--json"]:
    {auth_delay}
    print(json.dumps({{
        "loggedIn": {logged_in!r},
        "authMethod": {auth_method!r},
        "apiProvider": {api_provider!r},
        "orgId": {org_id!r},
        "orgName": {org_name!r},
    }}), flush=True)
    raise SystemExit(0)
if sys.argv[1:] == ["-v"]:
    print({f"{cli_version} (Claude Code)"!r}, flush=True)
    raise SystemExit(0)
for _line in sys.stdin:
    pass
""",
        encoding="utf-8",
    )
    cli.chmod(0o700)
    return cli.resolve(), capture


def _model_admission_cli(
    tmp_path: Path,
    *,
    outcome: str,
) -> tuple[Path, Path]:
    """Create a provider-free CLI double for the SDK control protocol."""

    frames = tmp_path / f"model-admission-{outcome}.jsonl"
    cli = tmp_path / f"model-admission-{outcome}-claude"
    cli.write_text(
        f"""#!{sys.executable}
import json
import sys

frames = {str(frames)!r}
outcome = {outcome!r}

if sys.argv[1:] == ["auth", "status", "--json"]:
    print(json.dumps({{
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "orgId": {CLAUDE_NVIDIA_ORG_ID!r},
        "orgName": {CLAUDE_NVIDIA_ORG_NAME!r},
    }}), flush=True)
    raise SystemExit(0)
if sys.argv[1:] == ["-v"]:
    print({f"{REVIEWED_CLAUDE_FABLE_CODE_VERSION} (Claude Code)"!r}, flush=True)
    raise SystemExit(0)

for line in sys.stdin:
    frame = json.loads(line)
    with open(frames, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(frame, sort_keys=True) + "\\n")
    if frame.get("type") != "control_request":
        if frame.get("type") == "user":
            raise SystemExit(0)
        continue
    request = frame.get("request", {{}})
    request_id = frame.get("request_id")
    if request.get("subtype") == "initialize":
        print(json.dumps({{
            "type": "control_response",
            "response": {{
                "subtype": "success",
                "request_id": request_id,
                "response": {{"commands": []}},
            }},
        }}), flush=True)
    elif request.get("subtype") == "set_model":
        if outcome == "success":
            print(json.dumps({{
                "type": "control_response",
                "response": {{
                    "subtype": "success",
                    "request_id": request_id,
                }},
            }}), flush=True)
        elif outcome == "error":
            print(json.dumps({{
                "type": "control_response",
                "response": {{
                    "subtype": "error",
                    "request_id": request_id,
                    "error": "model excluded",
                }},
            }}), flush=True)
        elif outcome == "eof":
            raise SystemExit(0)
""",
        encoding="utf-8",
    )
    cli.chmod(0o700)
    return cli.resolve(), frames


def _stderr_then_exit_cli(
    tmp_path: Path,
    *,
    unterminated_chars: int,
    last_line: str,
) -> Path:
    """Create a CLI double whose model process only writes stderr and fails.

    It answers the identity and version probes, then writes one run of
    ``unterminated_chars`` characters without a newline, then ``last_line``,
    and exits 1 without reading its input.
    """

    cli = tmp_path / "stderr-claude"
    cli.write_text(
        f"""#!{sys.executable}
import json
import sys

if sys.argv[1:] == ["auth", "status", "--json"]:
    print(json.dumps({{
        "loggedIn": True,
        "authMethod": "claude.ai",
        "apiProvider": "firstParty",
        "orgId": {CLAUDE_NVIDIA_ORG_ID!r},
        "orgName": {CLAUDE_NVIDIA_ORG_NAME!r},
    }}), flush=True)
    raise SystemExit(0)
if sys.argv[1:] == ["-v"]:
    print({f"{REVIEWED_CLAUDE_FABLE_CODE_VERSION} (Claude Code)"!r}, flush=True)
    raise SystemExit(0)

sys.stderr.write("x" * {unterminated_chars})
sys.stderr.write("\\n" + {last_line!r} + "\\n")
sys.stderr.flush()
raise SystemExit(1)
""",
        encoding="utf-8",
    )
    cli.chmod(0o700)
    return cli.resolve()


def _transport(
    tmp_path: Path,
    cli_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
    model: str = "claude-fable-5-1",
    resume: str | None = None,
    stderr: Callable[[str], None] | None = None,
) -> SanitizedClaudeSessionTransport:
    home = tmp_path / "session-home"
    home.mkdir(exist_ok=True)
    safe_env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "NODE_EXTRA_CA_CERTS": "/etc/ssl/certs/ca-certificates.crt",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "TMPDIR": str(tmp_path),
        "TZ": "UTC",
        **(extra_env or {}),
    }
    options = ClaudeAgentOptions(
        cli_path=str(cli_path),
        cwd=str(tmp_path),
        env=safe_env,
        system_prompt="test",
        model=model,
        max_turns=1,
        extra_args=dict(CLAUDE_SESSION_REQUIRED_EXTRA_ARGS),
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        plugins=[],
        disallowed_tools=["Task", "Agent"],
        resume=resume,
        stderr=stderr,
    )
    return SanitizedClaudeSessionTransport(
        prompt="offline transport test",
        options=options,
        cli=PinnedClaudeCLI.capture(str(cli_path)),
        expected_org_id=CLAUDE_NVIDIA_ORG_ID,
        expected_org_name=CLAUDE_NVIDIA_ORG_NAME,
    )


def _records(capture: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in capture.read_text(encoding="utf-8").splitlines()
    ]


def _protocol_frames(capture: Path) -> list[dict[str, object]]:
    if not capture.exists():
        return []
    return [
        json.loads(line)
        for line in capture.read_text(encoding="utf-8").splitlines()
    ]


def test_session_transport_exact_sdk_runtime_guard():
    assert _sdk_cli.__version__ == REVIEWED_CLAUDE_AGENT_SDK_VERSION
    assert REVIEWED_CLAUDE_LEGACY_CODE_VERSION == "2.1.220"
    assert REVIEWED_CLAUDE_CODE_VERSION == "2.1.261"
    assert REVIEWED_CLAUDE_FABLE_CODE_VERSION == "2.1.263"
    assert REVIEWED_CLAUDE_MYTHOS_CODE_VERSION == "2.1.263"
    assert dict(CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL) == {
        "claude-opus-5": frozenset({"2.1.220", "2.1.261"}),
        "claude-fable-5-1": frozenset({"2.1.261", "2.1.263"}),
        "claude-mythos-5-1": frozenset({"2.1.263"}),
    }


def test_pinned_sdk_parser_preserves_raw_model_not_found_category():
    from claude_agent_sdk._internal.message_parser import parse_message
    from claude_agent_sdk.types import AssistantMessage

    assert _sdk_cli.__version__ == REVIEWED_CLAUDE_AGENT_SDK_VERSION
    parsed = parse_message({
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": "untrusted detail"}],
            "model": "<synthetic>",
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
        "error": "model_not_found",
        "session_id": "offline-session",
        "uuid": "offline-message",
    })

    assert isinstance(parsed, AssistantMessage)
    assert parsed.error == "model_not_found"
    assert parsed.model == "<synthetic>"


def test_session_transport_exposes_cli_version_only_while_connected(tmp_path):
    cli, _capture = _recording_cli(tmp_path)
    transport = _transport(
        tmp_path,
        cli,
        model="claude-mythos-5-1",
    )

    async def exercise() -> None:
        assert transport.validated_cli_version is None
        await transport.connect()
        assert (
            transport.validated_cli_version
            == REVIEWED_CLAUDE_MYTHOS_CODE_VERSION
        )
        await transport.close()
        assert transport.validated_cli_version is None

    asyncio.run(exercise())


def test_session_transport_does_not_publish_version_when_spawn_fails(
    monkeypatch,
    tmp_path,
):
    cli, _capture = _recording_cli(tmp_path)
    transport = _transport(
        tmp_path,
        cli,
        model="claude-mythos-5-1",
    )
    original_open_process = anyio.open_process
    open_calls = 0

    async def fail_model_spawn(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        if open_calls == 3:
            raise OSError("offline model-spawn failure")
        return await original_open_process(*args, **kwargs)

    monkeypatch.setattr(anyio, "open_process", fail_model_spawn)

    with pytest.raises(CLIConnectionError, match="offline model-spawn failure"):
        asyncio.run(transport.connect())

    assert open_calls == 3
    assert transport.validated_cli_version is None
    assert transport._probed_cli_version is None
    assert transport._process is None


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1"])
def test_session_model_admission_precedes_first_user_frame(tmp_path, model):
    from crucible.backends import claude as claude_mod

    cli, capture = _model_admission_cli(tmp_path, outcome="success")
    transport = _transport(tmp_path, cli, model=model)
    active_before = set(_sdk_cli._ACTIVE_CHILDREN)

    async def exercise() -> list[object]:
        return [
            message
            async for message in claude_mod._query_claude_session_with_model_preflight(
                prompt="must follow admission",
                options=transport._options,
                transport=transport,
            )
        ]

    assert asyncio.run(exercise()) == []
    frames = _protocol_frames(capture)
    assert [frame["type"] for frame in frames] == [
        "control_request",
        "control_request",
        "user",
    ]
    assert [
        frame.get("request", {}).get("subtype")
        for frame in frames[:2]
    ] == ["initialize", "set_model"]
    assert frames[1]["request"]["model"] == model
    assert frames[2]["message"] == {
        "role": "user",
        "content": "must follow admission",
    }
    assert frames[2]["session_id"] == ""
    assert set(_sdk_cli._ACTIVE_CHILDREN) == active_before


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-mythos-5-1"])
def test_session_model_rejection_sends_no_user_and_reaps_child(
    monkeypatch,
    tmp_path,
    model,
):
    from crucible.backends import claude as claude_mod

    cli, capture = _model_admission_cli(tmp_path, outcome="error")
    transport = _transport(tmp_path, cli, model=model)
    active_before = set(_sdk_cli._ACTIVE_CHILDREN)
    monkeypatch.setattr(
        claude_mod,
        "_CLAUDE_SESSION_MODEL_ADMISSION_TIMEOUT_SECONDS",
        0.1,
    )

    async def exercise() -> None:
        async for _message in claude_mod._query_claude_session_with_model_preflight(
            prompt="must not be sent",
            options=transport._options,
            transport=transport,
        ):
            pytest.fail("admission failure yielded a model message")

    with pytest.raises(ClaudeSessionIdentityError, match="rejected before prompt"):
        asyncio.run(exercise())
    frames = _protocol_frames(capture)
    assert [
        frame.get("request", {}).get("subtype")
        for frame in frames
    ] == ["initialize", "set_model"]
    assert frames[1]["request"]["model"] == model
    assert all(frame.get("type") != "user" for frame in frames)
    assert set(_sdk_cli._ACTIVE_CHILDREN) == active_before


@pytest.mark.parametrize("outcome", ["eof", "hang"])
def test_session_model_admission_transport_failure_remains_retriable(
    monkeypatch,
    tmp_path,
    outcome,
):
    from crucible.backends import claude as claude_mod

    cli, capture = _model_admission_cli(tmp_path, outcome=outcome)
    transport = _transport(tmp_path, cli)
    active_before = set(_sdk_cli._ACTIVE_CHILDREN)
    monkeypatch.setattr(
        claude_mod,
        "_CLAUDE_SESSION_MODEL_ADMISSION_TIMEOUT_SECONDS",
        0.1,
    )

    async def exercise() -> None:
        async for _message in claude_mod._query_claude_session_with_model_preflight(
            prompt="must not be sent",
            options=transport._options,
            transport=transport,
        ):
            pytest.fail("indeterminate admission yielded a model message")

    with pytest.raises(TimeoutError):
        asyncio.run(exercise())
    frames = _protocol_frames(capture)
    assert [
        frame.get("request", {}).get("subtype")
        for frame in frames
    ] == ["initialize", "set_model"]
    assert all(frame.get("type") != "user" for frame in frames)
    assert set(_sdk_cli._ACTIVE_CHILDREN) == active_before


@pytest.mark.parametrize(
    ("model", "cli_version", "reviewed_versions"),
    [
        ("claude-fable-5-1", "2.1.220", {"2.1.261", "2.1.263"}),
        ("claude-mythos-5-1", "2.1.261", {"2.1.263"}),
        ("claude-opus-5", "2.1.263", {"2.1.220", "2.1.261"}),
    ],
)
def test_session_transport_rejects_cross_model_cli_version_swap(
    tmp_path,
    model,
    cli_version,
    reviewed_versions,
):
    cli, capture = _recording_cli(tmp_path, cli_version=cli_version)
    transport = _transport(tmp_path, cli, model=model)

    with pytest.raises(ClaudeSessionIdentityError) as caught:
        asyncio.run(transport.connect())

    assert str(caught.value) == (
        "Claude organization-session transport requires an exact reviewed "
        f"Claude Code version in {sorted(reviewed_versions)!r} for model "
        f"{model!r}"
    )
    assert [record["argv"] for record in _records(capture)] == [
        ["auth", "status", "--json"],
        ["-v"],
    ]


def test_session_transport_model_cli_matrix_is_closed_and_immutable():
    assert dict(CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL) == {
        "claude-opus-5": frozenset({"2.1.220", "2.1.261"}),
        "claude-fable-5-1": frozenset({"2.1.261", "2.1.263"}),
        "claude-mythos-5-1": frozenset({"2.1.263"}),
    }
    with pytest.raises(TypeError):
        CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL[
            "claude-sonnet-4-6"
        ] = frozenset({"2.1.261"})


def test_session_transport_options_are_detached_frozen_and_sdk_compatible(
    tmp_path,
):
    cli, _capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)
    options = transport._options

    assert isinstance(options, FrozenClaudeSessionOptions)
    assert not isinstance(options, ClaudeAgentOptions)
    assert isinstance(options.allowed_tools, list)
    assert isinstance(options.extra_args, dict)
    assert isinstance(options.mcp_servers, dict)

    with pytest.raises(TypeError, match="immutable"):
        options.model = "claude-sonnet-4-6"
    with pytest.raises(TypeError, match="immutable"):
        options.allowed_tools.append("Bash")
    with pytest.raises(TypeError, match="immutable"):
        options.extra_args["fallback-model"] = "claude-sonnet-4-6"
    with pytest.raises(TypeError, match="immutable"):
        options.mcp_servers["hostile"] = {"type": "stdio"}
    with pytest.raises(TypeError):
        vars(options)
    with pytest.raises(AttributeError):
        _ = options.__dict__
    with pytest.raises(TypeError, match="immutable"):
        options.__init__()
    with pytest.raises(TypeError, match="immutable"):
        options.allowed_tools.__init__(["Bash"])
    with pytest.raises(TypeError, match="immutable"):
        options.extra_args.__init__({"fallback-model": "claude-sonnet-4-6"})


def test_frozen_options_survive_actual_sdk_post_connect_extraction(
    monkeypatch,
):
    from claude_agent_sdk._internal import client as sdk_client

    captured: dict[str, object] = {}
    mutation_errors: list[BaseException] = []
    safe_server = object()
    options = FrozenClaudeSessionOptions.from_kwargs(
        system_prompt="test",
        model="claude-opus-5",
        strict_mcp_config=True,
        setting_sources=[],
        skills=[],
        plugins=[],
        disallowed_tools=["Task", "Agent"],
        mcp_servers={
            "crucible": {
                "type": "sdk",
                "name": "crucible",
                "instance": safe_server,
            },
        },
        extra_args=dict(CLAUDE_SESSION_REQUIRED_EXTRA_ARGS),
    )

    class HostileWindowTransport:
        async def connect(self) -> None:
            mutations = (
                lambda: vars(options).__setitem__("skills", ["hostile"]),
                lambda: options.skills.__init__(["hostile"]),
                lambda: options.mcp_servers["crucible"].__init__({
                    "type": "sdk",
                    "name": "crucible",
                    "instance": object(),
                }),
                lambda: setattr(options, "hooks", {"PreToolUse": []}),
            )
            for mutation in mutations:
                try:
                    mutation()
                except (AttributeError, TypeError) as exc:
                    mutation_errors.append(exc)

        async def write(self, data: str) -> None:
            captured["write"] = data

        async def close(self) -> None:
            pass

        def is_ready(self) -> bool:
            return True

        async def end_input(self) -> None:
            pass

    class CapturingQuery:
        def __init__(self, **kwargs):
            captured["query_kwargs"] = kwargs

        async def start(self) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def wait_for_result_and_end_input(self) -> None:
            pass

        def spawn_task(self, awaitable) -> None:
            awaitable.close()

        async def receive_messages(self):
            if False:  # pragma: no cover - declares an async generator
                yield {}

        async def close(self) -> None:
            pass

        def close_receive_stream(self) -> None:
            pass

    monkeypatch.setattr(sdk_client, "Query", CapturingQuery)

    async def exercise() -> None:
        messages = sdk_client.InternalClient().process_query(
            prompt="offline SDK extraction test",
            options=options,
            transport=HostileWindowTransport(),
        )
        assert [message async for message in messages] == []

    asyncio.run(exercise())

    assert len(mutation_errors) == 4
    query_kwargs = captured["query_kwargs"]
    assert query_kwargs["skills"] == []
    assert query_kwargs["hooks"] is None
    assert query_kwargs["agents"] is None
    assert query_kwargs["sdk_mcp_servers"] == {"crucible": safe_server}


def test_session_transport_rejects_unreviewed_sdk_version(
    monkeypatch,
    tmp_path,
):
    cli, _capture = _recording_cli(tmp_path)
    monkeypatch.setattr(_sdk_cli, "__version__", "0.2.111")

    with pytest.raises(
        ClaudeSessionIdentityError,
        match="claude-agent-sdk==0.2.121",
    ):
        _transport(tmp_path, cli)


@pytest.mark.parametrize(
    ("option_name", "replacement", "match"),
    [
        ("model", "claude-sonnet-4-6", "exact model from reviewed set"),
        ("fallback_model", "claude-sonnet-4-6", "fallback_model"),
        ("extra_args", {}, "exact auxiliary-call"),
        (
            "extra_args",
            {
                "disable-slash-commands": None,
                "prompt-suggestions": "true",
            },
            "exact auxiliary-call",
        ),
        (
            "extra_args",
            {
                "disable-slash-commands": None,
                "prompt-suggestions": "false",
                "fallback-model": "claude-sonnet-4-6",
            },
            "exact auxiliary-call",
        ),
        (
            "settings",
            '{"env":{"ANTHROPIC_BASE_URL":"https://gateway.invalid"}}',
            "forbids settings",
        ),
        ("settings", "/tmp/hostile-settings.json", "forbids settings"),
        ("disallowed_tools", [], "requires Task and Agent"),
        ("continue_conversation", True, "forbids implicit continue"),
        ("session_id", "hostile-session", "forbids caller session_id"),
        ("fork_session", True, "forbids fork_session"),
        (
            "permission_prompt_tool_name",
            "mcp__hostile__permission",
            "forbids a permission prompt tool",
        ),
        ("add_dirs", ["/tmp"], "forbids add_dirs"),
        ("user", "nobody", "forbids user override"),
        ("sandbox", {"enabled": True}, "forbids SDK sandbox overrides"),
        ("betas", ["hostile-beta"], "forbids beta overrides"),
        (
            "enable_file_checkpointing",
            True,
            "forbids file checkpointing",
        ),
        ("include_hook_events", True, "forbids hook events"),
        ("include_partial_messages", True, "forbids partial messages"),
        ("session_store", object(), "forbids session_store"),
        ("can_use_tool", lambda *_args: True, "forbids can_use_tool"),
        ("strict_mcp_config", False, "requires strict_mcp_config"),
        ("setting_sources", ["user"], "forbids setting sources"),
        ("skills", ["hostile"], "requires an empty skill set"),
        (
            "plugins",
            [{"type": "local", "path": "/tmp/hostile"}],
            "forbids plugins",
        ),
        ("hooks", {}, "forbids hooks"),
        ("agents", {}, "forbids custom agents"),
        ("system_prompt", {"type": "preset"}, "literal system prompt"),
        (
            "mcp_servers",
            {
                "hostile": {
                    "type": "stdio",
                    "command": "sh",
                    "args": [],
                },
            },
            "permits only",
        ),
    ],
)
def test_session_transport_rejects_command_policy_drift_at_admission(
    tmp_path,
    option_name,
    replacement,
    match,
):
    cli, _capture = _recording_cli(tmp_path)
    home = tmp_path / "session-home"
    home.mkdir()
    options_kw = {
        "cli_path": str(cli),
        "cwd": str(tmp_path),
        "env": {
            "HOME": str(home),
            "PATH": "/usr/bin:/bin",
        },
        "system_prompt": "test",
        "model": "claude-opus-5",
        "max_turns": 1,
        "extra_args": dict(CLAUDE_SESSION_REQUIRED_EXTRA_ARGS),
        "strict_mcp_config": True,
        "setting_sources": [],
        "skills": [],
        "plugins": [],
        "disallowed_tools": ["Task", "Agent"],
        option_name: replacement,
    }

    with pytest.raises(ClaudeSessionIdentityError, match=match):
        SanitizedClaudeSessionTransport(
            prompt="offline transport test",
            options=ClaudeAgentOptions(**options_kw),
            cli=PinnedClaudeCLI.capture(str(cli)),
            expected_org_id=CLAUDE_NVIDIA_ORG_ID,
            expected_org_name=CLAUDE_NVIDIA_ORG_NAME,
        )


def test_session_transport_rejects_unreviewed_cli_version_before_model(
    tmp_path,
):
    cli, capture = _recording_cli(tmp_path, cli_version="2.1.219")
    transport = _transport(tmp_path, cli)

    with pytest.raises(
        ClaudeSessionIdentityError,
        match="reviewed Claude Code version",
    ):
        asyncio.run(transport.connect())

    assert [record["argv"] for record in _records(capture)] == [
        ["auth", "status", "--json"],
        ["-v"],
    ]


def test_fable_session_requires_a_reviewed_cli_version(tmp_path):
    cli, capture = _recording_cli(tmp_path, cli_version="2.1.220")
    transport = _transport(tmp_path, cli, model="claude-fable-5-1")

    with pytest.raises(
        ClaudeSessionIdentityError,
        match=r"2\.1\.261.*2\.1\.263.*claude-fable-5-1",
    ):
        asyncio.run(transport.connect())

    assert [record["argv"] for record in _records(capture)] == [
        ["auth", "status", "--json"],
        ["-v"],
    ]


def test_fable_session_spawns_exact_model_with_current_cli(tmp_path):
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli, model="claude-fable-5-1")

    async def exercise() -> None:
        await transport.connect()
        await transport.close()

    asyncio.run(exercise())

    records = _records(capture)
    model_argv = records[2]["argv"]
    assert model_argv.count("--model") == 1
    assert model_argv[model_argv.index("--model") + 1] == (
        "claude-fable-5-1"
    )
    assert "--fallback-model" not in model_argv


def test_mythos_session_spawns_exact_model_with_current_cli(tmp_path):
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli, model="claude-mythos-5-1")

    async def exercise() -> None:
        await transport.connect()
        await transport.close()

    asyncio.run(exercise())

    records = _records(capture)
    model_argv = records[2]["argv"]
    assert model_argv.count("--model") == 1
    assert model_argv[model_argv.index("--model") + 1] == (
        "claude-mythos-5-1"
    )
    assert "--fallback-model" not in model_argv


def test_opus_session_retains_legacy_reviewed_cli_compatibility(tmp_path):
    cli, capture = _recording_cli(tmp_path, cli_version="2.1.220")
    transport = _transport(tmp_path, cli, model="claude-opus-5")

    async def exercise() -> None:
        await transport.connect()
        await transport.close()

    asyncio.run(exercise())

    model_argv = _records(capture)[2]["argv"]
    assert model_argv[model_argv.index("--model") + 1] == "claude-opus-5"


def test_session_resume_value_cannot_become_a_flag_of_its_own(tmp_path):
    """A stored conversation handle that begins with a dash stays a value.

    ``resume`` carries a handle taken from earlier output or from a workflow
    conversation request. Passed as a separate argv token, a dash-leading
    handle is parsed by the CLI as another flag (CVE-2026-96620); the
    reviewed SDK binds it to ``--resume`` in a single token.
    """
    hostile_handle = "--mcp-config=/nonexistent.json"
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli, resume=hostile_handle)

    async def exercise() -> None:
        await transport.connect()
        await transport.close()

    asyncio.run(exercise())

    model_argv = _records(capture)[2]["argv"]
    assert model_argv.count(f"--resume={hostile_handle}") == 1
    assert "--resume" not in model_argv
    assert not [
        token for token in model_argv if token.startswith("--mcp-config")
    ]


def test_session_stderr_reaches_the_callback_in_lines_of_bounded_length(
    tmp_path,
):
    """Every stderr entry the callback receives has a bounded length.

    The adapter bounds the stderr it keeps by counting entries, so its
    memory bound rests on the length of one entry. The reviewed SDK hands
    over a run without a newline in pieces of at most ``max_buffer_size``
    plus one pipe read, and the line that follows the run stays intact.
    """
    reason = "No conversation found with session ID: offline-session"
    max_buffer_size = _sdk_cli._DEFAULT_MAX_BUFFER_SIZE
    one_pipe_read = 64 * 1024
    unterminated_chars = 2 * max_buffer_size + 300_000
    cli = _stderr_then_exit_cli(
        tmp_path,
        unterminated_chars=unterminated_chars,
        last_line=reason,
    )
    entries: list[str] = []
    transport = _transport(tmp_path, cli, stderr=entries.append)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)

    async def exercise() -> None:
        await transport.connect()
        try:
            with pytest.raises(ProcessError):
                async for _message in transport.read_messages():
                    pytest.fail("the failing CLI double produced a message")
            # The SDK reads stderr in a task of its own; let it reach EOF.
            with anyio.fail_after(30):
                while reason not in entries:
                    await anyio.sleep(0.01)
        finally:
            await transport.close()

    asyncio.run(exercise())

    assert transport._max_buffer_size == max_buffer_size
    assert entries[-1] == reason
    # Two full pieces, the rest of the run, and the line after it.
    assert [len(entry) > max_buffer_size for entry in entries] == [
        True,
        True,
        False,
        False,
    ]
    assert max(len(entry) for entry in entries) <= (
        max_buffer_size + one_pipe_read
    )
    # Nothing of the run is lost or repeated at the seams.
    assert sum(len(entry) for entry in entries[:-1]) == unterminated_chars
    assert {char for entry in entries[:-1] for char in entry} == {"x"}
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


@pytest.mark.parametrize(
    "mutation",
    [
        lambda options: setattr(options, "model", "claude-sonnet-4-6"),
        lambda options: setattr(
            options,
            "fallback_model",
            "claude-sonnet-4-6",
        ),
        lambda options: options.extra_args.__setitem__(
            "fallback-model",
            "claude-sonnet-4-6",
        ),
        lambda options: options.extra_args.__setitem__(
            "prompt-suggestions",
            "true",
        ),
        lambda options: options.disallowed_tools.clear(),
        lambda options: options.allowed_tools.append("Bash"),
        lambda options: setattr(options, "strict_mcp_config", False),
        lambda options: setattr(options, "setting_sources", None),
        lambda options: setattr(options, "skills", ["all"]),
        lambda options: options.mcp_servers.__setitem__(
            "hostile",
            {
                "type": "stdio",
                "command": "sh",
                "args": [],
            },
        ),
    ],
)
def test_session_transport_options_reject_post_admission_mutation(
    tmp_path,
    mutation,
):
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)

    with pytest.raises(TypeError, match="immutable"):
        mutation(transport._options)

    assert not capture.exists()


def test_session_transport_rejects_generated_command_drift_before_model_spawn(
    monkeypatch,
    tmp_path,
):
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)
    drifted_command = [*transport._expected_command, "--hostile-drift"]
    monkeypatch.setattr(transport, "_build_command", lambda: drifted_command)

    with pytest.raises(ClaudeSessionIdentityError, match="command drifted"):
        asyncio.run(transport.connect())

    assert [record["argv"] for record in _records(capture)] == [
        ["auth", "status", "--json"],
        ["-v"],
    ]


def test_session_transport_rejects_cli_identity_drift_before_any_child(
    tmp_path,
):
    cli, capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)
    cli.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli.chmod(0o700)

    with pytest.raises(ClaudeSessionIdentityError, match="identity drifted"):
        asyncio.run(transport.connect())

    assert not capture.exists()
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


def test_session_transport_scrubs_actual_auth_version_and_model_children(
    monkeypatch,
    tmp_path,
):
    cli, capture = _recording_cli(tmp_path)
    hostile = {
        "INFER_BASE_URL": "https://gateway.invalid/v1",
        "INFER_AUTH_TOKEN": "infer-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
        "CLAUDE_CODE_OAUTH_TOKEN": "oauth-secret",
        "MOONSHOT_API_KEY": "moonshot-secret",
        "GITLAB_TOKEN": "gitlab-secret",
        "DATABASE_URL": "postgresql://secret",
        "DB_PASSWORD": "db-secret",
        "NVBUGS_TOKEN": "nvbugs-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "HTTPS_PROXY": "https://proxy.invalid",
        "HTTP_PROXY": "http://proxy.invalid",
        "OTEL_EXPORTER_OTLP_HEADERS": "authorization=secret",
        "NODE_OPTIONS": "--require=/tmp/hostile.js",
        "SSH_AUTH_SOCK": "/tmp/hostile-agent.sock",
        "CLAUDE_CONFIG_DIR": "/tmp/hostile-claude-config",
        "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK": "1",
        "DISABLE_AUTOUPDATER": "0",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "0",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "0",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "0",
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "0",
        "FORCE_PROMPT_CACHING_5M": "0",
        "GIT_OPTIONAL_LOCKS": "1",
        "CLAUDE_CODE_SSE_PORT": "9999",
        "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "1",
    }
    transport = _transport(tmp_path, cli, extra_env=hostile)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)

    # Mutating ambient state after transport construction cannot affect the
    # frozen child mapping, including operational names.
    for name, value in hostile.items():
        monkeypatch.setenv(name, f"ambient-{value}")
    monkeypatch.setenv("HOME", "/tmp/ambient-home")
    monkeypatch.setenv("PATH", "/tmp/ambient-path")
    monkeypatch.setenv("LANG", "ambient-locale")

    async def exercise() -> None:
        await transport.connect()
        await transport.close()

    asyncio.run(exercise())

    records = _records(capture)
    assert [record["argv"] for record in records[:2]] == [
        ["auth", "status", "--json"],
        ["-v"],
    ]
    assert "--output-format" in records[2]["argv"]
    assert len(records) == 3

    expected_fixed_process_env = {
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
        "FORCE_PROMPT_CACHING_5M": "1",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    assert (
        dict(CLAUDE_SESSION_FIXED_PROCESS_ENV)
        == expected_fixed_process_env
    )
    with pytest.raises(TypeError):
        CLAUDE_SESSION_FIXED_PROCESS_ENV["CLAUDE_CODE_NO_MODEL_FALLBACK"] = "0"

    expected_env = {
        "HOME": str(tmp_path / "session-home"),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_CTYPE": "C.UTF-8",
        "NODE_EXTRA_CA_CERTS": "/etc/ssl/certs/ca-certificates.crt",
        "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        "TMPDIR": str(tmp_path),
        "TZ": "UTC",
        "PWD": str(tmp_path),
        "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
        "CLAUDE_AGENT_SDK_VERSION": REVIEWED_CLAUDE_AGENT_SDK_VERSION,
        **expected_fixed_process_env,
    }
    for record in records:
        assert record["env"] == expected_env
        assert not (
            set(record["env"]).intersection(hostile)
            - set(expected_fixed_process_env)
        )
        for name, value in expected_fixed_process_env.items():
            assert record["env"][name] == value
    model_argv = records[2]["argv"]
    assert model_argv.count("--model") == 1
    assert model_argv[model_argv.index("--model") + 1] == "claude-fable-5-1"
    assert "--fallback-model" not in model_argv
    assert model_argv.count("--disable-slash-commands") == 1
    assert model_argv.count("--prompt-suggestions") == 1
    assert (
        model_argv[model_argv.index("--prompt-suggestions") + 1]
        == "false"
    )
    with pytest.raises(TypeError):
        transport._session_process_env["ANTHROPIC_BASE_URL"] = (
            "https://gateway.invalid"
        )
    with pytest.raises(TypeError):
        transport._session_process_env[
            "CLAUDE_CODE_DISABLE_TERMINAL_TITLE"
        ] = "0"
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("loggedIn", False),
        ("loggedIn", 1),
        ("authMethod", "apiKey"),
        ("apiProvider", "bedrock"),
        ("orgId", "00000000-0000-0000-0000-000000000000"),
        ("orgName", "Different Organization"),
    ],
)
def test_session_transport_rejects_auth_principal_before_version_or_model(
    tmp_path,
    field,
    replacement,
):
    status = {
        "logged_in": True,
        "auth_method": "claude.ai",
        "api_provider": "firstParty",
        "org_id": CLAUDE_NVIDIA_ORG_ID,
        "org_name": CLAUDE_NVIDIA_ORG_NAME,
    }
    status_key = {
        "loggedIn": "logged_in",
        "authMethod": "auth_method",
        "apiProvider": "api_provider",
        "orgId": "org_id",
        "orgName": "org_name",
    }[field]
    status[status_key] = replacement
    cli, capture = _recording_cli(tmp_path, **status)
    transport = _transport(tmp_path, cli)

    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)
    with pytest.raises(ClaudeSessionIdentityError, match=field):
        asyncio.run(transport.connect())

    records = _records(capture)
    assert [record["argv"] for record in records] == [
        ["auth", "status", "--json"],
    ]
    assert transport._process is None
    assert transport.is_ready() is False
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


def test_session_transport_cancellation_reaps_identity_probe(
    tmp_path,
):
    cli, capture = _recording_cli(tmp_path, auth_sleep=True)
    transport = _transport(tmp_path, cli)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)

    async def exercise() -> int:
        task = asyncio.create_task(transport.connect())
        for _ in range(500):
            if capture.exists() and capture.stat().st_size:
                break
            await asyncio.sleep(0.01)
        else:  # pragma: no cover - diagnostic timeout
            raise AssertionError("fake auth probe did not start")
        pid = int(_records(capture)[0]["pid"])
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(exercise())

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert transport._process is None
    assert transport.is_ready() is False
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


def test_session_transport_close_reaps_model_child_inside_cancellation(
    tmp_path,
):
    cli, _capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)

    async def exercise() -> None:
        await transport.connect()
        assert transport._process in _sdk_cli._ACTIVE_CHILDREN
        with anyio.CancelScope() as cancelled:
            cancelled.cancel()
            await transport.close()

    asyncio.run(exercise())

    assert transport._process is None
    assert transport.is_ready() is False
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children


def test_session_transport_partial_main_setup_failure_reaps_child(
    monkeypatch,
    tmp_path,
):
    cli, _capture = _recording_cli(tmp_path)
    transport = _transport(tmp_path, cli)
    baseline_children = set(_sdk_cli._ACTIVE_CHILDREN)

    def fail_stream_setup(_stream):
        raise RuntimeError("synthetic stream setup failure")

    monkeypatch.setattr(_sdk_cli, "TextReceiveStream", fail_stream_setup)

    with pytest.raises(CLIConnectionError, match="stream setup failure"):
        asyncio.run(transport.connect())

    assert transport._process is None
    assert transport.is_ready() is False
    assert set(_sdk_cli._ACTIVE_CHILDREN) == baseline_children
