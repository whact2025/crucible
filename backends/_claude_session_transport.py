# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed Claude Code transport for organization-session routes.

``claude-agent-sdk==0.2.121`` merges the complete parent environment into
both its normal Claude Code child and its preliminary version-check child.
That is appropriate for its general-purpose transport, but it is not an
acceptable credential boundary for Crucible's organization-session route.

This module is intentionally coupled to the exact reviewed SDK release.  It
subclasses the SDK transport only to preserve its command construction and
stream protocol while replacing every subprocess launch with an explicit,
frozen environment.  No process-global environment mutation is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from contextlib import suppress
from dataclasses import dataclass, fields
from pathlib import Path
from subprocess import PIPE
from types import MappingProxyType
from typing import Any, cast

import anyio
from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
from claude_agent_sdk._internal.transport import subprocess_cli as _sdk_cli
from claude_agent_sdk.types import ClaudeAgentOptions

logger = logging.getLogger(__name__)

# Reviewed on 2026-11-20 by an offline source comparison of the 0.2.110 and
# 0.2.121 releases. The live check of an organization session on 0.2.121 is
# still owed. Everything this module replaces (``connect``,
# ``_check_claude_version``), copies, or reads from ``SubprocessCLITransport``
# is the same text in both releases. Four inherited behaviors differ and are
# part of what this version admits; the owner accepted them on 2026-11-21:
# - ``--resume`` and ``--session-id`` reach the CLI as one ``--flag=value``
#   token, so a value that begins with a dash is no longer read as a flag
#   of its own (the fix for CVE-2026-96620);
# - the stderr callback receives complete lines instead of raw read chunks,
#   so the adapter's 200-entry stderr bound counts lines. The adapter keeps
#   the last 200 lines and drops earlier ones: a rejected-resume reason is
#   seen, and reported as ``stale_conversation``, unless the CLI writes 200
#   or more lines after it. One kept line is at most ``max_buffer_size``
#   plus one pipe read long, because a longer run without a newline is
#   handed over in pieces;
# - ``close()`` is shielded against cancellation, bounds every wait, and
#   keeps a child it could not reap tracked for the exit-time reaper;
# - stdout is framed by line: a large frame no longer loses whitespace at a
#   read boundary, and a malformed JSON line fails the turn at once.
# A further consequence of the second point, recorded on 2026-11-21 after
# that acceptance: the adapter redacts each kept stderr entry on its own,
# and an entry is now one line. A credential shape that spans a line break
# (a name and its value, ``Bearer`` and its token, a quoted value, or a
# secret that itself holds a newline) is not matched and stays readable in
# the adapter's ERROR record and in ``cli_stderr``. On 0.2.110 an entry was
# one pipe read, so such a shape was matched when both lines arrived in one
# read.
REVIEWED_CLAUDE_AGENT_SDK_VERSION = "0.2.121"
REVIEWED_CLAUDE_LEGACY_CODE_VERSION = "2.1.220"
REVIEWED_CLAUDE_CODE_VERSION = "2.1.261"
REVIEWED_CLAUDE_FABLE_CODE_VERSION = "2.1.263"
REVIEWED_CLAUDE_MYTHOS_CODE_VERSION = "2.1.263"
CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL = MappingProxyType({
    "claude-opus-5": frozenset({
        REVIEWED_CLAUDE_LEGACY_CODE_VERSION,
        REVIEWED_CLAUDE_CODE_VERSION,
    }),
    "claude-fable-5-1": frozenset({
        REVIEWED_CLAUDE_CODE_VERSION,
        REVIEWED_CLAUDE_FABLE_CODE_VERSION,
    }),
    "claude-mythos-5-1": frozenset({
        REVIEWED_CLAUDE_MYTHOS_CODE_VERSION,
    }),
})
CLAUDE_SESSION_FRONTIER_MODELS = frozenset(
    CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL
)
CLAUDE_SESSION_REQUIRED_EXTRA_ARGS: tuple[
    tuple[str, str | None],
    ...,
] = (
    ("disable-slash-commands", None),
    ("prompt-suggestions", "false"),
)

# These are fixed transport policy, not caller- or environment-controlled
# settings. They disable the native updater in each child, disable general
# nonessential traffic, explicitly suppress SDK session-title generation (an
# otherwise automatic small/fast-model call), activate the exact-reviewed
# CLI's private fail-closed model-substitution tripwire, bind prompt-cache
# writes to the reviewed five-minute rate, and prevent read-only Git discovery
# from opportunistically replacing repository indexes. An external installer
# can still reconcile its managed tree between child spawns, so the
# transport's exact executable identity checks remain authoritative. Exact
# terminal model-usage evidence remains mandatory, and a CLI upgrade requires
# renewed local semantic review.
CLAUDE_SESSION_FIXED_PROCESS_ENV = MappingProxyType({
    "DISABLE_AUTOUPDATER": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    "FORCE_PROMPT_CACHING_5M": "1",
    "GIT_OPTIONAL_LOCKS": "0",
})

# The session-authenticated CLI needs HOME to discover its first-party login.
# The remaining names are narrowly operational (locale, executable lookup,
# trust roots, temporary storage, and timezone).  Provider, proxy, telemetry,
# shell-injection, cloud, repository, and SDK/CLI-control variables are not
# admitted.
CLAUDE_SESSION_OPERATIONAL_ENV = frozenset({
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_EXTRA_CA_CERTS",
    "PATH",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "TMPDIR",
    "TZ",
})


class _FrozenList(list[Any]):
    """List-compatible read-only snapshot for the exact SDK boundary.

    The pinned SDK uses ``isinstance(value, list)`` for several option fields,
    so a tuple is not wire-compatible. Construction deliberately bypasses the
    public initializer; calling ``.__init__`` again is rejected.
    """

    __slots__ = ()

    @staticmethod
    def _deny(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Claude session SDK options are immutable")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> _FrozenList:
        raise TypeError("Claude session SDK option snapshots use _from_iterable")

    @classmethod
    def _from_iterable(cls, values: Any) -> _FrozenList:
        snapshot = list.__new__(cls)
        list.__init__(snapshot, values)
        return snapshot

    __init__ = _deny
    __setitem__ = _deny
    __delitem__ = _deny
    __iadd__ = _deny  # type: ignore[assignment]
    __imul__ = _deny  # type: ignore[assignment]
    append = _deny
    clear = _deny
    extend = _deny
    insert = _deny
    pop = _deny
    remove = _deny
    reverse = _deny
    sort = _deny


class _FrozenDict(dict[Any, Any]):
    """Dict-compatible read-only snapshot for the exact SDK boundary."""

    __slots__ = ()

    @staticmethod
    def _deny(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Claude session SDK options are immutable")

    def __new__(cls, *_args: Any, **_kwargs: Any) -> _FrozenDict:
        raise TypeError("Claude session SDK option snapshots use _from_mapping")

    @classmethod
    def _from_mapping(cls, values: Any) -> _FrozenDict:
        snapshot = dict.__new__(cls)
        dict.__init__(snapshot, values)
        return snapshot

    __init__ = _deny
    __setitem__ = _deny
    __delitem__ = _deny
    __ior__ = _deny  # type: ignore[assignment]
    clear = _deny
    pop = _deny
    popitem = _deny  # type: ignore[assignment]
    setdefault = _deny
    update = _deny


def _freeze_option_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _FrozenDict._from_mapping({
            key: _freeze_option_value(nested)
            for key, nested in value.items()
        })
    if isinstance(value, list):
        return _FrozenList._from_iterable(
            _freeze_option_value(item) for item in value
        )
    if isinstance(value, tuple):
        return tuple(_freeze_option_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_option_value(item) for item in value)
    return value


def _thaw_option_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _thaw_option_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_thaw_option_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw_option_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return {_thaw_option_value(item) for item in value}
    return value


class FrozenClaudeSessionOptions:
    """Read-only option snapshot compatible with the pinned SDK's reads.

    This is deliberately not a ``ClaudeAgentOptions`` subclass: that mutable
    dataclass exposes a writable ``__dict__``. The exact reviewed SDK treats
    its options structurally and reads only named fields on the admitted
    organization-session path. List and dict fields retain their concrete
    runtime types because SDK 0.2.121 checks them with ``isinstance``.

    Opaque values such as the in-process MCP server and stderr callback retain
    identity by design; the read-only guarantee covers the option control
    graph, not the internal state of workflow-owned runtime objects. As with
    every Python container, deliberately invoking a builtin base mutator
    (for example ``dict.__setitem__``) is outside this in-process boundary.
    """

    __slots__ = ("__values",)

    def __new__(
        cls,
        *_args: Any,
        **_kwargs: Any,
    ) -> FrozenClaudeSessionOptions:
        raise TypeError("Claude session SDK option snapshots use a factory")

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("Claude session SDK option snapshots are immutable")

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> FrozenClaudeSessionOptions:
        return cls.from_options(ClaudeAgentOptions(**kwargs))

    @classmethod
    def from_options(
        cls,
        options: ClaudeAgentOptions | FrozenClaudeSessionOptions,
    ) -> FrozenClaudeSessionOptions:
        values = {
            field.name: _freeze_option_value(getattr(options, field.name))
            for field in fields(ClaudeAgentOptions)
        }
        snapshot = object.__new__(cls)
        object.__setattr__(
            snapshot,
            "_FrozenClaudeSessionOptions__values",
            MappingProxyType(values),
        )
        return snapshot

    def __getattr__(self, name: str) -> Any:
        try:
            values = object.__getattribute__(
                self,
                "_FrozenClaudeSessionOptions__values",
            )
            return values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("Claude session SDK options are immutable")

    def __delattr__(self, _name: str) -> None:
        raise TypeError("Claude session SDK options are immutable")


def _copy_transport_options(
    options: ClaudeAgentOptions | FrozenClaudeSessionOptions,
) -> FrozenClaudeSessionOptions:
    """Detach every mutable CLI option from the SDK query control plane."""

    detached = ClaudeAgentOptions(**{
        field.name: _thaw_option_value(getattr(options, field.name))
        for field in fields(ClaudeAgentOptions)
    })
    return FrozenClaudeSessionOptions.from_options(detached)


class ClaudeSessionIdentityError(RuntimeError):
    """The local Claude session or executable violated its pinned identity."""


def _contained_identity_error(
    exc: BaseException,
) -> ClaudeSessionIdentityError | None:
    if isinstance(exc, ClaudeSessionIdentityError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for nested in exc.exceptions:
            if identity_error := _contained_identity_error(nested):
                return identity_error
    return None


@dataclass(frozen=True)
class PinnedClaudeCLI:
    """Canonical path plus immutable file identity captured at construction."""

    path: str
    device: int
    inode: int
    mode: int
    uid: int
    size: int
    mtime_ns: int
    ctime_ns: int
    sha256: str

    @classmethod
    def capture(cls, path: str) -> PinnedClaudeCLI:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ValueError("pinned Claude CLI path must be absolute")
        try:
            canonical = candidate.resolve(strict=True)
            metadata = os.stat(canonical, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                "pinned Claude CLI path must be an existing executable file"
            ) from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not os.access(canonical, os.X_OK)
        ):
            raise ValueError(
                "pinned Claude CLI path must be an existing executable file"
            )
        return cls(
            path=str(canonical),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
            uid=metadata.st_uid,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
            sha256=_file_sha256(canonical),
        )

    def revalidate(self) -> None:
        """Reject replacement, mutation, permission drift, or path retargeting."""

        try:
            candidate = Path(self.path)
            canonical = candidate.resolve(strict=True)
            metadata = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise ClaudeSessionIdentityError(
                "pinned Claude CLI disappeared before session transport"
            ) from exc
        observed = (
            str(canonical),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_uid,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        expected = (
            self.path,
            self.device,
            self.inode,
            self.mode,
            self.uid,
            self.size,
            self.mtime_ns,
            self.ctime_ns,
        )
        if (
            observed != expected
            or not stat.S_ISREG(metadata.st_mode)
            or not os.access(candidate, os.X_OK)
            or _file_sha256(candidate) != self.sha256
        ):
            raise ClaudeSessionIdentityError(
                "pinned Claude CLI identity drifted before session transport"
            )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


async def _read_bounded(
    stream: Any,
    *,
    limit: int,
) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while stream is not None:
        try:
            chunk = await stream.receive()
        except (anyio.EndOfStream, anyio.ClosedResourceError):
            break
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise ClaudeSessionIdentityError(
                "Claude CLI identity probe exceeded its output limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _stop_process(process: Any) -> None:
    """Reap a probe or partially initialized child on every exit path."""

    with anyio.CancelScope(shield=True):
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            with anyio.move_on_after(1):
                await process.wait()
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            with suppress(Exception):
                await process.wait()


class SanitizedClaudeSessionTransport(_sdk_cli.SubprocessCLITransport):
    """SDK-compatible transport whose real CLI children never inherit ambient env."""

    def __init__(
        self,
        *,
        prompt: str,
        options: ClaudeAgentOptions | FrozenClaudeSessionOptions,
        cli: PinnedClaudeCLI,
        expected_org_id: str,
        expected_org_name: str | None,
    ) -> None:
        if os.name != "posix":
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport is reviewed only for POSIX"
            )
        if _sdk_cli.__version__ != REVIEWED_CLAUDE_AGENT_SDK_VERSION:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires exact "
                f"claude-agent-sdk=={REVIEWED_CLAUDE_AGENT_SDK_VERSION}"
            )
        if options.cli_path is None or str(options.cli_path) != cli.path:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires its pinned CLI"
            )
        self._validate_command_policy(options)
        if (
            type(expected_org_id) is not str
            or not expected_org_id
            or expected_org_id != expected_org_id.strip()
        ):
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires expected_org_id"
            )
        if expected_org_name is not None and (
            type(expected_org_name) is not str
            or not expected_org_name
            or expected_org_name != expected_org_name.strip()
        ):
            raise ClaudeSessionIdentityError(
                "expected_org_name must be an exact non-empty string when set"
            )

        transport_options = _copy_transport_options(options)
        self._validate_command_policy(transport_options)
        super().__init__(
            prompt=prompt,
            options=transport_options,  # type: ignore[arg-type]
        )
        self._pinned_cli = cli
        self._expected_org_id = expected_org_id
        self._expected_org_name = expected_org_name
        self._session_process_env = MappingProxyType(
            self._build_session_process_env(transport_options)
        )
        self._expected_command = tuple(self._build_command())
        # This is deliberately absent until connect() has both validated the
        # exact CLI version and successfully spawned that same pinned binary.
        # Backend parsing may use it to admit version-specific CLI protocol
        # fields without broadening the pinned Agent SDK's schema globally.
        self._validated_cli_version: str | None = None
        self._probed_cli_version: str | None = None

    @property
    def validated_cli_version(self) -> str | None:
        """Return the exact CLI version bound to the live connection."""

        return self._validated_cli_version

    @staticmethod
    def _validate_command_policy(
        options: ClaudeAgentOptions | FrozenClaudeSessionOptions,
    ) -> None:
        """Reject model or auxiliary-policy drift before command construction."""

        if options.model not in CLAUDE_SESSION_FRONTIER_MODELS:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires exact model "
                "from reviewed set "
                f"{sorted(CLAUDE_SESSION_FRONTIER_MODELS)!r}"
            )
        if options.fallback_model is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids fallback_model"
            )
        if options.settings is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids settings"
            )
        if not {"Task", "Agent"} <= set(options.disallowed_tools):
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires Task and "
                "Agent to remain disallowed"
            )
        if options.continue_conversation:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids implicit continue"
            )
        if options.session_id is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids caller session_id"
            )
        if options.fork_session:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids fork_session"
            )
        if options.permission_prompt_tool_name is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids a permission "
                "prompt tool"
            )
        if options.add_dirs != []:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids add_dirs"
            )
        if options.user is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids user override"
            )
        if options.sandbox is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids SDK sandbox "
                "overrides"
            )
        if options.betas != []:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids beta overrides"
            )
        if options.enable_file_checkpointing:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids file "
                "checkpointing"
            )
        if options.include_hook_events:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids hook events"
            )
        if options.include_partial_messages:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids partial messages"
            )
        if options.session_store is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids session_store"
            )
        if options.can_use_tool is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids can_use_tool"
            )
        if options.strict_mcp_config is not True:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires strict_mcp_config"
            )
        if options.setting_sources != []:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids setting sources"
            )
        if options.skills != []:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires an empty skill set"
            )
        if options.plugins != []:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids plugins"
            )
        if options.hooks is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids hooks"
            )
        if options.agents is not None:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport forbids custom agents"
            )
        if type(options.system_prompt) is not str:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires a literal "
                "system prompt"
            )
        mcp_servers = options.mcp_servers
        if type(mcp_servers) not in (dict, _FrozenDict):
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires an explicit "
                "MCP mapping"
            )
        mcp_mapping = cast(dict[str, Any], mcp_servers)
        if set(mcp_mapping) - {"crucible"}:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport permits only the "
                "workflow-owned Crucible MCP server"
            )
        for config in mcp_mapping.values():
            if (
                type(config) not in (dict, _FrozenDict)
                or set(config) != {"type", "name", "instance"}
                or config.get("type") != "sdk"
                or config.get("name") != "crucible"
            ):
                raise ClaudeSessionIdentityError(
                    "Claude organization-session transport requires the "
                    "exact in-process Crucible SDK MCP shape"
                )
        if dict(options.extra_args) != dict(
            CLAUDE_SESSION_REQUIRED_EXTRA_ARGS
        ):
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires its exact "
                "auxiliary-call suppression arguments"
            )

    def _build_session_process_env(
        self,
        options: ClaudeAgentOptions | FrozenClaudeSessionOptions,
    ) -> dict[str, str]:
        supplied = options.env or {}
        process_env: dict[str, str] = {}
        for name, value in supplied.items():
            if name not in CLAUDE_SESSION_OPERATIONAL_ENV:
                continue
            if (
                type(name) is not str
                or type(value) is not str
                or "\x00" in name
                or "\x00" in value
                or "=" in name
            ):
                raise ClaudeSessionIdentityError(
                    "Claude session operational environment must contain "
                    "exact string names and values"
                )
            process_env[name] = value
        home = process_env.get("HOME")
        if (
            not home
            or "\x00" in home
            or not Path(home).is_absolute()
        ):
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires an absolute HOME"
            )
        process_env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-py"
        process_env["CLAUDE_AGENT_SDK_VERSION"] = _sdk_cli.__version__
        process_env.update(dict(CLAUDE_SESSION_FIXED_PROCESS_ENV))
        if options.enable_file_checkpointing:
            process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
        if self._cwd:
            process_env["PWD"] = self._cwd
        return process_env

    def _env_copy(self) -> dict[str, str]:
        """Give each spawn its own copy of the construction-time snapshot."""

        return dict(self._session_process_env)

    async def _run_probe(
        self,
        command: list[str],
        *,
        label: str,
        timeout: float,
        output_limit: int = 64 * 1024,
    ) -> tuple[int, bytes, bytes]:
        self._pinned_cli.revalidate()
        process = None
        stdout = b""
        stderr = b""
        try:
            with anyio.fail_after(timeout):
                process = await anyio.open_process(
                    command,
                    stdout=PIPE,
                    stderr=PIPE,
                    cwd=self._cwd,
                    env=self._env_copy(),
                    user=self._options.user,
                )
                _sdk_cli._ACTIVE_CHILDREN.add(process)

                async def read_stdout() -> None:
                    nonlocal stdout
                    stdout = await _read_bounded(
                        process.stdout,
                        limit=output_limit,
                    )

                async def read_stderr() -> None:
                    nonlocal stderr
                    stderr = await _read_bounded(
                        process.stderr,
                        limit=output_limit,
                    )

                async with anyio.create_task_group() as task_group:
                    task_group.start_soon(read_stdout)
                    task_group.start_soon(read_stderr)
                    returncode = await process.wait()
        except TimeoutError as exc:
            raise ClaudeSessionIdentityError(
                f"Claude CLI {label} timed out"
            ) from exc
        finally:
            if process is not None:
                await _stop_process(process)
                _sdk_cli._ACTIVE_CHILDREN.discard(process)
        return returncode, stdout, stderr

    async def _check_session_identity(self) -> None:
        returncode, stdout, _stderr = await self._run_probe(
            [self._pinned_cli.path, "auth", "status", "--json"],
            label="auth status",
            timeout=5,
        )
        if returncode != 0:
            raise ClaudeSessionIdentityError(
                "Claude CLI auth status failed before provider transport"
            )
        try:
            status_payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ClaudeSessionIdentityError(
                "Claude CLI auth status did not return valid JSON"
            ) from exc
        if not isinstance(status_payload, dict):
            raise ClaudeSessionIdentityError(
                "Claude CLI auth status must be a JSON object"
            )
        expected_fields: tuple[tuple[str, object], ...] = (
            ("loggedIn", True),
            ("authMethod", "claude.ai"),
            ("apiProvider", "firstParty"),
            ("orgId", self._expected_org_id),
        )
        for field, expected in expected_fields:
            observed = status_payload.get(field)
            exact_match = (
                observed is True
                if field == "loggedIn"
                else type(observed) is str and observed == expected
            )
            if not exact_match:
                raise ClaudeSessionIdentityError(
                    f"Claude CLI auth status {field} did not match the "
                    "configured organization-session identity"
                )
        if (
            self._expected_org_name is not None
            and (
                type(status_payload.get("orgName")) is not str
                or status_payload.get("orgName") != self._expected_org_name
            )
        ):
            raise ClaudeSessionIdentityError(
                "Claude CLI auth status orgName did not match the configured "
                "organization-session identity"
            )

    async def _check_claude_version(self) -> None:
        """Run the SDK's version gate with the same explicit safe environment."""

        self._probed_cli_version = None
        returncode, stdout, _stderr = await self._run_probe(
            [self._pinned_cli.path, "-v"],
            label="version check",
            timeout=2,
            output_limit=16 * 1024,
        )
        if returncode != 0:
            raise ClaudeSessionIdentityError(
                "Claude CLI version check failed before provider transport"
            )
        version_output = stdout.decode(errors="replace").strip()
        match = re.fullmatch(
            r"([0-9]+\.[0-9]+\.[0-9]+)(?: \(Claude Code\))?",
            version_output,
        )
        if match is None:
            raise ClaudeSessionIdentityError(
                "Claude CLI version check returned an unrecognized version"
            )
        version = match.group(1)
        expected_versions = CLAUDE_SESSION_REVIEWED_CLI_VERSIONS_BY_MODEL[
            str(self._options.model)
        ]
        if version not in expected_versions:
            raise ClaudeSessionIdentityError(
                "Claude organization-session transport requires an exact "
                f"reviewed Claude Code version in {sorted(expected_versions)!r} "
                "for model "
                f"{self._options.model!r}"
            )
        self._probed_cli_version = version

    async def _cleanup_failed_connect(self) -> None:
        with anyio.CancelScope(shield=True):
            self._validated_cli_version = None
            self._probed_cli_version = None
            process = self._process
            self._ready = False
            if self._stderr_task is not None and not self._stderr_task.done():
                self._stderr_task.cancel()
                with suppress(Exception):
                    await self._stderr_task.wait()
            self._stderr_task = None
            for stream_name in (
                "_stdin_stream",
                "_stdout_stream",
                "_stderr_stream",
            ):
                stream = getattr(self, stream_name)
                if stream is not None:
                    with suppress(Exception):
                        await stream.aclose()
                    setattr(self, stream_name, None)
            if process is not None:
                await _stop_process(process)
                _sdk_cli._ACTIVE_CHILDREN.discard(process)
            self._process = None

    async def connect(self) -> None:
        """Authenticate, version-check, and start Claude with no inherited env."""

        if self._process:
            return
        self._validated_cli_version = None
        self._probed_cli_version = None
        try:
            self._pinned_cli.revalidate()
            await self._check_session_identity()
            self._pinned_cli.revalidate()
            await self._check_claude_version()
            validated_cli_version = self._probed_cli_version
            if validated_cli_version is None:  # pragma: no cover - invariant
                raise ClaudeSessionIdentityError(
                    "Claude CLI version check did not bind a reviewed version"
                )
            self._pinned_cli.revalidate()

            self._validate_command_policy(self._options)
            command = self._build_command()
            if tuple(command) != self._expected_command:
                raise ClaudeSessionIdentityError(
                    "Claude organization-session command drifted before spawn"
                )
            stderr_dest = PIPE if self._options.stderr is not None else None
            self._process = await anyio.open_process(
                command,
                stdin=PIPE,
                stdout=PIPE,
                stderr=stderr_dest,
                cwd=self._cwd,
                env=self._env_copy(),
                user=self._options.user,
            )
            _sdk_cli._ACTIVE_CHILDREN.add(self._process)

            if self._process.stdout:
                self._stdout_stream = _sdk_cli.TextReceiveStream(
                    self._process.stdout
                )
            if stderr_dest is PIPE and self._process.stderr:
                self._stderr_stream = _sdk_cli.TextReceiveStream(
                    self._process.stderr
                )
                self._stderr_task = _sdk_cli.spawn_detached(
                    self._handle_stderr()
                )
            if self._process.stdin:
                self._stdin_stream = _sdk_cli.TextSendStream(
                    self._process.stdin
                )
            self._ready = True
            self._validated_cli_version = validated_cli_version
            self._probed_cli_version = None
        except BaseException as exc:
            await self._cleanup_failed_connect()
            if not isinstance(exc, Exception):
                raise
            identity_error = _contained_identity_error(exc)
            if identity_error is not None:
                self._exit_error = identity_error
                if identity_error is exc:
                    raise
                raise identity_error from exc
            self._exit_error = exc
            if isinstance(exc, FileNotFoundError):
                if self._cwd and not Path(self._cwd).exists():
                    raise CLIConnectionError(
                        f"Working directory does not exist: {self._cwd}"
                    ) from exc
                raise CLINotFoundError(
                    f"Claude Code not found at: {self._cli_path}"
                ) from exc
            raise CLIConnectionError(
                f"Failed to start sanitized Claude Code transport: {exc}"
            ) from exc

    async def close(self) -> None:
        """Preserve the SDK cleanup contract even inside caller cancellation."""

        with anyio.CancelScope(shield=True):
            try:
                await super().close()
            finally:
                self._validated_cli_version = None
                self._probed_cli_version = None


__all__ = [
    "CLAUDE_SESSION_OPERATIONAL_ENV",
    "ClaudeSessionIdentityError",
    "PinnedClaudeCLI",
    "REVIEWED_CLAUDE_AGENT_SDK_VERSION",
    "SanitizedClaudeSessionTransport",
]
