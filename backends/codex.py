# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""CodexBackend — assay.Backend implementation over codex-app-server-sdk.

Translates an ``assay.Invocation`` into a Codex thread + chat call and
returns an ``assay.InvocationResult``. Mirrors ``ClaudeBackend``'s
shape: same default policy, same retry + classification handling, same
budget enforcement.

The agent loop (tool use, approvals, multi-turn) is owned by the
Codex SDK; this adapter is translation + policy enforcement.

Confined roles run the CLI with an isolated ``CODEX_HOME``, an explicit MCP
catalogue, native executors disabled, and a read-only Codex sandbox. Scoped
mutation is performed only through Crucible's path-confined MCP handlers.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import math
import os
import platform
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback as _tb
from copy import deepcopy
from dataclasses import dataclass, field, replace as _replace
from pathlib import Path
from types import MappingProxyType, MethodType
from typing import Any, Literal

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ContentRejected,
    ConversationTransportOutcome,
    ExecutionCompleteness,
    ExternalCancellation,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    StaleConversation,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from ._classify import (
    RETRIABLE_HTTP_STATUSES,
    TERMINAL_EXCEPTION_NAMES,
    ClassifiedError,
)
from ._physical_telemetry import (
    CacheWriteFieldState,
    CacheWriteInterpretation,
    PhysicalUsageEvidence,
    PhysicalUsageResponse,
    capture_physical_usage,
    emit_physical_usage_attempt,
)
from ._sdk_debug import SdkDebugSink, redact_sdk_text

logger = logging.getLogger(__name__)

# GPT-6 Astra is the default frontier agentic-coding model. ``max`` remains
# the highest admitted non-delegating effort because Crucible owns
# orchestration and advertises ``native_delegation='none'``.
_DEFAULT_MODEL = "gpt-6-astra"
_DEFAULT_EFFORT = "max"
CODEX_CHATGPT_CREDENTIAL_SOURCE = "chatgpt_session"
_CODEX_CHATGPT_SESSION_MODELS = frozenset({_DEFAULT_MODEL, "gpt-5.6-sol"})
_CODEX_CHATGPT_SESSION_CLI_VERSIONS_BY_MODEL = MappingProxyType({
    "gpt-6-astra": frozenset({"0.153.4"}),
    "gpt-5.6-sol": None,
})
_CODEX_CREDENTIAL_SOURCES = frozenset(
    {
        "environment",
        CODEX_CHATGPT_CREDENTIAL_SOURCE,
    }
)
_CODEX_AUTH_MAX_BYTES = 1024 * 1024
_CODEX_CLI_VERSION_PROBE_TIMEOUT_SECONDS = 2.0
_CODEX_CLI_VERSION_PROBE_STOP_SECONDS = 1.0
_CODEX_CLI_VERSION_PROBE_OUTPUT_LIMIT_BYTES = 16 * 1024
_CODEX_CLI_VERSION_RE = re.compile(r"codex-cli ([0-9]+\.[0-9]+\.[0-9]+)\n")


class CodexSessionIdentityError(RuntimeError):
    """The local CLI does not match the reviewed ChatGPT-session route."""


@dataclass(frozen=True)
class _PinnedCodexCLI:
    """Canonical executable identity for a frozen ChatGPT-session route."""

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
    def capture(cls, value: object) -> _PinnedCodexCLI:
        requirement = (
            "credential_source='chatgpt_session' requires cli_path to "
            "resolve to an absolute existing executable Codex CLI"
        )
        if type(value) is not str or not value.strip():
            raise ValueError(requirement)
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(f"{requirement}; relative paths are not allowed")
        try:
            canonical = candidate.resolve(strict=True)
            metadata = os.stat(canonical, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"{requirement}; path does not exist") from exc
        if not stat.S_ISREG(metadata.st_mode) or not os.access(canonical, os.X_OK):
            raise ValueError(f"{requirement}; path is not an executable file")
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
        """Reject replacement or mutation immediately before transport."""

        try:
            candidate = Path(self.path)
            canonical = candidate.resolve(strict=True)
            metadata = os.stat(candidate, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                "pinned Codex CLI disappeared before provider transport"
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
            raise RuntimeError(
                "pinned Codex CLI identity drifted before provider transport"
            )


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_prefix(path: Path, size: int = 4) -> bytes:
    with path.open("rb") as stream:
        return stream.read(size)


def _is_native_codex_binary(path: Path) -> bool:
    prefix = _file_prefix(path)
    if sys.platform == "linux":
        return prefix == b"\x7fELF"
    if sys.platform == "darwin":
        return prefix in {
            b"\xca\xfe\xba\xbe",
            b"\xbe\xba\xfe\xca",
            b"\xca\xfe\xba\xbf",
            b"\xbf\xba\xfe\xca",
            b"\xce\xfa\xed\xfe",
            b"\xcf\xfa\xed\xfe",
            b"\xfe\xed\xfa\xce",
            b"\xfe\xed\xfa\xcf",
        }
    return False


def _resolve_native_codex_executable(value: object) -> str:
    """Resolve the reviewed native CLI, bypassing npm's Node launcher."""

    requirement = (
        "credential_source='chatgpt_session' requires an absolute native "
        "Codex executable or the canonical @openai/codex launcher"
    )
    if type(value) is not str or not value.strip():
        raise ValueError(requirement)
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{requirement}; relative paths are not allowed")
    try:
        canonical = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{requirement}; path does not exist") from exc
    if _is_native_codex_binary(canonical):
        return str(canonical)

    target_by_platform = {
        ("linux", "x86_64"): (
            "@openai/codex-linux-x64",
            "x86_64-unknown-linux-musl",
        ),
        ("linux", "aarch64"): (
            "@openai/codex-linux-arm64",
            "aarch64-unknown-linux-musl",
        ),
        ("darwin", "x86_64"): (
            "@openai/codex-darwin-x64",
            "x86_64-apple-darwin",
        ),
        ("darwin", "arm64"): (
            "@openai/codex-darwin-arm64",
            "aarch64-apple-darwin",
        ),
    }
    package_target = target_by_platform.get((sys.platform, platform.machine().lower()))
    if (
        package_target is None
        or canonical.name != "codex.js"
        or canonical.parent.name != "bin"
    ):
        raise ValueError(
            f"{requirement}; script launchers and unknown platforms are not accepted"
        )
    package_name, target = package_target
    package_root = canonical.parent.parent
    candidates = (
        package_root
        / "node_modules"
        / "@openai"
        / package_name.removeprefix("@openai/")
        / "vendor"
        / target
        / "bin"
        / "codex",
        package_root / "vendor" / target / "bin" / "codex",
    )
    matches = [
        path.resolve(strict=True)
        for path in candidates
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if len(matches) != 1 or not _is_native_codex_binary(matches[0]):
        raise ValueError(f"{requirement}; exactly one native package binary must exist")
    return str(matches[0])


def _terminate_codex_cli_probe_group_and_reap_leader(
    process: subprocess.Popen[bytes],
) -> None:
    """Kill one unreaped leader's isolated group, then reap it boundedly."""

    if process.returncode is not None:  # pragma: no cover - caller invariant
        raise CodexSessionIdentityError(
            "Codex CLI version probe leader was reaped before group settlement"
        )
    signal_error: OSError | None = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError as exc:
        signal_error = exc
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as direct_exc:
            signal_error = direct_exc
    try:
        process.wait(timeout=_CODEX_CLI_VERSION_PROBE_STOP_SECONDS)
    except subprocess.TimeoutExpired as exc:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as direct_exc:
            signal_error = signal_error or direct_exc
        try:
            process.wait(timeout=_CODEX_CLI_VERSION_PROBE_STOP_SECONDS)
        except subprocess.TimeoutExpired as final_exc:
            raise CodexSessionIdentityError(
                "Codex CLI version probe leader did not reap after SIGKILL"
            ) from final_exc
        raise CodexSessionIdentityError(
            "Codex CLI version probe leader exceeded its teardown deadline"
        ) from exc
    if signal_error is not None:
        raise CodexSessionIdentityError(
            "Codex CLI version probe process group could not be killed"
        ) from signal_error


def _observe_codex_cli_probe_returncode(
    process: subprocess.Popen[bytes],
    *,
    arguments: tuple[str, str],
    deadline: float,
) -> int:
    """Observe the direct child's status without freeing its PID/PGID."""

    required_waitid = ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if not hasattr(os, "waitid") or not all(
        hasattr(os, name) for name in required_waitid
    ):
        raise CodexSessionIdentityError(
            "Codex CLI version probe requires bounded non-reaping wait support"
        )
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(
                arguments,
                _CODEX_CLI_VERSION_PROBE_TIMEOUT_SECONDS,
            )
        try:
            status = os.waitid(
                os.P_PID,
                process.pid,
                os.WEXITED | os.WNOHANG | os.WNOWAIT,
            )
        except InterruptedError:
            continue
        if status is None or status.si_pid == 0:
            time.sleep(min(remaining, 0.01))
            continue
        if status.si_code == os.CLD_EXITED:
            return status.si_status
        if status.si_code in {os.CLD_KILLED, os.CLD_DUMPED}:
            return -status.si_status
        raise CodexSessionIdentityError(
            "Codex CLI version probe returned an unsupported child status"
        )


def _run_bounded_codex_cli_probe(
    cli_path: str,
    *,
    cwd: str,
) -> tuple[int, bytes, bytes]:
    """Bound a direct-child probe and its isolated process-group cleanup."""

    arguments = (cli_path, "--version")
    selector: selectors.BaseSelector | None = None
    stdout: Any = None
    stderr: Any = None
    process = subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        cwd=cwd,
        env={
            "CODEX_HOME": cwd,
            "HOME": cwd,
            "PATH": "/usr/bin:/bin",
        },
        start_new_session=True,
    )
    try:
        try:
            stdout = process.stdout
            stderr = process.stderr
            if stdout is None or stderr is None:  # pragma: no cover - Popen contract
                raise CodexSessionIdentityError(
                    "Codex CLI version probe pipes were not created"
                )
            selector = selectors.DefaultSelector()
            buffers = {
                stdout.fileno(): bytearray(),
                stderr.fileno(): bytearray(),
            }
            selector.register(stdout, selectors.EVENT_READ)
            selector.register(stderr, selectors.EVENT_READ)
            deadline = time.monotonic() + _CODEX_CLI_VERSION_PROBE_TIMEOUT_SECONDS
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(
                        arguments,
                        _CODEX_CLI_VERSION_PROBE_TIMEOUT_SECONDS,
                    )
                for key, _events in selector.select(timeout=min(remaining, 0.05)):
                    retained = buffers[key.fd]
                    payload = os.read(
                        key.fd,
                        min(
                            64 * 1024,
                            _CODEX_CLI_VERSION_PROBE_OUTPUT_LIMIT_BYTES
                            - len(retained)
                            + 1,
                        ),
                    )
                    if not payload:
                        selector.unregister(key.fileobj)
                        continue
                    retained.extend(payload)
                    if len(retained) > _CODEX_CLI_VERSION_PROBE_OUTPUT_LIMIT_BYTES:
                        raise CodexSessionIdentityError(
                            "Codex CLI version probe output exceeded its hard bound"
                        )
            returncode = _observe_codex_cli_probe_returncode(
                process,
                arguments=arguments,
                deadline=deadline,
            )
            result = (
                returncode,
                bytes(buffers[stdout.fileno()]),
                bytes(buffers[stderr.fileno()]),
            )
        except BaseException as probe_error:
            try:
                _terminate_codex_cli_probe_group_and_reap_leader(process)
            except BaseException as cleanup_error:
                raise cleanup_error from probe_error
            raise
        # Same-process-group descendants that close both captured pipes cannot
        # outlive a syntactically valid version response. A descendant that
        # deliberately escapes via setsid(2) or setpgid(2) is outside this
        # operator-selected, identity-pinned executable's probe threat model.
        _terminate_codex_cli_probe_group_and_reap_leader(process)
        if process.returncode != returncode:  # pragma: no cover - OS invariant
            raise CodexSessionIdentityError(
                "Codex CLI version probe status changed during bounded reap"
            )
        return result
    finally:
        if selector is not None:
            selector.close()
        if stdout is not None:
            stdout.close()
        if stderr is not None:
            stderr.close()


def _reviewed_codex_cli_versions(model: str) -> frozenset[str] | None:
    """Resolve one complete immutable ChatGPT-session compatibility row."""

    if (
        frozenset(_CODEX_CHATGPT_SESSION_CLI_VERSIONS_BY_MODEL)
        != _CODEX_CHATGPT_SESSION_MODELS
    ):
        raise CodexSessionIdentityError(
            "Codex ChatGPT-session model/CLI compatibility map is incomplete"
        )
    try:
        return _CODEX_CHATGPT_SESSION_CLI_VERSIONS_BY_MODEL[model]
    except KeyError as exc:  # pragma: no cover - constructor/model-map invariant
        raise CodexSessionIdentityError(
            "Codex ChatGPT-session model has no CLI compatibility row"
        ) from exc


def _validate_codex_cli_version(cli: _PinnedCodexCLI, *, model: str) -> str | None:
    """Fail closed unless a version-bound model uses its reviewed Codex CLI."""

    expected_versions = _reviewed_codex_cli_versions(model)
    if expected_versions is None:
        return None
    cli.revalidate()
    try:
        # Codex warns when CODEX_HOME is beneath the shared system temporary
        # root because it will not install helper aliases there. A fresh private
        # directory beneath the operator home admits no ambient Codex config and
        # lets an exact successful probe retain an empty stderr surface.
        with tempfile.TemporaryDirectory(
            prefix=".crucible-codex-version-",
            dir=Path.home(),
        ) as home:
            returncode, stdout, stderr = _run_bounded_codex_cli_probe(
                cli.path,
                cwd=home,
            )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodexSessionIdentityError(
            "Codex CLI version check failed before provider transport "
            f"({type(exc).__name__})"
        ) from exc
    finally:
        cli.revalidate()
    if returncode != 0:
        raise CodexSessionIdentityError(
            "Codex CLI version check failed before provider transport"
        )
    if stderr:
        raise CodexSessionIdentityError(
            "Codex CLI version check returned unexpected stderr"
        )
    try:
        output = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CodexSessionIdentityError(
            "Codex CLI version check returned an unrecognized version"
        ) from exc
    match = _CODEX_CLI_VERSION_RE.fullmatch(output)
    if match is None:
        raise CodexSessionIdentityError(
            "Codex CLI version check returned an unrecognized version"
        )
    version = match.group(1)
    if version not in expected_versions:
        raise CodexSessionIdentityError(
            "Codex ChatGPT-session transport requires an exact reviewed Codex "
            f"CLI version in {sorted(expected_versions)!r} for model {model!r}; "
            f"observed {version!r}"
        )
    return version


def _read_chatgpt_auth_payload(path: Path) -> bytes:
    """Read and validate a private ChatGPT login without following links."""

    nofollow = getattr(os, "O_NOFOLLOW", None)
    if os.name != "posix" or nofollow is None:
        raise RuntimeError(
            "Codex ChatGPT-session authentication is reviewed only for POSIX"
        )
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Codex ChatGPT auth file is unavailable at {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077
            or metadata.st_size > _CODEX_AUTH_MAX_BYTES
        ):
            raise RuntimeError(
                "Codex ChatGPT auth file must be a private, owner-held "
                "regular file within the size limit"
            )
        chunks: list[bytes] = []
        remaining = _CODEX_AUTH_MAX_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload_bytes = b"".join(chunks)
        if len(payload_bytes) > _CODEX_AUTH_MAX_BYTES:
            raise RuntimeError("Codex ChatGPT auth file exceeds its size limit")
    finally:
        os.close(descriptor)
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Codex ChatGPT auth file is not valid JSON") from exc
    if type(payload) is not dict or payload.get("auth_mode") != "chatgpt":
        raise RuntimeError("Codex frontier route requires auth_mode='chatgpt'")
    if payload.get("OPENAI_API_KEY") not in (None, ""):
        raise RuntimeError("Codex frontier route rejects API-key authentication")
    if type(payload.get("tokens")) is not dict:
        raise RuntimeError("Codex frontier route requires ChatGPT session tokens")
    return payload_bytes


# Prompt for a RECOVERY turn (a retry after a transport drop), whether we
# resumed the prior thread or had to start a fresh one (resume failed /
# unavailable). It EMBEDS the original task: on a resume the task is also in
# the thread history, but if the drop happened before the prompt was
# persisted — or the thread couldn't be resumed and we started fresh — the
# model would otherwise have no task to continue. Restating it is cheap
# insurance; the model is told to reconcile with prior progress on disk
# rather than blindly redo it. ``{task}`` is the original ``invocation.user``.
_CODEX_RESUME_NUDGE = (
    "A previous attempt at the task below was interrupted by a transport "
    "error before it finished. Any progress is preserved in this "
    "conversation and any edits are already on disk. FIRST review the current "
    "state (what's already changed), THEN continue and complete the task — do "
    "not blindly restart or duplicate work you've already done.\n\n"
    "--- TASK ---\n{task}"
)

# A 15-minute wall was too short for legitimate hard agentic-coding work:
# deep call-graph investigation, tool use, and test iteration can take 20-40
# minutes even when the turn is making progress.  Match the Claude agent
# adapter's 45-minute default so changing the backend does not silently give
# the same workflow a smaller completion window.  Explicit Invocation.policy
# values still override this per adapter call, and 45 minutes retains a hard
# bound for genuinely stuck turns.
_DEFAULT_POLICY = BackendPolicy(
    retry_budget=3,
    timeout_seconds=2700.0,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)

# ``codex-app-server-sdk`` 0.3.2 creates its subprocess pipes with asyncio's
# 64 KiB default StreamReader limit.  App-server emits one JSON object per
# line, and a review turn can legitimately echo Crucible's bounded 200 KiB
# artifact in a single item event.  The default reader then raises before the
# SDK can decode the event, surfacing only ``failed reading from stdio
# transport``.  Set a per-client reader bound before the SDK starts its
# receiver task.  Keep the allowance finite and comfortably above Crucible's
# current prompt/review envelopes instead of making the transport unbounded.
_CODEX_STDIO_READ_LIMIT_BYTES = 8 * 1024 * 1024


def _with_bounded_stdio_read_limit(client: Any) -> Any:
    """Apply an exact SDK stdio JSON-line bound before the client starts.

    ``CodexClient.connect_stdio`` returns an unstarted client.  Its private
    transport owns the subprocess and creates ``stdout`` only when
    ``connect`` runs, so wrap that one method and adjust the StreamReader
    before control returns to ``CodexClient.start`` (which starts the
    receiver).  Test doubles and future non-stdio clients without this SDK
    shape are returned unchanged.
    """
    transport = getattr(client, "_transport", None)
    connect = getattr(transport, "connect", None)
    if transport is None or not callable(connect):
        if type(client).__module__.startswith("codex_app_server_sdk"):
            raise RuntimeError(
                "codex-app-server-sdk stdio transport shape changed; "
                "refusing to run without a bounded JSON-line reader"
            )
        return client
    if getattr(transport, "_crucible_read_limit_wrapped", False):
        return client

    async def connect_with_bounded_read_limit() -> None:
        await connect()
        try:
            process = getattr(transport, "_proc", None)
            stdout = getattr(process, "stdout", None)
            current = getattr(stdout, "_limit", None)
            if not isinstance(current, int):
                raise RuntimeError(
                    "codex-app-server-sdk stdout reader shape changed; "
                    "cannot enforce the bounded JSON-line limit"
                )
            if current != _CODEX_STDIO_READ_LIMIT_BYTES:
                stdout._limit = _CODEX_STDIO_READ_LIMIT_BYTES
        except BaseException:
            close = getattr(transport, "close", None)
            if callable(close):
                await close()
            raise

    transport.connect = connect_with_bounded_read_limit
    transport._crucible_read_limit_wrapped = True
    return client


def _unwrap_exception_group(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup):  # noqa: E501
        if not exc.exceptions:
            break
        exc = exc.exceptions[0]
    return exc


def _is_external_cancellation(exc: BaseException) -> bool:
    """Per B32: external cancellation must convert to a backend error."""
    inner = _unwrap_exception_group(exc)
    return isinstance(inner, asyncio.CancelledError)


class _InvocationDeadlineExceeded(TimeoutError):
    """The adapter's logical wall deadline, not a provider timeout."""


async def _retain_attempt_result(
    awaitable: Any, sink: list[InvocationResult]
) -> InvocationResult:
    """Keep a normal result even if cancellation/join discards its return.

    The SDK may suppress cancellation or finish exactly at the deadline. A
    direct return remains authoritative for the normal completion path; the
    retained copy is only an incomplete diagnostic for the interrupted owner.
    No usage is re-harvested or added here.
    """
    result = await awaitable
    trace = result.execution_trace
    sink[:] = [_replace(
        result,
        completeness=_replace(
            result.completeness, token_usage_complete=False,
            cost_complete=False, terminal_complete=False,
        ),
        pricing_evidence=_pricing_evidence_sum(
            None, result.pricing_evidence, force_incomplete=True
        ),
        execution_trace=(None if trace is None else _replace(
            trace, events=tuple(
                _replace(event, name="partial") if event.kind == "usage" else event
                for event in trace.events
            ),
        )),
    )]
    return result


async def _await_with_deadline(
    awaitable: Any,
    timeout: float | None,
) -> Any:
    """Await without confusing a provider ``TimeoutError`` with our wall."""

    # Own a task even without a deadline: SDK teardown can replace or suppress
    # its cancellation. This owner must still join it and replay the original
    # external cancellation independently of that child behavior.
    task = asyncio.ensure_future(awaitable)
    try:
        done, _pending = await asyncio.wait(
            {task},
            timeout=None if timeout is None else max(0.0, timeout),
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        if not task.done():
            task.cancel()
        # ``asyncio.wait`` and the child can both settle in the same event-loop
        # turn.  A concurrently completed child may already carry an exception,
        # but it is still our task and must be observed before the owner's
        # cancellation/exception is replayed.
        await asyncio.gather(task, return_exceptions=True)
        raise
    if task in done:
        return await task
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _InvocationDeadlineExceeded


async def _await_with_terminal_tool_monitor(
    awaitable: Any,
    terminal_tool_server: Any | None,
) -> tuple[Any | None, BaseException | None]:
    """Race one provider attempt against a terminal MCP dispatch denial.

    The Codex app-server owns its internal tool loop. Returning an ordinary
    MCP ``isError`` result after a workflow-owned source budget is exhausted
    lets the model issue the same denied call repeatedly inside the same paid
    turn. The HTTP bridge therefore exposes a terminal event; this adapter
    cancels the provider attempt on the first such denial.

    A result that completed concurrently with the denial is returned only so
    its observed usage can be retained. The caller still replaces its content
    with a non-retriable terminal error.
    """

    event = getattr(
        terminal_tool_server,
        "terminal_dispatch_error_event",
        None,
    )
    message_attribute = "terminal_dispatch_error_message"
    if event is None or not callable(getattr(event, "is_set", None)):
        return await awaitable, None

    async def wait_for_denial() -> BaseException:
        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            TerminalToolDispatchError,
            ToolDispatchBudgetExceeded,
        )

        while not event.is_set():
            await asyncio.sleep(0.05)
        terminal = getattr(
            terminal_tool_server,
            "terminal_dispatch_error",
            None,
        )
        if isinstance(terminal, TerminalToolDispatchError):
            return terminal
        message = getattr(
            terminal_tool_server,
            message_attribute,
            None,
        )
        return ToolDispatchBudgetExceeded(
            message
            if isinstance(message, str) and message.strip()
            else "workflow-owned tool dispatch budget was denied"
        )

    attempt_task = asyncio.ensure_future(awaitable)
    denial_task = asyncio.create_task(wait_for_denial())
    try:
        done, _pending = await asyncio.wait(
            {attempt_task, denial_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if denial_task in done:
            denial_error = await denial_task
            if not attempt_task.done():
                attempt_task.cancel()
                await asyncio.gather(
                    attempt_task,
                    return_exceptions=True,
                )
                return None, denial_error
            try:
                result = await attempt_task
            except BaseException:
                # The terminal policy boundary takes precedence over a
                # concurrent transport teardown raised while cancellation
                # propagated through the app-server client.
                result = None
            return result, denial_error

        denial_task.cancel()
        await asyncio.gather(denial_task, return_exceptions=True)
        try:
            result = await attempt_task
        except BaseException:
            terminal = getattr(
                terminal_tool_server,
                "terminal_dispatch_error",
                None,
            )
            if isinstance(terminal, BaseException):
                return None, terminal
            message = getattr(
                terminal_tool_server,
                message_attribute,
                None,
            )
            if isinstance(message, str) and message.strip():
                from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                    ToolDispatchBudgetExceeded,
                )

                return None, ToolDispatchBudgetExceeded(message)
            raise
        terminal = getattr(
            terminal_tool_server,
            "terminal_dispatch_error",
            None,
        )
        if isinstance(terminal, BaseException):
            return result, terminal
        message = getattr(
            terminal_tool_server,
            message_attribute,
            None,
        )
        if isinstance(message, str) and message.strip():
            from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                ToolDispatchBudgetExceeded,
            )

            return result, ToolDispatchBudgetExceeded(message)
        return result, None
    except BaseException:
        for task in (attempt_task, denial_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            attempt_task,
            denial_task,
            return_exceptions=True,
        )
        raise


_MCP_URL_UNSET = object()
_CHAT_RESULT_IDENTITY_OMITTED = object()


def _process_hosted_toolset(
    toolset: Any,
    *,
    invocation_deadline: float | None,
    register_host: Any | None = None,
    unregister_host: Any | None = None,
    ensure_open: Any | None = None,
) -> tuple[Any, Any | None]:
    """Qualify mutators and bind them to one reapable child process."""

    write_names = frozenset(str(name) for name in getattr(toolset, "write_names", ()))
    if not write_names:
        return toolset, None
    if invocation_deadline is None:
        raise ValueError(
            "workspace-write Codex invocations require a finite wall deadline"
        )
    from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
        IsolatedMutationHost,
        ProcessHostedToolSet,
        build_isolated_toolset_spec,
    )

    spec = build_isolated_toolset_spec(
        toolset,
        expected_write_names=write_names,
    )
    host = IsolatedMutationHost(
        spec,
        invocation_deadline=invocation_deadline,
    )
    if register_host is not None:
        register_host(host)
    try:
        host.start()
        if ensure_open is not None:
            ensure_open()
        hosted = ProcessHostedToolSet(toolset, host)
    except BaseException:
        host.stop()
        if unregister_host is not None:
            unregister_host(host)
        raise
    return hosted, host


def _classify_max_turns(
    stop_reason: str | None, turns_used: int, turns_limit: int
) -> bool:
    """Recognize only an explicit provider max-turn terminal."""

    del turns_used, turns_limit
    return stop_reason in ("max_turns", "max-turns")


def _resolve_codex_auth_path() -> Path:
    codex_home = os.environ.get("CODEX_HOME", "").strip()
    if codex_home:
        return Path(codex_home) / "auth.json"
    return Path.home() / ".codex" / "auth.json"


def _trusted_repository_root(cwd: Path) -> Path:
    """Resolve the nearest repository boundary without invoking Git."""

    resolved = cwd.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("trusted Codex cwd must be an existing directory")
    for candidate in (resolved, *resolved.parents):
        marker = candidate / ".git"
        if marker.is_dir() or marker.is_file():
            return candidate
    raise ValueError(f"trusted Codex cwd is not inside a repository: {resolved}")


# Codex SDK token-usage notification method. The CLI emits one after every
# upstream response completion with cumulative thread counters in ``total``
# and that response's counters in ``last``. Resume first replays the prior
# turn's usage, so accounting binds IDs and cumulative deltas before summing
# only current-turn ``last`` snapshots.
_TOKEN_USAGE_METHOD = "thread/tokenUsage/updated"
_RAW_RESPONSE_COMPLETED_METHOD = "rawResponse/completed"
_TURN_COMPLETED_METHODS = frozenset(
    {"turn/completed", "turn.completed", "turnCompleted"}
)
_TURN_FAILED_METHODS = frozenset(
    {
        "turn/error",
        "turn.failed",
        "turn/failed",
        "turnFailed",
        "turn/errored",
    }
)
_HIDDEN_DELEGATION_ITEM_TYPES = frozenset(
    {
        "collabAgentToolCall",
        "subAgentActivity",
    }
)
_HIDDEN_DELEGATION_FAILURE_MESSAGE = (
    "Codex invocation observed forbidden collaboration/subagent "
    "activity in the app-server event stream"
)
_MAX_DELEGATION_METHOD_CHARS = 512
_MAX_TERMINAL_ITEMS_INSPECTED = 4096
_CAPTURED_TOKEN_FIELDS = (
    "inputTokens",
    "cachedInputTokens",
    "outputTokens",
)
_CAPTURED_OPTIONAL_TOKEN_FIELDS = ("cacheWriteInputTokens",)
_MAX_CAPTURED_TOKEN_COUNT = (1 << 63) - 1
_MAX_CAPTURED_USAGE_EVENTS = 1024
_MAX_CAPTURED_DELEGATION_SIGNALS = 64
_MAX_CAPTURED_TERMINAL_EVENTS = 64
_CAPTURE_SEQUENCE_INCOMPLETE_FIELD = "_captureSequenceIncomplete"
_MAX_CAPTURED_ERROR_TEXT_CHARS = 2000
_MAX_CAPTURED_IDENTIFIER_CHARS = 512
_MAX_CAPTURED_ERROR_CODE_CHARS = 128
_MAX_CAPTURED_ERROR_INFO_NODES = 64


@dataclass(frozen=True)
class _CodexDelegationSignal:
    """One bounded forbidden-delegation signal and its active-turn identity."""

    classification: str
    thread_id: str | None
    turn_id: str | None
    turn_ids: tuple[str, ...] = ()
    identity_complete: bool = True


_DELEGATION_CAPTURE_OVERFLOW = _CodexDelegationSignal(
    classification="capture_overflow",
    thread_id=None,
    turn_id=None,
    identity_complete=False,
)


def _codex_initialize_params() -> dict[str, dict[str, bool]]:
    """Opt into the exact per-response usage notification.

    App-server classifies ``rawResponse/completed`` as experimental and does
    not emit it unless the client declares ``experimentalApi`` during the
    initialize handshake. Return a fresh mapping so neither the SDK nor a
    test double can mutate shared backend state.
    """

    return {"capabilities": {"experimentalApi": True}}


async def _start_codex_thread_with_exact_usage(
    client: Any,
    config: Any,
) -> Any:
    """Start a thread with Codex 0.146's exact-usage event enabled.

    ``codex-app-server-sdk`` 0.3.2 does not yet project the experimental
    ``thread/start.experimentalRawEvents`` field through ``ThreadConfig``.
    Its public low-level request method and public ``ThreadHandle`` provide a
    narrow compatibility path until the pinned SDK gains that field. Test
    doubles without the low-level request surface retain the ordinary helper.
    """

    request = getattr(client, "request", None)
    if not callable(request):
        return await client.start_thread(config=config)

    params = {
        "model": config.model,
        "baseInstructions": config.base_instructions,
        "cwd": config.cwd,
        "approvalPolicy": config.approval_policy,
        "sandbox": config.sandbox,
        "experimentalRawEvents": True,
    }
    result = await request("thread/start", params)
    thread = result.get("thread") if isinstance(result, dict) else None
    thread_id = thread.get("id") if isinstance(thread, dict) else None
    if type(thread_id) is not str or not thread_id:
        raise RuntimeError("Codex thread/start with exact usage returned no thread id")
    from codex_app_server_sdk import (  # type: ignore[import-not-found]  # noqa: PLC0415
        ThreadHandle,
    )

    return ThreadHandle(client, thread_id, defaults=config)


_DIRECT_SUBSCHEMA_KEYS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_ARRAY_KEYS = frozenset(
    {
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems",
    }
)
_NAMED_SUBSCHEMA_MAP_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)


@dataclass(frozen=True)
class _CodexTerminalFailure:
    """Qualified non-success terminal carried by ``turn/completed``."""

    status: str
    message: str
    error_code: str | None = None
    http_status: int | None = None
    turn_id: str | None = None


def _without_unique_items_for_codex(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Copy a schema while omitting unsupported ``uniqueItems`` keywords.

    Codex forwards ``output_schema`` through OpenAI Structured Outputs, whose
    supported JSON-Schema subset does not include ``uniqueItems``. Traverse
    only subschema-bearing positions so a property literally named
    ``uniqueItems`` and literal values under ``enum``/``default`` survive.
    Assay retains the original schema and its strict response parser remains
    the authoritative client-side uniqueness validator.
    """

    normalized = deepcopy(schema)

    def strip(node: Any) -> int:
        if not isinstance(node, dict):
            return 0
        removed = int("uniqueItems" in node)
        node.pop("uniqueItems", None)

        for key in _DIRECT_SUBSCHEMA_KEYS:
            child = node.get(key)
            if isinstance(child, dict):
                removed += strip(child)
            elif key == "items" and isinstance(child, list):
                removed += sum(strip(item) for item in child)

        for key in _SUBSCHEMA_ARRAY_KEYS:
            children = node.get(key)
            if isinstance(children, list):
                removed += sum(strip(child) for child in children)

        for key in _NAMED_SUBSCHEMA_MAP_KEYS:
            children = node.get(key)
            if isinstance(children, dict):
                removed += sum(strip(child) for child in children.values())

        dependencies = node.get("dependencies")
        if isinstance(dependencies, dict):
            removed += sum(
                strip(child)
                for child in dependencies.values()
                if isinstance(child, dict)
            )
        return removed

    return normalized, strip(normalized)


def _nested_http_status(value: Any) -> int | None:
    """Extract an app-server upstream HTTP status without coercing booleans."""

    if isinstance(value, dict):
        for key in ("httpStatusCode", "http_status_code", "status_code"):
            candidate = value.get(key)
            if type(candidate) is int and 100 <= candidate <= 599:
                return candidate
        for child in value.values():
            status = _nested_http_status(child)
            if status is not None:
                return status
    elif isinstance(value, list):
        for child in value:
            status = _nested_http_status(child)
            if status is not None:
                return status
    return None


def _codex_error_code(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict) and len(value) == 1:
        key = next(iter(value))
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _codex_terminal_failure(
    raw_events: list[dict[str, Any]],
) -> _CodexTerminalFailure | None:
    """Resolve the pinned app-server's nested failed/interrupted terminal."""

    for event in reversed(raw_events):
        method = event.get("method")
        if method not in _TURN_COMPLETED_METHODS | _TURN_FAILED_METHODS:
            continue
        params = event.get("params")
        params = params if isinstance(params, dict) else {}
        turn = params.get("turn")
        turn = turn if isinstance(turn, dict) else {}
        status = turn.get("status")
        if method in _TURN_FAILED_METHODS:
            status = status if status in {"failed", "interrupted"} else "failed"
        if status not in {"failed", "interrupted"}:
            continue

        error = turn.get("error")
        if not isinstance(error, dict):
            candidate = params.get("error")
            error = candidate if isinstance(candidate, dict) else {}
        raw_message = error.get("message")
        message = (
            redact_sdk_text(raw_message.strip())
            if isinstance(raw_message, str) and raw_message.strip()
            else f"Codex turn ended with status={status}"
        )
        details = error.get("additionalDetails")
        redacted_details = (
            redact_sdk_text(details.strip())
            if isinstance(details, str) and details.strip()
            else ""
        )
        if redacted_details and redacted_details not in message:
            message = f"{message}; {redacted_details}"
        info = error.get("codexErrorInfo")
        turn_id = turn.get("id")
        return _CodexTerminalFailure(
            status=status,
            message=message[:2000],
            error_code=_codex_error_code(info),
            http_status=_nested_http_status(info),
            turn_id=(
                turn_id.strip()
                if isinstance(turn_id, str) and turn_id.strip()
                else None
            ),
        )
    return None


def _format_codex_terminal_failure(
    failure: _CodexTerminalFailure,
) -> str:
    qualifiers = [f"status={failure.status}"]
    if failure.error_code is not None:
        qualifiers.append(f"codex_error_info={failure.error_code}")
    if failure.http_status is not None:
        qualifiers.append(f"http_status={failure.http_status}")
    return f"Codex provider turn failed ({', '.join(qualifiers)}): {failure.message}"


_RETRIABLE_CODEX_TURN_FAILURES = frozenset(
    {
        "httpConnectionFailed",
        "internalServerError",
        "responseStreamConnectionFailed",
        "responseStreamDisconnected",
        "serverOverloaded",
    }
)


def _classify_codex_terminal_failure(
    failure: _CodexTerminalFailure,
) -> ClassifiedError:
    """Classify a provider-qualified turn terminal without blind retries."""

    message = _format_codex_terminal_failure(failure)
    if failure.status == "interrupted":
        return ExternalCancellation(
            cause="CodexTurnInterrupted",
            message=message,
            retriable=False,
        )
    if failure.error_code == "cyberPolicy":
        return ContentRejected(message=message, retriable=False)
    return TransportError(
        http_status=failure.http_status,
        message=message,
        retriable=(
            failure.error_code in _RETRIABLE_CODEX_TURN_FAILURES
            or (
                failure.http_status is not None
                and failure.http_status in RETRIABLE_HTTP_STATUSES
            )
        ),
    )


def _bounded_capture_text(value: Any) -> str | None:
    """Redact and bound provider error text retained past SDK cleanup."""

    if type(value) is not str:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return redact_sdk_text(stripped)[:_MAX_CAPTURED_ERROR_TEXT_CHARS]


def _bounded_capture_identifier(value: Any, *, max_chars: int) -> str | None:
    """Retain an identifier only when it fits without lossy truncation."""

    if type(value) is not str:
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > max_chars:
        return None
    return stripped


def _bounded_capture_http_status(value: Any) -> int | None:
    """Find a shallow provider HTTP status without following cycles forever."""

    pending = [value]
    seen: set[int] = set()
    examined = 0
    while pending and examined < _MAX_CAPTURED_ERROR_INFO_NODES:
        node = pending.pop()
        examined += 1
        if isinstance(node, dict):
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            for key in ("httpStatusCode", "http_status_code", "status_code"):
                candidate = dict.get(node, key)
                if type(candidate) is int and 100 <= candidate <= 599:
                    return candidate
            remaining = _MAX_CAPTURED_ERROR_INFO_NODES - examined
            if remaining > 0:
                children = []
                for child in dict.values(node):
                    children.append(child)
                    if len(children) >= remaining:
                        break
                pending.extend(reversed(children))
        elif isinstance(node, list):
            identity = id(node)
            if identity in seen:
                continue
            seen.add(identity)
            remaining = _MAX_CAPTURED_ERROR_INFO_NODES - examined
            if remaining > 0:
                count = min(list.__len__(node), remaining)
                pending.extend(
                    list.__getitem__(node, index) for index in range(count - 1, -1, -1)
                )
    return None


def _capture_codex_error_info(value: Any) -> Any:
    """Reduce arbitrary provider diagnostics to code + HTTP status evidence."""

    if type(value) is str:
        return _bounded_capture_identifier(
            value,
            max_chars=_MAX_CAPTURED_ERROR_CODE_CHARS,
        )

    code: str | None = None
    if isinstance(value, dict) and dict.__len__(value) == 1:
        key = next(iter(dict.keys(value)))
        code = _bounded_capture_identifier(
            key,
            max_chars=_MAX_CAPTURED_ERROR_CODE_CHARS,
        )
    http_status = _bounded_capture_http_status(value)
    if code is not None:
        nested = {"httpStatusCode": http_status} if http_status is not None else {}
        return {code: nested}
    if http_status is not None:
        # Keep two keys so _codex_error_code() does not mistake the normalized
        # metadata key for a provider error code.
        return {
            "_capture": "bounded",
            "httpStatusCode": http_status,
        }
    return None


def _capture_token_counter_map(value: Any) -> dict[str, int | None] | None:
    """Copy only billable counters while preserving malformed-map truthiness."""

    if not isinstance(value, dict):
        return None
    if dict.__len__(value) == 0:
        return {}
    captured: dict[str, int | None] = {}
    for field_name in _CAPTURED_TOKEN_FIELDS:
        candidate = dict.get(value, field_name)
        captured[field_name] = (
            candidate
            if (type(candidate) is int and 0 <= candidate <= _MAX_CAPTURED_TOKEN_COUNT)
            else None
        )
    for field_name in _CAPTURED_OPTIONAL_TOKEN_FIELDS:
        if not dict.__contains__(value, field_name):
            continue
        candidate = dict.get(value, field_name)
        captured[field_name] = (
            candidate
            if (type(candidate) is int and 0 <= candidate <= _MAX_CAPTURED_TOKEN_COUNT)
            else None
        )
    return captured


def _codex_hidden_delegation_signal(
    event: Any,
) -> _CodexDelegationSignal | None:
    """Capture one bounded app-server delegation signal without copying it.

    Current app-server emits collaboration calls and child-agent activity as
    ``item/started`` / ``item/completed`` notifications. A spawned child can
    also surface as ``thread/started`` with a parent or ``subAgent`` source.
    Keep a conservative method-name check for forward protocol spellings, but
    deliberately do not treat the unrelated ``process/spawn`` client API as
    agent delegation.
    """

    if not isinstance(event, dict):
        return None
    method = dict.get(event, "method")
    if type(method) is not str:
        return None
    params = dict.get(event, "params")
    thread_id: str | None = None
    turn_ids: list[str] = []
    identity_complete = not (
        dict.__contains__(event, "params") and not isinstance(params, dict)
    )
    if isinstance(params, dict):
        if dict.__contains__(params, "threadId"):
            thread_id = _bounded_capture_identifier(
                dict.get(params, "threadId"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
            if thread_id is None:
                identity_complete = False
        if dict.__contains__(params, "turnId"):
            turn_id = _bounded_capture_identifier(
                dict.get(params, "turnId"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
            if turn_id is None:
                identity_complete = False
            else:
                turn_ids.append(turn_id)
        turn = dict.get(params, "turn")
        if dict.__contains__(params, "turn"):
            if not isinstance(turn, dict):
                identity_complete = False
            elif dict.__contains__(turn, "id"):
                nested_turn_id = _bounded_capture_identifier(
                    dict.get(turn, "id"),
                    max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                )
                if nested_turn_id is None:
                    identity_complete = False
                else:
                    turn_ids.append(nested_turn_id)
        if method == "thread/started":
            thread = dict.get(params, "thread")
            if dict.__contains__(params, "thread"):
                if not isinstance(thread, dict):
                    identity_complete = False
                elif dict.__contains__(thread, "parentThreadId"):
                    parent_thread_id = _bounded_capture_identifier(
                        dict.get(thread, "parentThreadId"),
                        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                    )
                    if parent_thread_id is None:
                        identity_complete = False
                    else:
                        # The spawned thread's own ID is not the metered parent
                        # turn. Retain only its bounded parent identity here.
                        thread_id = parent_thread_id

    unique_turn_ids = tuple(dict.fromkeys(turn_ids))
    if len(unique_turn_ids) > 1:
        identity_complete = False

    def signal(classification: str) -> _CodexDelegationSignal:
        return _CodexDelegationSignal(
            classification=classification,
            thread_id=thread_id,
            turn_id=(unique_turn_ids[0] if len(unique_turn_ids) == 1 else None),
            turn_ids=unique_turn_ids,
            identity_complete=identity_complete,
        )

    if len(method) > _MAX_DELEGATION_METHOD_CHARS:
        return signal("unbounded_notification_method")
    if isinstance(params, dict):
        candidate_items: list[Any] = [dict.get(params, "item")]
        turn = dict.get(params, "turn")
        terminal_items = dict.get(turn, "items") if isinstance(turn, dict) else None
        if isinstance(terminal_items, list):
            if len(terminal_items) > _MAX_TERMINAL_ITEMS_INSPECTED:
                return signal("unbounded_terminal_item_ledger")
            candidate_items.extend(terminal_items)
        for item in candidate_items:
            if not isinstance(item, dict):
                continue
            item_type = dict.get(item, "type")
            if type(item_type) is str and item_type in _HIDDEN_DELEGATION_ITEM_TYPES:
                return signal(
                    "collab_agent_tool_call"
                    if item_type == "collabAgentToolCall"
                    else "subagent_activity",
                )
        if method == "thread/started":
            thread = dict.get(params, "thread")
            if isinstance(thread, dict):
                parent_thread_id = dict.get(thread, "parentThreadId")
                source = dict.get(thread, "source")
                if (type(parent_thread_id) is str and bool(parent_thread_id)) or (
                    isinstance(source, dict) and dict.__contains__(source, "subAgent")
                ):
                    return signal("spawned_subagent_thread")

    normalized_method = "".join(char for char in method.casefold() if char.isalnum())
    if "collabagent" in normalized_method or "agentcollab" in normalized_method:
        return signal("collaboration_agent_notification")
    if "subagent" in normalized_method:
        return signal("subagent_notification")
    if "spawnagent" in normalized_method or "agentspawn" in normalized_method:
        return signal("agent_spawn_notification")
    return None


def _retain_codex_delegation_signal(
    signals: list[_CodexDelegationSignal],
    signal: _CodexDelegationSignal,
) -> None:
    """Retain bounded, distinct signal identities and fail closed on overflow."""

    if signal in signals or _DELEGATION_CAPTURE_OVERFLOW in signals:
        return
    if len(signals) < _MAX_CAPTURED_DELEGATION_SIGNALS:
        signals.append(signal)
        return
    # The exact overflow contents are no longer knowable. Preserve that fact
    # within the original bound so later attribution cannot select an earlier,
    # apparently consistent identity from a truncated signal ledger.
    signals[-1] = _DELEGATION_CAPTURE_OVERFLOW


def _matched_delegation_turn_id(
    signals: list[_CodexDelegationSignal],
    *,
    expected_thread_id: str | None,
) -> str | None:
    """Select one delegation turn only when its active thread is proven."""

    expected_thread = _bounded_capture_identifier(
        expected_thread_id,
        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
    )
    if expected_thread is None or not signals:
        return None
    if any(
        not signal.identity_complete or signal.thread_id != expected_thread
        for signal in signals
    ):
        return None
    matched_turn_ids: set[str] = set()
    for signal in signals:
        matched_turn_ids.update(signal.turn_ids)
        if signal.turn_id is not None:
            matched_turn_ids.add(signal.turn_id)
    return next(iter(matched_turn_ids)) if len(matched_turn_ids) == 1 else None


def _qualified_resumed_delegation_turn_id(
    signals: list[_CodexDelegationSignal],
    raw_events: list[dict[str, Any]],
    *,
    expected_thread_id: str | None,
    result_thread_id: str | None = None,
    result_turn_id: str | None = None,
    result_identity_complete: bool = True,
    require_signal_thread_id: bool = True,
) -> str | None:
    """Bind delegation evidence to exactly one active provider turn.

    A resume restores historical cumulative usage, so selecting the wrong turn
    can rebill arbitrary prior work. Require every available signal, result,
    and terminal identity to agree before any usage or provider request ID is
    admitted. Resumed calls also require an explicit signal thread; a fresh
    call may use its already-retained handle when a legacy signal omits that
    field. A signal may omit its turn ID only when the result or terminal
    supplies the single consistent ID.
    """

    expected_thread = _bounded_capture_identifier(
        expected_thread_id,
        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
    )
    if expected_thread is None or not signals or not result_identity_complete:
        return None
    if any(not signal.identity_complete for signal in signals):
        return None
    if any(
        signal.thread_id is not None and signal.thread_id != expected_thread
        for signal in signals
    ):
        return None
    if require_signal_thread_id and any(signal.thread_id is None for signal in signals):
        return None

    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    if (
        not terminal_turn_shapes_complete
        or not terminal_thread_shapes_complete
        or any(thread_id != expected_thread for thread_id in terminal_thread_ids)
    ):
        return None

    candidate_turn_ids: list[str] = []
    for signal in signals:
        candidate_turn_ids.extend(signal.turn_ids)
        if signal.turn_id is not None and signal.turn_id not in signal.turn_ids:
            candidate_turn_ids.append(signal.turn_id)
    if result_thread_id is not None:
        bounded_result_thread = _bounded_capture_identifier(
            result_thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if bounded_result_thread != expected_thread:
            return None
    if result_turn_id is not None:
        bounded_result_turn = _bounded_capture_identifier(
            result_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if bounded_result_turn is None:
            return None
        candidate_turn_ids.append(bounded_result_turn)
    candidate_turn_ids.extend(terminal_turn_ids)
    unique_turn_ids = set(candidate_turn_ids)
    return next(iter(unique_turn_ids)) if len(unique_turn_ids) == 1 else None


def _capture_codex_evidence_event(event: Any) -> dict[str, Any] | None:
    """Return a bounded usage/terminal snapshot, ignoring all event deltas."""

    if not isinstance(event, dict):
        return None
    method = dict.get(event, "method")
    if type(method) is not str:
        return None

    if method == _TOKEN_USAGE_METHOD:
        captured: dict[str, Any] = {"method": method}
        params = dict.get(event, "params")
        if not isinstance(params, dict):
            if dict.__contains__(event, "params"):
                captured["params"] = None
            return captured
        captured_params: dict[str, Any] = {}
        for field_name in ("threadId", "turnId"):
            if dict.__contains__(params, field_name):
                captured_params[field_name] = _bounded_capture_identifier(
                    dict.get(params, field_name),
                    max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                )
        if dict.__contains__(params, "tokenUsage"):
            token_usage = dict.get(params, "tokenUsage")
            if isinstance(token_usage, dict):
                bounded_usage: dict[str, Any] = {}
                for field_name in ("total", "last"):
                    counters = _capture_token_counter_map(dict.get(token_usage, field_name))
                    if counters is not None:
                        bounded_usage[field_name] = counters
                captured_params["tokenUsage"] = bounded_usage
            else:
                captured_params["tokenUsage"] = None
        captured["params"] = captured_params
        return captured

    if method == _RAW_RESPONSE_COMPLETED_METHOD:
        captured = {"method": method}
        params = dict.get(event, "params")
        if not isinstance(params, dict):
            if dict.__contains__(event, "params"):
                captured["params"] = None
            return captured
        captured_params: dict[str, Any] = {}
        for field_name in ("threadId", "turnId", "responseId"):
            identifier = _bounded_capture_identifier(
                dict.get(params, field_name),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
            if identifier is not None:
                captured_params[field_name] = identifier
        if dict.__contains__(params, "usage"):
            usage = dict.get(params, "usage")
            captured_params["usage"] = (
                _capture_token_counter_map(usage) if isinstance(usage, dict) else None
            )
        captured["params"] = captured_params
        return captured

    if method not in _TURN_COMPLETED_METHODS | _TURN_FAILED_METHODS:
        return None

    captured = {"method": method}
    params = dict.get(event, "params")
    if not isinstance(params, dict):
        if dict.__contains__(event, "params"):
            # Preserve present-but-invalid containers so downstream identity
            # qualification cannot mistake them for a legacy ID-less event.
            captured["params"] = None
        return captured
    turn = dict.get(params, "turn")
    captured_params: dict[str, Any] = {}
    if dict.__contains__(params, "threadId"):
        captured_params["threadId"] = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
    captured_turn: dict[str, Any] = {}
    if isinstance(turn, dict):
        status = dict.get(turn, "status")
        if type(status) is str and status in {"failed", "interrupted"}:
            captured_turn["status"] = status
        if dict.__contains__(turn, "id"):
            captured_turn["id"] = _bounded_capture_identifier(
                dict.get(turn, "id"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
    elif dict.__contains__(params, "turn"):
        captured_params["turn"] = None

    error = dict.get(turn, "error") if isinstance(turn, dict) else None
    error_location = captured_turn
    if not isinstance(error, dict):
        error = dict.get(params, "error")
        error_location = captured_params
    if isinstance(error, dict):
        captured_error: dict[str, Any] = {}
        for field_name in ("message", "additionalDetails"):
            text = _bounded_capture_text(dict.get(error, field_name))
            if text is not None:
                captured_error[field_name] = text
        info = _capture_codex_error_info(dict.get(error, "codexErrorInfo"))
        if info is not None:
            captured_error["codexErrorInfo"] = info
        if captured_error:
            error_location["error"] = captured_error

    if captured_turn:
        captured_params["turn"] = captured_turn
    if captured_params:
        captured["params"] = captured_params
    return captured


def _install_codex_event_capture(
    client: Any,
    raw_events: list[dict[str, Any]],
    *,
    forbidden_delegation_events: list[_CodexDelegationSignal] | None = None,
) -> bool:
    """Capture turn evidence before SDK cleanup and qualify nested failure.

    Pinned ``codex-app-server-sdk==0.3.2`` recognizes failure only from
    failure *method names*. The app-server's canonical terminal instead uses
    ``turn/completed`` with ``params.turn.status=failed|interrupted`` and a
    nested error. Wrap the unstarted client so the receiver cannot race past
    this qualification. Retain the replay baseline and up to
    ``_MAX_CAPTURED_USAGE_EVENTS`` response updates; on overflow, retain one
    marked latest snapshot so accounting can fail closed to an aggregate
    bound. Terminal events and distinct delegation identities are retained up
    to explicit caps; overflow is marked incomplete so a conflicting identity
    cannot disappear behind replacement. Each snapshot contains only bounded
    fields needed by accounting and typed terminal classification. For a
    route whose feature profile forbids native delegation, the optional
    delegation sink additionally makes the active SDK session fail
    immediately. Provider events are never mutated. Return whether the
    dispatcher wrapper was installed.
    """

    original = getattr(client, "_apply_event_to_session", None)
    if not callable(original):
        if type(client).__module__.startswith("codex_app_server_sdk"):
            raise RuntimeError(
                "codex-app-server-sdk event-dispatch shape changed; cannot "
                "preserve qualified turn failures"
            )
        return False

    if forbidden_delegation_events is not None:
        original_event_is_for_turn = getattr(
            client,
            "_event_is_for_turn",
            None,
        )
        if callable(original_event_is_for_turn):

            def event_is_for_turn_with_delegation(
                _client: Any,
                event: Any,
                turn_id: str,
            ) -> bool:
                return bool(
                    _codex_hidden_delegation_signal(event) is not None
                    or original_event_is_for_turn(event, turn_id)
                )

            client._event_is_for_turn = MethodType(  # noqa: SLF001
                event_is_for_turn_with_delegation,
                client,
            )
        elif type(client).__module__.startswith("codex_app_server_sdk"):
            raise RuntimeError(
                "codex-app-server-sdk event-routing shape changed; cannot "
                "enforce the native-delegation boundary"
            )

    captured_usage_events = 0
    captured_raw_response_events = 0
    captured_terminal_events = 0
    overflow_usage_index: int | None = None
    overflow_raw_response_index: int | None = None
    overflow_terminal_index: int | None = None

    def apply_with_capture(
        _client: Any,
        session: Any,
        event: Any,
    ) -> Any:
        nonlocal captured_usage_events, captured_raw_response_events
        nonlocal captured_terminal_events
        nonlocal overflow_usage_index, overflow_raw_response_index
        nonlocal overflow_terminal_index
        failure = None
        forbidden_delegation = (
            _codex_hidden_delegation_signal(event)
            if forbidden_delegation_events is not None
            else None
        )
        if forbidden_delegation is not None:
            assert forbidden_delegation_events is not None
            _retain_codex_delegation_signal(
                forbidden_delegation_events,
                forbidden_delegation,
            )
        try:
            captured = _capture_codex_evidence_event(event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CodexBackend: could not retain bounded app-server event evidence (%s)",
                type(exc).__name__,
            )
            captured = None
        if captured is not None:
            if captured["method"] == _TOKEN_USAGE_METHOD:
                captured_usage_events += 1
                if captured_usage_events <= _MAX_CAPTURED_USAGE_EVENTS:
                    raw_events.append(captured)
                else:
                    captured[_CAPTURE_SEQUENCE_INCOMPLETE_FIELD] = True
                    if overflow_usage_index is None:
                        overflow_usage_index = len(raw_events)
                        raw_events.append(captured)
                    else:
                        raw_events[overflow_usage_index] = captured
            elif captured["method"] == _RAW_RESPONSE_COMPLETED_METHOD:
                captured_raw_response_events += 1
                if captured_raw_response_events <= _MAX_CAPTURED_USAGE_EVENTS:
                    raw_events.append(captured)
                else:
                    captured[_CAPTURE_SEQUENCE_INCOMPLETE_FIELD] = True
                    if overflow_raw_response_index is None:
                        overflow_raw_response_index = len(raw_events)
                        raw_events.append(captured)
                    else:
                        raw_events[overflow_raw_response_index] = captured
            else:
                failure = _codex_terminal_failure([captured])
                captured_terminal_events += 1
                if captured_terminal_events <= _MAX_CAPTURED_TERMINAL_EVENTS:
                    raw_events.append(captured)
                else:
                    overflow = {
                        "method": captured["method"],
                        "params": None,
                        _CAPTURE_SEQUENCE_INCOMPLETE_FIELD: True,
                    }
                    if overflow_terminal_index is None:
                        overflow_terminal_index = len(raw_events)
                        raw_events.append(overflow)
                    else:
                        raw_events[overflow_terminal_index] = overflow
        result = original(session, event)
        if failure is not None:
            # Qualify the nested canonical terminal for SDK 0.3.2's existing
            # failure path. Keep the structured copy above for typed adapter
            # classification after chat_once cleans its session state.
            session.failed = True
            session.failure_message = _format_codex_terminal_failure(failure)
        if forbidden_delegation is not None:
            # Do not wait for a model-controlled child to finish. ``chat_once``
            # raises on its next loop edge and the async client context closes
            # the app-server transport. The adapter classifies the latched
            # signal as non-retriable below, so transport recovery cannot
            # create another unauthorized turn.
            session.failed = True
            session.failure_message = _HIDDEN_DELEGATION_FAILURE_MESSAGE
        return result

    client._apply_event_to_session = MethodType(  # noqa: SLF001
        apply_with_capture,
        client,
    )
    return True


def _usage_sum(left: Usage, right: Usage) -> Usage:
    """Add usage observed on two physical attempts."""

    return Usage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cost_usd=left.cost_usd + right.cost_usd,
        cache_creation_tokens=(
            left.cache_creation_tokens + right.cache_creation_tokens
        ),
        cache_read_tokens=(left.cache_read_tokens + right.cache_read_tokens),
    )


def _pricing_evidence_sum(
    left: UsagePricingEvidence | None,
    right: UsagePricingEvidence | None,
    *,
    force_incomplete: bool = False,
) -> UsagePricingEvidence:
    """Concatenate ordered physical-response ledgers without inventing gaps."""

    left_segments = () if left is None else left.segments
    right_segments = () if right is None else right.segments
    return UsagePricingEvidence(
        segments=left_segments + right_segments,
        complete=(
            not force_incomplete
            and left is not None
            and right is not None
            and left.complete
            and right.complete
        ),
    )


def _validated_timeout_seconds(policy: BackendPolicy) -> float | None:
    value = policy.timeout_seconds
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(
            "BackendPolicy.timeout_seconds must be None or a finite number > 0"
        )
    return float(value)


def _conversation_outcome(
    invocation: Invocation,
    *,
    opaque_id: str | None,
    state: str = "active",
) -> ConversationTransportOutcome | None:
    """Bind provider continuity to the exact admitted logical operation."""

    request = invocation.conversation
    if request is None:
        return None
    if state == "active" and opaque_id is None:
        # A failure before ``thread/start`` produced no provider continuity
        # material. Do not fabricate an active provider identifier.
        return None
    return ConversationTransportOutcome(
        state=state,  # type: ignore[arg-type]
        opaque_id=opaque_id,
        identity_fingerprint=request.identity_fingerprint,
        operation_digest=request.operation_digest,
    )


def _codex_resume_was_rejected(exc: BaseException) -> bool:
    """Recognize provider protocol rejection without swallowing transport."""

    return type(_unwrap_exception_group(exc)).__name__ == "CodexProtocolError"


def _usage_observed(result: InvocationResult) -> bool:
    return (
        result.completeness.token_usage_complete
        or result.completeness.cost_complete
        or result.usage.input_tokens != 0
        or result.usage.output_tokens != 0
        or result.usage.cost_usd != 0
    )


_COST_UPPER_BOUND_EVENT_NAME = "cost_upper_bound"


def _cost_accounting_is_budget_safe(result: InvocationResult) -> bool:
    """Whether a physical attempt has exact cost or a traced upper bound."""

    if result.completeness.cost_complete:
        return True
    trace = result.execution_trace
    return trace is not None and any(
        event.kind == "usage" and event.name == _COST_UPPER_BOUND_EVENT_NAME
        for event in trace.events
    )


def _provider_terminal_complete(
    raw_events: list[dict[str, Any]],
    *,
    current_turn_id: str | None = None,
    expected_thread_id: str | None = None,
) -> bool:
    """Return whether app-server supplied the current turn's terminal."""

    terminal_observed = any(
        event.get("method") in _TURN_COMPLETED_METHODS | _TURN_FAILED_METHODS
        for event in raw_events
    )
    if not terminal_observed:
        return False
    (
        terminal_thread_ids,
        terminal_thread_shapes_complete,
    ) = _terminal_thread_identity_evidence(raw_events)
    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    if not terminal_thread_shapes_complete or not terminal_turn_shapes_complete:
        return False
    if expected_thread_id is not None:
        expected_thread = _bounded_capture_identifier(
            expected_thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if expected_thread is None or any(
            thread_id != expected_thread for thread_id in terminal_thread_ids
        ):
            return False
    elif terminal_thread_ids and any(
        thread_id != terminal_thread_ids[0] for thread_id in terminal_thread_ids
    ):
        return False
    if current_turn_id is None:
        return not terminal_turn_ids or all(
            turn_id == terminal_turn_ids[0] for turn_id in terminal_turn_ids
        )
    expected_turn_id = _bounded_capture_identifier(
        current_turn_id,
        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
    )
    if expected_turn_id is None:
        return False
    return all(
        terminal_turn_id == expected_turn_id for terminal_turn_id in terminal_turn_ids
    )


def _append_event(
    events: list[BackendExecutionEvent],
    kind: str,
    *,
    name: str | None = None,
    parent_id: str | None = None,
    child_id: str | None = None,
) -> None:
    events.append(
        BackendExecutionEvent(
            sequence=len(events),
            kind=kind,
            name=name,
            parent_id=parent_id,
            child_id=child_id,
        )
    )


def _merge_nonterminal_events(
    destination: list[BackendExecutionEvent],
    result: InvocationResult,
    *,
    request_name: str,
    include_cancellation: bool = True,
) -> None:
    """Merge one physical-attempt trace without its wrapper terminal."""

    source = result.execution_trace.events if result.execution_trace is not None else ()
    nonterminal = [
        event
        for event in source
        if event.kind != "terminal"
        and (include_cancellation or event.kind != "cancellation")
    ]
    if not any(event.kind == "request" for event in nonterminal):
        _append_event(destination, "request", name=request_name)
    for event in nonterminal:
        _append_event(
            destination,
            event.kind,
            name=event.name,
            parent_id=event.parent_id,
            child_id=event.child_id,
        )
    if not any(event.kind == "usage" for event in nonterminal) and _usage_observed(
        result
    ):
        _append_event(
            destination,
            "usage",
            name=(
                "complete"
                if (
                    result.completeness.token_usage_complete
                    and result.completeness.cost_complete
                )
                else "partial"
            ),
        )


@dataclass(frozen=True)
class _CodexTokenCounters:
    """One bounded app-server token counter mapping."""

    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int | None
    output_tokens: int
    cache_write_malformed: bool = False


@dataclass(frozen=True)
class _CodexUsageUpdate:
    """One response-completion usage notification."""

    thread_id: str
    turn_id: str
    total: _CodexTokenCounters
    last: _CodexTokenCounters
    capture_sequence_incomplete: bool


def _bounded_token_counters(value: Any) -> _CodexTokenCounters | None:
    """Validate one total/last mapping without coercing provider values."""

    if not isinstance(value, dict):
        return None
    required: dict[str, int] = {}
    for field_name in ("inputTokens", "cachedInputTokens", "outputTokens"):
        if not dict.__contains__(value, field_name):
            return None
        candidate = dict.get(value, field_name)
        if (
            type(candidate) is not int
            or candidate < 0
            or candidate > _MAX_CAPTURED_TOKEN_COUNT
        ):
            return None
        required[field_name] = candidate

    cache_write_tokens: int | None = None
    cache_write_malformed = False
    if dict.__contains__(value, "cacheWriteInputTokens"):
        candidate = dict.get(value, "cacheWriteInputTokens")
        if type(candidate) is int and 0 <= candidate <= _MAX_CAPTURED_TOKEN_COUNT:
            cache_write_tokens = candidate
        else:
            cache_write_malformed = True
    if required["cachedInputTokens"] > required["inputTokens"]:
        return None
    if (
        cache_write_tokens is not None
        and required["cachedInputTokens"] + cache_write_tokens > required["inputTokens"]
    ):
        cache_write_tokens = None
        cache_write_malformed = True
    return _CodexTokenCounters(
        input_tokens=required["inputTokens"],
        cache_read_tokens=required["cachedInputTokens"],
        cache_write_tokens=cache_write_tokens,
        output_tokens=required["outputTokens"],
        cache_write_malformed=cache_write_malformed,
    )


def _usage_updates(
    raw_events: "list[dict[str, Any]]",
) -> tuple[list[_CodexUsageUpdate], bool, bool, bool]:
    """Return usage rows and exact/required shape-complete claims.

    ``cacheWriteInputTokens`` is optional provider telemetry.  A malformed
    value prevents exact token/category accounting, but it must not erase an
    otherwise well-shaped input/cache-read/output response boundary.  Keep a
    separate required-shape claim so those always-present counters can still
    authorize a conservative per-response cost bound.
    """

    updates: list[_CodexUsageUpdate] = []
    observed = False
    shapes_complete = True
    required_shapes_complete = True
    for event in raw_events:
        if event.get("method") != _TOKEN_USAGE_METHOD:
            continue
        observed = True
        params = event.get("params")
        if not isinstance(params, dict):
            shapes_complete = False
            required_shapes_complete = False
            continue
        thread_id = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        turn_id = _bounded_capture_identifier(
            dict.get(params, "turnId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        token_usage = dict.get(params, "tokenUsage")
        if not isinstance(token_usage, dict):
            shapes_complete = False
            required_shapes_complete = False
            continue
        total = _bounded_token_counters(dict.get(token_usage, "total"))
        last = _bounded_token_counters(dict.get(token_usage, "last"))
        if thread_id is None or turn_id is None or total is None or last is None:
            shapes_complete = False
            required_shapes_complete = False
            continue
        if total.cache_write_malformed or last.cache_write_malformed:
            shapes_complete = False
        updates.append(
            _CodexUsageUpdate(
                thread_id=thread_id,
                turn_id=turn_id,
                total=total,
                last=last,
                capture_sequence_incomplete=(
                    dict.get(event, _CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True
                ),
            )
        )
    return updates, observed, shapes_complete, required_shapes_complete


def _counter_cache_write(counter: _CodexTokenCounters) -> int | None:
    """Infer the only possible write count for an all-cache-read input."""

    if counter.input_tokens == counter.cache_read_tokens:
        return 0
    return counter.cache_write_tokens


def _counter_values(counter: _CodexTokenCounters) -> tuple[int, int, int, int] | None:
    cache_write_tokens = _counter_cache_write(counter)
    if cache_write_tokens is None:
        return None
    return (
        counter.input_tokens,
        counter.cache_read_tokens,
        cache_write_tokens,
        counter.output_tokens,
    )


def _counter_delta_matches(
    current: _CodexTokenCounters,
    previous: _CodexTokenCounters,
    last: _CodexTokenCounters,
) -> bool:
    current_values = _counter_values(current)
    previous_values = _counter_values(previous)
    last_values = _counter_values(last)
    if current_values is None or previous_values is None or last_values is None:
        return False
    return all(
        current_value - previous_value == last_value
        for current_value, previous_value, last_value in zip(
            current_values,
            previous_values,
            last_values,
            strict=True,
        )
    )


def _required_counter_values(
    counter: _CodexTokenCounters,
) -> tuple[int, int, int]:
    """Return counters that exist independently of cache-write telemetry."""

    return (
        counter.input_tokens,
        counter.cache_read_tokens,
        counter.output_tokens,
    )


def _required_counter_delta_matches(
    current: _CodexTokenCounters,
    previous: _CodexTokenCounters,
    last: _CodexTokenCounters,
) -> bool:
    """Reconcile one response without claiming an exact cache-write split."""

    return all(
        current_value - previous_value == last_value
        for current_value, previous_value, last_value in zip(
            _required_counter_values(current),
            _required_counter_values(previous),
            _required_counter_values(last),
            strict=True,
        )
    )


def _terminal_turn_identity_evidence(
    raw_events: "list[dict[str, Any]]",
) -> tuple[tuple[str, ...], bool]:
    turn_ids: list[str] = []
    shapes_complete = True
    for event in reversed(raw_events):
        if event.get("method") not in _TURN_COMPLETED_METHODS | _TURN_FAILED_METHODS:
            continue
        params = event.get("params")
        if not isinstance(params, dict):
            if dict.__contains__(event, "params"):
                shapes_complete = False
            continue
        if not dict.__contains__(params, "turn"):
            continue
        turn = dict.get(params, "turn")
        if not isinstance(turn, dict):
            shapes_complete = False
            continue
        if not dict.__contains__(turn, "id"):
            continue
        turn_id = dict.get(turn, "id")
        bounded = _bounded_capture_identifier(
            turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if bounded is not None:
            turn_ids.append(bounded)
        else:
            shapes_complete = False
    return tuple(turn_ids), shapes_complete


def _terminal_thread_identity_evidence(
    raw_events: "list[dict[str, Any]]",
) -> tuple[tuple[str, ...], bool]:
    thread_ids: list[str] = []
    shapes_complete = True
    for event in reversed(raw_events):
        if event.get("method") not in _TURN_COMPLETED_METHODS | _TURN_FAILED_METHODS:
            continue
        params = event.get("params")
        if not isinstance(params, dict):
            if dict.__contains__(event, "params"):
                shapes_complete = False
            continue
        if not dict.__contains__(params, "threadId"):
            continue
        thread_id = dict.get(params, "threadId")
        bounded = _bounded_capture_identifier(
            thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if bounded is not None:
            thread_ids.append(bounded)
        else:
            shapes_complete = False
    return tuple(thread_ids), shapes_complete


def _usage_identity_evidence(
    raw_events: "list[dict[str, Any]]",
) -> tuple[tuple[tuple[str, str], ...], bool]:
    """Return bounded usage thread/turn identities independent of counters.

    Counter shape and pricing completeness belong to the harvester. Keeping
    that validation separate prevents a malformed optional counter from
    revoking otherwise useful conservative accounting, while any malformed,
    conflicting, or capture-incomplete identity still fails attribution.
    """

    identities: list[tuple[str, str]] = []
    shapes_complete = True
    for event in raw_events:
        if event.get("method") != _TOKEN_USAGE_METHOD:
            continue
        if event.get(_CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True:
            shapes_complete = False
        params = event.get("params")
        if not isinstance(params, dict):
            shapes_complete = False
            continue
        thread_id = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        turn_id = _bounded_capture_identifier(
            dict.get(params, "turnId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if thread_id is None or turn_id is None:
            shapes_complete = False
            continue
        identities.append((thread_id, turn_id))
    return tuple(identities), shapes_complete


def _completed_chat_identity_matches_final_evidence(
    raw_events: "list[dict[str, Any]]",
    *,
    current_turn_id: str | None,
    expected_thread_id: str | None,
    identity_complete: bool,
    resumed: bool,
) -> bool:
    """Reconcile a completed chat with every retained identity boundary.

    A legacy ChatResult may omit its turn ID. In that case validate the
    internally selected terminal/raw/current-usage turn without promoting it
    to a provider request ID. A resumed cumulative-only ledger with no current
    selector remains internally well-shaped but cannot authorize usage; the
    history-free harvester below keeps that case at zero.
    """

    expected_thread = _bounded_capture_identifier(
        expected_thread_id,
        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
    )
    if not identity_complete or expected_thread is None:
        return False

    selected_turn_id = (
        _bounded_capture_identifier(
            current_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if current_turn_id is not None
        else None
    )
    if current_turn_id is not None and selected_turn_id is None:
        return False

    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    if (
        not terminal_turn_shapes_complete
        or not terminal_thread_shapes_complete
        or any(thread_id != expected_thread for thread_id in terminal_thread_ids)
    ):
        return False

    usage_identities, usage_shapes_complete = _usage_identity_evidence(raw_events)
    if (
        not usage_shapes_complete
        or any(thread_id != expected_thread for thread_id, _turn_id in usage_identities)
    ):
        return False

    raw_turn_ids: list[str] = []
    response_ids: set[str] = set()
    for event in raw_events:
        if event.get("method") != _RAW_RESPONSE_COMPLETED_METHOD:
            continue
        if event.get(_CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True:
            return False
        params = event.get("params")
        if not isinstance(params, dict):
            return False
        response_thread_id = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        response_turn_id = _bounded_capture_identifier(
            dict.get(params, "turnId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        response_id = _bounded_capture_identifier(
            dict.get(params, "responseId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if (
            response_thread_id != expected_thread
            or response_turn_id is None
            or response_id is None
            or response_id in response_ids
        ):
            return False
        raw_turn_ids.append(response_turn_id)
        response_ids.add(response_id)

    if selected_turn_id is None:
        explicit_candidates = set(terminal_turn_ids) | set(raw_turn_ids)
        if len(explicit_candidates) > 1:
            return False
        if explicit_candidates:
            selected_turn_id = next(iter(explicit_candidates))
        elif usage_identities:
            selected_turn_id = usage_identities[-1][1]

    if selected_turn_id is None:
        # A legacy result with no identity-bearing evidence is vacuously
        # consistent. The harvester still cannot invent a request ID.
        return not terminal_turn_ids and not raw_turn_ids

    if any(turn_id != selected_turn_id for turn_id in terminal_turn_ids) or any(
        turn_id != selected_turn_id for turn_id in raw_turn_ids
    ):
        return False

    if usage_identities:
        current_indexes = [
            index
            for index, (_thread_id, turn_id) in enumerate(usage_identities)
            if turn_id == selected_turn_id
        ]
        if not current_indexes:
            return False
        first_current_index = current_indexes[0]
        prefix = usage_identities[:first_current_index]
        if any(
            turn_id != selected_turn_id
            for _thread_id, turn_id in usage_identities[first_current_index:]
        ):
            return False
        if prefix and (
            not resumed or len(prefix) != 1 or prefix[0][1] == selected_turn_id
        ):
            return False
    return True


def _current_usage_updates(
    raw_events: "list[dict[str, Any]]",
    *,
    current_turn_id: str | None = None,
    allow_last_update_fallback: bool = True,
    expected_thread_id: str | None = None,
    identity_complete: bool = True,
    require_cache_write: bool = True,
) -> tuple[list[_CodexUsageUpdate], bool, bool]:
    """Validate replay baseline and current-response cumulative deltas."""

    (
        updates,
        observed,
        shapes_complete,
        required_shapes_complete,
    ) = _usage_updates(raw_events)
    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    terminal_turn_id = terminal_turn_ids[0] if terminal_turn_ids else None
    selected_turn_id = (
        _bounded_capture_identifier(
            current_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if current_turn_id is not None
        else terminal_turn_id
    )
    expected_thread = (
        _bounded_capture_identifier(
            expected_thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if expected_thread_id is not None
        else None
    )
    identities_complete = (
        identity_complete
        and terminal_turn_shapes_complete
        and terminal_thread_shapes_complete
    )
    if expected_thread_id is not None and expected_thread is None:
        identities_complete = False
    terminal_expected_thread = (
        expected_thread
        if expected_thread is not None
        else (updates[0].thread_id if updates else None)
    )
    if terminal_expected_thread is not None and any(
        thread_id != terminal_expected_thread for thread_id in terminal_thread_ids
    ):
        identities_complete = False
    if selected_turn_id is not None and any(
        selected_turn_id != terminal_id for terminal_id in terminal_turn_ids
    ):
        identities_complete = False
    if (
        selected_turn_id is None
        and current_turn_id is None
        and updates
        and allow_last_update_fallback
    ):
        selected_turn_id = updates[-1].turn_id
    if not observed or selected_turn_id is None or not updates:
        return [], observed, False

    current_indexes = [
        index
        for index, update in enumerate(updates)
        if update.turn_id == selected_turn_id
    ]
    if not current_indexes:
        return [], observed, False
    first_current_index = current_indexes[0]
    prefix = updates[:first_current_index]
    current = updates[first_current_index:]
    complete = (
        identities_complete
        and (
            shapes_complete
            if require_cache_write
            else required_shapes_complete
        )
        and not any(update.capture_sequence_incomplete for update in updates)
    )
    if any(update.turn_id != selected_turn_id for update in current):
        complete = False
    if len({update.thread_id for update in updates}) != 1:
        complete = False
    if expected_thread is not None and any(
        update.thread_id != expected_thread for update in updates
    ):
        complete = False

    if prefix:
        # Resume emits one restored notification for the prior turn. It is a
        # cumulative baseline only and must never be billed as current work.
        if len(prefix) != 1 or prefix[0].turn_id == selected_turn_id:
            complete = False
        previous_total = prefix[-1].total
        counter_values = (
            _counter_values
            if require_cache_write
            else _required_counter_values
        )
        replay_total = counter_values(prefix[-1].total)
        replay_last = counter_values(prefix[-1].last)
        if (
            replay_total is None
            or replay_last is None
            or any(
                total_value < last_value
                for total_value, last_value in zip(
                    replay_total,
                    replay_last,
                    strict=True,
                )
            )
        ):
            complete = False
    else:
        previous_total = _CodexTokenCounters(0, 0, 0, 0)

    for index, update in enumerate(current):
        if index > 0 and update.total == previous_total:
            complete = False
        delta_matches = (
            _counter_delta_matches
            if require_cache_write
            else _required_counter_delta_matches
        )
        if not delta_matches(
            update.total,
            previous_total,
            update.last,
        ):
            complete = False
        previous_total = update.total
    return current, observed, complete


def _usage_events_complete(raw_events: "list[dict[str, Any]]") -> bool:
    """Whether response boundaries and all billable counters are proven."""

    _current, _observed, complete = _current_usage_updates(raw_events)
    return complete


@dataclass(frozen=True)
class _CodexUsageEvidence:
    """One SDK usage projection and the claims its counters authorize."""

    usage: Usage
    token_usage_complete: bool
    cost_complete: bool
    cost_upper_bound: bool = False
    pricing_evidence: UsagePricingEvidence = field(
        default_factory=lambda: UsagePricingEvidence(
            segments=(),
            complete=False,
        )
    )


def _as_incomplete_usage_lower_bound(
    evidence: _CodexUsageEvidence,
) -> _CodexUsageEvidence:
    """Retain observed usage without authorizing a complete-cost claim."""

    return _replace(
        evidence,
        token_usage_complete=False,
        cost_complete=False,
        # A locally terminated turn may accrue later provider usage, so even a
        # response-exact captured prefix is a lower bound for the full turn.
        cost_upper_bound=False,
        pricing_evidence=_replace(
            evidence.pricing_evidence,
            complete=False,
        ),
    )


def _codex_usage_cost_upper_bound(
    *,
    pricing: Any | None = None,
    model: str | None = None,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
) -> tuple[float, bool]:
    """Return a safe cost bound and whether it is also the exact cost.

    When a qualified Codex event lacks trustworthy cache-write telemetry, use
    its total input, cache-read input, and output counters to compute the
    maximum of the two endpoints allowed by Crucible's pricing contract: no
    observable non-read input was written, or all of it was written. The
    resulting value is safe for budget enforcement, but the exact-cost claim
    remains incomplete whenever the active cache-write rate differs from the
    ordinary input rate.
    """

    if pricing is None:
        if model is None:
            raise TypeError("pricing or model is required")
        from crucible.backends.pricing import (  # noqa: PLC0415
            pricing_for_model,
        )

        pricing = pricing_for_model(model)
    possible_cache_write_tokens = max(0, input_tokens - cache_read_tokens)
    cost_without_writes = pricing.compute_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    cost_if_all_non_read_input_was_written = pricing.compute_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_tokens=possible_cache_write_tokens,
        cache_read_tokens=cache_read_tokens,
    )

    use_long_context = (
        pricing.long_context_threshold_tokens is not None
        and input_tokens > pricing.long_context_threshold_tokens
    )
    input_rate = (
        pricing.long_context_input_per_mtok
        if use_long_context
        else pricing.input_per_mtok
    )
    cache_write_rate = (
        pricing.long_context_cache_write_per_mtok
        if use_long_context
        else pricing.cache_write_per_mtok
    )
    cost_complete = (
        possible_cache_write_tokens == 0
        or cache_write_rate is None
        or cache_write_rate == input_rate
    )
    return (
        max(cost_without_writes, cost_if_all_non_read_input_was_written),
        cost_complete,
    )


def _exact_cache_write_tokens(
    counters: _CodexTokenCounters,
) -> int | None:
    """Return writes only when telemetry or arithmetic proves the count."""

    candidate = counters.cache_write_tokens
    if counters.input_tokens == counters.cache_read_tokens:
        return 0
    # Codex 0.146 serde-defaults missing upstream write telemetry to zero.
    # Therefore zero is not presence proof when non-cache-read input exists.
    if candidate is None or candidate <= 0:
        return None
    return candidate


def _physical_usage_response(
    counters: _CodexTokenCounters,
    *,
    ordinal: int,
    boundary: Literal["raw_response_completed", "cumulative_last"],
) -> PhysicalUsageResponse:
    """Project one admitted raw response counter's cache-field state."""

    if boundary not in {"raw_response_completed", "cumulative_last"}:
        raise ValueError("unknown Codex physical-usage boundary")

    reported = counters.cache_write_tokens
    field_state: CacheWriteFieldState
    interpretation: CacheWriteInterpretation
    if counters.cache_write_malformed:
        field_state = "malformed"
        interpretation = "unavailable"
        reported = None
    elif reported is None and boundary == "raw_response_completed":
        field_state = "absent"
        interpretation = "raw_response_protocol_default_zero"
    elif reported is None:
        field_state = "absent"
        interpretation = (
            "arithmetic_all_read_zero"
            if counters.input_tokens == counters.cache_read_tokens
            else "unavailable"
        )
    elif reported > 0:
        field_state = "present"
        interpretation = "explicit_positive"
    elif boundary == "raw_response_completed":
        field_state = "present"
        interpretation = "raw_response_explicit_zero"
    elif counters.input_tokens == counters.cache_read_tokens:
        field_state = "present"
        interpretation = "explicit_zero_all_read"
    else:
        field_state = "present"
        interpretation = "explicit_zero_untrusted"
    return PhysicalUsageResponse(
        response_ordinal=ordinal,
        input_tokens=counters.input_tokens,
        output_tokens=counters.output_tokens,
        cache_read_tokens=counters.cache_read_tokens,
        reported_cache_write_tokens=reported,
        cache_write_field_state=field_state,
        cache_write_interpretation=interpretation,
    )


def _aggregate_pricing_upper_bound(
    *,
    pricing: Any,
    counters: _CodexTokenCounters,
) -> float:
    """Price unpartitioned input without trusting cache-category labels."""

    input_rates = [
        pricing.input_per_mtok,
        (
            pricing.cache_write_per_mtok
            if pricing.cache_write_per_mtok is not None
            else pricing.input_per_mtok
        ),
        (
            pricing.cache_read_per_mtok
            if pricing.cache_read_per_mtok is not None
            else pricing.input_per_mtok
        ),
    ]
    output_rates = [pricing.output_per_mtok]
    use_long_context_rates = (
        pricing.long_context_threshold_tokens is not None
        and counters.input_tokens > pricing.long_context_threshold_tokens
    )
    if use_long_context_rates:
        assert pricing.long_context_input_per_mtok is not None
        input_rates.extend(
            [
                pricing.long_context_input_per_mtok,
                (
                    pricing.long_context_cache_write_per_mtok
                    if pricing.long_context_cache_write_per_mtok is not None
                    else pricing.long_context_input_per_mtok
                ),
                (
                    pricing.long_context_cache_read_per_mtok
                    if pricing.long_context_cache_read_per_mtok is not None
                    else pricing.long_context_input_per_mtok
                ),
            ]
        )
        output_rates.append(
            pricing.long_context_output_per_mtok
            if pricing.long_context_output_per_mtok is not None
            else pricing.output_per_mtok
        )
    return (
        counters.input_tokens * max(input_rates)
        + counters.output_tokens * max(output_rates)
    ) / 1_000_000.0


def _safe_unpartitioned_usage(
    raw_events: "list[dict[str, Any]]",
    *,
    pricing: Any,
    current_turn_id: str | None,
    allow_last_update_fallback: bool,
    expected_thread_id: str | None,
    identity_complete: bool,
) -> _CodexUsageEvidence:
    """Return a safe aggregate bound when response boundaries are unproven."""

    (
        updates,
        _observed,
        shapes_complete,
        required_shapes_complete,
    ) = _usage_updates(raw_events)
    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    terminal_turn_id = terminal_turn_ids[0] if terminal_turn_ids else None
    selected_turn_id = (
        _bounded_capture_identifier(
            current_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if current_turn_id is not None
        else terminal_turn_id
    )
    expected_thread = (
        _bounded_capture_identifier(
            expected_thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if expected_thread_id is not None
        else None
    )
    identities_complete = (
        identity_complete
        and terminal_turn_shapes_complete
        and terminal_thread_shapes_complete
    )
    if expected_thread_id is not None and expected_thread is None:
        identities_complete = False
    terminal_expected_thread = (
        expected_thread
        if expected_thread is not None
        else (updates[0].thread_id if updates else None)
    )
    if terminal_expected_thread is not None and any(
        thread_id != terminal_expected_thread for thread_id in terminal_thread_ids
    ):
        identities_complete = False
    if selected_turn_id is not None and any(
        selected_turn_id != terminal_id for terminal_id in terminal_turn_ids
    ):
        identities_complete = False
    if expected_thread is not None and any(
        update.thread_id != expected_thread for update in updates
    ):
        identities_complete = False
    if (
        selected_turn_id is None
        and current_turn_id is None
        and updates
        and allow_last_update_fallback
    ):
        selected_turn_id = updates[-1].turn_id
    current = [update for update in updates if update.turn_id == selected_turn_id]
    if not current:
        return _CodexUsageEvidence(Usage(), False, False, False)

    totals_contain_last = all(
        update.total.input_tokens >= update.last.input_tokens
        and update.total.cache_read_tokens >= update.last.cache_read_tokens
        and update.total.output_tokens >= update.last.output_tokens
        for update in updates
    )
    overflowed = any(update.capture_sequence_incomplete for update in updates)
    same_thread = len({update.thread_id for update in updates}) == 1
    totals_monotonic = same_thread
    previous_total: tuple[int, int, int] | None = None
    for update in updates:
        current_total = (
            update.total.input_tokens,
            update.total.cache_read_tokens,
            update.total.output_tokens,
        )
        if previous_total is not None and any(
            current_value < previous_value
            for current_value, previous_value in zip(
                current_total,
                previous_total,
                strict=True,
            )
        ):
            totals_monotonic = False
            break
        previous_total = current_total
    # A monotonic cumulative thread total already contains every preceding
    # response. Charge its final snapshot once instead of summing each
    # cumulative notification and repeatedly rebilling the same history. This
    # remains an upper bound when response boundaries are duplicated or one
    # bounded-capture gap exists. Fall back to the deliberately loose sum when
    # thread identity or monotonicity is not proven. If a malformed total does
    # not contain its advertised last, include the last counters as well.
    include_last = not totals_contain_last
    # Optional cache-write corruption cannot make repeated cumulative totals
    # independent billable responses.  Compact whenever every required row is
    # shaped and the same-thread required counters are monotonic; otherwise
    # retain the deliberately loose sum for malformed required evidence.
    compact_cumulative = required_shapes_complete and totals_monotonic
    billed_updates = updates[-1:] if compact_cumulative else updates
    aggregate = _CodexTokenCounters(
        input_tokens=sum(
            update.total.input_tokens
            + (update.last.input_tokens if include_last else 0)
            for update in billed_updates
        ),
        cache_read_tokens=sum(
            update.total.cache_read_tokens
            + (update.last.cache_read_tokens if include_last else 0)
            for update in billed_updates
        ),
        cache_write_tokens=None,
        output_tokens=sum(
            update.total.output_tokens
            + (update.last.output_tokens if include_last else 0)
            for update in billed_updates
        ),
    )
    if aggregate.cache_read_tokens > aggregate.input_tokens:
        return _CodexUsageEvidence(Usage(), False, False, False)

    return _CodexUsageEvidence(
        usage=Usage(
            input_tokens=aggregate.input_tokens,
            output_tokens=aggregate.output_tokens,
            cost_usd=_aggregate_pricing_upper_bound(
                pricing=pricing,
                counters=aggregate,
            ),
            cache_read_tokens=aggregate.cache_read_tokens,
        ),
        token_usage_complete=False,
        cost_complete=False,
        cost_upper_bound=(
            identities_complete
            and required_shapes_complete
            and totals_contain_last
            and (not overflowed or totals_monotonic)
        ),
    )


def _incomplete_resumed_usage_from_counters(
    counters: list[_CodexTokenCounters],
    *,
    pricing: Any,
    source: str,
    physical_counters: list[_CodexTokenCounters] | None = None,
) -> _CodexUsageEvidence:
    """Price a proven resumed-turn prefix without claiming finality."""

    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    cost_usd = 0.0
    segments: list[UsagePricingSegment] = []
    for counter in counters:
        input_tokens += counter.input_tokens
        output_tokens += counter.output_tokens
        cache_read_tokens += counter.cache_read_tokens
        cache_write_tokens = _exact_cache_write_tokens(counter)
        if cache_write_tokens is None:
            response_cost, _response_cost_complete = _codex_usage_cost_upper_bound(
                pricing=pricing,
                input_tokens=counter.input_tokens,
                output_tokens=counter.output_tokens,
                cache_read_tokens=counter.cache_read_tokens,
            )
        else:
            response_cost = pricing.compute_cost(
                input_tokens=counter.input_tokens,
                output_tokens=counter.output_tokens,
                cache_creation_tokens=cache_write_tokens,
                cache_read_tokens=counter.cache_read_tokens,
            )
            cache_creation_tokens += cache_write_tokens
            segments.append(
                UsagePricingSegment(
                    input_tokens=counter.input_tokens,
                    output_tokens=counter.output_tokens,
                    cache_creation_tokens=cache_write_tokens,
                    cache_read_tokens=counter.cache_read_tokens,
                    cost_usd=response_cost,
                )
            )
        cost_usd += response_cost

    emit_physical_usage_attempt(
        backend="codex",
        source=source,
        complete=False,
        responses=tuple(
            _physical_usage_response(
                counter,
                ordinal=ordinal,
                boundary="cumulative_last",
            )
            for ordinal, counter in enumerate(
                counters if physical_counters is None else physical_counters
            )
        ),
    )
    return _CodexUsageEvidence(
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
        token_usage_complete=False,
        cost_complete=False,
        cost_upper_bound=False,
        pricing_evidence=UsagePricingEvidence(
            segments=tuple(segments),
            complete=False,
        ),
    )


def _current_usage_lower_bound_without_history(
    raw_events: list[dict[str, Any]],
    *,
    pricing: Any,
    current_turn_id: str | None,
    expected_thread_id: str | None,
    identity_complete: bool,
) -> _CodexUsageEvidence | None:
    """Exclude resumed history from an incomplete cumulative sequence.

    Prefer the componentwise difference between the final current cumulative
    total and exactly one qualified replay snapshot. When that proof is not
    available, admit only duplicate-free ``last`` rows uniquely bound to the
    selected turn. Never use an unpartitioned cumulative historical total.
    """

    selected_turn_id = (
        _bounded_capture_identifier(
            current_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if current_turn_id is not None
        else None
    )
    expected_thread = _bounded_capture_identifier(
        expected_thread_id,
        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
    )
    updates, observed, shapes_complete, _required_shapes_complete = (
        _usage_updates(raw_events)
    )
    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    if selected_turn_id is None and current_turn_id is None:
        unique_terminal_turn_ids = set(terminal_turn_ids)
        if len(unique_terminal_turn_ids) == 1:
            selected_turn_id = next(iter(unique_terminal_turn_ids))
    if (
        not identity_complete
        or selected_turn_id is None
        or expected_thread is None
        or not observed
        or not shapes_complete
        or not terminal_turn_shapes_complete
        or not terminal_thread_shapes_complete
        or not updates
        or any(update.thread_id != expected_thread for update in updates)
        or any(turn_id != selected_turn_id for turn_id in terminal_turn_ids)
        or any(thread_id != expected_thread for thread_id in terminal_thread_ids)
    ):
        return None

    current_indexes = [
        index
        for index, update in enumerate(updates)
        if update.turn_id == selected_turn_id
    ]
    if not current_indexes:
        return None
    first_current_index = current_indexes[0]
    prefix = updates[:first_current_index]
    current = updates[first_current_index:]
    if any(update.turn_id != selected_turn_id for update in current):
        return None

    # The app-server resume contract emits exactly one restored prior-turn
    # cumulative snapshot. Its ``last`` is historical context; only the
    # componentwise growth after its ``total`` belongs to this attempt.
    if (
        len(prefix) == 1
        and prefix[0].turn_id != selected_turn_id
        and not prefix[0].capture_sequence_incomplete
    ):
        baseline_values = _counter_values(prefix[0].total)
        baseline_last_values = _counter_values(prefix[0].last)
        total_values = [_counter_values(update.total) for update in current]
        if (
            baseline_values is not None
            and baseline_last_values is not None
            and all(values is not None for values in total_values)
            and all(
                total >= last
                for total, last in zip(
                    baseline_values,
                    baseline_last_values,
                    strict=True,
                )
            )
        ):
            previous_values = baseline_values
            totals_monotonic = True
            for values in total_values:
                assert values is not None
                if any(
                    current_value < previous_value
                    for current_value, previous_value in zip(
                        values,
                        previous_values,
                        strict=True,
                    )
                ):
                    totals_monotonic = False
                    break
                previous_values = values
            if totals_monotonic:
                final_values = previous_values
                delta_values = tuple(
                    final_value - baseline_value
                    for final_value, baseline_value in zip(
                        final_values,
                        baseline_values,
                        strict=True,
                    )
                )
                delta = _CodexTokenCounters(
                    input_tokens=delta_values[0],
                    cache_read_tokens=delta_values[1],
                    cache_write_tokens=delta_values[2],
                    output_tokens=delta_values[3],
                )
                current_last_values = [
                    _counter_values(update.last) for update in current
                ]
                # An incomplete capture may omit response rows, but every
                # retained selected-turn row still has to fit inside the
                # cumulative current-turn delta.
                if (
                    delta.cache_read_tokens + (delta.cache_write_tokens or 0)
                    <= delta.input_tokens
                    and all(values is not None for values in current_last_values)
                    and all(
                        all(
                            delta_value >= last_value
                            for delta_value, last_value in zip(
                                delta_values,
                                values,
                                strict=True,
                            )
                        )
                        for values in current_last_values
                        if values is not None
                    )
                ):
                    return _incomplete_resumed_usage_from_counters(
                        [delta],
                        pricing=pricing,
                        source=("codex_app_server.thread_token_usage_updated.last.v1"),
                        physical_counters=[update.last for update in current],
                    )

    # A response row is usable without a replay baseline only when the
    # selected-turn suffix is intact and each cumulative snapshot advances
    # uniquely. This admits a known current prefix while rejecting duplicate
    # notifications that could otherwise double bill one response.
    if any(update.capture_sequence_incomplete for update in current):
        return None
    last_counters: list[_CodexTokenCounters] = []
    previous_total: _CodexTokenCounters | None = None
    seen_totals: set[tuple[int, int, int, int]] = set()
    for update in current:
        total_values = _counter_values(update.total)
        last_values = _counter_values(update.last)
        if total_values is None or last_values is None:
            return None
        if total_values in seen_totals or any(
            total_value < last_value
            for total_value, last_value in zip(
                total_values,
                last_values,
                strict=True,
            )
        ):
            return None
        if previous_total is not None and not _counter_delta_matches(
            update.total,
            previous_total,
            update.last,
        ):
            return None
        seen_totals.add(total_values)
        previous_total = update.total
        last_counters.append(update.last)
    return _incomplete_resumed_usage_from_counters(
        last_counters,
        pricing=pricing,
        source="codex_app_server.thread_token_usage_updated.last.v1",
    )


def _ordered_noncumulative_raw_response_rows(
    raw_events: list[dict[str, Any]],
    *,
    selected_turn_id: str,
    selected_thread_id: str,
    baseline: _CodexTokenCounters,
    final_total: _CodexTokenCounters,
) -> int | None:
    """Prove exact raw rows that Codex intentionally omits from thread totals.

    Codex 0.146 remote-v2 compaction emits one exact raw-response usage row,
    records it against the rollout budget, then emits a token-usage update
    whose billable ``total`` is unchanged and whose billable ``last`` is zero.
    Ordinary responses instead advance ``total`` by exactly the raw row.

    This recognizer is deliberately stricter than the aggregate fast path. It
    requires transport order, a one-to-one raw/update pairing for every
    advancing cumulative update, and a one-to-one raw/zero-update pairing for
    every non-cumulative row.  Missing, reordered, duplicated, cross-identity,
    reset, or capture-incomplete evidence cannot qualify.
    """

    previous_total = baseline
    pending_raw: _CodexTokenCounters | None = None
    advancing_sum = [0, 0, 0]
    noncumulative_rows = 0
    selected_usage_observed = False
    prefix_usage_rows = 0

    for event in raw_events:
        method = event.get("method")
        if method == _RAW_RESPONSE_COMPLETED_METHOD:
            if event.get(_CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True:
                return None
            params = event.get("params")
            if not isinstance(params, dict):
                return None
            thread_id = _bounded_capture_identifier(
                dict.get(params, "threadId"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
            turn_id = _bounded_capture_identifier(
                dict.get(params, "turnId"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
            if thread_id != selected_thread_id or turn_id != selected_turn_id:
                return None
            counter = _bounded_token_counters(dict.get(params, "usage"))
            if (
                counter is None
                or counter.cache_write_malformed
                or pending_raw is not None
            ):
                return None
            pending_raw = counter
            continue

        if method != _TOKEN_USAGE_METHOD:
            continue
        if event.get(_CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True:
            return None
        params = event.get("params")
        if not isinstance(params, dict):
            return None
        thread_id = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        turn_id = _bounded_capture_identifier(
            dict.get(params, "turnId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        token_usage = dict.get(params, "tokenUsage")
        if thread_id != selected_thread_id or not isinstance(token_usage, dict):
            return None
        total = _bounded_token_counters(dict.get(token_usage, "total"))
        last = _bounded_token_counters(dict.get(token_usage, "last"))
        if (
            total is None
            or last is None
            or total.cache_write_malformed
            or last.cache_write_malformed
        ):
            return None

        if turn_id != selected_turn_id:
            # Resume may replay exactly one prior-turn cumulative baseline,
            # but never after current-turn raw or usage evidence has begun.
            prefix_usage_rows += 1
            if (
                prefix_usage_rows != 1
                or selected_usage_observed
                or pending_raw is not None
                or total != baseline
            ):
                return None
            continue

        selected_usage_observed = True
        total_values = _required_counter_values(total)
        previous_values = _required_counter_values(previous_total)
        last_values = _required_counter_values(last)
        delta = tuple(
            current - previous
            for current, previous in zip(
                total_values,
                previous_values,
                strict=True,
            )
        )
        if any(value < 0 for value in delta):
            return None

        if delta == (0, 0, 0) and last_values == (0, 0, 0):
            # Optional cache-write telemetry is tolerated when absent, but a
            # present positive ``last`` is billable and therefore cannot be a
            # zero boundary. Likewise, two comparable cumulative write
            # counters must remain unchanged even though the required
            # input/read/output projection is unchanged.
            if (
                last.cache_write_tokens not in (None, 0)
                or (
                    previous_total.cache_write_tokens is not None
                    and total.cache_write_tokens is not None
                    and total.cache_write_tokens
                    != previous_total.cache_write_tokens
                )
            ):
                return None
            if pending_raw is not None:
                # The exact response is billable, but Codex deliberately did
                # not append it to the thread's billable cumulative counters.
                noncumulative_rows += 1
                pending_raw = None
        else:
            if (
                delta != last_values
                or pending_raw is None
                or _required_counter_values(pending_raw) != delta
            ):
                return None
            for index, value in enumerate(delta):
                advancing_sum[index] += value
            pending_raw = None
        previous_total = total

    final_values = _required_counter_values(final_total)
    baseline_values = _required_counter_values(baseline)
    expected_advancing = tuple(
        final - initial
        for final, initial in zip(
            final_values,
            baseline_values,
            strict=True,
        )
    )
    if (
        not selected_usage_observed
        or pending_raw is not None
        or noncumulative_rows == 0
        or _required_counter_values(previous_total) != final_values
        or tuple(advancing_sum) != expected_advancing
    ):
        return None
    return noncumulative_rows


def _exact_raw_response_usage_evidence(
    raw_events: "list[dict[str, Any]]",
    *,
    pricing: Any,
    current_turn_id: str | None,
    allow_last_update_fallback: bool,
    expected_thread_id: str | None,
    identity_complete: bool,
    force_incomplete: bool = False,
) -> _CodexUsageEvidence | None:
    """Return reconciled exact per-response usage when Codex exposes it.

    Codex 0.146's generated protocol describes ``rawResponse/completed`` as
    carrying exact usage for one upstream Responses API completion. Its
    optional cache-write field has protocol default zero, which is authoritative
    at this exact-response boundary but not in the older cumulative projection.
    Exact response rows are accepted only when their identities are unique and
    their known totals reconcile with the cumulative current-turn counters.

    Remote-v2 compaction in Codex 0.146 is a deliberate exception to the
    aggregate equality: its exact upstream response is emitted through
    ``rawResponse/completed`` and charged to Codex's rollout budget, while the
    thread's cumulative billable counters are left unchanged before a
    zero-billable recomputation notification.  The narrow ordered fallback
    below recognizes only that protocol shape.  It does not generally relax
    raw-versus-cumulative reconciliation.
    """

    raw_rows = [
        event
        for event in raw_events
        if event.get("method") == _RAW_RESPONSE_COMPLETED_METHOD
    ]
    if not raw_rows:
        return None

    terminal_turn_ids, terminal_turn_shapes_complete = _terminal_turn_identity_evidence(
        raw_events
    )
    terminal_thread_ids, terminal_thread_shapes_complete = (
        _terminal_thread_identity_evidence(raw_events)
    )
    terminal_turn_id = terminal_turn_ids[0] if terminal_turn_ids else None
    selected_turn_id = (
        _bounded_capture_identifier(
            current_turn_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if current_turn_id is not None
        else terminal_turn_id
    )
    if (
        selected_turn_id is None
        and current_turn_id is None
        and allow_last_update_fallback
    ):
        params = raw_rows[-1].get("params")
        if isinstance(params, dict):
            selected_turn_id = _bounded_capture_identifier(
                dict.get(params, "turnId"),
                max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
            )
    expected_thread = (
        _bounded_capture_identifier(
            expected_thread_id,
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if expected_thread_id is not None
        else None
    )
    if (
        not identity_complete
        or not terminal_turn_shapes_complete
        or not terminal_thread_shapes_complete
        or selected_turn_id is None
        or (expected_thread_id is not None and expected_thread is None)
    ):
        return None

    response_ids: set[str] = set()
    counters: list[_CodexTokenCounters] = []
    selected_threads: set[str] = set()
    for event in raw_rows:
        params = event.get("params")
        if not isinstance(params, dict):
            return None
        turn_id = _bounded_capture_identifier(
            dict.get(params, "turnId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        if turn_id != selected_turn_id:
            # Raw-response notifications are neither persisted nor replayed,
            # so a different turn cannot be a legitimate resume baseline.
            return None
        thread_id = _bounded_capture_identifier(
            dict.get(params, "threadId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        response_id = _bounded_capture_identifier(
            dict.get(params, "responseId"),
            max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
        )
        usage = _bounded_token_counters(dict.get(params, "usage"))
        if (
            thread_id is None
            or response_id is None
            or response_id in response_ids
            or usage is None
            or usage.cache_write_malformed
            or event.get(_CAPTURE_SEQUENCE_INCOMPLETE_FIELD) is True
        ):
            return None
        if expected_thread is not None and thread_id != expected_thread:
            return None
        selected_threads.add(thread_id)
        response_ids.add(response_id)
        counters.append(usage)
    if not counters or len(selected_threads) != 1:
        return None
    selected_thread_id = next(iter(selected_threads))
    if terminal_turn_ids and any(
        turn_id != selected_turn_id for turn_id in terminal_turn_ids
    ):
        return None
    if any(thread_id != selected_thread_id for thread_id in terminal_thread_ids):
        return None

    updates, observed_updates, shapes_complete, _required_shapes_complete = (
        _usage_updates(raw_events)
    )
    if (
        not observed_updates
        or not shapes_complete
        or any(update.capture_sequence_incomplete for update in updates)
    ):
        return None
    if any(update.thread_id != selected_thread_id for update in updates):
        return None
    current_indexes = [
        index
        for index, update in enumerate(updates)
        if update.turn_id == selected_turn_id
    ]
    if not current_indexes:
        return None
    first_index = current_indexes[0]
    if any(update.turn_id != selected_turn_id for update in updates[first_index:]):
        return None
    baseline = (
        updates[first_index - 1].total
        if first_index > 0
        else _CodexTokenCounters(0, 0, 0, 0)
    )
    final_total = updates[current_indexes[-1]].total
    expected_known = (
        final_total.input_tokens - baseline.input_tokens,
        final_total.cache_read_tokens - baseline.cache_read_tokens,
        final_total.output_tokens - baseline.output_tokens,
    )
    observed_known = (
        sum(counter.input_tokens for counter in counters),
        sum(counter.cache_read_tokens for counter in counters),
        sum(counter.output_tokens for counter in counters),
    )
    if any(value < 0 for value in expected_known):
        return None
    if observed_known != expected_known:
        noncumulative_raw_rows = _ordered_noncumulative_raw_response_rows(
            raw_events,
            selected_turn_id=selected_turn_id,
            selected_thread_id=selected_thread_id,
            baseline=baseline,
            final_total=final_total,
        )
        if noncumulative_raw_rows is None:
            return None
        logger.info(
            "CodexBackend: reconciled %d exact non-cumulative raw-response "
            "boundary/boundaries against zero-billable unchanged-total "
            "thread usage updates",
            noncumulative_raw_rows,
        )

    segments: list[UsagePricingSegment] = []
    cache_creation_tokens = 0
    cost_usd = 0.0
    for counter in counters:
        cache_write_tokens = (
            0 if counter.cache_write_tokens is None else counter.cache_write_tokens
        )
        response_cost = pricing.compute_cost(
            input_tokens=counter.input_tokens,
            output_tokens=counter.output_tokens,
            cache_creation_tokens=cache_write_tokens,
            cache_read_tokens=counter.cache_read_tokens,
        )
        cache_creation_tokens += cache_write_tokens
        cost_usd += response_cost
        segments.append(
            UsagePricingSegment(
                input_tokens=counter.input_tokens,
                output_tokens=counter.output_tokens,
                cache_creation_tokens=cache_write_tokens,
                cache_read_tokens=counter.cache_read_tokens,
                cost_usd=response_cost,
            )
        )
    usage = Usage(
        input_tokens=observed_known[0],
        output_tokens=observed_known[2],
        cost_usd=cost_usd,
        cache_creation_tokens=cache_creation_tokens,
        cache_read_tokens=observed_known[1],
    )
    emit_physical_usage_attempt(
        backend="codex",
        source="codex_app_server.raw_response_completed.usage.v1",
        # A forbidden-delegation boundary can stop local capture before the
        # provider turn itself is terminal. Its retained response rows remain
        # useful, but cannot qualify the physical attempt as complete.
        complete=not force_incomplete,
        responses=tuple(
            _physical_usage_response(
                counter,
                ordinal=ordinal,
                boundary="raw_response_completed",
            )
            for ordinal, counter in enumerate(counters)
        ),
    )
    evidence = _CodexUsageEvidence(
        usage=usage,
        token_usage_complete=True,
        cost_complete=True,
        cost_upper_bound=False,
        pricing_evidence=UsagePricingEvidence(
            segments=tuple(segments),
            complete=True,
        ),
    )
    return _as_incomplete_usage_lower_bound(evidence) if force_incomplete else evidence


def _harvest_usage_evidence(
    raw_events: "list[dict[str, Any]]",
    *,
    model: str,
    current_turn_id: str | None = None,
    allow_last_update_fallback: bool = True,
    expected_thread_id: str | None = None,
    identity_complete: bool = True,
    force_incomplete: bool = False,
    require_history_free_lower_bound: bool = False,
) -> _CodexUsageEvidence:
    """Price current-turn snapshots without rebilling thread total.

    ``force_incomplete`` retains a captured prefix as a numeric lower bound
    when local policy terminated the provider turn before complete accounting.
    ``require_history_free_lower_bound`` forbids the generic cumulative
    fallback when a resumed thread's totals include restored history or local
    delegation termination makes the retained rows only a prefix. This
    applies to every resumed completion, not only delegation-terminated ones.
    """

    incomplete = _CodexUsageEvidence(
        usage=Usage(),
        token_usage_complete=False,
        cost_complete=False,
    )
    # Resolve the complete immutable rate card once. Every physical response
    # in this harvest must be priced against the same reviewed snapshot.
    from crucible.backends.pricing import pricing_for_model  # noqa: PLC0415

    pricing = pricing_for_model(model)
    exact_raw = _exact_raw_response_usage_evidence(
        raw_events,
        pricing=pricing,
        current_turn_id=current_turn_id,
        allow_last_update_fallback=allow_last_update_fallback,
        expected_thread_id=expected_thread_id,
        identity_complete=identity_complete,
        force_incomplete=force_incomplete,
    )
    if exact_raw is not None:
        return exact_raw
    if any(
        event.get("method") == _RAW_RESPONSE_COMPLETED_METHOD for event in raw_events
    ):
        logger.warning(
            "CodexBackend: exact %s evidence did not reconcile with the "
            "bounded cumulative counters; using cumulative accounting",
            _RAW_RESPONSE_COMPLETED_METHOD,
        )
    else:
        logger.warning(
            "CodexBackend: app-server emitted no %s evidence; using "
            "cumulative accounting",
            _RAW_RESPONSE_COMPLETED_METHOD,
        )
    current_updates, observed, boundaries_complete = _current_usage_updates(
        raw_events,
        current_turn_id=current_turn_id,
        allow_last_update_fallback=allow_last_update_fallback,
        expected_thread_id=expected_thread_id,
        identity_complete=identity_complete,
    )
    required_only_boundaries = False
    required_observed = False
    required_boundaries_complete = False
    if not boundaries_complete:
        (
            required_updates,
            required_observed,
            required_boundaries_complete,
        ) = _current_usage_updates(
            raw_events,
            current_turn_id=current_turn_id,
            allow_last_update_fallback=allow_last_update_fallback,
            expected_thread_id=expected_thread_id,
            identity_complete=identity_complete,
            require_cache_write=False,
        )
        if required_observed and required_boundaries_complete:
            current_updates = required_updates
            required_only_boundaries = True
    if (
        require_history_free_lower_bound
        and not boundaries_complete
        and not required_only_boundaries
    ):
        resumed_lower_bound = _current_usage_lower_bound_without_history(
            raw_events,
            pricing=pricing,
            current_turn_id=current_turn_id,
            expected_thread_id=expected_thread_id,
            identity_complete=identity_complete,
        )
        if resumed_lower_bound is not None:
            return resumed_lower_bound
        emit_physical_usage_attempt(
            backend="codex",
            source=("codex_app_server.thread_token_usage_updated.last.v1"),
            complete=False,
            responses=(),
        )
        logger.warning(
            "CodexBackend: retained usage did not prove a "
            "history-free current-turn lower bound; reporting zero usage"
        )
        return incomplete
    physical_responses = tuple(
        _physical_usage_response(
            update.last,
            ordinal=ordinal,
            boundary="cumulative_last",
        )
        for ordinal, update in enumerate(current_updates)
    )
    emit_physical_usage_attempt(
        backend="codex",
        source="codex_app_server.thread_token_usage_updated.last.v1",
        # Cumulative notifications are retained to diagnose a missing or
        # non-reconciling raw-response boundary, but cannot authorize a live
        # canary or cohort result.
        complete=False,
        responses=physical_responses,
    )
    if not observed:
        logger.warning(
            "CodexBackend: no %s event in raw_events; cost reported "
            "as $0 for this invocation. Check whether the Codex SDK renamed "
            "the token-usage notification method or schema.",
            _TOKEN_USAGE_METHOD,
        )
        return incomplete
    if not identity_complete:
        # Physical telemetry above remains append-only evidence that the
        # provider did work, but an incomplete protocol identity cannot
        # authorize those counters on this logical invocation. In particular,
        # do not let the conservative aggregate fallback turn an identity
        # qualification failure back into attributed usage.
        logger.warning(
            "CodexBackend: usage identity was incomplete; reporting zero "
            "attributed usage"
        )
        return incomplete
    if not boundaries_complete:
        if required_only_boundaries:
            logger.warning(
                "CodexBackend: cache-write response-boundary evidence in %s "
                "is incomplete; using required-counter response bounds",
                _TOKEN_USAGE_METHOD,
            )
        else:
            logger.warning(
                "CodexBackend: incomplete response-boundary evidence in %s; "
                "using a conservative aggregate cost bound",
                _TOKEN_USAGE_METHOD,
            )
            evidence = _safe_unpartitioned_usage(
                raw_events,
                pricing=pricing,
                current_turn_id=current_turn_id,
                allow_last_update_fallback=allow_last_update_fallback,
                expected_thread_id=expected_thread_id,
                identity_complete=identity_complete,
            )
            return (
                _as_incomplete_usage_lower_bound(evidence)
                if force_incomplete
                else evidence
            )

    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    cost_usd = 0.0
    cost_complete = True
    token_usage_complete = True
    pricing_complete = True
    segments: list[UsagePricingSegment] = []
    for update in current_updates:
        response = update.last
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        cache_read_tokens += response.cache_read_tokens
        # Exact reconciliation failed for the optional cache-write dimension.
        # Even a syntactically valid per-response value is untrusted here: it
        # may contradict the cumulative cache-write delta that forced this
        # required-counter-only branch.
        cache_write_tokens = (
            None
            if required_only_boundaries
            else _exact_cache_write_tokens(response)
        )
        if cache_write_tokens is None:
            response_cost, response_cost_complete = _codex_usage_cost_upper_bound(
                pricing=pricing,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_read_tokens=response.cache_read_tokens,
            )
            # Preserve the safe aggregate cost bound, but do not encode an
            # assumed write partition as exact Usage or segment evidence.
            token_usage_complete = False
            pricing_complete = False
        else:
            response_cost = pricing.compute_cost(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cache_creation_tokens=cache_write_tokens,
                cache_read_tokens=response.cache_read_tokens,
            )
            response_cost_complete = True
            cache_creation_tokens += cache_write_tokens
            segments.append(
                UsagePricingSegment(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cache_creation_tokens=cache_write_tokens,
                    cache_read_tokens=response.cache_read_tokens,
                    cost_usd=response_cost,
                )
            )
        cost_usd += response_cost
        cost_complete = cost_complete and response_cost_complete
    evidence = _CodexUsageEvidence(
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        ),
        token_usage_complete=token_usage_complete,
        cost_complete=cost_complete,
        cost_upper_bound=not cost_complete,
        pricing_evidence=UsagePricingEvidence(
            segments=tuple(segments),
            complete=pricing_complete,
        ),
    )
    return _as_incomplete_usage_lower_bound(evidence) if force_incomplete else evidence


def _harvest_usage_from_events(
    raw_events: "list[dict[str, Any]]",
    *,
    model: str,
) -> Usage:
    """Extract token counts from a Codex ``chat_once`` raw_events list
    and compute a budget-safe cost bound via the pricing table.

    The app-server emits a ``thread/tokenUsage/updated`` notification after
    each upstream response completion::

        {"method": "thread/tokenUsage/updated",
         "params": {
            "threadId": "...", "turnId": "...",
            "tokenUsage": {
               "total": {"inputTokens":  N, "cachedInputTokens": C,
                         "cacheWriteInputTokens": W, "outputTokens": M,
                         "reasoningOutputTokens": R, "totalTokens": T},
               "last":  {...same shape...},
            }}}

    ``total`` is cumulative thread history; ``last`` is one upstream response.
    A fresh turn must start with ``total == last``. Resume instead emits one
    prior-turn replay as the baseline. Current-turn cumulative deltas must
    equal each corresponding ``last`` snapshot componentwise before those
    responses are priced independently and summed. This prevents both resumed
    history rebilling and an aggregate token count from selecting the wrong
    272K pricing tier.

    The protocol's ``outputTokens`` already includes its
    ``reasoningOutputTokens`` subset, so adding reasoning again would double
    bill it. ``cachedInputTokens`` and ``cacheWriteInputTokens`` are disjoint
    subsets of ``inputTokens``. Codex 0.146 defaults missing upstream write
    telemetry to zero; a zero is exact only when all input was cache-read.

    Returns ``Usage()`` (all zeros) when no token-usage event is
    found. Completeness remains false and a warning identifies a possible SDK
    schema rename. A malformed, duplicated, discontinuous, or overflowed
    sequence uses a conservative aggregate cost bound instead of claiming an
    exact response partition. A proven monotonic cumulative thread total is
    charged once at its final snapshot; only unqualified/reset sequences fall
    back to summing retained totals, avoiding quadratic rebilling of repeated
    cumulative notifications.

    ``Usage`` has no separate reasoning field, so the reasoning subset remains
    available only in the raw SDK event. ``Usage.output_tokens`` retains the
    protocol's billable ``outputTokens`` value. Positive, bounded, disjoint
    write telemetry (or the mathematically forced all-read zero case) permits
    exact tiered pricing. Other write evidence retains a traced upper bound.
    """
    return _harvest_usage_evidence(raw_events, model=model).usage


class CodexBackend:
    """assay.Backend implementation over codex-app-server-sdk."""

    name = "codex"
    transport = "legacy_app_server"

    @property
    def features(self) -> Any:
        """Return the observed profile for this concrete Codex route."""

        from .features import feature_profile_for_route

        return feature_profile_for_route(self.name, model=self.model)

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        effort: str = _DEFAULT_EFFORT,
        cwd: str | None = None,
        output_schema: dict | None = None,
        log_dir: Path | str | None = None,
        capability: str = "text",
        native_shell: bool | None = None,
        trusted_project_instructions: bool = False,
        credential_source: str = "environment",
        cli_path: str | None = None,
        timeout_seconds: float = 2700.0,
    ) -> None:
        from .capabilities import normalize_capability

        self.capability = normalize_capability(capability)
        if native_shell is True and self.capability != "workspace_write":
            raise ValueError("native_shell requires workspace_write capability")
        self._native_shell = (
            self.capability == "workspace_write"
            if native_shell is None
            else native_shell
        )
        if type(trusted_project_instructions) is not bool:
            raise TypeError("trusted_project_instructions must be a bool")
        if trusted_project_instructions and (
            self.capability != "read_only" or self._native_shell or cwd is None
        ):
            raise ValueError(
                "trusted_project_instructions requires capability='read_only', "
                "native_shell=False, and an explicit cwd"
            )
        if not isinstance(model, str) or not model.strip():
            raise ValueError("CodexBackend model must be a non-empty string")
        # ``model`` is part of the assay Backend protocol (public attr).
        self.model = model
        if (
            type(credential_source) is not str
            or credential_source not in _CODEX_CREDENTIAL_SOURCES
        ):
            raise ValueError(
                "Codex credential_source must be one of "
                f"{sorted(_CODEX_CREDENTIAL_SOURCES)}"
            )
        self._credential_source = credential_source
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError(
                "CodexBackend timeout_seconds must be a finite positive number"
            )
        self._default_policy = _replace(
            _DEFAULT_POLICY,
            timeout_seconds=float(timeout_seconds),
        )
        self._cli_pin: _PinnedCodexCLI | None = None
        self._chatgpt_auth_path: Path | None = None
        if credential_source == CODEX_CHATGPT_CREDENTIAL_SOURCE:
            if model not in _CODEX_CHATGPT_SESSION_MODELS:
                raise ValueError(
                    "credential_source='chatgpt_session' requires exact "
                    "model from reviewed set "
                    f"{sorted(_CODEX_CHATGPT_SESSION_MODELS)!r}"
                )
            if self._native_shell:
                raise ValueError(
                    "credential_source='chatgpt_session' requires native_shell=False"
                )
            selected_cli = cli_path or shutil.which("codex")
            native_cli = _resolve_native_codex_executable(selected_cli)
            self._cli_pin = _PinnedCodexCLI.capture(native_cli)
            self._chatgpt_auth_path = Path.home() / ".codex" / "auth.json"
        elif cli_path is not None:
            raise ValueError(
                "Codex cli_path requires credential_source='chatgpt_session'"
            )
        if not effort:
            raise ValueError("effort must be a non-empty string")
        if effort == "ultra":
            raise ValueError(
                "Codex effort 'ultra' enables automatic task delegation, "
                "which is not qualified by this backend; use 'max' for "
                "maximum non-delegating reasoning"
            )
        self._effort = effort
        self._cwd = cwd or os.getcwd()
        self._trusted_project_instructions = trusted_project_instructions
        self._trusted_repository_root = (
            _trusted_repository_root(Path(self._cwd))
            if trusted_project_instructions
            else None
        )
        self._output_schema = output_schema
        self._logged_unique_items_compat = False
        # See ClaudeBackend for the SDK-debug sink contract — same
        # discipline here: each attempt dumps to
        # ``sdk_<seq>_codex.json`` in log_dir (typically the workflow's
        # run folder).
        self._sdk_sink = SdkDebugSink(
            Path(log_dir) if log_dir is not None else None,
            backend_name="codex",
        )
        # MCP attachment — populated by attach_toolset(). The HTTP
        # server is lazily started on the first invoke that needs it,
        # so a backend instance that's never invoked pays no
        # uvicorn-spawn cost.
        self._mcp_toolset: Any = None
        self._mcp_server: Any = None  # CrucibleHttpMcpServer | None
        self._mcp_context_budget: int | None = None
        # Confined app-server subprocesses must not inherit user/project MCP,
        # plugin, hook, skill, or settings state. The isolated home retains
        # only a private copy of auth.json and survives retry/resume attempts
        # for this backend instance.
        self._isolated_home: tempfile.TemporaryDirectory[str] | None = None
        # ``invoke`` is synchronous, but callers may fan out backend turns
        # from worker threads. Serialize lazy start / reattach / close so two
        # concurrent first turns cannot each bind a server and orphan one.
        self._mcp_lock = threading.RLock()
        self._active_mcp_servers: dict[int, Any] = {}
        self._active_mutation_hosts: dict[int, Any] = {}
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, invocation: Invocation) -> InvocationResult:
        from .pricing import pricing_for_model

        pricing_for_model(self.model)
        return asyncio.run(self._invoke_with_retries(invocation))

    def invoke_with_physical_telemetry(
        self,
        invocation: Invocation,
    ) -> tuple[InvocationResult, PhysicalUsageEvidence]:
        """Invoke once logically and return source-owned response telemetry.

        This qualification-only surface is deliberately separate from the
        provider-neutral :class:`assay.InvocationResult`.  Evidence is emitted
        from reconciled ``rawResponse/completed.usage`` admission, never
        reconstructed from the result's pricing segments. Cumulative
        ``thread/tokenUsage/updated.last`` is retained only as explicitly
        incomplete diagnostic fallback evidence.
        """

        with capture_physical_usage("codex") as collector:
            result = self.invoke(invocation)
        return result, collector.freeze()

    def attach_toolset(
        self, toolset: object, *, context_budget: int | None = None
    ) -> None:
        """Stash the toolset for HTTP-MCP bridging on the next invoke.

        The crucible ``ToolSet`` is hosted by an in-process HTTP MCP
        server (see ``crucible.tools._mcp_http_server``) on the first
        ``invoke()`` after attach. Codex CLI is told to connect to that
        server via ``-c mcp_servers.crucible.url=...`` configuration
        overrides. Tool calls flow:

            Codex CLI subproc -> HTTP -> uvicorn daemon thread in this
                              proc -> lowlevel MCP server ->
                              toolset.dispatch(name, args)

        Lifecycle guarantees: the HTTP server is torn down by
        :py:meth:`close` (graceful → forced → give-up), with an
        ``atexit`` backstop registered on first start so the listening
        socket can't outlive the Python process.

        Idempotent — calling ``attach_toolset`` twice replaces the
        previous toolset; the previous HTTP server (if running) is
        stopped first.

        ``context_budget`` caps cumulative UTF-8 tool-result bytes returned
        through the HTTP bridge and is reset for each invocation.
        """
        from .capabilities import validate_toolset_capability

        if context_budget is not None and (
            isinstance(context_budget, bool)
            or not isinstance(context_budget, int)
            or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        validate_toolset_capability(self.capability, toolset)
        write_names = frozenset(
            str(name) for name in getattr(toolset, "write_names", ())
        )
        if write_names:
            from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
                build_isolated_toolset_spec,
            )

            build_isolated_toolset_spec(
                toolset,
                expected_write_names=write_names,
            )

        # Settle the prior physical server before publishing replacement
        # authority. A failed stop retains the exact server and the old
        # attachment, so later close() can retry it.
        with self._mcp_lock:
            if self._closed:
                raise RuntimeError("CodexBackend is closed")
            previous_server = self._mcp_server
            if previous_server is not None:
                previous_server.stop()
                if self._mcp_server is previous_server:
                    self._mcp_server = None
            self._mcp_toolset = toolset
            self._mcp_context_budget = context_budget if toolset is not None else None

        if toolset is not None:
            tool_count = "?"
            try:
                tool_count = str(len(toolset.to_anthropic_tools()))
            except (AttributeError, TypeError):
                pass
            logger.info(
                "CodexBackend: attach_toolset accepted %s tool(s); "
                "HTTP MCP server will start on first invoke.",
                tool_count,
            )

    def close(self) -> None:
        """Positively settle every server, mutation host, and CLI home.

        Stages graceful → forced → give-up via
        ``CrucibleHttpMcpServer.stop``. Safe to call from cleanup paths
        that may also have already run (workflow ``finally`` block +
        ``atexit`` hook + interpreter shutdown).
        """
        errors: list[BaseException] = []
        with self._mcp_lock:
            self._closed = True
            srv = self._mcp_server
            active_servers = tuple(self._active_mcp_servers.values())
            mutation_hosts = tuple(self._active_mutation_hosts.values())
            isolated_home = self._isolated_home
        if srv is not None:
            try:
                srv.stop()
            except BaseException as exc:
                errors.append(exc)
            else:
                with self._mcp_lock:
                    if self._mcp_server is srv:
                        self._mcp_server = None
        servers_settled = not errors
        for server in active_servers:
            try:
                server.stop()
            except BaseException as exc:
                errors.append(exc)
                servers_settled = False
            else:
                self._unregister_mcp_server(server)
        if servers_settled:
            for host in mutation_hosts:
                try:
                    host.stop()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self._unregister_mutation_host(host)
        # A still-owned server/host may need its private rollout evidence for
        # settlement. Retain the exact home (and its existing private modes)
        # until every earlier cleanup stage has positively completed.
        if isolated_home is not None and not errors:
            try:
                isolated_home.cleanup()
            except BaseException as exc:
                errors.append(exc)
            else:
                with self._mcp_lock:
                    if self._isolated_home is isolated_home:
                        self._isolated_home = None
        for cleanup_error in errors:
            if isinstance(cleanup_error, KeyboardInterrupt):
                # Earlier ordinary cleanup failures must not consume a later
                # cancellation. The backend pool defers this signal and retries
                # every retained exact owner before it replays Ctrl-C.
                raise cleanup_error
        if errors:
            raise errors[0]
        with self._mcp_lock:
            resources_remain = bool(
                self._mcp_server is not None
                or self._active_mcp_servers
                or self._active_mutation_hosts
                or self._isolated_home is not None
            )
        if resources_remain:
            raise RuntimeError(
                "CodexBackend.close could not confirm all resources settled"
            )

    def _register_mcp_server(self, server: Any) -> None:
        with self._mcp_lock:
            closed = self._closed
            self._active_mcp_servers[id(server)] = server
        if closed:
            try:
                server.stop()
            except BaseException:
                raise
            else:
                self._unregister_mcp_server(server)
            raise RuntimeError("CodexBackend is closed")

    def _ensure_lifecycle_open(self) -> None:
        with self._mcp_lock:
            if self._closed:
                raise RuntimeError("CodexBackend is closed")

    def _unregister_mcp_server(self, server: Any) -> None:
        with self._mcp_lock:
            if self._active_mcp_servers.get(id(server)) is server:
                del self._active_mcp_servers[id(server)]

    def _register_mutation_host(self, host: Any) -> None:
        with self._mcp_lock:
            closed = self._closed
            self._active_mutation_hosts[id(host)] = host
        if closed:
            try:
                host.stop()
            except BaseException:
                raise
            else:
                self._unregister_mutation_host(host)
            raise RuntimeError("CodexBackend is closed")

    def _unregister_mutation_host(self, host: Any) -> None:
        with self._mcp_lock:
            if self._active_mutation_hosts.get(id(host)) is host:
                del self._active_mutation_hosts[id(host)]

    def _confined_subprocess_env(
        self,
        auth_path: Path,
        *,
        auth_payload: bytes | None = None,
    ) -> dict[str, str] | None:
        """Return an environment rooted at a config-free Codex home.

        Native-shell roles are an explicit administrative surface and retain
        their operator configuration. Every confined role gets a private home
        containing only its credential file, so user settings, skills,
        plugins, hooks, and MCP servers cannot enter the model tool catalogue.
        """
        if self._native_shell:
            return None
        with self._mcp_lock:
            created = False
            if self._isolated_home is None:
                # These homes contain only a copied credential and exist for
                # one backend lifetime. Create the random 0700 directory
                # atomically beneath the process temp root: a shared fixed
                # parent would need separate ownership, mode, and symlink
                # validation before it could safely hold credentials.
                # ``tempfile.gettempdir()`` honors TMPDIR, so benchmark
                # operators can keep all ephemeral state on a mounted volume.
                isolated = tempfile.TemporaryDirectory(
                    prefix="crucible-codex-home-",
                    dir=tempfile.gettempdir(),
                )
                self._isolated_home = isolated
                created = True
            home = self._isolated_home.name
            destination = Path(home) / "auth.json"
            try:
                if auth_payload is not None:
                    nofollow = getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(
                        destination,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_TRUNC
                        | nofollow
                        | getattr(os, "O_CLOEXEC", 0),
                        0o600,
                    )
                    try:
                        os.fchmod(descriptor, 0o600)
                        view = memoryview(auth_payload)
                        while view:
                            written = os.write(descriptor, view)
                            if written <= 0:  # pragma: no cover - OS invariant
                                raise OSError("short write while copying Codex auth")
                            view = view[written:]
                    finally:
                        os.close(descriptor)
                elif created:
                    shutil.copyfile(auth_path, destination)
                    destination.chmod(0o600)
            except BaseException:
                if created:
                    self._isolated_home = None
                    isolated.cleanup()
                raise
        safe_names = {
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
        }
        child_env = {
            **{name: value for name, value in os.environ.items() if name in safe_names},
            "CODEX_HOME": home,
        }
        if self._credential_source == CODEX_CHATGPT_CREDENTIAL_SOURCE:
            # The frozen route executes the native binary directly; no Node
            # or package-manager lookup is needed. Keep a fixed minimal PATH
            # for any ordinary system helper the CLI itself may require.
            child_env["PATH"] = "/usr/bin:/bin"
        return child_env

    def _ensure_mcp_server_running(self) -> str | None:
        """Lazily start the HTTP MCP server. Returns the URL or None.

        Called from ``_invoke_once`` before each Codex turn. If no toolset is
        attached, returns None. Confined roles still receive no ambient native
        tools or inherited MCP servers. If the server is already running,
        returns the cached URL.
        """
        with self._mcp_lock:
            if self._closed:
                raise RuntimeError("CodexBackend is closed")
            if self._mcp_toolset is None:
                return None
            if self._mcp_server is not None and self._mcp_server.is_running:
                self._mcp_server.reset_context_budget(self._mcp_context_budget)
                reset_terminal = getattr(
                    self._mcp_server,
                    "reset_terminal_dispatch_error",
                    None,
                )
                if callable(reset_terminal):
                    reset_terminal()
                return self._mcp_server.url

            # A server whose thread died after startup is not reusable. Stop
            # any remaining resources before replacing it with a fresh
            # single-use instance.
            stale_server = self._mcp_server
            if stale_server is not None:
                stale_server.stop()
                if self._mcp_server is stale_server:
                    self._mcp_server = None

            # Lazy import: keeps the codex backend importable without
            # uvicorn + mcp on the path (useful for code that introspects
            # the Backend protocol without intending to run anything).
            from crucible.tools._mcp_http_server import (  # noqa: PLC0415
                CrucibleHttpMcpServer,
            )

            server = CrucibleHttpMcpServer(
                self._mcp_toolset,
                context_budget=self._mcp_context_budget,
            )
            self._mcp_server = server
            try:
                return server.start()
            except BaseException:
                # ``start`` may itself fail while its listener/dispatch pool
                # is still alive. Keep the exact instance for close() retry.
                raise

    def _attached_tool_authority_digest(self) -> str | None:
        with self._mcp_lock:
            toolset = self._mcp_toolset
            context_budget = self._mcp_context_budget
        if toolset is None:
            return None
        from .tool_authority import (
            APPROVAL_POLICY_PER_CALL_EXACT,
            TOOL_AUTHORITY_BRIDGE_PROTOCOL,
            TOOL_AUTHORITY_BRIDGE_VERSION,
            tool_authority_digest,
        )

        return tool_authority_digest(
            toolset,
            capability=self.capability,
            approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
            bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
            bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
            context_budget=context_budget,
        )

    def _trusted_context_digest(self) -> str | None:
        if not self._trusted_project_instructions:
            return None
        assert self._trusted_repository_root is not None
        from .trusted_codex import trusted_codex_context_digest

        return trusted_codex_context_digest(
            self._trusted_repository_root,
            self._cwd,
        )

    def chat_with_tools(
        self,
        invocation: Invocation,
        toolset: object,
        *,
        cache: object = None,
        max_rounds: int | None = None,
        max_response_tokens: int | None = None,
        retry_budget: int | None = None,
        strict: bool = False,
    ) -> InvocationResult:
        """Run a tool-using turn via the Codex CLI + HTTP MCP bridge.

        ``ClaudeBackend.chat_with_tools`` is the *direct Anthropic
        Messages-API* path — it bypasses the bundled Claude CLI to
        sidestep that CLI's 25K-token Read cap and exit-1 quirks.
        Codex has no direct-API equivalent, so the Codex path just
        attaches the toolset to this backend (starting the HTTP MCP
        server lazily) and calls ``invoke()``. The Codex CLI sees
        the tool catalog through MCP exactly the same way as a
        sustained-attach workflow.

        Behavior:

        - Attaches ``toolset`` (replaces any previously-attached
          toolset). The HTTP MCP server starts on the first
          invocation that needs it.
        - Calls ``invoke(invocation)`` — Codex CLI dispatches tools
          through MCP itself.
        - The current app-server surface cannot enforce portable
          ``cache``, ``max_rounds``, ``max_response_tokens``, or
          ``retry_budget`` overrides. Explicit requests therefore reject
          before attaching the toolset or starting transport. ``None`` means
          no portable override; invocation policy remains authoritative.
        - ``strict=True`` requires a non-null toolset. With one supplied, the
          MCP path satisfies that requirement.

        The toolset remains attached after the call returns so a
        workflow's subsequent ``invoke()`` calls reuse the same MCP
        server (cheaper). Callers that want one-shot tool exposure
        should call ``backend.attach_toolset(None)`` after to drop
        the reference (does not force HTTP server shutdown — that's
        :py:meth:`close`'s job).
        """
        unsupported: list[str] = []
        if cache is not None:
            unsupported.append("cache")
        if max_rounds is not None:
            unsupported.append("max_rounds")
        if max_response_tokens is not None:
            unsupported.append("max_response_tokens")
        if retry_budget is not None:
            unsupported.append("retry_budget")
        if unsupported:
            raise NotImplementedError(
                "CodexBackend.chat_with_tools cannot enforce explicit "
                f"{', '.join(unsupported)} overrides; express supported "
                "retry/timeout policy on Invocation.policy or select a "
                "route whose feature profile declares these controls"
            )
        if strict and toolset is None:
            raise NotImplementedError(
                "CodexBackend.chat_with_tools strict mode requires a toolset"
            )
        if toolset is not None:
            self.attach_toolset(toolset)
        return self.invoke(invocation)

    # ------------------------------------------------------------------
    # Retry + policy enforcement
    # ------------------------------------------------------------------

    def _result_with_evidence(
        self,
        *,
        final_message: str,
        transcript: tuple[Message, ...],
        usage: Usage,
        stop_reason: str,
        pricing_evidence: UsagePricingEvidence | None = None,
        error: Any = None,
        token_usage_complete: bool = False,
        cost_complete: bool = False,
        terminal_complete: bool = False,
        events: list[BackendExecutionEvent] | None = None,
        trace_template: BackendExecutionTrace | None = None,
        provider_request_id: str | None = None,
        tool_authority_digest: str | None = None,
        trusted_context_digest: str | None = None,
        conversation: ConversationTransportOutcome | None = None,
    ) -> InvocationResult:
        """Build one normalized logical-invocation terminal envelope."""

        normalized: list[BackendExecutionEvent] = []
        for event in events or ():
            if event.kind == "terminal":
                continue
            _append_event(
                normalized,
                event.kind,
                name=event.name,
                parent_id=event.parent_id,
                child_id=event.child_id,
            )
        if not any(event.kind == "request" for event in normalized):
            _append_event(normalized, "request", name="codex_app_server")
        _append_event(normalized, "terminal", name=stop_reason)

        if trace_template is None:
            from .features import feature_profile_digest

            trace_template = BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                requested_effort=self._effort,
                # The reviewed app-server event stream does not expose a
                # provider-served model/effort identity. Preserve the
                # requested route without presenting it as observed.
                served_model=None,
                served_effort=None,
                provider_request_id=provider_request_id,
                feature_profile_digest=feature_profile_digest(self.features),
                tool_authority_digest=tool_authority_digest,
                trusted_context_digest=(
                    self._trusted_context_digest()
                    if trusted_context_digest is None
                    else trusted_context_digest
                ),
            )
        trace = _replace(trace_template, events=tuple(normalized))
        if pricing_evidence is None:
            pricing_evidence = UsagePricingEvidence(
                segments=(),
                complete=False,
            )
        return InvocationResult(
            final_message=final_message,
            transcript=transcript,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            completeness=ExecutionCompleteness(
                token_usage_complete=token_usage_complete,
                cost_complete=cost_complete,
                terminal_complete=terminal_complete,
                # This is the complete normalized adapter envelope even when
                # the provider terminal itself is absent; terminal_complete
                # independently carries that distinction.
                trace_complete=True,
            ),
            execution_trace=trace,
            conversation=conversation,
            pricing_evidence=pricing_evidence,
        )

    def _finalize_attempts(
        self,
        result: InvocationResult,
        *,
        events: list[BackendExecutionEvent],
        prior_usage: Usage,
        prior_pricing_evidence: UsagePricingEvidence,
        prior_token_usage_complete: bool,
        prior_cost_complete: bool,
    ) -> InvocationResult:
        """Merge physical-attempt evidence into one logical result."""

        _merge_nonterminal_events(
            events,
            result,
            request_name="codex_app_server",
            include_cancellation=False,
        )
        error_kind = getattr(result.error, "kind", None)
        if error_kind == "budget_exhausted" and not any(
            event.kind == "limit" and event.name == "budget_usd" for event in events
        ):
            _append_event(events, "limit", name="budget_usd")
        elif (
            error_kind == "max_turns"
            and result.stop_reason != "tool_budget_exhausted"
            and not any(
                event.kind == "limit" and event.name == "max_turns" for event in events
            )
        ):
            _append_event(events, "limit", name="max_turns")
        elif error_kind == "cancellation" and not any(
            event.kind == "cancellation" for event in events
        ):
            _append_event(events, "cancellation", name="external")
        if result.stop_reason == "timeout" and not any(
            event.kind == "limit" and event.name == "deadline" for event in events
        ):
            _append_event(events, "limit", name="deadline")

        usage = _usage_sum(prior_usage, result.usage)
        pricing_evidence = _pricing_evidence_sum(
            prior_pricing_evidence,
            result.pricing_evidence,
        )
        return self._result_with_evidence(
            final_message=result.final_message,
            transcript=result.transcript,
            usage=usage,
            pricing_evidence=pricing_evidence,
            stop_reason=result.stop_reason,
            error=result.error,
            token_usage_complete=(
                prior_token_usage_complete and result.completeness.token_usage_complete
            ),
            cost_complete=(prior_cost_complete and result.completeness.cost_complete),
            terminal_complete=result.completeness.terminal_complete,
            events=events,
            trace_template=result.execution_trace,
            tool_authority_digest=self._attached_tool_authority_digest(),
            conversation=result.conversation,
        )

    def _override_with_terminal_dispatch(
        self,
        invocation: Invocation,
        result: InvocationResult,
        terminal_error: BaseException,
    ) -> InvocationResult:
        """Make a settled late MCP terminal authoritative over provider output."""

        from crucible.tools._dispatch_budget import (  # noqa: PLC0415
            ToolDispatchBudgetExceeded,
            ToolMutationIndeterminate,
        )

        mutation_indeterminate = isinstance(
            terminal_error,
            ToolMutationIndeterminate,
        )
        budget_exhausted = isinstance(
            terminal_error,
            ToolDispatchBudgetExceeded,
        )
        events: list[BackendExecutionEvent] = []
        _merge_nonterminal_events(
            events,
            result,
            request_name="codex_app_server",
            include_cancellation=False,
        )
        if budget_exhausted:
            _append_event(events, "limit", name="tool_dispatch_budget")
        error = (
            TransportError(
                http_status=None,
                message=(
                    "Codex tool loop terminated after an indeterminate "
                    f"source mutation: {terminal_error}"
                ),
                retriable=False,
            )
            if mutation_indeterminate
            else (
                MaxTurnsExceeded(
                    turns_used=0,
                    turns_limit=invocation.max_turns,
                    message=(
                        "Codex tool loop terminated after a workflow-owned "
                        "tool-dispatch budget denial: "
                        f"{terminal_error}"
                    ),
                )
                if budget_exhausted
                else TransportError(
                    http_status=None,
                    message=(
                        "Codex tool loop terminated at a workflow-owned "
                        f"dispatch boundary: {terminal_error}"
                    ),
                    retriable=False,
                )
            )
        )
        stop_reason = (
            "tool_mutation_indeterminate"
            if mutation_indeterminate
            else (
                "tool_budget_exhausted"
                if budget_exhausted
                else "tool_dispatch_terminal"
            )
        )
        return self._result_with_evidence(
            final_message="",
            transcript=result.transcript,
            usage=result.usage,
            pricing_evidence=result.pricing_evidence,
            stop_reason=stop_reason,
            error=error,
            token_usage_complete=result.completeness.token_usage_complete,
            cost_complete=result.completeness.cost_complete,
            terminal_complete=True,
            events=events,
            trace_template=result.execution_trace,
            tool_authority_digest=(
                None
                if result.execution_trace is None
                else result.execution_trace.tool_authority_digest
            ),
            conversation=result.conversation,
        )

    async def _invoke_with_retries(self, invocation: Invocation) -> InvocationResult:
        policy = invocation.policy or self._default_policy
        timeout_seconds = _validated_timeout_seconds(policy)
        deadline = (
            None if timeout_seconds is None else time.monotonic() + timeout_seconds
        )
        with self._mcp_lock:
            if self._closed:
                raise RuntimeError("CodexBackend is closed")
            attached_toolset = self._mcp_toolset
            context_budget = self._mcp_context_budget
        if self._credential_source == CODEX_CHATGPT_CREDENTIAL_SOURCE:
            if self._cli_pin is None:  # pragma: no cover - constructor invariant
                raise CodexSessionIdentityError(
                    "Codex ChatGPT-session route has no pinned CLI"
                )
            _validate_codex_cli_version(self._cli_pin, model=self.model)
        mutation_host: Any | None = None
        hosted_server: Any | None = None
        terminal_tool_server: Any | None = None
        mcp_url_override: Any = _MCP_URL_UNSET
        if attached_toolset is not None and getattr(
            attached_toolset, "write_names", ()
        ):
            hosted_toolset, mutation_host = _process_hosted_toolset(
                attached_toolset,
                invocation_deadline=deadline,
                register_host=self._register_mutation_host,
                unregister_host=self._unregister_mutation_host,
                ensure_open=self._ensure_lifecycle_open,
            )
            from crucible.tools._mcp_http_server import (  # noqa: PLC0415
                CrucibleHttpMcpServer,
            )

            hosted_server = CrucibleHttpMcpServer(
                hosted_toolset,
                context_budget=context_budget,
                invocation_deadline=deadline,
                start_timeout=(
                    10.0
                    if deadline is None
                    else min(10.0, max(0.001, deadline - time.monotonic()))
                ),
            )
            self._register_mcp_server(hosted_server)
            try:
                mcp_url_override = hosted_server.start()
                self._ensure_lifecycle_open()
            except BaseException as invocation_error:
                cleanup_error: BaseException | None = None
                server_settled = True
                try:
                    hosted_server.stop()
                except BaseException as exc:
                    cleanup_error = exc
                    server_settled = False
                else:
                    self._unregister_mcp_server(hosted_server)
                if server_settled:
                    try:
                        mutation_host.stop()
                    except BaseException as exc:
                        cleanup_error = cleanup_error or exc
                    else:
                        self._unregister_mutation_host(mutation_host)
                if cleanup_error is not None:
                    raise cleanup_error from invocation_error
                raise
            terminal_tool_server = hosted_server
        elif attached_toolset is not None:
            # Start/reset the reusable read-only bridge once per logical
            # invocation. Physical transport retries must share the same
            # cumulative UTF-8 result ledger.
            mcp_url_override = self._ensure_mcp_server_running()
            with self._mcp_lock:
                terminal_tool_server = self._mcp_server
        result: InvocationResult | None = None
        provider_error: BaseException | None = None
        late_terminal: BaseException | None = None
        try:
            try:
                result = await self._invoke_with_retries_core(
                    invocation,
                    policy=policy,
                    timeout_seconds=timeout_seconds,
                    deadline=deadline,
                    mcp_url_override=mcp_url_override,
                    mutation_host=mutation_host,
                    terminal_tool_server=terminal_tool_server,
                )
            except BaseException as exc:
                provider_error = exc
        finally:
            cleanup_error: BaseException | None = None
            server_settled = True
            if hosted_server is not None:
                try:
                    hosted_server.stop()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    server_settled = False
                else:
                    self._unregister_mcp_server(hosted_server)
            if mutation_host is not None and server_settled:
                try:
                    mutation_host.stop()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                else:
                    self._unregister_mutation_host(mutation_host)
            if hosted_server is not None:
                observed_terminal = getattr(
                    hosted_server,
                    "terminal_dispatch_error",
                    None,
                )
                if isinstance(observed_terminal, BaseException):
                    late_terminal = observed_terminal
        if late_terminal is not None:
            from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                ToolMutationIndeterminate,
            )

            if isinstance(late_terminal, ToolMutationIndeterminate):
                if (
                    provider_error is not None
                    and provider_error is not late_terminal
                ):
                    late_terminal.add_note(
                        "Codex provider execution also raised "
                        f"{type(provider_error).__name__}; the indeterminate "
                        "mutation remains primary."
                    )
                if cleanup_error is not None:
                    late_terminal.add_note(
                        "Codex retained invocation-scoped cleanup authority; "
                        "CodexBackend.close() must settle the exact registered "
                        "server and mutation host before journal rollback."
                    )
                    if cleanup_error is late_terminal:
                        raise late_terminal
                    raise late_terminal from cleanup_error
                if provider_error is not None:
                    if provider_error is late_terminal:
                        raise late_terminal
                    raise late_terminal from provider_error
        if cleanup_error is not None:
            if provider_error is not None:
                raise cleanup_error from provider_error
            raise cleanup_error
        if provider_error is not None:
            raise provider_error
        if late_terminal is not None:
            assert result is not None
            result = self._override_with_terminal_dispatch(
                invocation,
                result,
                late_terminal,
            )
        assert result is not None
        return result

    async def _invoke_with_retries_core(
        self,
        invocation: Invocation,
        *,
        policy: BackendPolicy,
        timeout_seconds: float | None,
        deadline: float | None,
        mcp_url_override: Any,
        mutation_host: Any | None,
        terminal_tool_server: Any | None,
    ) -> InvocationResult:
        conversation_request = invocation.conversation
        if conversation_request is not None and conversation_request.mode == "none":
            raise NotImplementedError(
                "CodexBackend cannot attach provider continuity in "
                "conversation mode 'none'"
            )
        external_thread_id = (
            conversation_request.opaque_id
            if (
                conversation_request is not None
                and conversation_request.action == "reuse"
            )
            else None
        )
        # Carried across retries: the thread opened by a failed attempt, so
        # a transport retry resumes it (full context) instead of starting
        # fresh. None until an attempt actually opens a thread.
        resume_thread_id: str | None = external_thread_id
        conversation_thread_id: str | None = external_thread_id
        external_resume_accepted = False
        # True on any retry → the turn uses the task-restating recovery
        # prompt (so a seeded-fresh fallback still carries the task).
        recovery = False
        events: list[BackendExecutionEvent] = []
        prior_usage = Usage()
        prior_pricing_evidence = UsagePricingEvidence(
            segments=(),
            complete=True,
        )
        prior_token_usage_complete = True
        prior_cost_complete = True

        for attempt in range(max(1, policy.retry_budget)):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                _append_event(events, "limit", name="deadline")
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=prior_usage,
                    pricing_evidence=prior_pricing_evidence,
                    stop_reason="timeout",
                    error=TransportError(
                        http_status=None,
                        message=(
                            "Codex invocation exhausted its cumulative wall deadline"
                        ),
                        retriable=False,
                    ),
                    token_usage_complete=prior_token_usage_complete,
                    cost_complete=prior_cost_complete,
                    terminal_complete=False,
                    events=events,
                    tool_authority_digest=(self._attached_tool_authority_digest()),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_thread_id,
                    ),
                )
            remaining_budget = (
                None
                if invocation.budget_usd is None
                else max(0.0, invocation.budget_usd - prior_usage.cost_usd)
            )
            if remaining_budget is not None and remaining_budget <= 0:
                _append_event(events, "limit", name="budget_usd")
                return self._result_with_evidence(
                    final_message="",
                    transcript=(),
                    usage=prior_usage,
                    pricing_evidence=prior_pricing_evidence,
                    stop_reason="budget_exhausted",
                    error=BudgetExhausted(
                        spent_usd=prior_usage.cost_usd,
                        limit_usd=invocation.budget_usd or 0.0,
                        message=(
                            f"Invocation spent ${prior_usage.cost_usd:.4f} "
                            "before a retry could start"
                        ),
                    ),
                    token_usage_complete=prior_token_usage_complete,
                    cost_complete=prior_cost_complete,
                    terminal_complete=False,
                    events=events,
                    tool_authority_digest=(self._attached_tool_authority_digest()),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_thread_id,
                    ),
                )
            started_iso = datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            )
            start_mono = time.monotonic()
            result: InvocationResult | None = None
            # The deadline/dispatch monitors deliberately cancel and join the
            # attempt before raising. Its return value cannot carry evidence
            # through that cancellation; this one-entry sink survives it.
            interrupted_evidence: list[InvocationResult] = []
            traceback_str: str | None = None
            thread_id_holder: list[str] = []
            resume_failed_holder: list[bool] = []
            external_resume = (
                external_thread_id is not None
                and not external_resume_accepted
                and resume_thread_id == external_thread_id
            )

            try:
                (
                    result,
                    terminal_tool_error,
                ) = await _await_with_terminal_tool_monitor(
                    _await_with_deadline(
                        _retain_attempt_result(
                            self._invoke_once(
                                invocation,
                                policy,
                                resume_thread_id=resume_thread_id,
                                thread_id_holder=thread_id_holder,
                                resume_failed_holder=resume_failed_holder,
                                external_resume=external_resume,
                                recovery=recovery,
                                budget_usd_override=remaining_budget,
                                mcp_url_override=mcp_url_override,
                                interrupted_evidence=interrupted_evidence,
                            ),
                            interrupted_evidence,
                        ),
                        remaining,
                    ),
                    terminal_tool_server,
                )
                if terminal_tool_error is not None:
                    if result is None and interrupted_evidence:
                        result = interrupted_evidence[0]
                    from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                        ToolDispatchBudgetExceeded,
                        ToolMutationIndeterminate,
                    )

                    mutation_indeterminate = isinstance(
                        terminal_tool_error,
                        ToolMutationIndeterminate,
                    )
                    budget_exhausted = isinstance(
                        terminal_tool_error,
                        ToolDispatchBudgetExceeded,
                    )
                    denial_events: list[BackendExecutionEvent] = []
                    if result is not None:
                        _merge_nonterminal_events(
                            denial_events,
                            result,
                            request_name=(f"codex_app_server_attempt_{attempt + 1}"),
                            include_cancellation=False,
                        )
                    else:
                        _append_event(
                            denial_events,
                            "request",
                            name=(f"codex_app_server_attempt_{attempt + 1}"),
                        )
                    if budget_exhausted:
                        _append_event(
                            denial_events,
                            "limit",
                            name="tool_dispatch_budget",
                        )
                    denial_error = (
                        TransportError(
                            http_status=None,
                            message=(
                                "Codex tool loop terminated after an "
                                "indeterminate source mutation: "
                                f"{terminal_tool_error}"
                            ),
                            retriable=False,
                        )
                        if mutation_indeterminate
                        else (
                            MaxTurnsExceeded(
                            turns_used=0,
                            turns_limit=invocation.max_turns,
                            message=(
                                "Codex tool loop terminated after a "
                                "workflow-owned tool-dispatch budget denial: "
                                f"{terminal_tool_error}"
                            ),
                            )
                            if budget_exhausted
                            else TransportError(
                                http_status=None,
                                message=(
                                    "Codex tool loop terminated at a "
                                    "workflow-owned dispatch boundary: "
                                    f"{terminal_tool_error}"
                                ),
                                retriable=False,
                            )
                        )
                    )
                    result = self._result_with_evidence(
                        final_message="",
                        transcript=(() if result is None else result.transcript),
                        usage=(Usage() if result is None else result.usage),
                        pricing_evidence=(
                            None if result is None else result.pricing_evidence
                        ),
                        stop_reason=(
                                "tool_mutation_indeterminate"
                                if mutation_indeterminate
                                else (
                                    "tool_budget_exhausted"
                                    if budget_exhausted
                                    else "tool_dispatch_terminal"
                                )
                        ),
                        error=denial_error,
                        token_usage_complete=(
                            False
                            if result is None
                            else result.completeness.token_usage_complete
                        ),
                        cost_complete=(
                            False
                            if result is None
                            else result.completeness.cost_complete
                        ),
                        # The workflow-owned limit signal is an exact adapter
                        # terminal even when cancelling the provider turn
                        # leaves its token/cost snapshot unavailable.
                        terminal_complete=True,
                        events=denial_events,
                        trace_template=(
                            None if result is None else result.execution_trace
                        ),
                        tool_authority_digest=(self._attached_tool_authority_digest()),
                        conversation=(
                            _conversation_outcome(
                                invocation,
                                opaque_id=(
                                    thread_id_holder[0]
                                    if thread_id_holder
                                    else conversation_thread_id
                                ),
                            )
                            if result is None
                            else result.conversation
                        ),
                    )
            except _InvocationDeadlineExceeded:
                # The joined attempt may retain qualified bounded capture even
                # though buffered chat_once never returned. It is a lower
                # bound, never evidence that the interrupted turn was free.
                partial = interrupted_evidence[0] if interrupted_evidence else None
                err = TransportError(
                    http_status=None,
                    message=(
                        "Codex invocation exhausted its cumulative wall "
                        f"deadline after {timeout_seconds}s"
                    ),
                    retriable=False,
                )
                logger.warning(
                    "CodexBackend timeout on attempt %d: %s",
                    attempt + 1,
                    err.message,
                )
                if partial is not None:
                    _merge_nonterminal_events(
                        events,
                        partial,
                        request_name=f"codex_app_server_attempt_{attempt + 1}",
                        include_cancellation=False,
                    )
                else:
                    _append_event(
                        events,
                        "request",
                        name=f"codex_app_server_attempt_{attempt + 1}",
                    )
                _append_event(events, "limit", name="deadline")
                if thread_id_holder:
                    conversation_thread_id = thread_id_holder[0]
                    if external_thread_id is not None:
                        external_resume_accepted = True
                result = self._result_with_evidence(
                    final_message="",
                    transcript=(() if partial is None else partial.transcript),
                    usage=_usage_sum(
                        prior_usage, Usage() if partial is None else partial.usage
                    ),
                    pricing_evidence=_pricing_evidence_sum(
                        prior_pricing_evidence,
                        None if partial is None else partial.pricing_evidence,
                        force_incomplete=True,
                    ),
                    stop_reason="timeout",
                    error=err,
                    token_usage_complete=False,
                    cost_complete=False,
                    terminal_complete=False,
                    events=events,
                    trace_template=(None if partial is None else partial.execution_trace),
                    tool_authority_digest=(self._attached_tool_authority_digest()),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=conversation_thread_id,
                    ),
                )
                self._sdk_sink.dump(
                    invocation=invocation,
                    result=result,
                    started_iso=started_iso,
                    duration_s=time.monotonic() - start_mono,
                    attempt=attempt,
                )
                return result
            except BaseException as exc:  # noqa: BLE001
                traceback_str = _tb.format_exc()
                err = self._classify_exception(exc, policy)
                partial = interrupted_evidence[0] if interrupted_evidence else None
                if err is None:
                    diagnostic_source = partial
                    if partial is not None and _is_external_cancellation(exc):
                        cancellation = _unwrap_exception_group(exc)
                        diagnostic_source = _replace(
                            partial,
                            stop_reason="cancelled",
                            error=ExternalCancellation(
                                cause=type(cancellation).__name__,
                                message=str(cancellation),
                                retriable=False,
                            ),
                        )
                    diagnostic = (
                        None
                        if diagnostic_source is None
                        else self._finalize_attempts(
                            diagnostic_source,
                            events=events,
                            prior_usage=prior_usage,
                            prior_pricing_evidence=prior_pricing_evidence,
                            prior_token_usage_complete=prior_token_usage_complete,
                            prior_cost_complete=prior_cost_complete,
                        )
                    )
                    self._sdk_sink.dump(
                        invocation=invocation,
                        result=diagnostic,
                        started_iso=started_iso,
                        duration_s=time.monotonic() - start_mono,
                        traceback_str=traceback_str,
                        re_raised=True,
                        attempt=attempt,
                    )
                    raise
                local_events: list[BackendExecutionEvent] = []
                if partial is not None:
                    _merge_nonterminal_events(
                        local_events,
                        partial,
                        request_name=f"codex_app_server_attempt_{attempt + 1}",
                    )
                else:
                    _append_event(
                        local_events,
                        "request",
                        name=f"codex_app_server_attempt_{attempt + 1}",
                    )
                if getattr(err, "kind", None) == "cancellation" and not any(
                    event.kind == "cancellation" for event in local_events
                ):
                    _append_event(
                        local_events,
                        "cancellation",
                        name="external",
                    )
                result = self._result_with_evidence(
                    final_message="",
                    transcript=(() if partial is None else partial.transcript),
                    usage=(Usage() if partial is None else partial.usage),
                    pricing_evidence=(
                        None if partial is None else partial.pricing_evidence
                    ),
                    stop_reason=(
                        "cancelled"
                        if getattr(err, "kind", None) == "cancellation"
                        else "error"
                    ),
                    error=err,
                    terminal_complete=False,
                    events=local_events,
                    trace_template=(None if partial is None else partial.execution_trace),
                    tool_authority_digest=(self._attached_tool_authority_digest()),
                    conversation=_conversation_outcome(
                        invocation,
                        opaque_id=(
                            thread_id_holder[0]
                            if thread_id_holder
                            else conversation_thread_id
                        ),
                    ),
                )

            if thread_id_holder:
                conversation_thread_id = thread_id_holder[0]
                if external_thread_id is not None:
                    external_resume_accepted = True

            # ``_invoke_once`` may have caught a
            # transport-class exception (e.g. ``CodexTransportError:
            # failed reading from stdio transport`` when the
            # ``codex app-server`` subprocess closes its stdio
            # mid-turn) and packaged it as ``result.error`` instead
            # of raising. The exception-driven retry branch above
            # only fires when ``_invoke_once`` itself raises; we
            # also have to honor ``retriable=True`` on a structured
            # error in the returned result, otherwise transport
            # flakes silently terminate the cascade after one
            # attempt regardless of ``retry_budget``.
            self._sdk_sink.dump(
                invocation=invocation,
                result=result,
                started_iso=started_iso,
                duration_s=time.monotonic() - start_mono,
                traceback_str=traceback_str,
                attempt=attempt,
            )
            retriable = (
                result.error is not None
                and getattr(result.error, "retriable", False)
                and attempt + 1 < policy.retry_budget
                and getattr(result.error, "kind", None) != "cancellation"
            )
            if (
                retriable
                and mutation_host is not None
                and mutation_host.ever_started_write
            ):
                result = _replace(
                    result,
                    error=_replace(
                        result.error,
                        message=(
                            f"{result.error.message}; automatic retry "
                            "suppressed after a mutating tool call"
                        ),
                        retriable=False,
                    ),
                )
                retriable = False
            if retriable and not (
                prior_token_usage_complete
                and prior_cost_complete
                and _usage_observed(result)
                and _cost_accounting_is_budget_safe(result)
            ):
                current_incomplete = not (
                    _usage_observed(result) and _cost_accounting_is_budget_safe(result)
                )
                reason = (
                    "the failed physical attempt did not return complete "
                    "token/cost accounting that is budget-safe"
                    if current_incomplete
                    else (
                        "a prior automatic retry made cumulative versus "
                        "incremental usage ambiguous"
                    )
                )
                result = _replace(
                    result,
                    error=_replace(
                        result.error,
                        message=(
                            f"{result.error.message}; automatic retry "
                            f"suppressed because {reason}"
                        ),
                        retriable=False,
                    ),
                )
                retriable = False
            if retriable:
                _merge_nonterminal_events(
                    events,
                    result,
                    request_name=f"codex_app_server_attempt_{attempt + 1}",
                )
                prior_usage = _usage_sum(prior_usage, result.usage)
                prior_pricing_evidence = _pricing_evidence_sum(
                    prior_pricing_evidence,
                    result.pricing_evidence,
                    # Interrupted retry boundaries retain segments but cannot
                    # prove that every paid response reached local capture.
                    force_incomplete=True,
                )
                prior_token_usage_complete = (
                    prior_token_usage_complete
                    and result.completeness.token_usage_complete
                )
                prior_cost_complete = (
                    prior_cost_complete and result.completeness.cost_complete
                )
                # Each physical attempt now excludes a qualified replay
                # baseline, but an interrupted retry chain still cannot attest
                # that every provider response reached one and only one local
                # terminal result. Preserve the existing aggregate debt.
                prior_token_usage_complete = False
                prior_cost_complete = False
                recovery = True
                if thread_id_holder:
                    resume_thread_id = thread_id_holder[0]
                elif resume_thread_id is not None and resume_failed_holder:
                    # ``_invoke_once`` abandoned a client whose resume did
                    # not produce a handle.  Clear the stale id so the next
                    # retry cannot re-enter the same failed rollout.
                    if external_thread_id is None:
                        resume_thread_id = None
                backoff = min(2**attempt, 16) + (time.monotonic() % 1.0)
                _append_event(
                    events,
                    "retry",
                    name=getattr(
                        result.error,
                        "kind",
                        type(result.error).__name__,
                    ),
                )
                logger.info(
                    "CodexBackend retriable error in result on "
                    "attempt %d (%s); retrying in %.1fs%s",
                    attempt + 1,
                    getattr(result.error, "kind", type(result.error).__name__),
                    backoff,
                    f" (resuming thread {resume_thread_id})"
                    if resume_thread_id
                    else "",
                )
                try:
                    if deadline is None:
                        await asyncio.sleep(backoff)
                    else:
                        backoff_remaining = deadline - time.monotonic()
                        if backoff_remaining <= 0:
                            raise asyncio.TimeoutError
                        await asyncio.wait_for(
                            asyncio.sleep(backoff),
                            timeout=backoff_remaining,
                        )
                except asyncio.TimeoutError:
                    _append_event(events, "limit", name="deadline")
                    timeout_result = self._result_with_evidence(
                        final_message="",
                        transcript=(),
                        usage=prior_usage,
                        pricing_evidence=prior_pricing_evidence,
                        stop_reason="timeout",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "Codex retry backoff exhausted the cumulative "
                                "invocation wall deadline"
                            ),
                            retriable=False,
                        ),
                        token_usage_complete=prior_token_usage_complete,
                        cost_complete=prior_cost_complete,
                        terminal_complete=False,
                        events=events,
                        tool_authority_digest=(self._attached_tool_authority_digest()),
                        conversation=_conversation_outcome(
                            invocation,
                            opaque_id=conversation_thread_id,
                        ),
                    )
                    return timeout_result
                except asyncio.CancelledError as exc:
                    if policy.on_cancellation == "raise":
                        raise
                    _append_event(events, "cancellation", name="external")
                    return self._result_with_evidence(
                        final_message="",
                        transcript=(),
                        usage=prior_usage,
                        pricing_evidence=prior_pricing_evidence,
                        stop_reason="cancelled",
                        error=ExternalCancellation(
                            cause=type(exc).__name__,
                            message=str(exc),
                            retriable=True,
                        ),
                        token_usage_complete=prior_token_usage_complete,
                        cost_complete=prior_cost_complete,
                        terminal_complete=False,
                        events=events,
                        tool_authority_digest=(self._attached_tool_authority_digest()),
                        conversation=_conversation_outcome(
                            invocation,
                            opaque_id=conversation_thread_id,
                        ),
                    )
                continue

            logical_usage = _usage_sum(prior_usage, result.usage)
            if (
                invocation.budget_usd is not None
                and logical_usage.cost_usd > invocation.budget_usd
            ):
                result = _replace(
                    result,
                    stop_reason="budget_exhausted",
                    error=BudgetExhausted(
                        spent_usd=logical_usage.cost_usd,
                        limit_usd=invocation.budget_usd,
                        message=(
                            f"Logical invocation spent "
                            f"${logical_usage.cost_usd:.4f} exceeding budget "
                            f"${invocation.budget_usd:.4f}"
                        ),
                    ),
                )
            return self._finalize_attempts(
                result,
                events=events,
                prior_usage=prior_usage,
                prior_pricing_evidence=prior_pricing_evidence,
                prior_token_usage_complete=prior_token_usage_complete,
                prior_cost_complete=prior_cost_complete,
            )

        fallback_events = list(events)
        return self._result_with_evidence(
            final_message="",
            transcript=(),
            usage=prior_usage,
            pricing_evidence=prior_pricing_evidence,
            stop_reason="error",
            error=TransportError(
                message="retry loop exhausted with no result",
                retriable=False,
            ),
            token_usage_complete=prior_token_usage_complete,
            cost_complete=prior_cost_complete,
            terminal_complete=False,
            events=fallback_events,
            tool_authority_digest=self._attached_tool_authority_digest(),
            conversation=_conversation_outcome(
                invocation,
                opaque_id=conversation_thread_id,
            ),
        )

    # ------------------------------------------------------------------
    # Single SDK invocation
    # ------------------------------------------------------------------

    async def _invoke_once(
        self,
        invocation: Invocation,
        policy: BackendPolicy,
        *,
        resume_thread_id: str | None = None,
        thread_id_holder: list[str] | None = None,
        resume_failed_holder: list[bool] | None = None,
        external_resume: bool = False,
        recovery: bool = False,
        budget_usd_override: float | None = None,
        mcp_url_override: Any = _MCP_URL_UNSET,
        interrupted_evidence: list[InvocationResult] | None = None,
    ) -> InvocationResult:
        """One pass through a Codex thread chat call.

        ``resume_thread_id``: when set (a transport-retry after a prior
        attempt opened a thread), RESUME that thread instead of starting a
        fresh one. Codex persists each conversation as an on-disk rollout,
        so a new ``connect_stdio`` (a freshly-reconnected app-server) can
        pick the thread back up with its full context — exploration,
        reasoning, and any edits already on disk — intact. This is the
        transport-hardening path: a dropped stdio transport no longer
        discards the turn's accumulated context; we reconnect and continue.

        ``thread_id_holder``: a mutable sink the caller passes so the retry
        loop learns the thread id even when ``chat_once`` later fails — the
        id is captured right after ``start_thread``/``resume_thread``, which
        complete before the turn runs.

        ``resume_failed_holder``: records that this attempt reached and
        failed ``resume_thread``.  The retry loop clears a stale thread id
        only for an adapter-created retry identity.

        ``external_resume``: distinguishes a workflow-supplied opaque ID from
        an adapter-internal transport retry. A provider protocol rejection of
        the former becomes a typed stale outcome; it never silently starts a
        fresh conversation."""
        from codex_app_server_sdk import (  # type: ignore[import-not-found]
            CodexClient,
            ThreadConfig,
            TurnOverrides,
        )

        # Lazy auth probe — fail loud if the user hasn't `codex login`-ed.
        # Frozen Arm A ignores ambient CODEX_HOME and validates a private
        # ChatGPT session record before any provider process is started.
        auth_payload: bytes | None = None
        if self._credential_source == CODEX_CHATGPT_CREDENTIAL_SOURCE:
            if os.environ.get("CODEX_HOME", "").strip():
                raise RuntimeError(
                    "Codex ChatGPT-session route rejects ambient CODEX_HOME"
                )
            if self._chatgpt_auth_path is None or self._cli_pin is None:
                raise RuntimeError("Codex ChatGPT-session route is not fully bound")
            auth_path = self._chatgpt_auth_path
            self._cli_pin.revalidate()
            auth_payload = _read_chatgpt_auth_payload(auth_path)
        else:
            auth_path = _resolve_codex_auth_path()
            if not auth_path.is_file():
                raise RuntimeError(
                    f"Codex auth file not found at {auth_path}. "
                    f"Run `codex login` (or set $CODEX_HOME) before invoking "
                    f"the codex backend."
                )

        transcript: list[Message] = []
        final_message = ""
        # ``usage`` is populated from bounded
        # ``thread/tokenUsage/updated`` response events. Cost is computed from
        # tokens via the local pricing table because app-server does not
        # surface per-response cost.
        usage = Usage()
        pricing_evidence = UsagePricingEvidence(
            segments=(),
            complete=False,
        )
        usage_complete = False
        cost_complete = False
        cost_upper_bound = False
        stop_reason = "end_turn"
        terminal_complete = False
        provider_request_id: str | None = None
        provider_conversation_id: str | None = resume_thread_id
        events: list[BackendExecutionEvent] = []
        captured_raw_events: list[dict[str, Any]] = []
        forbidden_delegation_events: list[_CodexDelegationSignal] = []
        handle_thread_id: str | None = None
        handle_identity_complete = False
        chat_once_completed = False
        current_turn_id: str | None = None
        result_thread_id: str | None = None
        result_identity_complete = False
        _append_event(events, "request", name="codex_app_server")

        # The Codex SDK is async-context-managed; entering the `async
        # with` opens an stdio transport to the codex CLI subprocess.
        #
        # If an MCP toolset is attached, ensure the HTTP server is running and
        # tell Codex CLI to use it via per-subprocess config overrides.
        tool_authority_digest = self._attached_tool_authority_digest()
        trusted_context_digest = self._trusted_context_digest()
        mcp_url = (
            self._ensure_mcp_server_running()
            if mcp_url_override is _MCP_URL_UNSET
            else mcp_url_override
        )
        # The concrete Codex feature contract denies native delegation for
        # every capability.  Keep the event-stream guard armed for planner,
        # read-only, and workspace-write turns alike, including MCP-backed
        # invocations.  CLI feature flags are defense in depth; observed
        # delegation must still fail closed if an upstream default regresses.
        forbid_hidden_delegation = self.features.native_delegation == "none"
        subprocess_env = self._confined_subprocess_env(
            auth_path,
            auth_payload=auth_payload,
        )
        # Start from a deterministic, least-privilege feature surface. The
        # empty MCP table clears lower-precedence project/user configuration;
        # the Crucible endpoint below is then the only server added back.
        codex_executable = self._cli_pin.path if self._cli_pin is not None else "codex"
        codex_command: list[str] = [
            codex_executable,
            "--disable",
            "apps",
            "--disable",
            "browser_use",
            "--disable",
            "computer_use",
            "--disable",
            "hooks",
            "--disable",
            "multi_agent",
            "--disable",
            "multi_agent_v2",
            "--disable",
            "plugins",
            "--disable",
            "plugin_sharing",
            "--disable",
            "skill_mcp_dependency_install",
            "--disable",
            "tool_suggest",
            "--disable",
            "workspace_dependencies",
            "-c",
            "mcp_servers={}",
        ]
        if self._trusted_project_instructions:
            # Trusted first-party read-only roles deliberately admit the
            # reviewed root-to-cwd AGENTS chain. Pin the byte ceiling rather
            # than inheriting a project/user setting.
            codex_command.extend(["-c", "project_doc_max_bytes=262144"])
        else:
            # Confined invocations receive their complete role instructions
            # through Invocation.system. Do not let root or nested AGENTS.md
            # files silently append ambient authority.
            codex_command.extend(["-c", "project_doc_max_bytes=0"])
        if not self._native_shell:
            codex_command.extend(
                ["--disable", "shell_tool", "--disable", "unified_exec"]
            )
        if mcp_url is not None:
            # ``mcp_servers.<name>.url=<URL>`` — Codex CLI auto-detects
            # streamable_http transport from the URL form. The TOML
            # parser on the CLI side wants quoted strings; wrap the URL
            # in literal double-quotes inside the value.
            codex_command.extend(["-c", f'mcp_servers.crucible.url="{mcp_url}"'])
            if not self._native_shell:
                # Confined roles keep the native approval handler fail-closed.
                # The only exception is this explicitly attached, capability-
                # checked Crucible server: Codex may call its advertised tools
                # without turning each MCP call into an approval request that
                # the fail-closed handler would (correctly) decline. This does
                # not grant filesystem access to Codex's native executors;
                # those stay disabled and the thread sandbox stays read-only.
                codex_command.extend(
                    [
                        "-c",
                        ('mcp_servers.crucible.default_tools_approval_mode="approve"'),
                    ]
                )
        codex_command.append("app-server")

        try:
            if self._cli_pin is not None:
                self._cli_pin.revalidate()
            stdio_client = _with_bounded_stdio_read_limit(
                CodexClient.connect_stdio(
                    cwd=self._cwd,
                    command=codex_command,
                    env=subprocess_env,
                    connect_timeout=60.0,
                    request_timeout=600.0,
                    inactivity_timeout=1800.0,
                )
            )
            # Install before ``async with`` starts the receiver task. A
            # canonical failed ``turn/completed`` can otherwise be consumed
            # and cleaned up before the adapter can preserve its nested error.
            capture_installed = _install_codex_event_capture(
                stdio_client,
                captured_raw_events,
                forbidden_delegation_events=(
                    forbidden_delegation_events if forbid_hidden_delegation else None
                ),
            )
            async with stdio_client as client:
                await client.initialize(_codex_initialize_params())
                client.set_approval_handler(
                    _permissive_auto_approve if self._native_shell else _deny_approval
                )

                sandbox = "workspace-write" if self._native_shell else "read-only"
                cfg = ThreadConfig(
                    model=self.model,
                    base_instructions=invocation.system,
                    cwd=self._cwd,
                    approval_policy="never",
                    sandbox=sandbox,
                )
                handle = None
                if resume_thread_id is not None:
                    # Reconnect to the prior (interrupted) conversation —
                    # full context restored from the on-disk rollout.
                    try:
                        handle = await client.resume_thread(
                            resume_thread_id,
                            overrides=cfg,
                        )
                    except Exception as exc:  # noqa: BLE001
                        if external_resume and _codex_resume_was_rejected(exc):
                            if resume_failed_holder is not None:
                                resume_failed_holder.append(True)
                            return self._result_with_evidence(
                                final_message="",
                                transcript=(),
                                usage=Usage(),
                                stop_reason="stale_conversation",
                                error=StaleConversation(
                                    message=(
                                        "Codex provider rejected the requested "
                                        "conversation handle"
                                    )
                                ),
                                terminal_complete=False,
                                events=events,
                                tool_authority_digest=tool_authority_digest,
                                trusted_context_digest=trusted_context_digest,
                                conversation=_conversation_outcome(
                                    invocation,
                                    opaque_id=None,
                                    state="stale",
                                ),
                            )
                        # A failed resume can mean the receiver task already
                        # died (for example on an oversized stdio line).
                        # Never issue ``thread/start`` on that same client:
                        # without a receiver its pending request can only sit
                        # until the 600-second request timeout.  Let the outer
                        # retry close this client, clear the stale id, and
                        # start a fresh context-seeded thread on a new client.
                        logger.warning(
                            "CodexBackend: resume_thread(%s) failed (%s); "
                            "abandoning this client before fresh fallback",
                            resume_thread_id,
                            type(exc).__name__,
                        )
                        if resume_failed_holder is not None:
                            resume_failed_holder.append(True)
                        raise
                if handle is None:
                    handle = await _start_codex_thread_with_exact_usage(
                        client,
                        cfg,
                    )
                # Capture the thread id BEFORE the turn runs, so a
                # transport drop during chat_once still lets the retry loop
                # resume this exact thread.
                observed_thread_id = getattr(handle, "thread_id", None)
                handle_thread_id = _bounded_capture_identifier(
                    observed_thread_id,
                    max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                )
                expected_resume_thread_id = (
                    _bounded_capture_identifier(
                        resume_thread_id,
                        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                    )
                    if resume_thread_id is not None
                    else None
                )
                handle_identity_complete = handle_thread_id is not None and (
                    resume_thread_id is None
                    or handle_thread_id == expected_resume_thread_id
                )
                if handle_thread_id is not None:
                    if thread_id_holder is not None:
                        thread_id_holder.append(handle_thread_id)
                    provider_conversation_id = handle_thread_id

                # The app-server protocol deliberately accepts forward-
                # compatible, non-empty effort strings.  SDK 0.3.2's static
                # Literal predates current Codex ``max``/``ultra`` values, but its
                # serializer forwards this value verbatim to ``turn/start``.
                turn_kwargs: dict[str, Any] = {"effort": self._effort}
                # Invocation-local schema is the semantic contract. The
                # constructor value is only a compatibility fallback for
                # direct adapter users. Copy before handing it to the SDK so
                # neither a reused backend nor a provider shim can mutate the
                # caller's mapping across turns.
                output_schema = (
                    invocation.response_schema
                    if invocation.response_schema is not None
                    else self._output_schema
                )
                if output_schema is not None:
                    if isinstance(output_schema, dict):
                        (
                            provider_output_schema,
                            removed_unique_items,
                        ) = _without_unique_items_for_codex(output_schema)
                    else:
                        provider_output_schema = deepcopy(output_schema)
                        removed_unique_items = 0
                    if removed_unique_items:
                        if not self._logged_unique_items_compat:
                            logger.warning(
                                "CodexBackend: removed %d unsupported "
                                "uniqueItems keyword(s) from the provider "
                                "output schema; Assay retains the original "
                                "schema for client-side validation",
                                removed_unique_items,
                            )
                            self._logged_unique_items_compat = True
                        else:
                            logger.debug(
                                "CodexBackend: removed %d unsupported "
                                "uniqueItems keyword(s) from output schema",
                                removed_unique_items,
                            )
                    turn_kwargs["output_schema"] = provider_output_schema
                turn_overrides = TurnOverrides(**turn_kwargs)
                # A recovery turn (resumed thread OR seeded-fresh fallback)
                # gets the task-restating nudge so the model always has the
                # task; the first attempt gets the prompt verbatim.
                turn_input = (
                    _CODEX_RESUME_NUDGE.format(task=invocation.user)
                    if recovery
                    else invocation.user
                )

                # chat_once (buffered) over chat (streamed): we need
                # access to ChatResult.raw_events to harvest the
                # ``thread/tokenUsage/updated`` notification for cost.
                # The chat() iterator cleans up the session before it
                # returns, so raw_events is unreachable from outside.
                # The streaming view wasn't operator-facing anyway — the
                # iterator body only collected final assistant text.
                chat_result = await handle.chat_once(
                    turn_input, turn_overrides=turn_overrides
                )
                final_message = chat_result.final_text
                if final_message:
                    transcript.append(Message(role="assistant", content=final_message))
                observed_turn_id = getattr(
                    chat_result,
                    "turn_id",
                    _CHAT_RESULT_IDENTITY_OMITTED,
                )
                turn_id_present = (
                    observed_turn_id is not _CHAT_RESULT_IDENTITY_OMITTED
                )
                current_turn_id = (
                    _bounded_capture_identifier(
                        observed_turn_id,
                        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                    )
                    if turn_id_present
                    else None
                )
                raw_result_thread_id = getattr(
                    chat_result,
                    "thread_id",
                    _CHAT_RESULT_IDENTITY_OMITTED,
                )
                thread_id_present = (
                    raw_result_thread_id is not _CHAT_RESULT_IDENTITY_OMITTED
                )
                result_thread_id = (
                    _bounded_capture_identifier(
                        raw_result_thread_id,
                        max_chars=_MAX_CAPTURED_IDENTIFIER_CHARS,
                    )
                    if thread_id_present
                    else None
                )
                result_identity_complete = handle_identity_complete
                if thread_id_present and (
                    result_thread_id is None or result_thread_id != handle_thread_id
                ):
                    result_identity_complete = False
                if turn_id_present and current_turn_id is None:
                    result_identity_complete = False
                # The pre-installed capture retains bounded identifiers,
                # replay baselines, every response boundary up to its cap,
                # and an explicit overflow marker. A non-SDK test double may
                # not expose the dispatcher; production SDK shape drift raises
                # before starting the turn rather than falling back unbounded.
                evidence_events = (
                    captured_raw_events if capture_installed else chat_result.raw_events
                )
                if forbid_hidden_delegation and not capture_installed:
                    delegation_events = getattr(
                        chat_result,
                        "raw_events",
                        evidence_events,
                    )
                    if not isinstance(delegation_events, list):
                        delegation_events = evidence_events
                    for event in delegation_events:
                        forbidden_delegation = _codex_hidden_delegation_signal(event)
                        if forbidden_delegation is not None:
                            _retain_codex_delegation_signal(
                                forbidden_delegation_events,
                                forbidden_delegation,
                            )
                # Only a fully read and bounded ChatResult can participate in
                # recovery from a later client-context shutdown failure.
                chat_once_completed = True

            # Qualify only after the client context has closed successfully.
            # ``__aexit__`` may append final terminal evidence or itself fail;
            # harvesting inside the context would publish a stale pre-exit
            # attribution and duplicate physical telemetry during recovery.
            resumed_delegation = bool(
                resume_thread_id is not None and forbidden_delegation_events
            )
            usage_turn_id = current_turn_id
            usage_identity_complete = result_identity_complete
            if forbidden_delegation_events:
                usage_turn_id = _qualified_resumed_delegation_turn_id(
                    forbidden_delegation_events,
                    evidence_events,
                    expected_thread_id=handle_thread_id,
                    result_thread_id=result_thread_id,
                    result_turn_id=current_turn_id,
                    result_identity_complete=result_identity_complete,
                    require_signal_thread_id=resumed_delegation,
                )
                usage_identity_complete = (
                    usage_turn_id is not None
                    and _completed_chat_identity_matches_final_evidence(
                        evidence_events,
                        current_turn_id=usage_turn_id,
                        expected_thread_id=handle_thread_id,
                        identity_complete=result_identity_complete,
                        resumed=resume_thread_id is not None,
                    )
                )
            else:
                usage_identity_complete = (
                    _completed_chat_identity_matches_final_evidence(
                        evidence_events,
                        current_turn_id=current_turn_id,
                        expected_thread_id=handle_thread_id,
                        identity_complete=result_identity_complete,
                        resumed=resume_thread_id is not None,
                    )
                )
            task = asyncio.current_task()
            interrupted = task is not None and task.cancelling() > 0
            usage_evidence = _harvest_usage_evidence(
                evidence_events,
                model=self.model,
                current_turn_id=usage_turn_id,
                allow_last_update_fallback=(resume_thread_id is None),
                expected_thread_id=handle_thread_id,
                identity_complete=usage_identity_complete,
                force_incomplete=(interrupted or bool(forbidden_delegation_events)),
                require_history_free_lower_bound=(
                    resume_thread_id is not None or bool(forbidden_delegation_events)
                ),
            )
            usage = usage_evidence.usage
            pricing_evidence = usage_evidence.pricing_evidence
            usage_complete = usage_evidence.token_usage_complete
            cost_complete = usage_evidence.cost_complete
            cost_upper_bound = usage_evidence.cost_upper_bound
            if any(
                event.get("method") == _TOKEN_USAGE_METHOD for event in evidence_events
            ):
                _append_event(
                    events,
                    "usage",
                    name=(
                        "complete"
                        if usage_complete and cost_complete
                        else (
                            _COST_UPPER_BOUND_EVENT_NAME
                            if cost_upper_bound
                            else "partial"
                        )
                    ),
                )
            terminal_complete = _provider_terminal_complete(
                evidence_events,
                current_turn_id=usage_turn_id,
                expected_thread_id=handle_thread_id,
            )
            if not usage_identity_complete or forbidden_delegation_events or interrupted:
                terminal_complete = False
            if usage_identity_complete and usage_turn_id is not None:
                provider_request_id = usage_turn_id
        except BaseException as exc:  # noqa: BLE001
            cancelled = _is_external_cancellation(_unwrap_exception_group(exc))
            if not isinstance(exc, Exception) and not cancelled:
                raise
            task = asyncio.current_task()
            interrupted = cancelled or (task is not None and task.cancelling() > 0)
            # ``chat_once`` discards its session and raw events before raising
            # a protocol error. Recover the provider's qualified terminal and
            # any usage notification from the pre-installed event capture.
            # Discard every pre-exit attribution first: ``__aexit__`` can add
            # a conflicting terminal and then raise after the normal body has
            # already populated these locals. Only the final retained ledger
            # may authorize the result fields below.
            usage = Usage()
            pricing_evidence = UsagePricingEvidence(segments=(), complete=False)
            usage_complete = False
            cost_complete = False
            cost_upper_bound = False
            terminal_complete = False
            provider_request_id = None
            # A qualification failure after appending the provisional usage
            # marker must not leave a duplicate or stale trace claim. Source
            # physical telemetry remains append-only; only this derived result
            # projection is rebuilt from the final ledger.
            events[:] = [event for event in events if event.kind != "usage"]
            terminal_failure = _codex_terminal_failure(captured_raw_events)
            if captured_raw_events or chat_once_completed:
                resumed_delegation = bool(
                    resume_thread_id is not None and forbidden_delegation_events
                )
                if forbidden_delegation_events:
                    current_usage_turn_id = _qualified_resumed_delegation_turn_id(
                        forbidden_delegation_events,
                        captured_raw_events,
                        expected_thread_id=handle_thread_id,
                        result_thread_id=(
                            result_thread_id if chat_once_completed else None
                        ),
                        result_turn_id=(
                            current_turn_id if chat_once_completed else None
                        ),
                        result_identity_complete=(
                            result_identity_complete
                            if chat_once_completed
                            else handle_identity_complete
                        ),
                        require_signal_thread_id=resumed_delegation,
                    )
                    usage_identity_complete = current_usage_turn_id is not None
                elif chat_once_completed:
                    usage_identity_complete = (
                        _completed_chat_identity_matches_final_evidence(
                            captured_raw_events,
                            current_turn_id=current_turn_id,
                            expected_thread_id=handle_thread_id,
                            identity_complete=result_identity_complete,
                            resumed=resume_thread_id is not None,
                        )
                    )
                    current_usage_turn_id = (
                        current_turn_id if usage_identity_complete else None
                    )
                    if not usage_identity_complete:
                        terminal_failure = None
                else:
                    terminal_turn_ids, terminal_turn_shapes_complete = (
                        _terminal_turn_identity_evidence(captured_raw_events)
                    )
                    terminal_thread_ids, terminal_thread_shapes_complete = (
                        _terminal_thread_identity_evidence(captured_raw_events)
                    )
                    unique_terminal_turn_ids = set(terminal_turn_ids)
                    usage_identity_complete = (
                        handle_identity_complete
                        and terminal_turn_shapes_complete
                        and terminal_thread_shapes_complete
                        and len(unique_terminal_turn_ids) <= 1
                        and all(
                            thread_id == handle_thread_id
                            for thread_id in terminal_thread_ids
                        )
                    )
                    current_usage_turn_id = (
                        next(iter(unique_terminal_turn_ids))
                        if usage_identity_complete and unique_terminal_turn_ids
                        else None
                    )
                    if interrupted and current_usage_turn_id is None:
                        # Unlike restored cumulative usage, raw response rows
                        # are not replayed on resume. A unique qualified raw
                        # turn can select the interrupted current prefix; the
                        # final-ledger identity check below still rejects all
                        # conflicting/malformed IDs and replay boundaries.
                        raw_turn_ids = {
                            event["params"].get("turnId")
                            for event in captured_raw_events
                            if event.get("method") == _RAW_RESPONSE_COMPLETED_METHOD
                            and isinstance(event.get("params"), dict)
                            and isinstance(event["params"].get("turnId"), str)
                        }
                        if len(raw_turn_ids) == 1:
                            current_usage_turn_id = next(iter(raw_turn_ids))
                usage_identity_complete = (
                    usage_identity_complete
                    and _completed_chat_identity_matches_final_evidence(
                        captured_raw_events,
                        current_turn_id=current_usage_turn_id,
                        expected_thread_id=handle_thread_id,
                        identity_complete=(
                            result_identity_complete
                            if chat_once_completed
                            else handle_identity_complete
                        ),
                        resumed=resume_thread_id is not None,
                    )
                )
                if not usage_identity_complete:
                    current_usage_turn_id = None
                    terminal_failure = None
                usage_evidence = _harvest_usage_evidence(
                    captured_raw_events,
                    model=self.model,
                    current_turn_id=current_usage_turn_id,
                    allow_last_update_fallback=(resume_thread_id is None),
                    expected_thread_id=handle_thread_id,
                    identity_complete=usage_identity_complete,
                    force_incomplete=(interrupted or bool(forbidden_delegation_events)),
                    require_history_free_lower_bound=(
                        # This mode excludes restored cumulative history. A
                        # completed non-delegation turn needs the same rule
                        # when only the subsequent client shutdown failed.
                        bool(forbidden_delegation_events)
                        or resume_thread_id is not None
                    ),
                )
                if usage_identity_complete:
                    usage = usage_evidence.usage
                    pricing_evidence = usage_evidence.pricing_evidence
                    usage_complete = usage_evidence.token_usage_complete
                    cost_complete = usage_evidence.cost_complete
                    cost_upper_bound = usage_evidence.cost_upper_bound
                    if current_usage_turn_id is not None:
                        provider_request_id = current_usage_turn_id
                terminal_complete = _provider_terminal_complete(
                    captured_raw_events,
                    current_turn_id=current_usage_turn_id,
                    expected_thread_id=handle_thread_id,
                )
                if not usage_identity_complete:
                    terminal_complete = False
                if any(
                    event.get("method") == _TOKEN_USAGE_METHOD
                    for event in captured_raw_events
                ):
                    _append_event(
                        events,
                        "usage",
                        name=(
                            "complete"
                            if usage_complete and cost_complete
                            else (
                                _COST_UPPER_BOUND_EVENT_NAME
                                if cost_upper_bound
                                else "partial"
                            )
                        ),
                    )
                if forbidden_delegation_events:
                    terminal_complete = False
            if interrupted:
                terminal_complete = False
            if cancelled:
                # Preserve cancellation itself, independently of the outer
                # policy. The sink is diagnostic and read only after joining;
                # neither this prefix nor a terminal seen during shutdown can
                # prove complete accounting for the interrupted attempt.
                err = ExternalCancellation(
                    cause=type(exc).__name__, message=str(exc), retriable=False
                )
            elif forbidden_delegation_events:
                _append_event(
                    events,
                    "limit",
                    name="delegation_authority",
                )
                err = TransportError(
                    http_status=None,
                    message=(
                        f"{_HIDDEN_DELEGATION_FAILURE_MESSAGE}: "
                        f"{forbidden_delegation_events[0].classification}"
                    ),
                    retriable=False,
                )
            elif terminal_failure is not None:
                err = _classify_codex_terminal_failure(terminal_failure)
            else:
                err = self._classify_exception(exc, policy)
            reraise = cancelled or err is None
            if err is None:
                err = ExternalCancellation(
                    cause=type(exc).__name__, message=str(exc), retriable=False
                )
            result_stop_reason = (
                "cancelled"
                if getattr(err, "kind", None) == "cancellation"
                else (
                    "delegation_forbidden" if forbidden_delegation_events else "error"
                )
            )
            if getattr(err, "kind", None) == "cancellation":
                _append_event(events, "cancellation", name="external")
            failed_result = self._result_with_evidence(
                final_message=final_message,
                transcript=tuple(transcript),
                usage=usage,
                pricing_evidence=pricing_evidence,
                stop_reason=result_stop_reason,
                error=err,
                token_usage_complete=usage_complete,
                cost_complete=cost_complete,
                terminal_complete=terminal_complete,
                events=events,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                trusted_context_digest=trusted_context_digest,
                conversation=_conversation_outcome(
                    invocation,
                    opaque_id=provider_conversation_id,
                ),
            )
            if interrupted and interrupted_evidence is not None:
                interrupted_evidence[:] = [failed_result]
            if reraise:
                raise
            return failed_result

        def result_with_evidence(
            *,
            error: Any = None,
            result_stop_reason: str = stop_reason,
            result_final_message: str = final_message,
            result_transcript: tuple[Message, ...] = tuple(transcript),
        ) -> InvocationResult:
            return self._result_with_evidence(
                final_message=result_final_message,
                transcript=result_transcript,
                usage=usage,
                pricing_evidence=pricing_evidence,
                stop_reason=result_stop_reason,
                error=error,
                token_usage_complete=usage_complete,
                cost_complete=cost_complete,
                terminal_complete=terminal_complete,
                events=events,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                trusted_context_digest=trusted_context_digest,
                conversation=_conversation_outcome(
                    invocation,
                    opaque_id=provider_conversation_id,
                ),
            )

        if forbidden_delegation_events:
            _append_event(
                events,
                "limit",
                name="delegation_authority",
            )
            return result_with_evidence(
                error=TransportError(
                    http_status=None,
                    message=(
                        f"{_HIDDEN_DELEGATION_FAILURE_MESSAGE}: "
                        f"{forbidden_delegation_events[0].classification}"
                    ),
                    retriable=False,
                ),
                result_stop_reason="delegation_forbidden",
                result_final_message="",
            )

        # Budget gate. ``usage.cost_usd`` is computed from the bounded SDK
        # token-usage sequence via ``_harvest_usage_evidence`` and the local
        # pricing table; the evidence object separately carries exact versus
        # conservative-upper-bound authority.
        budget_limit = (
            invocation.budget_usd
            if budget_usd_override is None
            else budget_usd_override
        )
        if budget_limit is not None and usage.cost_usd > budget_limit:
            _append_event(events, "limit", name="budget_usd")
            return result_with_evidence(
                result_stop_reason="budget_exhausted",
                error=BudgetExhausted(
                    spent_usd=usage.cost_usd,
                    limit_usd=budget_limit,
                    message=(
                        f"Invocation spent ${usage.cost_usd:.4f} "
                        f"exceeding remaining budget ${budget_limit:.4f}"
                    ),
                ),
            )

        turns_used = sum(1 for m in transcript if m.role == "assistant")
        if _classify_max_turns(stop_reason, turns_used, invocation.max_turns):
            if policy.on_max_turns == "raise":
                _append_event(events, "limit", name="max_turns")
                return result_with_evidence(
                    result_stop_reason="max_turns",
                    error=MaxTurnsExceeded(
                        turns_used=turns_used,
                        turns_limit=invocation.max_turns,
                        message=f"hit max_turns={invocation.max_turns}",
                    ),
                )
            logger.warning(
                "CodexBackend hit max_turns=%d but policy=continue_with_warning",
                invocation.max_turns,
            )

        # Stalled / empty completion guard — mirrors ClaudeBackend. A
        # clean chat_once that yields empty ``final_text`` and zero
        # output tokens is a content-free stall: returned as-is it looks
        # like success, the retry loop never re-tries it, and downstream
        # JSON-parsing consumers (assay.review / assay.plan) fail the
        # gate on a parse_failure. Convert to a retriable TransportError
        # so the backend's own retry budget gets a fresh attempt. Gated
        # on no tools AND no attached toolset: a tool-driven turn can
        # legitimately end with empty assistant text after a tool call.
        if (
            not final_message
            and usage.output_tokens == 0
            and not invocation.tools
            and self._mcp_toolset is None
        ):
            logger.warning(
                "CodexBackend: empty completion (stop_reason=%s, "
                "0 output tokens, empty final_text); treating as a "
                "retriable transport stall",
                stop_reason,
            )
            return result_with_evidence(
                result_final_message="",
                error=TransportError(
                    http_status=None,
                    message=(
                        f"empty completion (stop_reason={stop_reason}, "
                        "0 output tokens) — codex returned a clean but "
                        "content-free turn"
                    ),
                    retriable=True,
                ),
            )

        return result_with_evidence()

    # ------------------------------------------------------------------
    # Exception → BackendError classification
    # ------------------------------------------------------------------

    def _classify_exception(
        self,
        exc: BaseException,
        policy: BackendPolicy,
    ) -> ClassifiedError | None:
        """Map a Python exception to an assay BackendError subtype.

        Mirrors ``ClaudeBackend._classify_exception``'s policy via the
        shared ``backends._classify.TERMINAL_EXCEPTION_NAMES`` abort-list
        so the hybrid swebench surface (BackendPool routing through
        both backends) sees consistent classification. See the Claude
        version's docstring for the full per-branch rationale; the
        order here matches exactly: cancellation → refusal →
        abort-list → status_code → default-retry.

        Codex-specific differences from Claude:

        - No bare-Exception ``Command failed`` carve-out (that shape is
          bundled-claude-CLI specific with its own documented
          cost-amplification + idempotency reasoning).
        - Broken pipe, transport, process, timeout, connection, and OS errors
          reach the default-retry path unless they carry a terminal status.

        Returns ``None`` only when the caller explicitly opted out of
        classification (``policy.on_cancellation == "raise"`` through
        ``_is_external_cancellation``). Every other input returns a
        BackendError instance.
        """
        inner = _unwrap_exception_group(exc)

        if _is_external_cancellation(inner):
            if policy.on_cancellation == "raise":
                return None
            return ExternalCancellation(
                cause=type(inner).__name__,
                message=str(inner),
                retriable=True,
            )

        cls_name = type(inner).__name__
        msg = str(inner)

        if "refusal" in msg.lower() or cls_name in (
            "ContentFilterError",
            "RefusalError",
        ):
            return ContentRejected(message=msg, retriable=False)

        if cls_name == "CodexProtocolError":
            # Protocol failures are deterministic request/terminal outcomes,
            # not blind transport drops. In particular, SDK 0.3.2's
            # "turn completed but no final assistant message" has already
            # performed a thread/read fallback; replaying the same turn cannot
            # recover it. Preserve bounded JSON-RPC diagnostics when present.
            details: list[str] = []
            code = getattr(inner, "code", None)
            if type(code) is int:
                details.append(f"code={code}")
            data = getattr(inner, "data", None)
            if isinstance(data, dict):
                for key in ("type", "status", "code", "message"):
                    value = data.get(key)
                    if isinstance(value, (str, int)) and not isinstance(
                        value,
                        bool,
                    ):
                        details.append(f"{key}={str(value)[:500]}")
            qualifier = f" ({', '.join(details)})" if details else ""
            return TransportError(
                http_status=None,
                message=f"{cls_name}{qualifier}: {msg}",
                retriable=False,
            )

        # Abort-list — terminal class names that retry can't recover.
        # Shared with ClaudeBackend via ``backends._classify`` so the
        # hybrid swebench surface (codex + claude through BackendPool)
        # sees consistent classification. Adding a new terminal class
        # only requires editing TERMINAL_EXCEPTION_NAMES; both backends
        # pick it up.
        if cls_name in TERMINAL_EXCEPTION_NAMES:
            return TransportError(
                http_status=None,
                message=f"{cls_name}: {msg}",
                retriable=False,
            )

        if hasattr(inner, "status_code"):
            status = int(getattr(inner, "status_code", 0) or 0)
            return TransportError(
                http_status=status,
                message=f"{cls_name}: {msg}",
                retriable=status in RETRIABLE_HTTP_STATUSES,
            )

        # Default: retry. Unknown Codex SDK exception shapes (BrokenPipeError,
        # OSError, ProcessError, TimeoutError, CodexTransportError, and any
        # future shape the SDK introduces) get a retriable
        # TransportError instead of leaking as a raw Python exception
        # to the workflow layer. The intentional terminal cases
        # (abort-list, non-retriable HTTP 4xx via status_code) are
        # caught above this fall-through.
        return TransportError(
            http_status=None,
            message=f"{cls_name}: {msg}",
            retriable=True,
        )


async def _permissive_auto_approve(request: Any) -> str:
    """Approve requests for an explicitly native-shell-enabled role."""
    return "accept"


async def _deny_approval(request: Any) -> str:
    """Deny any tool that escaped a confined role's explicit catalogue."""
    return "decline"


__all__ = ["CodexBackend"]
