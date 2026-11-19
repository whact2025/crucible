# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Dedicated, project-locked DeepSeek native Responses backend.

The DSK-00a contract established the exact no-fallback construction. DSK-00b
and DSK-02 remain promotion gates, so workflow routing admits only an explicit
diagnostic lane bound to the captured worker identity. The parent process
never imports the OpenAI SDK. Every physical response runs in a fresh
``backends/deepseek_worker`` subprocess whose lock, source, installed SBOM,
interpreter, and client policy are fingerprinted before dispatch.

The only transport identity is ``POST https://api.deepseek.com/responses``
with request alias ``deepseek-flash``, high reasoning, streaming, and
client-executed function tools.  The vendor moves that alias between Flash
snapshots without an id change and exposes no dated id, so the returned
alias is attested and the served snapshot is not.  ``DEEPSEEK_API_KEY`` is
read only immediately before subprocess dispatch and crosses the boundary on
standard input.  It is never retained in routing, argv, the worker
environment, traces, or debug records. Response schemas use the canonical
Assay prompt ABI, and diagnostic writers execute only through an
invocation-scoped isolated mutation host.

DeepSeek bills by time of day and the price table has no time dimension, so
recorded cost uses the vendor's published peak rate: it equals the charge at
peak and is twice the charge off-peak.  ``cost_complete`` therefore attests
complete token counters priced at that ceiling, not the amount billed.
"""

from __future__ import annotations

import datetime
import fcntl
import functools
import hashlib
import json
import math
import os
import re
import selectors
import signal
import socket
import stat
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendPolicy,
    BudgetExhausted,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)

from ._sdk_debug import SdkDebugSink
from .pricing import compute_cost, pricing_for_model


_PROTOCOL = "crucible.deepseek-worker.v1"
_MODEL = "deepseek-flash"
_BASE_URL = "https://api.deepseek.com"
_ENDPOINT = "https://api.deepseek.com/responses"
_OPENAI_VERSION = "2.53.0"
_OPENAI_WHEEL_SHA256 = (
    "c694ffc747a3c4d1663ef2b07b811315a476164ee5efa3a993967349ebca7618"
)
_OPENAI_SOURCE_SHA256 = (
    "baf5802ad08980e1d9d561e1b996e800c8bcd14af5847c6d0e7a5cc59e4d4116"
)
_WORKER_LOCK_SHA256 = (
    "c2622cb4b213068c3db4e1e39abc419a975df5a4e3113b5b2ab4a9625d3137e3"
)
_LOCKED_LINUX_COMPONENT_VERSIONS = (
    ("annotated-types", "0.8.0"),
    ("anyio", "4.14.2"),
    ("certifi", "2026.7.22"),
    ("crucible-deepseek-worker", "0.1.0"),
    ("distro", "1.9.0"),
    ("h11", "0.16.0"),
    ("httpcore", "1.0.9"),
    ("httpx", "0.28.1"),
    ("idna", "3.18"),
    ("jiter", "0.16.0"),
    ("openai", _OPENAI_VERSION),
    ("pydantic", "2.13.4"),
    ("pydantic-core", "2.46.4"),
    ("sniffio", "1.3.1"),
    ("tqdm", "4.70.0"),
    ("typing-extensions", "4.16.0"),
    ("typing-inspection", "0.4.3"),
)
_LOCKED_ALL_COMPONENT_VERSIONS = tuple(
    sorted((*_LOCKED_LINUX_COMPONENT_VERSIONS, ("colorama", "0.4.6")))
)
_MAX_OUTPUT_TOKENS = 32_000
_MAX_TOOL_ROUNDS = 64
_DEFAULT_TIMEOUT_SECONDS = 600.0
_MAX_STDOUT_BYTES = 24 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_STOP_GRACE_SECONDS = 2.0
_STREAM_DRAIN_GRACE_SECONDS = 0.5
_STREAM_CHUNK_BYTES = 64 * 1024
_PROBE_TIMEOUT_SECONDS = 15.0
_SUPERVISOR_CONTROL_MAX_BYTES = 4096
_SUPERVISOR_SCHEMA = "crucible.process-supervisor.v1"
_WORKER_PROJECT = Path(__file__).with_name("deepseek_worker")
_DEFAULT_WORKER_PYTHON = _WORKER_PROJECT / ".venv" / "bin" / "python"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_TERMINAL_TYPES = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)
_DEFAULT_POLICY = BackendPolicy(
    retry_budget=1,
    timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    on_cancellation="convert_to_backend_error",
    on_max_turns="continue_with_warning",
)
DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA = (
    "crucible.deepseek-project-locked-runtime-identity.v1"
)
DEEPSEEK_OPENAI_DISTRIBUTION_HASHES = {
    "wheel_sha256": f"sha256:{_OPENAI_WHEEL_SHA256}",
    "source_sha256": f"sha256:{_OPENAI_SOURCE_SHA256}",
}


class DeepSeekSDKUnavailable(RuntimeError):
    """The exact project-owned DeepSeek worker is not installed."""


class DeepSeekSDKQualificationError(RuntimeError):
    """Worker/runtime/provider evidence differs from the frozen contract."""


class DeepSeekWorkerInvocationError(RuntimeError):
    """The isolated worker reported a sanitized invocation failure."""

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type
        super().__init__(f"DeepSeek worker failed with {error_type}")


class DeepSeekWorkerTimeout(TimeoutError):
    """The local worker deadline elapsed after process settlement."""


@dataclass(frozen=True, slots=True)
class DeepSeekSBOMComponent:
    """One prompt- and credential-free installed worker distribution."""

    name: str
    version: str
    record_sha256: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or self.name != self.name.lower()
            or not self.version
            or _DIGEST_RE.fullmatch(self.record_sha256) is None
        ):
            raise ValueError("DeepSeek SBOM component is malformed")

    def to_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "record_sha256": self.record_sha256,
        }


@dataclass(frozen=True, slots=True)
class DeepSeekProjectLockedRuntimeIdentity:
    """Closed public projection of the worker execution/SBOM authority."""

    logical_worker_python: str
    resolved_worker_python: str
    resolved_worker_python_sha256: str
    lock_source_fingerprint: str
    installed_tree_fingerprint: str
    installed_sbom_fingerprint: str
    runtime_fingerprint: str
    openai_version: str
    sbom: tuple[DeepSeekSBOMComponent, ...]

    def __post_init__(self) -> None:
        if (
            not Path(self.logical_worker_python).is_absolute()
            or not Path(self.resolved_worker_python).is_absolute()
        ):
            raise ValueError("DeepSeek worker identity paths must be absolute")
        for value in (
            self.resolved_worker_python_sha256,
            self.lock_source_fingerprint,
            self.installed_tree_fingerprint,
            self.installed_sbom_fingerprint,
            self.runtime_fingerprint,
        ):
            if _DIGEST_RE.fullmatch(value) is None:
                raise ValueError("DeepSeek worker identity digest is malformed")
        if self.openai_version != _OPENAI_VERSION:
            raise ValueError("DeepSeek worker OpenAI SDK version is not exact")
        if tuple(
            (component.name, component.version) for component in self.sbom
        ) != _LOCKED_LINUX_COMPONENT_VERSIONS:
            raise ValueError("DeepSeek worker SBOM differs from frozen closure")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
            "worker_protocol": _PROTOCOL,
            "logical_worker_python": self.logical_worker_python,
            "resolved_worker_python": self.resolved_worker_python,
            "resolved_worker_python_sha256": self.resolved_worker_python_sha256,
            "lock_source_fingerprint": self.lock_source_fingerprint,
            "installed_tree_fingerprint": self.installed_tree_fingerprint,
            "installed_sbom_fingerprint": self.installed_sbom_fingerprint,
            "runtime_fingerprint": self.runtime_fingerprint,
            "openai_version": self.openai_version,
            "openai_distribution_hashes": dict(DEEPSEEK_OPENAI_DISTRIBUTION_HASHES),
            "client_policy": {
                "base_url": _BASE_URL,
                "responses_endpoint": _ENDPOINT,
                "max_retries": 0,
                "trust_env": False,
                "follow_redirects": False,
            },
            "sbom": [component.to_mapping() for component in self.sbom],
        }


@dataclass(frozen=True, slots=True)
class DeepSeekResponseEvidence:
    """Prompt-free terminal/accounting evidence for one physical response."""

    response_ordinal: int
    response_id: str
    served_model: str
    system_fingerprint: str | None
    terminal_type: str
    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None
    reasoning_output_tokens: int | None
    cost_usd: float | None
    usage_complete: bool

    def __post_init__(self) -> None:
        if type(self.response_ordinal) is not int or self.response_ordinal < 0:
            raise ValueError("DeepSeek response ordinal is invalid")
        _bounded_identity(self.response_id, field="response id", maximum=512)
        _bounded_identity(self.served_model, field="served model", maximum=256)
        if self.system_fingerprint is not None:
            _bounded_identity(
                self.system_fingerprint,
                field="system fingerprint",
                maximum=512,
            )
        if self.terminal_type not in _TERMINAL_TYPES:
            raise ValueError("DeepSeek response terminal type is invalid")
        for name in (
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_output_tokens",
        ):
            value = getattr(self, name)
            if value is not None:
                _exact_count(value, field=name)
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            raise ValueError("DeepSeek response cache count exceeds input")
        if (
            self.output_tokens is not None
            and self.reasoning_output_tokens is not None
            and self.reasoning_output_tokens > self.output_tokens
        ):
            raise ValueError("DeepSeek response reasoning count exceeds output")
        if self.cost_usd is not None and (
            not isinstance(self.cost_usd, (int, float))
            or not math.isfinite(self.cost_usd)
            or self.cost_usd < 0
        ):
            raise ValueError("DeepSeek response cost is invalid")
        if type(self.usage_complete) is not bool:
            raise TypeError("DeepSeek response usage_complete must be a bool")
        if self.usage_complete and (
            self.input_tokens is None
            or self.cached_input_tokens is None
            or self.output_tokens is None
            or self.reasoning_output_tokens is None
            or self.cost_usd is None
        ):
            raise ValueError("complete DeepSeek response evidence lacks counters")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "response_ordinal": self.response_ordinal,
            "response_id": self.response_id,
            "served_model": self.served_model,
            "system_fingerprint": self.system_fingerprint,
            "terminal_type": self.terminal_type,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "cost_usd": self.cost_usd,
            "usage_complete": self.usage_complete,
        }


@dataclass(frozen=True, slots=True)
class DeepSeekInvocationEvidence:
    """Exact runtime-bound physical responses from one logical invocation."""

    worker_runtime_fingerprint: str
    logical_result_complete: bool
    responses: tuple[DeepSeekResponseEvidence, ...]

    def __post_init__(self) -> None:
        if _DIGEST_RE.fullmatch(self.worker_runtime_fingerprint) is None:
            raise ValueError("DeepSeek invocation runtime fingerprint is invalid")
        if type(self.logical_result_complete) is not bool:
            raise TypeError("DeepSeek logical_result_complete must be a bool")
        if any(
            response.response_ordinal != ordinal
            for ordinal, response in enumerate(self.responses)
        ):
            raise ValueError("DeepSeek response evidence ordinals are not contiguous")
        response_ids = [response.response_id for response in self.responses]
        if len(response_ids) != len(set(response_ids)):
            raise ValueError("DeepSeek physical response IDs are not unique")

    @property
    def complete(self) -> bool:
        return self.logical_result_complete and bool(self.responses) and all(
            response.served_model == _MODEL
            and response.terminal_type == "response.completed"
            and response.usage_complete
            for response in self.responses
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": "crucible.deepseek-invocation-evidence.v1",
            "worker_runtime_fingerprint": self.worker_runtime_fingerprint,
            "logical_result_complete": self.logical_result_complete,
            "responses": [response.to_mapping() for response in self.responses],
            "complete": self.complete,
        }


def _sha256_file(path: Path, *, label: str) -> str:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DeepSeekSDKUnavailable(
            f"project-locked DeepSeek worker file is unavailable: {label}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise DeepSeekSDKQualificationError(
            f"project-locked DeepSeek worker file is not single-link regular: {label}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _hash_tree_into(digest: Any, root: Path, *, label: str) -> None:
    """Hash every installed entry and reject links/devices/FIFOs."""

    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise DeepSeekSDKQualificationError(
            f"project-locked DeepSeek worker tree is unavailable: {label}"
        ) from exc
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise DeepSeekSDKQualificationError(
            f"project-locked DeepSeek worker tree is not a directory: {label}"
        )
    try:
        paths = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for path in paths:
            relative = path.relative_to(root).as_posix()
            entry_label = f"{label}/{relative}"
            metadata = path.lstat()
            if stat.S_ISDIR(metadata.st_mode):
                digest.update(b"D\0")
                digest.update(entry_label.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
                digest.update(b"\0")
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                digest.update(b"F\0")
                digest.update(entry_label.encode("utf-8"))
                digest.update(b"\0")
                digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
                digest.update(b"\0")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                digest.update(b"\0")
            else:
                raise DeepSeekSDKQualificationError(
                    "project-locked DeepSeek worker tree contains an "
                    f"unsupported entry: {entry_label}"
                )
    except OSError as exc:
        raise DeepSeekSDKQualificationError(
            f"project-locked DeepSeek worker tree changed while hashing: {label}"
        ) from exc


def _installed_tree_fingerprint() -> str:
    """Bind venv metadata and all importable installed bytes."""

    digest = hashlib.sha256()
    for relative in (".python-version", ".venv/pyvenv.cfg"):
        path = _WORKER_PROJECT / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path, label=relative).encode("ascii"))
        digest.update(b"\0")
    site_packages = tuple(
        (_WORKER_PROJECT / ".venv" / "lib").glob("python*/site-packages")
    )
    if len(site_packages) != 1:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker must contain exactly one "
            "site-packages tree"
        )
    editable_bindings = tuple(
        site_packages[0].glob("_editable_impl_crucible_deepseek_worker.pth")
    )
    expected_binding = str(
        (_WORKER_PROJECT / "src").resolve(strict=True)
    ).encode("utf-8")
    try:
        binding = editable_bindings[0].read_bytes()
    except (IndexError, OSError) as exc:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker editable binding is unavailable"
        ) from exc
    if len(editable_bindings) != 1 or binding.rstrip(b"\r\n") != expected_binding:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker editable binding escaped its source"
        )
    _hash_tree_into(digest, site_packages[0], label="site-packages")
    # The editable worker distribution executes this source tree through its
    # project-owned .pth file. Bind every bytecode/cache entry as well as the
    # reviewed source so an untracked .pyc cannot outrank a hashed .py file.
    _hash_tree_into(
        digest,
        _WORKER_PROJECT / "src" / "crucible_deepseek_worker",
        label="editable-worker-source",
    )
    return "sha256:" + digest.hexdigest()


def _lock_source_fingerprint() -> str:
    _validate_frozen_openai_lock()
    digest = hashlib.sha256()
    paths = [
        _WORKER_PROJECT / "pyproject.toml",
        _WORKER_PROJECT / "uv.lock",
        *sorted(
            (_WORKER_PROJECT / "src").rglob("*.py"),
            key=lambda path: path.as_posix(),
        ),
    ]
    for path in paths:
        try:
            relative = path.relative_to(_WORKER_PROJECT).as_posix()
        except ValueError as exc:  # pragma: no cover - constant invariant
            raise AssertionError("DeepSeek worker source escaped project") from exc
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path, label=relative).encode("ascii"))
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _validate_frozen_openai_lock() -> None:
    """Require the exact reviewed lock, closure, and OpenAI artifacts."""

    lock_path = _WORKER_PROJECT / "uv.lock"
    if _sha256_file(lock_path, label="uv.lock") != (
        f"sha256:{_WORKER_LOCK_SHA256}"
    ):
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker lock digest differs from DSK-00a"
        )
    try:
        document = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker lock is unreadable"
        ) from exc
    if (
        set(document) != {"version", "revision", "requires-python", "package"}
        or document.get("version") != 1
        or document.get("revision") != 3
        or document.get("requires-python") != "==3.12.*"
    ):
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker lock header changed"
        )
    packages = document["package"]
    if not isinstance(packages, list):
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker lock has no package inventory"
        )
    versions: list[tuple[str, str]] = []
    for package in packages:
        if (
            not isinstance(package, Mapping)
            or not isinstance(package.get("name"), str)
            or not isinstance(package.get("version"), str)
        ):
            raise DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker lock package is malformed"
            )
        versions.append((package["name"], package["version"]))
    if tuple(sorted(versions)) != _LOCKED_ALL_COMPONENT_VERSIONS:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker dependency closure changed"
        )
    openai_entries = [
        package
        for package in packages
        if isinstance(package, Mapping) and package.get("name") == "openai"
    ]
    if len(openai_entries) != 1:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker lock must contain one openai package"
        )
    package = openai_entries[0]
    sdist = package.get("sdist")
    wheels = package.get("wheels")
    if (
        package.get("version") != _OPENAI_VERSION
        or not isinstance(sdist, Mapping)
        or set(sdist) != {"url", "hash", "size", "upload-time"}
        or sdist.get("hash") != f"sha256:{_OPENAI_SOURCE_SHA256}"
        or not isinstance(wheels, list)
        or len(wheels) != 1
        or not isinstance(wheels[0], Mapping)
        or set(wheels[0]) != {"url", "hash", "size", "upload-time"}
        or wheels[0].get("hash") != f"sha256:{_OPENAI_WHEEL_SHA256}"
    ):
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker openai artifacts differ from DSK-00a"
        )


def _worker_environment() -> dict[str, str]:
    """Return a closed environment with no proxy/provider/user-site input."""

    return {"PYTHONNOUSERSITE": "1"}


def _validate_probe(payload: Any) -> tuple[DeepSeekSBOMComponent, ...]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "protocol",
        "type",
        "ok",
        "openai_version",
        "client_policy",
        "sbom",
    }:
        raise DeepSeekSDKQualificationError("DeepSeek worker probe fields changed")
    if (
        payload["protocol"] != _PROTOCOL
        or payload["type"] != "probe"
        or payload["ok"] is not True
        or payload["openai_version"] != _OPENAI_VERSION
        or payload["client_policy"]
        != {
            "base_url": _BASE_URL,
            "responses_endpoint": _ENDPOINT,
            "max_retries": 0,
            "trust_env": False,
            "follow_redirects": False,
        }
    ):
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker probe differs from the DSK-00a client contract"
        )
    raw_sbom = payload["sbom"]
    if not isinstance(raw_sbom, list) or not 1 <= len(raw_sbom) <= 128:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker probe returned an invalid SBOM"
        )
    components: list[DeepSeekSBOMComponent] = []
    names: set[str] = set()
    for raw in raw_sbom:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "version",
            "record_sha256",
        }:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker SBOM component fields changed"
            )
        try:
            component = DeepSeekSBOMComponent(
                name=raw["name"],
                version=raw["version"],
                record_sha256=raw["record_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker SBOM component is malformed"
            ) from exc
        if component.name in names:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker SBOM contains a duplicate package"
            )
        names.add(component.name)
        components.append(component)
    if components != sorted(components, key=lambda item: item.name):
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker SBOM is not canonically ordered"
        )
    observed_versions = tuple(
        (component.name, component.version) for component in components
    )
    if observed_versions != _LOCKED_LINUX_COMPONENT_VERSIONS:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker SBOM differs from the exact Linux lock closure"
        )
    return tuple(components)


@dataclass(frozen=True, slots=True)
class _SupervisedCommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool
    output_complete: bool


@dataclass(eq=False, slots=True)
class _ActiveDeepSeekWorker:
    """One backend-owned lease on the trusted subreaper controller."""

    process: subprocess.Popen[bytes]
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    _settled: bool = field(default=False, init=False, repr=False)

    def request_stop(self) -> None:
        _signal_supervisor_group(self.process, signal.SIGTERM)

    def finish(self, *, settled: bool) -> None:
        self._settled = settled
        self._done.set()

    def wait_for_settlement(self, timeout: float) -> bool:
        return self._done.wait(timeout) and self._settled


def _signal_supervisor_group(
    process: subprocess.Popen[bytes],
    signal_number: int,
) -> bool:
    """Signal only the still-unreaped supervisor process-group identity."""

    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise DeepSeekSDKQualificationError(
            "could not signal trusted DeepSeek process supervisor"
        ) from exc
    return True


def _raise_descriptor(descriptor: int) -> int:
    """Keep supervisor authority descriptors outside standard I/O."""

    if descriptor >= 3:
        return descriptor
    try:
        raised = fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
    finally:
        os.close(descriptor)
    return int(raised)


def _open_exact_executable(path: Path, *, expected_sha256: str) -> int:
    """Open and re-hash the exact executable inode passed to the supervisor."""

    try:
        descriptor = _raise_descriptor(
            os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        )
    except OSError as exc:
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker executable cannot be pinned"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_mode & 0o111 == 0
            or (metadata.st_dev, metadata.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker executable identity changed"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if f"sha256:{digest.hexdigest()}" != expected_sha256:
            raise DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker executable bytes changed"
            )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _close_selector_stream(
    selector: selectors.BaseSelector,
    stream: Any,
) -> None:
    try:
        selector.unregister(stream)
    except (KeyError, ValueError):
        pass
    try:
        stream.close()
    except (OSError, ValueError):
        pass


def _bounded_supervisor_communicate(
    process: subprocess.Popen[bytes],
    *,
    stdin_bytes: bytes,
    controller_deadline: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    stop_immediately: bool = False,
) -> tuple[bytes, bytes, bool, bool]:
    """Drain supervisor pipes incrementally without unbounded accumulation."""

    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    if stdin is None or stdout is None or stderr is None:
        raise DeepSeekSDKQualificationError(
            "trusted DeepSeek process supervisor pipes are unavailable"
        )
    selector = selectors.DefaultSelector()
    stdout_capture = bytearray()
    stderr_capture = bytearray()
    streams = {
        stdout.fileno(): (stdout, stdout_capture, max_stdout_bytes),
        stderr.fileno(): (stderr, stderr_capture, max_stderr_bytes),
    }
    input_offset = 0
    output_complete = True
    forced = False
    shutdown_deadline: float | None = None
    exited_at: float | None = None
    for stream in (stdin, stdout, stderr):
        os.set_blocking(stream.fileno(), False)
    if stdin_bytes:
        selector.register(stdin, selectors.EVENT_WRITE, "stdin")
    else:
        stdin.close()
    selector.register(stdout, selectors.EVENT_READ, "stdout")
    selector.register(stderr, selectors.EVENT_READ, "stderr")
    if stop_immediately:
        _signal_supervisor_group(process, signal.SIGTERM)
        shutdown_deadline = time.monotonic() + _STOP_GRACE_SECONDS
    try:
        while selector.get_map() or process.poll() is None:
            now = time.monotonic()
            returncode = process.poll()
            if returncode is not None and exited_at is None:
                exited_at = now
                if stdin in (
                    key.fileobj for key in selector.get_map().values()
                ):
                    _close_selector_stream(selector, stdin)
            if (
                now >= controller_deadline
                and returncode is None
                and shutdown_deadline is None
            ):
                _signal_supervisor_group(process, signal.SIGKILL)
                forced = True
                shutdown_deadline = now + _STOP_GRACE_SECONDS
            if (
                shutdown_deadline is not None
                and now >= shutdown_deadline
                and process.poll() is None
            ):
                if forced:
                    raise DeepSeekSDKQualificationError(
                        "trusted DeepSeek process supervisor survived SIGKILL"
                    )
                _signal_supervisor_group(process, signal.SIGKILL)
                forced = True
                shutdown_deadline = now + _STOP_GRACE_SECONDS
            output_open = any(
                key.data in {"stdout", "stderr"}
                for key in selector.get_map().values()
            )
            if returncode is not None and not output_open:
                break
            if (
                returncode is not None
                and exited_at is not None
                and now - exited_at >= _STREAM_DRAIN_GRACE_SECONDS
            ):
                output_complete = False
                _close_selector_stream(selector, stdout)
                _close_selector_stream(selector, stderr)
                break
            wait = 0.05
            if returncode is None:
                wait = min(wait, max(0.0, controller_deadline - now))
            if shutdown_deadline is not None:
                wait = min(wait, max(0.0, shutdown_deadline - now))
            if exited_at is not None:
                wait = min(
                    wait,
                    max(
                        0.0,
                        _STREAM_DRAIN_GRACE_SECONDS - (now - exited_at),
                    ),
                )
            for key, _mask in selector.select(wait):
                selected_stream: Any = key.fileobj
                if key.data == "stdin":
                    try:
                        written = os.write(
                            selected_stream.fileno(),
                            stdin_bytes[
                                input_offset : input_offset + _STREAM_CHUNK_BYTES
                            ],
                        )
                    except BrokenPipeError:
                        _close_selector_stream(selector, selected_stream)
                        continue
                    input_offset += written
                    if input_offset >= len(stdin_bytes):
                        _close_selector_stream(selector, selected_stream)
                    continue
                try:
                    chunk = os.read(
                        selected_stream.fileno(), _STREAM_CHUNK_BYTES
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    _close_selector_stream(selector, selected_stream)
                    continue
                _stream, capture, limit = streams[selected_stream.fileno()]
                available = max(0, limit - len(capture))
                capture.extend(chunk[:available])
                if len(chunk) > available:
                    output_complete = False
                    if shutdown_deadline is None and process.poll() is None:
                        _signal_supervisor_group(process, signal.SIGTERM)
                        shutdown_deadline = (
                            time.monotonic() + _STOP_GRACE_SECONDS
                        )
    finally:
        selector.close()
        for stream in (stdin, stdout, stderr):
            if not stream.closed:
                stream.close()
    if process.poll() is None:
        _signal_supervisor_group(process, signal.SIGKILL)
        forced = True
    try:
        process.wait(timeout=_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise DeepSeekSDKQualificationError(
            "trusted DeepSeek process supervisor did not terminate"
        ) from exc
    return bytes(stdout_capture), bytes(stderr_capture), output_complete, forced


def _read_supervisor_control(descriptor: int) -> tuple[int, bool]:
    """Authenticate one trusted subreaper descendant-settlement receipt."""

    control = socket.socket(fileno=descriptor)
    try:
        if control.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != (
            socket.SOCK_SEQPACKET
        ):
            raise DeepSeekSDKQualificationError(
                "DeepSeek supervisor control transport is not seqpacket"
            )
        payload = control.recv(_SUPERVISOR_CONTROL_MAX_BYTES + 1)
        if len(payload) > _SUPERVISOR_CONTROL_MAX_BYTES:
            raise DeepSeekSDKQualificationError(
                "DeepSeek supervisor control record exceeded its bound"
            )
        control.setblocking(False)
        try:
            extra = control.recv(1)
        except BlockingIOError:
            extra = None
        if extra not in {None, b""}:
            raise DeepSeekSDKQualificationError(
                "DeepSeek supervisor emitted multiple control packets"
            )
    finally:
        control.detach()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        record: dict[str, Any] = {}
        for name, value in pairs:
            if name in record:
                raise ValueError("duplicate process supervisor field")
            record[name] = value
        return record

    try:
        record = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DeepSeekSDKQualificationError(
            "DeepSeek supervisor settlement record is malformed"
        ) from exc
    if (
        not isinstance(record, Mapping)
        or set(record) != {"schema_version", "returncode", "settled", "error"}
        or record.get("schema_version") != _SUPERVISOR_SCHEMA
        or record.get("settled") is not True
        or type(record.get("returncode")) is not int
        or record.get("error") not in {None, "background_descendants"}
    ):
        raise DeepSeekSDKQualificationError(
            "DeepSeek supervisor did not prove complete descendant settlement"
        )
    return int(record["returncode"]), (
        record.get("error") == "background_descendants"
    )


def _run_supervised_command(
    argv: tuple[str, ...],
    *,
    stdin_bytes: bytes,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    max_stdout_bytes: int,
    max_stderr_bytes: int,
    executable_path: Path,
    executable_sha256: str,
    on_process_start: Callable[[_ActiveDeepSeekWorker], None] | None = None,
    on_process_stop: Callable[[_ActiveDeepSeekWorker], None] | None = None,
) -> _SupervisedCommandResult:
    """Execute under Crucible's trusted subreaper/PDEATHSIG boundary.

    This closes ordinary ``setsid`` and double-fork escape and positively
    authenticates descendant-empty settlement. Live routing remains blocked
    until the shared delegated-cgroup Gate also closes the supervisor-SIGKILL
    case, which an unprivileged subreaper cannot prove.
    """

    if (
        not argv
        or argv[0] != str(Path(argv[0]))
        or not Path(argv[0]).is_absolute()
        or any(not isinstance(item, str) or not item or "\x00" in item for item in argv)
    ):
        raise ValueError("DeepSeek supervised argv is invalid")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("DeepSeek supervised timeout is invalid")
    from ..workflows._process_supervisor_contract import (
        acquire_trusted_supervisor_source,
        open_target_working_directory,
        sealed_target_environment_descriptor,
        supervisor_environment,
        supervisor_interpreter_path,
    )

    target_deadline_ns = time.monotonic_ns() + max(
        1, int(timeout_seconds * 1_000_000_000)
    )
    descriptors: set[int] = set()
    control_read: int | None = None
    process: subprocess.Popen[bytes] | None = None
    active: _ActiveDeepSeekWorker | None = None
    registered = False
    settlement_proven = False
    callback_error: BaseException | None = None
    try:
        executable_descriptor = _open_exact_executable(
            executable_path,
            expected_sha256=executable_sha256,
        )
        descriptors.add(executable_descriptor)
        target_env_descriptor = _raise_descriptor(
            sealed_target_environment_descriptor(env)
        )
        descriptors.add(target_env_descriptor)
        target_cwd_descriptor = _raise_descriptor(
            open_target_working_directory(cwd)
        )
        descriptors.add(target_cwd_descriptor)
        supervisor_source_descriptor, supervisor_source_path = (
            acquire_trusted_supervisor_source()
        )
        supervisor_source_descriptor = _raise_descriptor(
            supervisor_source_descriptor
        )
        supervisor_source_path = (
            f"/proc/self/fd/{supervisor_source_descriptor}"
        )
        descriptors.add(supervisor_source_descriptor)
        controller_control, supervisor_control = socket.socketpair(
            socket.AF_UNIX,
            socket.SOCK_SEQPACKET | socket.SOCK_CLOEXEC,
        )
        control_read = _raise_descriptor(controller_control.detach())
        control_write = _raise_descriptor(supervisor_control.detach())
        descriptors.add(control_write)
        supervisor_argv = (
            supervisor_interpreter_path(),
            "-I",
            "-S",
            supervisor_source_path,
            str(control_write),
            str(target_env_descriptor),
            str(supervisor_source_descriptor),
            str(target_cwd_descriptor),
            format(_STOP_GRACE_SECONDS, ".17g"),
            str(os.getpid()),
            str(target_deadline_ns),
            "--held-executable-fd",
            str(executable_descriptor),
            argv[0],
            "--",
            *argv,
        )
        process = subprocess.Popen(
            supervisor_argv,
            cwd=Path("/"),
            env=supervisor_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=tuple(sorted(descriptors)),
        )
        for descriptor in tuple(descriptors):
            os.close(descriptor)
            descriptors.remove(descriptor)
        active = _ActiveDeepSeekWorker(process)
        if on_process_start is not None:
            try:
                on_process_start(active)
                registered = True
            except BaseException as exc:  # noqa: BLE001
                callback_error = exc
        controller_deadline = (
            target_deadline_ns / 1_000_000_000
            + _STOP_GRACE_SECONDS
            + _STREAM_DRAIN_GRACE_SECONDS
            + 0.25
        )
        stdout, stderr, output_complete, forced = (
            _bounded_supervisor_communicate(
                process,
                stdin_bytes=stdin_bytes,
                controller_deadline=controller_deadline,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
                stop_immediately=callback_error is not None,
            )
        )
        if process.returncode != 0 or forced or control_read is None:
            raise DeepSeekSDKQualificationError(
                "DeepSeek supervisor exited without trusted settlement"
            )
        target_returncode, background_descendants = _read_supervisor_control(
            control_read
        )
        settlement_proven = True
        if callback_error is not None:
            raise callback_error
        if background_descendants:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker left background descendants; all were settled"
            )
        return _SupervisedCommandResult(
            returncode=target_returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=target_returncode == -124,
            output_complete=output_complete,
        )
    finally:
        if process is not None and process.poll() is None:
            try:
                _signal_supervisor_group(process, signal.SIGTERM)
                process.wait(timeout=_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _signal_supervisor_group(process, signal.SIGKILL)
                try:
                    process.wait(timeout=_STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            except DeepSeekSDKQualificationError:
                pass
        if active is not None:
            active.finish(settled=settlement_proven)
            if registered and on_process_stop is not None:
                on_process_stop(active)
        if control_read is not None:
            try:
                os.close(control_read)
            except OSError:
                pass
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _probe_worker(worker_python: Path) -> tuple[DeepSeekSBOMComponent, ...]:
    if not worker_python.is_file() or not os.access(worker_python, os.X_OK):
        raise DeepSeekSDKUnavailable(
            "project-locked DeepSeek worker is not installed; run "
            f"`uv sync --project {_WORKER_PROJECT} --frozen`"
        )
    try:
        resolved = worker_python.resolve(strict=True)
        completed = _run_supervised_command(
            (
                str(worker_python.absolute()),
                "-B",
                "-I",
                "-m",
                "crucible_deepseek_worker",
                "--probe",
            ),
            stdin_bytes=b"",
            cwd=_WORKER_PROJECT.resolve(strict=True),
            env=_worker_environment(),
            timeout_seconds=_PROBE_TIMEOUT_SECONDS,
            max_stdout_bytes=_MAX_STDERR_BYTES,
            max_stderr_bytes=_MAX_STDERR_BYTES,
            executable_path=resolved,
            executable_sha256=_sha256_file(
                resolved, label="resolved worker Python"
            ),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeepSeekSDKUnavailable(
            f"could not probe project-locked DeepSeek worker ({type(exc).__name__})"
        ) from exc
    if not completed.output_complete:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker probe exceeded its output bound"
        )
    try:
        payload = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker probe did not return one JSON object"
        ) from exc
    if completed.timed_out:
        raise DeepSeekSDKQualificationError("DeepSeek worker probe timed out")
    if completed.returncode != 0 or completed.stderr:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker probe did not terminate cleanly"
        )
    return _validate_probe(payload)


@dataclass(frozen=True, slots=True)
class _PinnedDeepSeekWorkerRuntime:
    logical_python: Path
    resolved_python: Path
    resolved_python_sha256: str
    lock_source_fingerprint: str
    installed_tree_fingerprint: str
    installed_sbom_fingerprint: str
    runtime_fingerprint: str
    sbom: tuple[DeepSeekSBOMComponent, ...]

    def __post_init__(self) -> None:
        if tuple(
            (component.name, component.version) for component in self.sbom
        ) != _LOCKED_LINUX_COMPONENT_VERSIONS:
            raise ValueError("pinned DeepSeek worker SBOM differs from lock closure")

    @classmethod
    def capture(cls) -> _PinnedDeepSeekWorkerRuntime:
        logical = Path(os.path.abspath(_DEFAULT_WORKER_PYTHON))
        try:
            resolved = logical.resolve(strict=True)
        except OSError as exc:
            raise DeepSeekSDKUnavailable(
                "project-locked DeepSeek worker interpreter is unavailable"
            ) from exc
        executable_sha = _sha256_file(resolved, label="resolved worker Python")
        source_fingerprint = _lock_source_fingerprint()
        installed_tree_fingerprint = _installed_tree_fingerprint()
        sbom = _probe_worker(logical)
        sbom_encoded = json.dumps(
            [component.to_mapping() for component in sbom],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        sbom_fingerprint = "sha256:" + hashlib.sha256(sbom_encoded).hexdigest()
        runtime_payload = {
            "protocol": _PROTOCOL,
            "logical_python": str(logical),
            "resolved_python": str(resolved),
            "resolved_python_sha256": executable_sha,
            "lock_source_fingerprint": source_fingerprint,
            "installed_tree_fingerprint": installed_tree_fingerprint,
            "installed_sbom_fingerprint": sbom_fingerprint,
        }
        runtime_fingerprint = (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    runtime_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
        return cls(
            logical_python=logical,
            resolved_python=resolved,
            resolved_python_sha256=executable_sha,
            lock_source_fingerprint=source_fingerprint,
            installed_tree_fingerprint=installed_tree_fingerprint,
            installed_sbom_fingerprint=sbom_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            sbom=sbom,
        )

    def revalidate(self) -> None:
        if type(self).capture() != self:
            raise DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker runtime drifted after construction"
            )

    def public_identity(self) -> DeepSeekProjectLockedRuntimeIdentity:
        return DeepSeekProjectLockedRuntimeIdentity(
            logical_worker_python=str(self.logical_python),
            resolved_worker_python=str(self.resolved_python),
            resolved_worker_python_sha256=self.resolved_python_sha256,
            lock_source_fingerprint=self.lock_source_fingerprint,
            installed_tree_fingerprint=self.installed_tree_fingerprint,
            installed_sbom_fingerprint=self.installed_sbom_fingerprint,
            runtime_fingerprint=self.runtime_fingerprint,
            openai_version=_OPENAI_VERSION,
            sbom=self.sbom,
        )


def capture_project_locked_deepseek_runtime_identity() -> (
    DeepSeekProjectLockedRuntimeIdentity
):
    """Capture the exact installed worker, lock/source, and SBOM identity."""

    if os.environ.get("CRUCIBLE_DEEPSEEK_WORKER_PYTHON", "").strip():
        raise DeepSeekSDKQualificationError(
            "project-locked DeepSeek worker rejects ambient interpreter overrides"
        )
    return _PinnedDeepSeekWorkerRuntime.capture().public_identity()


@dataclass(frozen=True, slots=True)
class _DeepSeekUsage:
    input_tokens: int
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int
    reasoning_output_tokens: int | None
    total_tokens: int

    @property
    def complete(self) -> bool:
        return (
            self.cached_input_tokens is not None
            and self.reasoning_output_tokens is not None
        )


@dataclass(frozen=True, slots=True)
class _DeepSeekToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class _DeepSeekWorkerTerminal:
    terminal_type: str
    response_id: str
    model: str
    system_fingerprint: str | None
    output_text: str
    tool_calls: tuple[_DeepSeekToolCall, ...]
    continuation_json: str = field(repr=False)
    usage: _DeepSeekUsage | None
    event_types: tuple[str, ...]


def _bounded_identity(value: Any, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise DeepSeekSDKQualificationError(f"worker returned invalid {field}")
    return value


def _exact_count(value: Any, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise DeepSeekSDKQualificationError(f"worker returned invalid {field} counter")
    return value


def _continuation_json(
    value: Any,
    *,
    calls: tuple[_DeepSeekToolCall, ...],
    output_text: str,
) -> str:
    """Validate the private stateless-replay payload without retaining it publicly."""

    if not isinstance(value, list) or len(value) > 512:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker returned invalid continuation items"
        )
    if not calls:
        if value:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker crossed unnecessary terminal continuation"
            )
        return "[]"
    observed_calls: list[tuple[str, str, str]] = []
    text_parts: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise DeepSeekSDKQualificationError(
                "DeepSeek continuation item is not an object"
            )
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
                raise DeepSeekSDKQualificationError(
                    "DeepSeek continuation message fields changed"
                )
            content = item.get("content")
            if not isinstance(content, list):
                raise DeepSeekSDKQualificationError(
                    "DeepSeek continuation content is not a list"
                )
            for part in content:
                if (
                    not isinstance(part, Mapping)
                    or set(part) != {"type", "text"}
                    or part.get("type") != expected_part_type
                    or not isinstance(part.get("text"), str)
                ):
                    raise DeepSeekSDKQualificationError(
                        "DeepSeek continuation content fields changed"
                    )
                if item_type == "message":
                    text_parts.append(part["text"])
            continue
        if item_type == "function_call":
            if set(item) != {"type", "call_id", "name", "arguments"}:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek continuation function-call fields changed"
                )
            observed_calls.append(
                (
                    _bounded_identity(
                        item.get("call_id"),
                        field="continuation function call id",
                        maximum=512,
                    ),
                    _bounded_identity(
                        item.get("name"),
                        field="continuation function name",
                        maximum=128,
                    ),
                    _bounded_identity(
                        item.get("arguments"),
                        field="continuation function arguments",
                        maximum=2 * 1024 * 1024,
                    ),
                )
            )
            continue
        raise DeepSeekSDKQualificationError(
            f"DeepSeek continuation item {index} has unsupported type"
        )
    expected_calls = [
        (call.call_id, call.name, call.arguments) for call in calls
    ]
    if observed_calls != expected_calls or "".join(text_parts) != output_text:
        raise DeepSeekSDKQualificationError(
            "DeepSeek continuation contradicts sanitized terminal output"
        )
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DeepSeekSDKQualificationError(
            "DeepSeek continuation is not canonical JSON"
        ) from exc


def _parse_terminal(
    raw: Mapping[str, Any], events: list[str]
) -> _DeepSeekWorkerTerminal:
    expected_fields = {
        "protocol",
        "type",
        "sequence",
        "provider_sequence",
        "terminal_type",
        "response_id",
        "model",
        "system_fingerprint",
        "output_text",
        "tool_calls",
        "continuation_items",
        "usage",
    }
    if set(raw) != expected_fields or raw["terminal_type"] not in _TERMINAL_TYPES:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker terminal fields or type changed"
        )
    output_text = raw["output_text"]
    if not isinstance(output_text, str):
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker terminal output is not text"
        )
    fingerprint = raw["system_fingerprint"]
    if fingerprint is not None:
        fingerprint = _bounded_identity(
            fingerprint, field="system fingerprint", maximum=512
        )
    raw_calls = raw["tool_calls"]
    if not isinstance(raw_calls, list) or len(raw_calls) > 256:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker returned invalid function calls"
        )
    calls: list[_DeepSeekToolCall] = []
    call_ids: set[str] = set()
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping) or set(raw_call) != {
            "call_id",
            "name",
            "arguments",
        }:
            raise DeepSeekSDKQualificationError("DeepSeek function-call fields changed")
        call = _DeepSeekToolCall(
            call_id=_bounded_identity(
                raw_call["call_id"], field="function call id", maximum=512
            ),
            name=_bounded_identity(
                raw_call["name"], field="function name", maximum=128
            ),
            arguments=_bounded_identity(
                raw_call["arguments"],
                field="function arguments",
                maximum=2 * 1024 * 1024,
            ),
        )
        if call.call_id in call_ids:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker returned duplicate function call ids"
            )
        call_ids.add(call.call_id)
        calls.append(call)
    usage: _DeepSeekUsage | None = None
    raw_usage = raw["usage"]
    if raw_usage is not None:
        if not isinstance(raw_usage, Mapping) or set(raw_usage) != {
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "reasoning_output_tokens",
            "total_tokens",
        }:
            raise DeepSeekSDKQualificationError("DeepSeek worker usage fields changed")
        input_tokens = _exact_count(raw_usage["input_tokens"], field="input")
        output_tokens = _exact_count(raw_usage["output_tokens"], field="output")
        total_tokens = _exact_count(raw_usage["total_tokens"], field="total")
        cached = raw_usage["cached_input_tokens"]
        if cached is not None:
            cached = _exact_count(cached, field="cached input")
            if cached > input_tokens:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek cached input exceeds total input"
                )
        cache_write = raw_usage["cache_write_tokens"]
        if cache_write is not None:
            cache_write = _exact_count(cache_write, field="cache write")
            if cache_write + (cached or 0) > input_tokens:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek cache write plus cached input exceeds total input"
                )
        reasoning = raw_usage["reasoning_output_tokens"]
        if reasoning is not None:
            reasoning = _exact_count(reasoning, field="reasoning output")
            if reasoning > output_tokens:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek reasoning output exceeds total output"
                )
        if total_tokens != input_tokens + output_tokens:
            raise DeepSeekSDKQualificationError(
                "DeepSeek terminal total token count is contradictory"
            )
        usage = _DeepSeekUsage(
            input_tokens=input_tokens,
            cached_input_tokens=cached,
            cache_write_tokens=cache_write,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning,
            total_tokens=total_tokens,
        )
    frozen_calls = tuple(calls)
    return _DeepSeekWorkerTerminal(
        terminal_type=raw["terminal_type"],
        response_id=_bounded_identity(
            raw["response_id"], field="response id", maximum=512
        ),
        model=_bounded_identity(raw["model"], field="model", maximum=256),
        system_fingerprint=fingerprint,
        output_text=output_text,
        tool_calls=frozen_calls,
        continuation_json=_continuation_json(
            raw["continuation_items"],
            calls=frozen_calls,
            output_text=output_text,
        ),
        usage=usage,
        event_types=tuple(events),
    )


def _parse_worker_stdout(stdout: bytes, *, returncode: int) -> _DeepSeekWorkerTerminal:
    if len(stdout) > _MAX_STDOUT_BYTES:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker stdout exceeded its byte bound"
        )
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker stdout is not UTF-8"
        ) from exc
    raw_lines = text.splitlines()
    if not raw_lines:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker reached EOF without a provider terminal"
        )
    expected_sequence = 0
    last_provider_sequence = -1
    events: list[str] = []
    terminal: _DeepSeekWorkerTerminal | None = None
    worker_error: str | None = None
    for index, line in enumerate(raw_lines):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker emitted malformed JSONL"
            ) from exc
        if not isinstance(raw, Mapping) or raw.get("protocol") != _PROTOCOL:
            raise DeepSeekSDKQualificationError("DeepSeek worker protocol mismatch")
        if raw.get("sequence") != expected_sequence:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker event sequence is not monotonic and contiguous"
            )
        expected_sequence += 1
        record_type = raw.get("type")
        if terminal is not None:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker emitted data after its terminal"
            )
        if record_type in {"event", "terminal"}:
            provider_sequence = raw.get("provider_sequence")
            if (
                type(provider_sequence) is not int
                or provider_sequence < 0
                or provider_sequence <= last_provider_sequence
            ):
                raise DeepSeekSDKQualificationError(
                    "DeepSeek provider SSE sequence is not strictly monotonic"
                )
            last_provider_sequence = provider_sequence
        if record_type == "event":
            if set(raw) != {
                "protocol",
                "type",
                "sequence",
                "provider_sequence",
                "event_type",
            }:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek worker event fields changed"
                )
            event_type = _bounded_identity(
                raw["event_type"], field="event type", maximum=256
            )
            if event_type in _TERMINAL_TYPES:
                raise DeepSeekSDKQualificationError(
                    "provider terminal was disguised as a progress event"
                )
            events.append(event_type)
        elif record_type == "terminal":
            terminal = _parse_terminal(raw, events)
            if index != len(raw_lines) - 1:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek worker terminal was not its final record"
                )
        elif record_type == "error":
            if set(raw) != {
                "protocol",
                "type",
                "sequence",
                "error_type",
            }:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek worker error fields changed"
                )
            worker_error = _bounded_identity(
                raw["error_type"], field="worker error type", maximum=128
            )
            if index != len(raw_lines) - 1:
                raise DeepSeekSDKQualificationError(
                    "DeepSeek worker error was not its final record"
                )
        else:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker emitted an unknown record type"
            )
    if returncode != 0:
        if worker_error is not None:
            raise DeepSeekWorkerInvocationError(worker_error)
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker exited nonzero without sanitized error evidence"
        )
    if worker_error is not None:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker reported an error with a zero exit status"
        )
    if terminal is None:
        raise DeepSeekSDKQualificationError(
            "DeepSeek worker reached EOF without exactly one provider terminal"
        )
    return terminal


def _run_worker_subprocess(
    request: Mapping[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
    worker_pin: _PinnedDeepSeekWorkerRuntime,
    on_process_start: Callable[[_ActiveDeepSeekWorker], None] | None = None,
    on_process_stop: Callable[[_ActiveDeepSeekWorker], None] | None = None,
) -> _DeepSeekWorkerTerminal:
    worker_pin.revalidate()
    wire_request = {**request, "api_key": api_key}
    encoded = json.dumps(
        wire_request,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > _MAX_REQUEST_BYTES:
        raise ValueError("DeepSeek worker request exceeds its byte bound")
    try:
        completed = _run_supervised_command(
            (
                str(worker_pin.logical_python.absolute()),
                "-B",
                "-I",
                "-m",
                "crucible_deepseek_worker",
            ),
            stdin_bytes=encoded,
            cwd=_WORKER_PROJECT.resolve(strict=True),
            env=_worker_environment(),
            timeout_seconds=timeout_seconds,
            max_stdout_bytes=_MAX_STDOUT_BYTES,
            max_stderr_bytes=_MAX_STDERR_BYTES,
            executable_path=worker_pin.resolved_python,
            executable_sha256=worker_pin.resolved_python_sha256,
            on_process_start=on_process_start,
            on_process_stop=on_process_stop,
        )
        if completed.timed_out:
            raise DeepSeekWorkerTimeout(
                "DeepSeek worker deadline elapsed; trusted subreaper proved "
                "local descendant settlement "
                "but provider-side billing/settlement remains unproven"
            )
        if not completed.output_complete:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker output exceeded its incremental byte bounds"
            )
        if completed.stderr:
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker wrote unexpected stderr"
            )
        return _parse_worker_stdout(
            completed.stdout,
            returncode=completed.returncode,
        )
    finally:
        encoded = b""
        wire_request.clear()


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
            "DeepSeekBackend response_schema must be JSON serializable"
        ) from exc


def _prompt_system_text(system: str, schema_json: str | None) -> str:
    """Bind one canonical response schema into the prompt-only route."""

    if schema_json is None:
        return system
    prompt_abi_marker = "<!-- assay.prompt.v2 -->"
    if prompt_abi_marker in system:
        canonical_block = f"```json\n{schema_json}\n```"
        if (
            not system.startswith(prompt_abi_marker + "\n")
            or system.count(prompt_abi_marker) != 1
            or system.count("## Required output contract") != 1
            or system.count(canonical_block) != 1
            or system.count(schema_json) != 1
            or "CRUCIBLE OUTPUT CONTRACT:" in system
        ):
            raise DeepSeekSDKQualificationError(
                "DeepSeek prompt ABI v2 output contract does not contain "
                "exactly one matching canonical response schema"
            )
        return system
    return (
        system.rstrip()
        + "\n\nCRUCIBLE OUTPUT CONTRACT:\n"
        + "Return only one JSON value matching the following JSON Schema. "
        + "Do not wrap it in Markdown fences or add prose outside the value.\n"
        + schema_json
    )


class _FrozenFunctionToolSet:
    """Immutable authority view over a live DeepSeek dispatcher."""

    def __init__(self, toolset: Any, *, authority_snapshot: Any) -> None:
        self._toolset = toolset
        self._authority_snapshot = authority_snapshot
        self.write_names = authority_snapshot.write_names
        self.read_only_names = authority_snapshot.read_only_names

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return self._authority_snapshot.to_anthropic_tools()

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        return self._toolset.dispatch(name, arguments)


def _function_tools(
    toolset: Any,
    *,
    authority_snapshot: Any | None = None,
) -> tuple[list[dict[str, Any]], frozenset[str]]:
    from .tool_authority import snapshot_tool_authority

    snapshot = (
        snapshot_tool_authority(toolset)
        if authority_snapshot is None
        else authority_snapshot
    )
    descriptors = snapshot.to_anthropic_tools()
    tools: list[dict[str, Any]] = []
    for descriptor in descriptors:
        name = descriptor["name"]
        description = descriptor["description"]
        parameters = descriptor["input_schema"]
        if not isinstance(name, str) or _TOOL_NAME_RE.fullmatch(name) is None:
            raise ValueError(
                "DeepSeek function names must match ^[A-Za-z0-9_-]{1,128}$"
            )
        if not isinstance(description, str) or len(description) > 16_384:
            raise ValueError("DeepSeek function descriptions exceed provider bounds")
        if not isinstance(parameters, Mapping):
            raise ValueError("DeepSeek function parameters must be JSON objects")
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return tools, snapshot.tool_names


class DeepSeekBackend:
    """Assay backend for the exact diagnostic native Responses route."""

    name = "deepseek"

    @property
    def features(self) -> Any:
        from .features import feature_profile_for_route

        return feature_profile_for_route(self.name, model=self.model)

    def __init__(
        self,
        *,
        model: str = _MODEL,
        cwd: str | None = None,
        log_dir: Path | str | None = None,
        capability: str = "text",
        max_response_tokens: int = _MAX_OUTPUT_TOKENS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        expected_worker_runtime_fingerprint: str | None = None,
        diagnostic_workspace_write: bool = False,
        _worker_pin: _PinnedDeepSeekWorkerRuntime | None = None,
        _worker_runner: Callable[..., _DeepSeekWorkerTerminal] | None = None,
    ) -> None:
        from .capabilities import normalize_capability

        self.capability = normalize_capability(capability)
        if type(diagnostic_workspace_write) is not bool:
            raise TypeError("diagnostic_workspace_write must be an exact bool")
        if (
            self.capability == "workspace_write"
            and diagnostic_workspace_write is not True
        ):
            raise ValueError(
                "DeepSeek workspace_write is promotion-blocked pending "
                "DSK-00b/DSK-02; explicit diagnostic_workspace_write=true "
                "is required for the confined diagnostic route"
            )
        if model != _MODEL:
            raise ValueError(
                f"DeepSeekBackend supports only exact request alias {_MODEL!r}"
            )
        self.model = _MODEL
        if (
            type(max_response_tokens) is not int
            or not 1 <= max_response_tokens <= _MAX_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"max_response_tokens must be from 1 through {_MAX_OUTPUT_TOKENS}"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if expected_worker_runtime_fingerprint is not None and (
            not isinstance(expected_worker_runtime_fingerprint, str)
            or _DIGEST_RE.fullmatch(expected_worker_runtime_fingerprint) is None
        ):
            raise ValueError(
                "expected_worker_runtime_fingerprint must be sha256:<64 lowercase hex>"
            )
        if _worker_pin is None and expected_worker_runtime_fingerprint is None:
            raise DeepSeekSDKQualificationError(
                "DeepSeek construction requires a predeclared worker runtime "
                "fingerprint from provider-free capture"
            )
        if os.environ.get("CRUCIBLE_DEEPSEEK_WORKER_PYTHON", "").strip():
            raise DeepSeekSDKQualificationError(
                "project-locked DeepSeek worker rejects ambient interpreter overrides"
            )
        self._worker_pin = _worker_pin or _PinnedDeepSeekWorkerRuntime.capture()
        if (
            expected_worker_runtime_fingerprint is not None
            and self._worker_pin.runtime_fingerprint
            != expected_worker_runtime_fingerprint
        ):
            raise DeepSeekSDKQualificationError(
                "DeepSeek worker does not match configured runtime fingerprint"
            )
        self._worker_runner = _worker_runner or functools.partial(
            _run_worker_subprocess,
            worker_pin=self._worker_pin,
        )
        self._max_response_tokens = max_response_tokens
        self._timeout_seconds = float(timeout_seconds)
        self._cwd = cwd or os.getcwd()
        self._sdk_sink = SdkDebugSink(
            Path(log_dir) if log_dir is not None else None,
            backend_name=self.name,
        )
        self._invoke_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._active_processes: set[_ActiveDeepSeekWorker] = set()
        self._active_mutation_hosts: set[Any] = set()
        self._toolset: Any = None
        self._tool_names: frozenset[str] = frozenset()
        self._tool_write_names: frozenset[str] = frozenset()
        self._tool_descriptor_json = "[]"
        self._context_budget: int | None = None
        self._response_evidence_collector: list[DeepSeekResponseEvidence] | None = None
        self._closed = False

    @property
    def worker_runtime_fingerprint(self) -> str:
        return self._worker_pin.runtime_fingerprint

    @property
    def worker_runtime_identity(self) -> DeepSeekProjectLockedRuntimeIdentity:
        return self._worker_pin.public_identity()

    @property
    def max_response_tokens(self) -> int:
        return self._max_response_tokens

    def _register_process(self, process: _ActiveDeepSeekWorker) -> None:
        with self._state_lock:
            if self._closed:
                process.request_stop()
                raise RuntimeError("DeepSeekBackend closed before worker start")
            self._active_processes.add(process)

    def _unregister_process(self, process: _ActiveDeepSeekWorker) -> None:
        with self._state_lock:
            self._active_processes.discard(process)

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
            raise RuntimeError("DeepSeekBackend is closed")

    def _unregister_mutation_host(self, host: Any) -> None:
        with self._state_lock:
            self._active_mutation_hosts.discard(host)

    def attach_toolset(
        self,
        toolset: Any,
        *,
        context_budget: int | None = None,
    ) -> None:
        from .capabilities import validate_toolset_capability

        if context_budget is not None and (
            type(context_budget) is not int or context_budget <= 0
        ):
            raise ValueError("context_budget must be a positive integer")
        validate_toolset_capability(self.capability, toolset)
        names: frozenset[str] = frozenset()
        writes: frozenset[str] = frozenset()
        descriptor_json = "[]"
        if toolset is not None:
            from .tool_authority import snapshot_tool_authority

            authority_snapshot = snapshot_tool_authority(toolset)
            descriptors, names = _function_tools(
                toolset,
                authority_snapshot=authority_snapshot,
            )
            writes = authority_snapshot.write_names
            descriptor_json = json.dumps(
                descriptors,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        with self._invoke_lock, self._state_lock:
            if self._closed:
                raise RuntimeError("DeepSeekBackend is closed")
            self._toolset = toolset
            self._tool_names = names
            self._tool_write_names = writes
            self._tool_descriptor_json = descriptor_json
            self._context_budget = context_budget if toolset is not None else None

    def chat_with_tools(
        self,
        invocation: Invocation,
        toolset: Any,
        *,
        cache: Any = None,
        max_rounds: int = 10,
        max_response_tokens: int | None = None,
        retry_budget: int = 1,
        strict: bool = False,
    ) -> InvocationResult:
        if cache is not None:
            raise NotImplementedError("DeepSeekBackend does not support ToolCallCache")
        if retry_budget != 1:
            raise ValueError("DeepSeekBackend permits exactly one physical attempt")
        if type(max_rounds) is not int or not 1 <= max_rounds <= _MAX_TOOL_ROUNDS:
            raise ValueError(f"max_rounds must be from 1 through {_MAX_TOOL_ROUNDS}")
        if toolset is None:
            if strict:
                raise NotImplementedError(
                    "DeepSeekBackend.chat_with_tools requires a toolset"
                )
            return self.invoke(invocation)
        selected = (
            self._max_response_tokens
            if max_response_tokens is None
            else max_response_tokens
        )
        if type(selected) is not int or not 1 <= selected <= self._max_response_tokens:
            raise ValueError(
                "max_response_tokens cannot exceed the route-owned ceiling"
            )
        self.attach_toolset(toolset)
        bounded = Invocation(
            system=invocation.system,
            user=invocation.user,
            tools=invocation.tools,
            max_turns=min(invocation.max_turns, max_rounds),
            budget_usd=invocation.budget_usd,
            policy=invocation.policy,
            response_schema=invocation.response_schema,
            conversation=invocation.conversation,
        )
        with self._invoke_lock:
            return self._invoke_serialized(bounded, max_response_tokens=selected)

    def invoke(self, invocation: Invocation) -> InvocationResult:
        with self._invoke_lock:
            return self._invoke_serialized(
                invocation,
                max_response_tokens=self._max_response_tokens,
            )

    def invoke_with_response_evidence(
        self,
        invocation: Invocation,
    ) -> tuple[InvocationResult, DeepSeekInvocationEvidence]:
        """Invoke and return raw-reasoning-free physical response evidence."""

        with self._invoke_lock:
            if self._response_evidence_collector is not None:
                raise RuntimeError("DeepSeek response evidence collector is active")
            collector: list[DeepSeekResponseEvidence] = []
            self._response_evidence_collector = collector
            try:
                result = self._invoke_serialized(
                    invocation,
                    max_response_tokens=self._max_response_tokens,
                )
            finally:
                self._response_evidence_collector = None
            return result, DeepSeekInvocationEvidence(
                worker_runtime_fingerprint=self.worker_runtime_fingerprint,
                logical_result_complete=(
                    result.error is None
                    and result.completeness.token_usage_complete
                    and result.completeness.cost_complete
                    and result.completeness.terminal_complete
                    and result.completeness.trace_complete
                ),
                responses=tuple(collector),
            )

    def _record_response_evidence(
        self,
        terminal: _DeepSeekWorkerTerminal,
    ) -> None:
        collector = self._response_evidence_collector
        if collector is None:
            return
        usage = terminal.usage
        cost_usd: float | None = None
        if usage is not None:
            cost_usd = compute_cost(
                self.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_creation_tokens=usage.cache_write_tokens or 0,
                cache_read_tokens=usage.cached_input_tokens or 0,
            )
        collector.append(
            DeepSeekResponseEvidence(
                response_ordinal=len(collector),
                response_id=terminal.response_id,
                served_model=terminal.model,
                system_fingerprint=terminal.system_fingerprint,
                terminal_type=terminal.terminal_type,
                input_tokens=None if usage is None else usage.input_tokens,
                cached_input_tokens=(
                    None if usage is None else usage.cached_input_tokens
                ),
                cache_write_tokens=(
                    None if usage is None else usage.cache_write_tokens
                ),
                output_tokens=None if usage is None else usage.output_tokens,
                reasoning_output_tokens=(
                    None if usage is None else usage.reasoning_output_tokens
                ),
                cost_usd=cost_usd,
                usage_complete=usage is not None and usage.complete,
            )
        )

    def _invoke_serialized(
        self,
        invocation: Invocation,
        *,
        max_response_tokens: int,
    ) -> InvocationResult:
        if invocation.tools:
            raise ValueError(
                "DeepSeekBackend accepts only attached client function tools"
            )
        schema_json = _schema_json(invocation.response_schema)
        system_text = _prompt_system_text(invocation.system, schema_json)
        if invocation.conversation is not None:
            raise ValueError(
                "DeepSeek provider conversation continuity is not qualified"
            )
        if type(invocation.max_turns) is not int or not (
            1 <= invocation.max_turns <= _MAX_TOOL_ROUNDS
        ):
            raise ValueError(
                f"DeepSeek max_turns must be from 1 through {_MAX_TOOL_ROUNDS}"
            )
        if invocation.budget_usd is not None and (
            isinstance(invocation.budget_usd, bool)
            or not isinstance(invocation.budget_usd, (int, float))
            or not math.isfinite(invocation.budget_usd)
            or invocation.budget_usd <= 0
        ):
            raise ValueError("DeepSeek budget_usd must be finite and positive")
        policy = invocation.policy or _DEFAULT_POLICY
        if policy.retry_budget != 1:
            raise ValueError(
                "DeepSeek no-retry route requires policy.retry_budget == 1"
            )
        timeout_seconds = (
            self._timeout_seconds
            if policy.timeout_seconds is None
            else min(self._timeout_seconds, policy.timeout_seconds)
        )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("DeepSeek invocation timeout must be finite and positive")
        logical_deadline = time.monotonic() + timeout_seconds
        pricing_for_model(self.model)
        with self._state_lock:
            if self._closed:
                raise RuntimeError("DeepSeekBackend is closed")
            toolset = self._toolset
            attached_names = self._tool_names
            attached_write_names = self._tool_write_names
            attached_descriptor_json = self._tool_descriptor_json
            context_budget = self._context_budget
        tools: list[dict[str, Any]] = []
        tool_authority_digest: str | None = None
        dispatch_toolset = toolset
        mutation_host: Any = None
        if toolset is not None:
            from .tool_authority import snapshot_tool_authority

            authority_snapshot = snapshot_tool_authority(toolset)
            tools, current_names = _function_tools(
                toolset,
                authority_snapshot=authority_snapshot,
            )
            current_descriptor_json = json.dumps(
                tools,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if (
                current_names != attached_names
                or authority_snapshot.write_names != attached_write_names
                or current_descriptor_json != attached_descriptor_json
            ):
                raise ValueError(
                    "DeepSeek ToolSet descriptors or classification changed "
                    "after capability validation"
                )
            from .tool_authority import (
                APPROVAL_POLICY_PER_CALL_EXACT,
                tool_authority_digest as compute_tool_authority_digest,
            )

            tool_authority_digest = compute_tool_authority_digest(
                authority_snapshot,
                capability=self.capability,
                approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
                bridge_protocol="deepseek-client-functions",
                bridge_version=1,
                context_budget=context_budget,
            )
            frozen_toolset = _FrozenFunctionToolSet(
                toolset,
                authority_snapshot=authority_snapshot,
            )
            if authority_snapshot.write_names:
                from crucible.tools._isolated_mutation_host import (  # noqa: PLC0415
                    IsolatedMutationHost,
                    MutationHostQualificationError,
                    ProcessHostedToolSet,
                    build_isolated_toolset_spec,
                )

                try:
                    mutation_spec = build_isolated_toolset_spec(
                        toolset,
                        expected_write_names=authority_snapshot.write_names,
                        authority_snapshot=authority_snapshot,
                    )
                except MutationHostQualificationError as exc:
                    raise DeepSeekSDKQualificationError(str(exc)) from exc
                mutation_host = IsolatedMutationHost(
                    mutation_spec,
                    invocation_deadline=logical_deadline,
                )
                self._register_mutation_host(mutation_host)
                try:
                    mutation_host.start()
                    with self._state_lock:
                        if self._closed:
                            raise RuntimeError(
                                "DeepSeekBackend closed during mutation-host startup"
                            )
                    dispatch_toolset = ProcessHostedToolSet(
                        frozen_toolset,
                        mutation_host,
                    )
                except BaseException:
                    try:
                        mutation_host.stop()
                    except BaseException:
                        # Retain the exact host so close() can retry settlement.
                        raise
                    else:
                        self._unregister_mutation_host(mutation_host)
                        mutation_host = None
                    raise

        def settle_mutation_host() -> None:
            nonlocal mutation_host
            if mutation_host is None:
                return
            mutation_host.stop()
            self._unregister_mutation_host(mutation_host)
            mutation_host = None

        started_iso = datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
        started = time.monotonic()
        result: InvocationResult | None = None
        try:
            try:
                result = self._drive_tool_loop(
                    invocation,
                    policy=policy,
                    logical_deadline=logical_deadline,
                    system_text=system_text,
                    max_response_tokens=max_response_tokens,
                    toolset=dispatch_toolset,
                    tools=tools,
                    attached_names=attached_names,
                    context_budget=context_budget,
                    tool_authority_digest=tool_authority_digest,
                )
            except BaseException as invocation_error:
                try:
                    settle_mutation_host()
                except BaseException as cleanup_error:
                    from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                        ToolMutationIndeterminate,
                    )

                    if isinstance(invocation_error, ToolMutationIndeterminate):
                        invocation_error.add_note(
                            "DeepSeek retained its isolated mutation host; "
                            "DeepSeekBackend.close() must prove settlement "
                            "before workflow rollback."
                        )
                        raise invocation_error from cleanup_error
                    cleanup_error.add_note(
                        "DeepSeek invocation also raised "
                        f"{type(invocation_error).__name__}."
                    )
                    raise DeepSeekSDKQualificationError(
                        "confined DeepSeek mutation-host lifecycle did not settle"
                    ) from cleanup_error
                raise
            try:
                settle_mutation_host()
            except BaseException as cleanup_error:
                result = None
                raise DeepSeekSDKQualificationError(
                    "confined DeepSeek mutation-host lifecycle did not settle"
                ) from cleanup_error
            return result
        finally:
            self._sdk_sink.dump(
                invocation=invocation,
                result=result,
                started_iso=started_iso,
                duration_s=time.monotonic() - started,
                re_raised=result is None,
            )

    def _drive_tool_loop(
        self,
        invocation: Invocation,
        *,
        policy: BackendPolicy,
        logical_deadline: float,
        system_text: str,
        max_response_tokens: int,
        toolset: Any,
        tools: list[dict[str, Any]],
        attached_names: frozenset[str],
        context_budget: int | None,
        tool_authority_digest: str | None,
    ) -> InvocationResult:
        events: list[BackendExecutionEvent] = []
        transcripts: list[Message] = [Message(role="user", content=invocation.user)]
        segments: list[UsagePricingSegment] = []
        total_input = 0
        total_output = 0
        total_cached = 0
        total_cost = 0.0
        evidence_complete = True
        trace_complete = True
        response_input: Any = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": invocation.user},
        ]
        final_message = ""
        served_model: str | None = None
        provider_request_id: str | None = None
        cumulative_tool_bytes = 0
        seen_response_ids: set[str] = set()
        seen_tool_call_ids: set[str] = set()
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not isinstance(api_key, str) or not api_key.strip():
            raise RuntimeError(
                "DeepSeek dispatch requires DEEPSEEK_API_KEY; no generic or "
                "OpenAI credential fallback is permitted"
            )
        if api_key != api_key.strip() or len(api_key) > 4096 or "\x00" in api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is malformed")
        deadline = logical_deadline

        def add_event(
            kind: str, *, name: str | None = None, child_id: str | None = None
        ) -> None:
            events.append(
                BackendExecutionEvent(
                    sequence=len(events),
                    kind=kind,
                    name=name,
                    child_id=child_id,
                )
            )

        try:
            for turn in range(invocation.max_turns):
                add_event("request", name="deepseek_responses", child_id=str(turn))
                request = {
                    "protocol": _PROTOCOL,
                    "input": response_input,
                    "max_output_tokens": max_response_tokens,
                    "model": _MODEL,
                    "reasoning": {"effort": "high"},
                    "stream": True,
                    "tools": tools,
                }
                try:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise DeepSeekWorkerTimeout(
                            "DeepSeek invocation deadline elapsed before the "
                            "next physical response"
                        )
                    terminal = self._worker_runner(
                        request,
                        api_key=api_key,
                        timeout_seconds=remaining_seconds,
                        on_process_start=self._register_process,
                        on_process_stop=self._unregister_process,
                    )
                    if terminal.response_id in seen_response_ids:
                        raise DeepSeekSDKQualificationError(
                            "DeepSeek returned a duplicate physical response id"
                        )
                    response_call_ids = {
                        call.call_id for call in terminal.tool_calls
                    }
                    if len(response_call_ids) != len(terminal.tool_calls):
                        raise DeepSeekSDKQualificationError(
                            "DeepSeek returned duplicate function call ids"
                        )
                    if response_call_ids & seen_tool_call_ids:
                        raise DeepSeekSDKQualificationError(
                            "DeepSeek reused a function call id across responses"
                        )
                    # Establish invocation-wide identity uniqueness before
                    # recording usage/evidence or dispatching any tool.
                    seen_response_ids.add(terminal.response_id)
                    seen_tool_call_ids.update(response_call_ids)
                except DeepSeekWorkerTimeout as exc:
                    add_event("limit", name="deadline")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message=str(exc),
                            retriable=False,
                        ),
                        token_cost_complete=False,
                        terminal_complete=False,
                        trace_complete=False,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                except DeepSeekWorkerInvocationError as exc:
                    add_event("terminal", name="worker_error")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "DeepSeek worker returned sanitized error "
                                f"{exc.error_type}"
                            ),
                            retriable=False,
                        ),
                        token_cost_complete=False,
                        terminal_complete=False,
                        trace_complete=False,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                except (DeepSeekSDKQualificationError, DeepSeekSDKUnavailable) as exc:
                    # This may follow a paid but unparseable response. Retain
                    # every earlier physical segment, but never assert that
                    # the current response cost zero.
                    add_event("terminal", name="worker_qualification_error")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "DeepSeek worker qualification failed with "
                                f"{type(exc).__name__}"
                            ),
                            retriable=False,
                        ),
                        token_cost_complete=False,
                        terminal_complete=False,
                        trace_complete=False,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                self._record_response_evidence(terminal)
                served_model = terminal.model
                provider_request_id = terminal.response_id
                usage = terminal.usage
                if usage is None:
                    evidence_complete = False
                else:
                    cached_for_cost = usage.cached_input_tokens or 0
                    cost = compute_cost(
                        self.model,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_creation_tokens=usage.cache_write_tokens or 0,
                        cache_read_tokens=cached_for_cost,
                    )
                    segments.append(
                        UsagePricingSegment(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_creation_tokens=usage.cache_write_tokens or 0,
                            cache_read_tokens=cached_for_cost,
                            cost_usd=cost,
                        )
                    )
                    total_input += usage.input_tokens
                    total_output += usage.output_tokens
                    total_cached += cached_for_cost
                    total_cost += cost
                    evidence_complete = evidence_complete and usage.complete
                    add_event(
                        "usage",
                        name="complete" if usage.complete else "incomplete",
                        child_id=terminal.response_id,
                    )
                if terminal.model != self.model:
                    add_event("terminal", name="served_model_mismatch")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message="DeepSeek returned a model other than the exact request alias",
                            retriable=False,
                        ),
                        # Token counts are retained, but the requested-model
                        # price cannot prove cost for a mismatched served ID.
                        token_cost_complete=False,
                        terminal_complete=True,
                        trace_complete=False,
                        served_model=terminal.model,
                        provider_request_id=terminal.response_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if terminal.output_text:
                    final_message = terminal.output_text
                    transcripts.append(
                        Message(role="assistant", content=terminal.output_text)
                    )
                add_event(
                    "terminal",
                    name=terminal.terminal_type,
                    child_id=terminal.response_id,
                )
                if terminal.terminal_type != "response.completed":
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message=(
                                "DeepSeek Responses ended with "
                                f"{terminal.terminal_type}"
                            ),
                            retriable=False,
                        ),
                        token_cost_complete=evidence_complete,
                        terminal_complete=True,
                        trace_complete=trace_complete,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if (
                    invocation.budget_usd is not None
                    and total_cost > invocation.budget_usd
                ):
                    add_event("limit", name="budget_usd")
                    # A limit exit ends with its own logical terminal;
                    # ``_result`` keeps only the last terminal event.
                    add_event("terminal", name="budget_exhausted")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="budget_exhausted",
                        error=BudgetExhausted(
                            spent_usd=total_cost,
                            limit_usd=float(invocation.budget_usd),
                            message="DeepSeek post-response budget ceiling exceeded",
                        ),
                        token_cost_complete=evidence_complete,
                        terminal_complete=True,
                        trace_complete=trace_complete,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if not terminal.tool_calls:
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="end_turn",
                        error=None,
                        token_cost_complete=evidence_complete,
                        terminal_complete=True,
                        trace_complete=trace_complete,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if toolset is None:
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="error",
                        error=TransportError(
                            http_status=None,
                            message="DeepSeek returned a function call without attached authority",
                            retriable=False,
                        ),
                        token_cost_complete=evidence_complete,
                        terminal_complete=True,
                        trace_complete=False,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                if turn == invocation.max_turns - 1:
                    add_event("limit", name="max_turns")
                    add_event("terminal", name="max_turns")
                    return self._result(
                        final_message=final_message,
                        transcript=transcripts,
                        segments=segments,
                        total_input=total_input,
                        total_output=total_output,
                        total_cached=total_cached,
                        total_cost=total_cost,
                        stop_reason="max_turns",
                        error=MaxTurnsExceeded(
                            turns_used=turn + 1,
                            turns_limit=invocation.max_turns,
                            message="DeepSeek client function loop reached its hard round bound",
                        ),
                        token_cost_complete=evidence_complete,
                        terminal_complete=True,
                        trace_complete=trace_complete,
                        served_model=served_model,
                        provider_request_id=provider_request_id,
                        events=events,
                        tool_authority_digest=tool_authority_digest,
                    )
                outputs: list[dict[str, str]] = []
                for call in terminal.tool_calls:
                    add_event("tool_call", name=call.name, child_id=call.call_id)
                    if call.name not in attached_names:
                        rendered = f"ERROR: function {call.name!r} is outside attached authority"
                        trace_complete = False
                    else:
                        try:
                            arguments = json.loads(call.arguments)
                        except json.JSONDecodeError:
                            arguments = None
                        if not isinstance(arguments, dict):
                            rendered = "ERROR: function arguments must be a JSON object"
                        else:
                            from crucible.tools._dispatch_budget import (  # noqa: PLC0415
                                TerminalToolDispatchError,
                                ToolDispatchBudgetExceeded,
                                ToolMutationIndeterminate,
                            )

                            try:
                                rendered = str(toolset.dispatch(call.name, arguments))
                            except ToolMutationIndeterminate:
                                # An uncertain source mutation is terminal.
                                # Never ask the provider for another turn.
                                raise
                            except TerminalToolDispatchError as exc:
                                budget_exhausted = isinstance(
                                    exc,
                                    ToolDispatchBudgetExceeded,
                                )
                                if budget_exhausted:
                                    add_event(
                                        "limit",
                                        name="tool_dispatch_budget",
                                    )
                                dispatch_stop_reason = (
                                    "tool_budget_exhausted"
                                    if budget_exhausted
                                    else "tool_dispatch_terminal"
                                )
                                add_event(
                                    "terminal",
                                    name=dispatch_stop_reason,
                                )
                                return self._result(
                                    final_message=final_message,
                                    transcript=transcripts,
                                    segments=segments,
                                    total_input=total_input,
                                    total_output=total_output,
                                    total_cached=total_cached,
                                    total_cost=total_cost,
                                    stop_reason=dispatch_stop_reason,
                                    error=(
                                        MaxTurnsExceeded(
                                            turns_used=turn + 1,
                                            turns_limit=invocation.max_turns,
                                            message=(
                                                "DeepSeek tool loop stopped at "
                                                "a workflow-owned dispatch "
                                                f"boundary: {exc}"
                                            ),
                                        )
                                        if budget_exhausted
                                        else TransportError(
                                            http_status=None,
                                            message=(
                                                "DeepSeek tool loop terminated "
                                                "at a workflow-owned dispatch "
                                                f"boundary: {exc}"
                                            ),
                                            retriable=False,
                                        )
                                    ),
                                    token_cost_complete=evidence_complete,
                                    terminal_complete=True,
                                    trace_complete=trace_complete,
                                    served_model=served_model,
                                    provider_request_id=provider_request_id,
                                    events=events,
                                    tool_authority_digest=(
                                        tool_authority_digest
                                    ),
                                )
                            except Exception as exc:  # noqa: BLE001
                                rendered = (
                                    f"ERROR: {type(exc).__name__}: tool dispatch failed"
                                )
                    encoded_size = len(rendered.encode("utf-8"))
                    if (
                        context_budget is not None
                        and cumulative_tool_bytes + encoded_size > context_budget
                    ):
                        add_event("limit", name="context_bytes")
                        add_event("terminal", name="context_limit")
                        return self._result(
                            final_message=final_message,
                            transcript=transcripts,
                            segments=segments,
                            total_input=total_input,
                            total_output=total_output,
                            total_cached=total_cached,
                            total_cost=total_cost,
                            stop_reason="context_limit",
                            error=TransportError(
                                http_status=None,
                                message="DeepSeek tool-result context bound exceeded",
                                retriable=False,
                            ),
                            token_cost_complete=evidence_complete,
                            terminal_complete=True,
                            trace_complete=trace_complete,
                            served_model=served_model,
                            provider_request_id=provider_request_id,
                            events=events,
                            tool_authority_digest=tool_authority_digest,
                        )
                    cumulative_tool_bytes += encoded_size
                    add_event("tool_result", name=call.name, child_id=call.call_id)
                    transcripts.append(Message(role="tool", content=rendered))
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": rendered,
                        }
                    )
                continuation = json.loads(terminal.continuation_json)
                if not isinstance(continuation, list):  # pragma: no cover
                    raise AssertionError("validated DeepSeek continuation changed")
                response_input = [*response_input, *continuation, *outputs]
            raise AssertionError("DeepSeek tool loop escaped its hard bound")
        finally:
            api_key = ""

    def _result(
        self,
        *,
        final_message: str,
        transcript: list[Message],
        segments: list[UsagePricingSegment],
        total_input: int,
        total_output: int,
        total_cached: int,
        total_cost: float,
        stop_reason: str,
        error: Any,
        token_cost_complete: bool,
        terminal_complete: bool,
        trace_complete: bool,
        served_model: str | None,
        provider_request_id: str | None,
        events: list[BackendExecutionEvent],
        tool_authority_digest: str | None,
    ) -> InvocationResult:
        normalized_events = list(events)
        if trace_complete:
            terminal_indexes = [
                index
                for index, event in enumerate(normalized_events)
                if event.kind == "terminal"
            ]
            # Assay's trace contract has one logical terminal. Earlier
            # provider-response terminals in a client-owned function loop
            # remain evidenced by their response-bound usage records; they
            # are not additional logical invocation terminals.
            if terminal_indexes:
                final_terminal_index = terminal_indexes[-1]
                normalized_events = [
                    event
                    for index, event in enumerate(normalized_events)
                    if event.kind != "terminal" or index == final_terminal_index
                ]
            normalized_events = [
                BackendExecutionEvent(
                    sequence=sequence,
                    kind=event.kind,
                    name=event.name,
                    parent_id=event.parent_id,
                    child_id=event.child_id,
                )
                for sequence, event in enumerate(normalized_events)
            ]
        return InvocationResult(
            final_message=final_message,
            transcript=tuple(transcript),
            usage=Usage(
                input_tokens=total_input,
                output_tokens=total_output,
                cost_usd=total_cost,
                cache_creation_tokens=sum(
                    segment.cache_creation_tokens for segment in segments
                ),
                cache_read_tokens=total_cached,
            ),
            stop_reason=stop_reason,
            error=error,
            completeness=ExecutionCompleteness(
                token_usage_complete=token_cost_complete,
                cost_complete=token_cost_complete,
                terminal_complete=terminal_complete,
                trace_complete=trace_complete,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                served_model=served_model,
                requested_effort="high",
                # DeepSeek attests the returned model but does not return a
                # served reasoning-effort field. Never infer it from alias
                # equality with our requested high effort.
                served_effort=None,
                endpoint=_ENDPOINT,
                provider_request_id=provider_request_id,
                tool_authority_digest=tool_authority_digest,
                events=tuple(normalized_events),
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=tuple(segments),
                complete=token_cost_complete,
            ),
        )

    def close(self) -> None:
        with self._state_lock:
            self._closed = True

        def settle_current() -> list[BaseException]:
            with self._state_lock:
                processes = tuple(self._active_processes)
                mutation_hosts = tuple(self._active_mutation_hosts)
            errors: list[BaseException] = []
            for process in processes:
                try:
                    process.request_stop()
                    if not process.wait_for_settlement(
                        _STOP_GRACE_SECONDS * 2
                        + _STREAM_DRAIN_GRACE_SECONDS
                        + 1.0
                    ):
                        raise DeepSeekSDKQualificationError(
                            "DeepSeek worker close lacks trusted settlement proof"
                        )
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                else:
                    self._unregister_process(process)
            for host in mutation_hosts:
                try:
                    host.stop()
                except BaseException as exc:  # noqa: BLE001
                    errors.append(exc)
                else:
                    self._unregister_mutation_host(host)
            return errors

        initial_errors = settle_current()
        drain_timeout = (
            _STOP_GRACE_SECONDS * 2 + _STREAM_DRAIN_GRACE_SECONDS + 1.0
        )
        acquired = self._invoke_lock.acquire(timeout=drain_timeout)
        if acquired:
            try:
                final_errors = settle_current()
            finally:
                self._invoke_lock.release()
        else:
            final_errors = [
                DeepSeekSDKQualificationError(
                    "DeepSeekBackend.close could not drain the active invocation"
                )
            ]
        with self._state_lock:
            resources_remain = bool(
                self._active_processes or self._active_mutation_hosts
            )
        for cleanup_error in (*initial_errors, *final_errors):
            if isinstance(cleanup_error, KeyboardInterrupt):
                raise cleanup_error
        if final_errors or resources_remain:
            causes = final_errors or initial_errors
            cause = (
                causes[0]
                if causes
                else DeepSeekSDKQualificationError(
                    "a confined resource appeared during final close"
                )
            )
            raise DeepSeekSDKQualificationError(
                "DeepSeekBackend close could not settle every confined resource"
            ) from cause

    def __enter__(self) -> DeepSeekBackend:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


__all__ = [
    "DEEPSEEK_OPENAI_DISTRIBUTION_HASHES",
    "DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA",
    "DeepSeekBackend",
    "DeepSeekInvocationEvidence",
    "DeepSeekProjectLockedRuntimeIdentity",
    "DeepSeekResponseEvidence",
    "DeepSeekSBOMComponent",
    "DeepSeekSDKQualificationError",
    "DeepSeekSDKUnavailable",
    "capture_project_locked_deepseek_runtime_identity",
]
