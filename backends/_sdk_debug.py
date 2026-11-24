# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""SDK-debug sink for crucible backend adapters.

Every successful or failed Backend invocation can be dumped to a
per-attempt JSON file alongside the workflow's other run-folder outputs
(``events.jsonl``, ``result.json``, ``build.log``, ``verify.log``).

File naming: ``sdk_<seq>_<backend>.json`` where each sink proposes a
thread-safe monotonic sequence and atomically advances past names already
claimed in the run directory. One attempt (including SDK retries inside the
adapter's own retry budget) produces one uniquely named file.

Payload shape:

.. code-block:: json

    {
      "schema_version": "crucible.sdk-debug.v2",
      "parse_authority": "diagnostic_only",
      "authority": {
        "sdk_debug_payload": "diagnostic_only",
        "workflow_parse": "typed_workflow_records"
      },
      "seq": 0,
      "backend": "claude",
      "attempt": 0,
      "started_at": "2026-05-18T12:00:00+00:00",
      "duration_s": 12.5,
      "re_raised": false,
      "invocation": {
        "system": "<truncated>",
        "user": "<truncated>",
        "tools": [...],
        "max_turns": 5,
        "budget_usd": null,
        "policy": {...}
      },
      "result": {
        "final_message": "<truncated>",
        "transcript": [{"role": "...", "content": "<truncated>"}, ...],
        "usage": {"input_tokens": ..., "output_tokens": ..., "cost_usd": ...,
                  "cache_creation_tokens": ..., "cache_read_tokens": ...},
        "pricing_evidence": {"segments": [...], "complete": false},
        "stop_reason": "end_turn",
        "error": null | {"kind": "...", "message": "...", "retriable": ..., "http_status": ...}
      },
      "text_captures": [...],
      "full_redacted_result_ref": "sha256:<digest>" | null,
      "full_redacted_result_path": "sdk_artifacts/<digest>.json" | null,
      "traceback": null | "Traceback (most recent call last): ..."
    }

Redaction always happens before truncation or hashing. ``text_captures`` gives
exact redacted character counts and a SHA-256 for every bounded text field.
When a result exists, its complete redacted diagnostic projection is stored in
``sdk_artifacts/<digest>.json``. Exact environment secrets are removed as one
union of original-text spans before shape heuristics run, so delimiters and
overlapping credentials cannot expose a suffix. Invalid Unicode surrogate code
points become the replacement character before hashing or JSON encoding.
Typed workflow records remain the semantic parse authority; redaction can
intentionally make embedded JSON text invalid.

Version 2 adds the structured pricing ledger. Results from older adapters (or
historical v1 records where the field is absent) have incomplete pricing
evidence; absence must never be upgraded to an exact zero-cost ledger.

An already-present result sidecar is accepted only through a non-blocking,
no-follow descriptor after it proves to be a matching regular single-link
file. Symlinks, FIFOs, hard-linked files, and corrupt content fail only the
diagnostic dump and are never read or chmodded through their paths.

Failure isolation: dump writes are wrapped in a try/except so a
``log_dir`` permission error never breaks the workflow itself.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_SCHEMA_VERSION = "crucible.sdk-debug.v2"
_RESULT_ARTIFACT_SCHEMA_VERSION = "crucible.sdk-debug-result-artifact.v2"
_PARSE_AUTHORITY = "diagnostic_only"
_AUTHORITY = {
    "sdk_debug_payload": "diagnostic_only",
    "workflow_parse": "typed_workflow_records",
}

# Per-field truncation limits. Tune up if your debug needs more
# context, down if log_dir sizes are a concern.
_MAX_SYSTEM_PROMPT = 4_000
_MAX_USER_PROMPT = 16_000
_MAX_FINAL_MESSAGE = 8_000
_MAX_TRANSCRIPT_CONTENT_PER_MESSAGE = 4_000

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|"
    r"passwd|secret|token)\b(\s*[:=]\s*)([^\s,;\]\}\"']+|"
    r"\"[^\"]*\"|'[^']*')"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_SENSITIVE_ENV_RE = re.compile(
    r"(?i)(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH)"
)
_SURROGATE_RE = re.compile(r"[\uD800-\uDFFF]")
_LINK_SETTLE_ATTEMPTS = 20
_LINK_SETTLE_DELAY_S = 0.001


def _unicode_safe_text(value: str) -> str:
    """Replace non-encodable surrogate code points deterministically."""

    return _SURROGATE_RE.sub("\N{REPLACEMENT CHARACTER}", value)


def _redact_exact_values(value: str, secrets_to_remove: set[str]) -> str:
    """Replace the union of all exact-secret spans in the original text."""

    spans: list[tuple[int, int]] = []
    for secret in secrets_to_remove:
        start = 0
        while True:
            index = value.find(secret, start)
            if index < 0:
                break
            spans.append((index, index + len(secret)))
            # Advancing one character retains overlapping occurrences of the
            # same value as well as overlaps between distinct credentials.
            start = index + 1
    if not spans:
        return value

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    chunks: list[str] = []
    cursor = 0
    for start, end in merged:
        chunks.extend((value[cursor:start], "[REDACTED]"))
        cursor = end
    chunks.append(value[cursor:])
    return "".join(chunks)


def _truncate_redacted_text(
    value: str,
    limit: int,
    *,
    json_pointer: str,
) -> tuple[str, dict[str, Any]]:
    """Bound already-redacted text and return exact capture metadata."""

    original_chars = len(value)
    retained_chars = min(original_chars, limit)
    omitted_chars = original_chars - retained_chars
    truncated = omitted_chars > 0
    rendered = (
        value[:limit] + f"\n...[truncated {omitted_chars} chars]"
        if truncated
        else value
    )
    return rendered, {
        "json_pointer": json_pointer,
        "truncated": truncated,
        "original_chars": original_chars,
        "retained_chars": retained_chars,
        "omitted_chars": omitted_chars,
        "redacted_sha256": (
            "sha256:"
            + hashlib.sha256(value.encode("utf-8")).hexdigest()
        ),
    }


def _sensitive_env_values() -> set[str]:
    """Values of the parent's variables with a sensitive name, 8 characters up."""

    return {
        secret
        for name, secret in os.environ.items()
        if _SENSITIVE_ENV_RE.search(name) and len(secret) >= 8
    }


def _redact_text(value: str) -> str:
    """Remove common credentials and exact sensitive environment values."""

    sensitive_values = _sensitive_env_values()
    # Discover every exact match against the original string and replace the
    # union in one pass. Sequential replacement is unsafe: an assignment
    # heuristic can split a delimiter-containing secret, while replacing one
    # of two non-nested overlapping secrets can destroy the other match and
    # leave its distinguishing suffix.
    redacted = _redact_exact_values(value, sensitive_values)
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        redacted,
    )
    return _unicode_safe_text(redacted)


def redact_sdk_text(value: str) -> str:
    """Redact credential-shaped text before it reaches ordinary logs."""

    return _redact_text(value)


_LINE_WITHHELD = "[REDACTED: line withheld]"


def _framed_secret_lines() -> set[str]:
    """Lines of sensitive values that line-framed output does not hold whole.

    A value that holds a line break, or begins or ends in white space, is
    not found as it stands in output that was split at line breaks and
    right-stripped, or that a child printed after stripping the value. Its
    lines of at least 8 characters, stripped, are returned.
    """

    framed: set[str] = set()
    for secret in _sensitive_env_values():
        parts = [part.strip() for part in secret.split("\n")]
        if parts != [secret]:
            framed.update(part for part in parts if len(part) >= 8)
    return framed


def redact_sdk_lines(lines: Sequence[str]) -> list[str]:
    """Redact the lines of one bounded text; return one item per line.

    Each line is redacted on its own, each pair of neighbouring lines as
    one text, and all lines joined by line breaks as one text. A line is
    returned, as redacted on its own, only where the pair results and the
    joined result hold that same line and it holds none of the lines
    ``_framed_secret_lines`` names; any other line is replaced by a fixed
    marker. The caller bounds the text: every line is read four times.
    """

    per_line = [_redact_text(line) for line in lines]
    joined = _redact_text("\n".join(lines)).split("\n")
    if len(joined) == len(per_line):
        shown = [
            alone == together
            for alone, together in zip(per_line, joined, strict=True)
        ]
    else:
        shown = [False] * len(per_line)
        matcher = difflib.SequenceMatcher(a=per_line, b=joined, autojunk=False)
        for block in matcher.get_matching_blocks():
            end = block.a + block.size
            shown[block.a:end] = [True] * block.size
    for index in range(len(lines) - 1):
        pair = _redact_text(f"{lines[index]}\n{lines[index + 1]}").split("\n")
        if len(pair) != 2:
            shown[index] = shown[index + 1] = False
            continue
        shown[index] = shown[index] and pair[0] == per_line[index]
        shown[index + 1] = shown[index + 1] and pair[1] == per_line[index + 1]
    framed = _framed_secret_lines()
    return [
        alone
        if keep and not any(part in line for part in framed)
        else _LINE_WITHHELD
        for line, alone, keep in zip(lines, per_line, shown, strict=True)
    ]


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def _unicode_safe(value: Any) -> Any:
    """Make every payload string and mapping key strict-UTF-8 encodable."""

    if isinstance(value, str):
        return _unicode_safe_text(value)
    if isinstance(value, list):
        return [_unicode_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_unicode_safe(item) for item in value]
    if isinstance(value, dict):
        return {
            _unicode_safe_text(str(key)): _unicode_safe(item)
            for key, item in value.items()
        }
    return value


def _redact_payload_content(payload: dict[str, Any]) -> dict[str, Any]:
    """Redact captured content without mutating fixed schema identifiers."""

    redacted = dict(payload)
    for field in ("invocation", "result", "traceback"):
        redacted[field] = _redact(redacted.get(field))
    return redacted


def _write_private_atomic(path: Path, payload: bytes) -> None:
    """Atomically publish a mode-0600 debug record without replacement.

    Multiple backend-scoped sinks can target one run directory.  Linking the
    fully written temporary inode gives publication no-replace semantics, so
    two fresh sinks cannot overwrite the same ``sdk_0000_*`` record.
    """

    tmp = path.parent / f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("short write while publishing SDK debug record")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _serialize_dataclass(obj: Any) -> Any:
    """Recursively render dataclasses (and nested dataclasses) into dicts."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize_dataclass(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_dataclass(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _serialize_dataclass(v) for k, v in obj.items()}
    return obj


def _serialize_pricing_evidence(result: Any) -> dict[str, Any]:
    """Project a ledger, treating legacy/absent evidence as incomplete."""

    value = getattr(result, "pricing_evidence", None)
    if value is None:
        return {"segments": [], "complete": False}
    serialized = _serialize_dataclass(value)
    if not isinstance(serialized, dict):
        return {"segments": [], "complete": False}
    return serialized


def _json_safe(value: Any) -> Any:
    """Render every eventual JSON string before the redaction boundary."""

    serialized = _serialize_dataclass(value)
    if serialized is None or isinstance(serialized, (str, int, float, bool)):
        return serialized
    if isinstance(serialized, (list, tuple)):
        return [_json_safe(item) for item in serialized]
    if isinstance(serialized, dict):
        return {
            str(key): _json_safe(item) for key, item in serialized.items()
        }
    return str(serialized)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _no_follow_flags(*, directory: bool = False) -> int:
    """Return fail-closed POSIX flags for non-blocking diagnostic opens."""

    required = ("O_NOFOLLOW", "O_NONBLOCK")
    missing = [name for name in required if not hasattr(os, name)]
    if directory and not hasattr(os, "O_DIRECTORY"):
        missing.append("O_DIRECTORY")
    if missing:
        raise OSError(
            "secure SDK artifact inspection is unavailable; missing "
            + ", ".join(missing)
        )
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if directory:
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _verify_directory_path_binding(path: Path, directory_fd: int) -> None:
    """Require ``path`` to still name the opened directory object."""

    opened = os.fstat(directory_fd)
    current = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or current.st_dev != opened.st_dev
        or current.st_ino != opened.st_ino
    ):
        raise OSError(
            f"SDK artifact directory changed during publication: {path}"
        )


def _ensure_private_artifact_dir(log_dir: Path) -> tuple[Path, int]:
    """Open and return the private artifact directory without following it.

    The caller owns the returned descriptor and must keep it open for every
    sidecar operation.  Returning the descriptor, rather than only the path,
    prevents a later parent-directory rename or symlink substitution from
    redirecting publication into an attacker-selected directory.
    """

    artifact_dir = log_dir / "sdk_artifacts"
    try:
        artifact_dir.mkdir(mode=0o700)
    except FileExistsError:
        pass
    fd = os.open(artifact_dir, _no_follow_flags(directory=True))
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISDIR(metadata.st_mode):
            raise OSError(
                f"SDK artifact path is not a private directory: {artifact_dir}"
            )
        os.fchmod(fd, 0o700)
        _verify_directory_path_binding(artifact_dir, fd)
    except Exception:
        os.close(fd)
        raise
    return artifact_dir, fd


def _write_private_atomic_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> None:
    """Publish a private no-replace record relative to an open directory."""

    tmp_name = f".tmp_{name}_{os.getpid()}_{secrets.token_hex(4)}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(tmp_name, flags, 0o600, dir_fd=directory_fd)
        try:
            os.fchmod(fd, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError(
                        "short write while publishing SDK debug artifact"
                    )
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.link(
            tmp_name,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    finally:
        try:
            os.unlink(tmp_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


def _verify_existing_private_artifact_at(
    directory_fd: int,
    name: str,
    expected: bytes,
) -> None:
    """Validate matching sidecar bytes relative to its verified directory."""

    fd = os.open(name, _no_follow_flags(), dir_fd=directory_fd)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError(
                f"SDK debug result artifact is not a regular file: {name}"
            )

        # Atomic publication temporarily gives the inode a second link: its
        # random .tmp name and the final digest name. Give a winning publisher
        # a small bounded window to unlink that temporary name, but never
        # accept the pre-existing inode until it has exactly one link.
        for _attempt in range(_LINK_SETTLE_ATTEMPTS):
            if metadata.st_nlink == 1:
                break
            if metadata.st_nlink != 2:
                break
            time.sleep(_LINK_SETTLE_DELAY_S)
            metadata = os.fstat(fd)
        if metadata.st_nlink != 1:
            raise OSError(
                "SDK debug result artifact must have exactly one link: "
                f"{name}"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise OSError(
                "SDK debug result artifact is not mode 0600: "
                f"{name}"
            )
        if metadata.st_size != len(expected):
            raise OSError(
                "SDK debug result artifact digest collision or corruption: "
                f"{name}"
            )

        retained = bytearray()
        while len(retained) < len(expected):
            chunk = os.read(fd, min(64 * 1024, len(expected) - len(retained)))
            if not chunk:
                break
            retained.extend(chunk)
        extra = os.read(fd, 1)
        if bytes(retained) != expected or extra:
            raise OSError(
                "SDK debug result artifact digest collision or corruption: "
                f"{name}"
            )

        path_metadata = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
            or path_metadata.st_nlink != 1
        ):
            raise OSError(
                "SDK debug result artifact changed during inspection: "
                f"{name}"
            )
    finally:
        os.close(fd)


def _persist_full_redacted_result(
    log_dir: Path,
    result_payload: dict[str, Any],
) -> tuple[str, str]:
    """Publish one deduplicated, content-addressed redacted result artifact."""

    sidecar = {
        "schema_version": _RESULT_ARTIFACT_SCHEMA_VERSION,
        "parse_authority": _PARSE_AUTHORITY,
        "authority": dict(_AUTHORITY),
        "result": result_payload,
    }
    encoded = _canonical_json_bytes(sidecar)
    digest = hashlib.sha256(encoded).hexdigest()
    artifact_dir, artifact_dir_fd = _ensure_private_artifact_dir(log_dir)
    name = f"{digest}.json"
    try:
        try:
            _write_private_atomic_at(artifact_dir_fd, name, encoded)
        except FileExistsError:
            _verify_existing_private_artifact_at(
                artifact_dir_fd,
                name,
                encoded,
            )
        _verify_directory_path_binding(artifact_dir, artifact_dir_fd)
    finally:
        os.close(artifact_dir_fd)
    return f"sha256:{digest}", f"sdk_artifacts/{name}"


def _truncate_payload(
    full_redacted_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply display bounds after redaction and retain integrity metadata."""

    payload = dict(full_redacted_payload)
    captures: list[dict[str, Any]] = []
    invocation = payload.get("invocation")
    if isinstance(invocation, dict):
        invocation = dict(invocation)
        payload["invocation"] = invocation
        for field, limit in (
            ("system", _MAX_SYSTEM_PROMPT),
            ("user", _MAX_USER_PROMPT),
        ):
            rendered, capture = _truncate_redacted_text(
                str(invocation.get(field) or ""),
                limit,
                json_pointer=f"/invocation/{field}",
            )
            invocation[field] = rendered
            captures.append(capture)

    result = payload.get("result")
    if isinstance(result, dict):
        result = dict(result)
        payload["result"] = result
        rendered, capture = _truncate_redacted_text(
            str(result.get("final_message") or ""),
            _MAX_FINAL_MESSAGE,
            json_pointer="/result/final_message",
        )
        result["final_message"] = rendered
        captures.append(capture)
        transcript = []
        for index, raw_message in enumerate(result.get("transcript") or ()):
            message = dict(raw_message)
            rendered, capture = _truncate_redacted_text(
                str(message.get("content") or ""),
                _MAX_TRANSCRIPT_CONTENT_PER_MESSAGE,
                json_pointer=f"/result/transcript/{index}/content",
            )
            message["content"] = rendered
            transcript.append(message)
            captures.append(capture)
        result["transcript"] = transcript
    return payload, captures


class SdkDebugSink:
    """Writes one ``sdk_<seq>_<backend>.json`` per invocation attempt.

    Thread-safe: ``assay.parallel`` may invoke a single backend concurrently,
    and independent sinks may share one run directory. The local counter is
    guarded by a lock; no-replace publication resolves cross-sink collisions.

    When ``log_dir`` is None the sink is a no-op — callers can always
    construct one without paying the dump cost when debug capture
    isn't wanted.
    """

    def __init__(self, log_dir: Path | None, backend_name: str) -> None:
        self.log_dir: Path | None = Path(log_dir) if log_dir is not None else None
        self.backend_name = backend_name
        self._lock = threading.Lock()
        self._seq = 0

    def enabled(self) -> bool:
        return self.log_dir is not None

    def _next_seq(self) -> int:
        with self._lock:
            seq = self._seq
            self._seq += 1
            return seq

    def dump(
        self,
        *,
        invocation: Any,
        result: Any,
        started_iso: str,
        duration_s: float,
        traceback_str: str | None = None,
        re_raised: bool = False,
        attempt: int = 0,
    ) -> Path | None:
        """Write one sdk_<seq>.json. No-op when log_dir is None.

        ``result`` may be None when the SDK raised an unclassified
        exception that we re-raised — in that case ``traceback_str``
        carries the formatted exception chain.
        """
        if self.log_dir is None:
            return None
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            full_payload = _unicode_safe(
                _redact_payload_content(
                    _json_safe(
                        _build_payload(
                            invocation=invocation,
                            result=result,
                            started_iso=started_iso,
                            duration_s=duration_s,
                            traceback_str=traceback_str,
                            re_raised=re_raised,
                            attempt=attempt,
                            seq=-1,
                            backend=self.backend_name,
                        )
                    )
                )
            )
            payload, text_captures = _truncate_payload(full_payload)
            full_result_ref: str | None = None
            full_result_path: str | None = None
            full_result = full_payload.get("result")
            if isinstance(full_result, dict):
                full_result_ref, full_result_path = (
                    _persist_full_redacted_result(self.log_dir, full_result)
                )
            payload["text_captures"] = text_captures
            payload["full_redacted_result_ref"] = full_result_ref
            payload["full_redacted_result_path"] = full_result_path
            while True:
                seq = self._next_seq()
                path = self.log_dir / f"sdk_{seq:04d}_{self.backend_name}.json"
                payload["seq"] = seq
                encoded = json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ).encode("utf-8")
                try:
                    _write_private_atomic(path, encoded)
                except FileExistsError:
                    # A different sink/process won this directory-scoped
                    # sequence. Preserve its record and claim the next one.
                    continue
                return path
        except Exception as exc:
            # No diagnostic serialization or filesystem failure may mask the
            # authoritative provider result. Even logging is best-effort: a
            # hostile custom handler must not turn this boundary into a throw.
            try:
                detail = _redact_text(str(exc))
            except Exception:
                detail = "[unavailable diagnostic failure detail]"
            try:
                redacted_log_dir = _redact_text(str(self.log_dir))
            except Exception:
                redacted_log_dir = "[unavailable diagnostic log directory]"
            try:
                logger.warning(
                    "SdkDebugSink: failed to write sdk_<seq>.json under %s: %s",
                    redacted_log_dir,
                    detail,
                )
            except Exception:
                pass
            return None


def _build_payload(
    *,
    invocation: Any,
    result: Any,
    started_iso: str,
    duration_s: float,
    traceback_str: str | None,
    re_raised: bool,
    attempt: int,
    seq: int,
    backend: str,
) -> dict:
    inv_payload: dict | None = None
    if invocation is not None:
        inv_payload = {
            "system": getattr(invocation, "system", "") or "",
            "user": getattr(invocation, "user", "") or "",
            "tools": [
                {"name": t.name, "description": t.description}
                for t in (getattr(invocation, "tools", None) or ())
            ],
            "max_turns": getattr(invocation, "max_turns", None),
            "budget_usd": getattr(invocation, "budget_usd", None),
            "policy": _serialize_dataclass(
                getattr(invocation, "policy", None)
            ),
        }

    result_payload: dict | None = None
    if result is not None:
        transcript_payload = []
        for msg in (getattr(result, "transcript", None) or ()):
            transcript_payload.append({
                "role": getattr(msg, "role", "?"),
                "content": getattr(msg, "content", "") or "",
            })
        err = getattr(result, "error", None)
        err_payload: dict | None = None
        if err is not None:
            err_payload = {
                "kind": getattr(err, "kind", None),
                "message": getattr(err, "message", None),
                "retriable": getattr(err, "retriable", None),
                "http_status": getattr(err, "http_status", None),
            }
        result_payload = {
            "final_message": getattr(result, "final_message", "") or "",
            "transcript": transcript_payload,
            "usage": _serialize_dataclass(
                getattr(result, "usage", None)
            ),
            "pricing_evidence": _serialize_pricing_evidence(result),
            "stop_reason": getattr(result, "stop_reason", None),
            "error": err_payload,
        }

    return {
        "schema_version": _SCHEMA_VERSION,
        "parse_authority": _PARSE_AUTHORITY,
        "authority": dict(_AUTHORITY),
        "seq": seq,
        "backend": backend,
        "attempt": attempt,
        "started_at": started_iso,
        "duration_s": duration_s,
        "re_raised": re_raised,
        "invocation": inv_payload,
        "result": result_payload,
        "traceback": traceback_str,
    }


__all__ = ["SdkDebugSink"]
