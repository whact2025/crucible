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
#   so the adapter's 200-entry stderr bound counts lines: a rejected-resume
#   reason is seen, and reported as ``stale_conversation``, unless the CLI
#   writes 200 or more lines after it. One line handed over is at most
#   ``max_buffer_size`` plus one pipe read long, because a longer run
#   without a newline is handed over in pieces. What the adapter keeps of
#   these lines is recorded below;
# - ``close()`` is shielded against cancellation, bounds every wait, and
#   keeps a child it could not reap tracked for the exit-time reaper;
# - stdout is framed by line: a large frame no longer loses whitespace at a
#   read boundary, and a malformed JSON line fails the turn at once.
#
# What the adapter does with the stderr lines of the second point, recorded
# on 2026-11-23 and decided by a panel on that day. It replaces the
# consequence recorded here on 2026-11-21, that the adapter redacts each
# kept entry on its own and a credential shape across a line break stays
# readable. Three words have one meaning each below. An entry is "dropped"
# when it is gone, and the header counts it. "Not kept" are the words of a
# size note: an entry whose text was replaced by its size, in no number. A
# line is "withheld" when a fixed marker stands in its place.
#
# 1. What is kept (``_CliStderrWindow`` in ``claude.py``). Of one physical
#    attempt the adapter keeps the last 200 entries and at most 65 536 bytes
#    of UTF-8 over them joined by line breaks, in whole entries, the oldest
#    dropped first. An entry that alone is over 65 536 bytes is not kept:
#    its text is not in the record, only a note of its size in bytes as
#    received, before redaction, also when it is the last or the only thing
#    the CLI wrote. "Line", in the note and in the header, is an entry as
#    the SDK hands it over, so a run without a line break gives one note for
#    each piece over the bound. The note counts as one line of "last N of
#    M". The bound is on what is kept, and what is published can be longer:
#    a redacted line can be longer than the kept one, and a withheld line is
#    a marker of 25 characters whatever its length, so neither the ERROR
#    record nor ``cli_stderr`` is held to 65 536 bytes.
# 2. What the header counts. "earlier stderr lines dropped: N" is printed
#    whenever the adapter received more entries than it holds, for either
#    bound: N is the entries received minus the entries kept. An entry is
#    what the SDK hands over, so a run without a line break counts once for
#    each piece. An entry replaced by its note is not in N: the note stands
#    for it and is one of the M lines. Blank lines, and output the SDK did
#    not hand over (whatever follows a byte sequence that is not UTF-8), are
#    in no number. ``cli_stderr`` carries no trace of dropped entries. For
#    every output, entries received = N dropped + M kept. No number says how
#    many of the M lines are size notes or withheld lines, and a header
#    without the clause does not say that the whole error output is shown.
# 3. What the record loses to the byte bound. It shows fewer than 50 lines
#    once the last 50 lines, each at most 65 536 bytes, hold more than
#    65 487 bytes together (an average of 1 310 bytes or more for lines of
#    one length). The line that states the cause is gone when more than
#    about 64 KB, in lines of at most 65 536 bytes each, follow it within
#    the last 49 lines. Neither sentence holds for a line over the bound:
#    its note has about 50 bytes and does not push the lines before it out.
# 4. The rejected resume. For a resumed handle the markers of a rejected
#    resume are read as each entry arrives, for the last 200 entries
#    received, kept or not. A consequence of that window, recorded with
#    this change: a rejection reason followed by 200 or more entries is not
#    recognized. The attempt then ends as a transport error and is not
#    retried; the handle stays active; in a planner session the operation is
#    left indeterminate, the next call needs an operator recovery, and after
#    it the same handle is sent again; the reason is then in no log record
#    and not in ``cli_stderr``.
# 5. What the published tail is for: a session process that ends without a
#    terminal of its protocol. Under the rule of point 6 the adapter
#    publishes the last 50 kept entries in one ERROR record and all kept
#    entries in ``cli_stderr``.
# 6. The rule of redaction (``redact_sdk_lines`` in ``_sdk_debug.py``): never
#    joined instead of per entry, and never the function applied to its own
#    output. Every logged line is the per-entry result of its entry or one
#    fixed marker. The form is a line agreement (per entry and joined, on
#    the bounded raw kept entries; a line of the CLI is printed only where
#    both hold it); the union of spans is not taken. Beside it two checks, and
#    only these: (1) the same agreement over each pair of neighbouring
#    entries; (2) an entry is withheld when it holds a stripped line, 8
#    characters or more, of a sensitive environment value that holds a line
#    break or begins or ends in white space. Each reads bounded raw kept
#    entries, never output of the function, replaces neither pass and can
#    only withhold. A size note is no line of the CLI: it is printed as
#    redacted on its own, never as the marker. A further check, text held
#    beside the kept entries, or other printed text needs its own ruling and
#    a run counting the plain lines it withholds.
# 7. What the rule costs. Whole lines are withheld where a pass disagrees
#    or a line of check 2 is held, harmless ones among them; the line that
#    states a cause is a likely one. The pair agreement withholds the plain
#    line after a name whose own line follows another name, also in output
#    without any credential. The lines of check 2 are taken from every
#    variable whose name the shared pattern calls sensitive (``KEY``,
#    ``TOKEN``, ``SECRET``, ``PASSWORD``, ``PASSWD``, ``CREDENTIAL`` or
#    ``AUTH`` in its name), so a value under such a name that is no secret
#    withholds lines as well: one of several lines, every line that holds
#    one of its lines; one of one line that begins or ends in white space,
#    every kept line that holds its stripped form, where the redaction of
#    that entry alone prints the line. The handler takes about twice the
#    time of redacting each entry alone, as far as measured, and that time
#    grows with the number of environment variables.
# 8. What is not covered, and is not claimed.
#    A. What the shared patterns miss on one line they miss across a line
#    break as well. The rule adds no pattern: it lets the three steps of
#    ``_redact_text`` see across a line break and nothing more. In the forms
#    a credential has on the wire these are a quoted key as in a token
#    response, a name inside a longer name such as a refresh token or a
#    variable name, and a scheme other than ``Bearer``. On the
#    organization-session route the adapter knows no exact value (it hands
#    the CLI no credential), so only the shape rules apply there. The
#    classes: (1) a scheme other than ``Bearer``, where the name takes the
#    scheme word as its value; (2) a name inside a longer name, an
#    underscore being no word boundary, and a quoted JSON key, whose closing
#    quote stands between the name and the separator; (3) a quote that is
#    never closed, and a quote inside the quoted value; (4) a delimiter
#    inside the value, where the part after it stays; (5) two names on one
#    line before one value, where the first takes the second as its value;
#    (6) a name without a separator, color codes between the name and the
#    separator, and the marker text as the CLI's own output before a value
#    on the same line.
#    B. What a line break still hides. (1) A shape whose first or last line
#    the bound took: dropped by either bound, replaced by its size note, or
#    cut at the SDK's piece boundary of a run without a line break. The
#    value line is then logged as its own redaction leaves it; nothing is
#    built for this class. A first line dropped by the entry bound, or by
#    the byte bound with more than 50 lines kept, leaves the value in
#    ``cli_stderr`` only. A first line dropped by the byte bound with at
#    most 50 lines kept, replaced by a note among the last 50 lines, or cut
#    at a piece boundary leaves it in the ERROR record and in ``stderr.log``
#    as well, whenever the value line is among the last 50 kept. The header
#    carries the clause for a drop and a kept note is never the marker, so
#    such a seam can be seen in the tail; a note that stands 51 or more
#    lines from the end is not among the lines shown. A value of 8
#    characters or more that the parent holds in a sensitive variable is
#    hidden at a seam, as anywhere else, where it stands whole in one kept
#    entry: the value as held on one line, or, of a value that holds a line
#    break or begins or ends in white space, each of its lines that has 8
#    characters or more after stripping (check 2). Not hidden at a seam: a
#    value, or a line of such a value, under 8 characters; a value cut by
#    the SDK's piece boundary or printed with a line break inside it; a
#    value printed in another form. On the organization-session route the
#    adapter knows no such value. (2) A value broken in its middle, by a
#    line break, by a lone carriage return inside one entry, or by the SDK's
#    piece boundary: the part before the break is redacted when its name
#    stands before it, the rest is readable. A name, or the word ``Bearer``,
#    broken in its middle is the same class. (3) A line between the name and
#    the value. (4) The same value a second time where no pass matches it:
#    the withheld line is hidden, the repeated line has no name before it in
#    any pass and is logged (an exact environment value is found wherever it
#    stands; this is about values the adapter does not know). (5) An
#    environment value that holds no line break and no white space at its
#    ends and is printed with a break inside it, or any environment value
#    printed in another form (encoded, or wrapped at other places than its
#    own line breaks). (6) A line of an environment value that is shorter
#    than 8 characters after stripping, where the passes do not match the
#    value whole.
#    The two checks do not cover B (6), B (5), A (5) and B (3). Check 2
#    mends this one path: the same miss of the exact-value step, which looks
#    for a value as the parent holds it, stays for every other caller of
#    ``redact_sdk_text``. Both the exact-value step and check 2 know values
#    the parent's environment holds, nothing else: on the
#    organization-session route neither reaches the session's own
#    credential; on the other routes check 2 reaches the token that the
#    adapter hands to the CLI stripped.
#    Besides. Against the joined pass the agreement's promise is by string
#    where the alignment decides: a line is shown only if the very same line
#    is a line of the joined result. The entry in flight is the SDK's: at
#    most ``max_buffer_size`` plus one pipe read in characters, held by the
#    SDK's framer before the callback is called; the adapter's bound starts
#    at the callback. A byte sequence that is not UTF-8 ends the SDK's
#    stderr reader, and nothing of that read or after it is delivered. Other
#    text leaves the adapter without this rule: the message of a typed error
#    is built from the exception text. ``stderr.log`` has no second
#    redaction: what the handler logs is what is written. Nothing was
#    observed of what the real CLI writes to stderr.
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
