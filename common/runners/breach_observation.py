# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Trusted binding for exact Python breach-test observations.

The repository's normal ``test_command`` is a project-owned suite gate.  It
may be a shell script, build-system driver, or another non-pytest command and
must never be rewritten to make Crucible's exact-node observer work.  Python
projects that need a distinct interpreter therefore declare the static,
content-hashed ``per_language.python.breach_test_command`` parameter.

Only a direct Python ``-m pytest`` command, a direct pytest console launcher,
or one of the deliberately enumerated environment launchers is accepted.
The explicit command carries no selectors or filters; Crucible appends the
single canonical node itself through its private observer bootstrap.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from hashlib import sha256
from collections.abc import Mapping, Sequence
from pathlib import Path


BREACH_TEST_COMMAND_KEY = "breach_test_command"
BREACH_OBSERVATION_FAILURE_KEY = "breach_observation_failure"
BREACH_OBSERVATION_FAILURE_SCHEMA = (
    "crucible.breach-observation-failure.v1"
)
BREACH_OBSERVATION_FAILURE_KINDS = frozenset({
    "channel",
    "configuration",
    "infrastructure",
})

_PYTHON_EXECUTABLE_RE = re.compile(r"python(?:\d+(?:\.\d+)*)?\Z", re.I)
_PYTEST_EXECUTABLES = frozenset({"pytest", "py.test"})
_ENVIRONMENT_LAUNCHERS = frozenset({"pipenv", "pdm", "poetry", "uv"})
_MAX_FAILURE_DETAIL_CHARS = 1_000
_PYTEST_IMPORT_TIMEOUT_S = 30.0


class BreachObservationConfigurationError(ValueError):
    """The static exact-observation command is absent or unsafe."""


class BreachObservationChannelError(ValueError):
    """The trusted observer process returned an invalid protocol result."""


class BreachObservationInfrastructureError(RuntimeError):
    """The trusted observer process could not be executed."""


@dataclass(frozen=True)
class _FileIdentity:
    lexical_path: str
    link_target: str | None
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    digest: str


@dataclass(frozen=True)
class PythonObservationBinding:
    """Immutable runtime binding retained across the aggregate suite gate."""

    python_prefix: tuple[str, ...]
    executable_identity: _FileIdentity
    pytest_identity: _FileIdentity
    candidate_root: str


def _command_tokens(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BreachObservationConfigurationError(
            f"runner parameter {path!r} must be a non-empty list[str]"
        )
    if not all(
        isinstance(token, str) and token and "\x00" not in token
        for token in value
    ):
        raise BreachObservationConfigurationError(
            f"runner parameter {path!r} must be a non-empty list[str]"
        )
    return tuple(value)


def _basename(token: str) -> str:
    return Path(token).name.lower()


def _parse_pytest_command(
    value: object,
    *,
    path: str,
    allow_suite_arguments: bool,
) -> tuple[str, ...]:
    command = _command_tokens(value, path=path)
    first = _basename(command[0])

    if (
        _PYTHON_EXECUTABLE_RE.fullmatch(first) is not None
        and len(command) >= 3
        and command[1:3] == ("-m", "pytest")
    ):
        prefix = command[:3]
    elif first in _PYTEST_EXECUTABLES:
        prefix = command[:1]
    elif (
        first in _ENVIRONMENT_LAUNCHERS
        and len(command) >= 3
        and command[1] == "run"
        and _basename(command[2]) in _PYTEST_EXECUTABLES
    ):
        prefix = command[:3]
    else:
        raise BreachObservationConfigurationError(
            f"runner parameter {path!r} must be direct 'python -m pytest', "
            "a pytest launcher, or a supported '<environment> run pytest' "
            "launcher"
        )

    if not allow_suite_arguments and len(prefix) != len(command):
        raise BreachObservationConfigurationError(
            f"runner parameter {path!r} may not contain pytest selectors, "
            "filters, options, or node arguments"
        )
    return prefix


def validate_breach_test_command(value: object, *, path: str) -> None:
    """Validate the content-hashed explicit command without touching the host."""

    command = _parse_pytest_command(
        value,
        path=path,
        allow_suite_arguments=False,
    )
    if (
        len(command) != 3
        or not Path(command[0]).is_absolute()
        or _PYTHON_EXECUTABLE_RE.fullmatch(_basename(command[0])) is None
        or command[1:] != ("-m", "pytest")
    ):
        raise BreachObservationConfigurationError(
            f"runner parameter {path!r} must be an absolute direct "
            "'python -m pytest' command"
        )


def _resolved_executable(token: str, *, cwd: str | Path) -> Path:
    candidate: Path | None
    if os.sep in token or (os.altsep is not None and os.altsep in token):
        path = Path(token)
        candidate = Path(os.path.abspath(
            path if path.is_absolute() else Path(cwd) / path,
        ))
    else:
        found = shutil.which(token)
        candidate = Path(found) if found is not None else None
    if (
        candidate is None
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
    ):
        raise BreachObservationConfigurationError(
            f"exact pytest observation executable is unavailable: {token!r}"
        )
    # Preserve the lexical invocation path.  Python uses it to discover
    # ``pyvenv.cfg``; resolving ``.venv/bin/python`` to the system binary
    # silently escapes the selected virtual environment.
    return candidate


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _trusted_file_identity(
    path: str | Path,
    *,
    candidate_root: str | Path,
    executable: bool,
) -> _FileIdentity:
    lexical = Path(os.path.abspath(path))
    trusted_root = Path(candidate_root).resolve()
    try:
        real = lexical.resolve(strict=True)
        lexical.lstat()
        target_stat = lexical.stat()
        link_target = os.readlink(lexical) if lexical.is_symlink() else None
        digest = sha256(lexical.read_bytes()).hexdigest()
    except OSError as exc:
        raise BreachObservationInfrastructureError(
            "cannot bind an exact-observation runtime file: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if _is_within(lexical, trusted_root) or _is_within(real, trusted_root):
        raise BreachObservationConfigurationError(
            "exact-observation runtime files may not reside in the mutable "
            f"candidate workspace: {lexical}"
        )
    if executable and not os.access(lexical, os.X_OK):
        raise BreachObservationConfigurationError(
            f"exact-observation executable is not executable: {lexical}"
        )
    return _FileIdentity(
        lexical_path=str(lexical),
        link_target=link_target,
        device=target_stat.st_dev,
        inode=target_stat.st_ino,
        mode=target_stat.st_mode,
        size=target_stat.st_size,
        mtime_ns=target_stat.st_mtime_ns,
        digest=digest,
    )


def _executable_identity(
    executable: str | Path,
    *,
    candidate_root: str | Path,
) -> _FileIdentity:
    return _trusted_file_identity(
        executable,
        candidate_root=candidate_root,
        executable=True,
    )


def assert_python_observation_binding(
    binding: PythonObservationBinding,
) -> None:
    """Fail closed if the bound interpreter changed after the suite gate."""

    current = _executable_identity(
        binding.executable_identity.lexical_path,
        candidate_root=binding.candidate_root,
    )
    if current != binding.executable_identity:
        raise BreachObservationInfrastructureError(
            "exact-observation Python executable changed after preflight"
        )
    current_pytest = _trusted_file_identity(
        binding.pytest_identity.lexical_path,
        candidate_root=binding.candidate_root,
        executable=False,
    )
    if current_pytest != binding.pytest_identity:
        raise BreachObservationInfrastructureError(
            "exact-observation pytest runtime changed after preflight"
        )


def _absolute_pytest_interpreter(
    pytest_path: Path,
    *,
    explicit: bool,
) -> Path:
    try:
        with pytest_path.open("rb") as stream:
            first_line = stream.readline(512)
    except OSError as exc:
        raise BreachObservationConfigurationError(
            "cannot inspect the configured pytest launcher"
        ) from exc
    if first_line.startswith(b"#!"):
        try:
            shebang = first_line[2:].decode("utf-8").strip().split()
        except UnicodeDecodeError:
            shebang = []
        if shebang and Path(shebang[0]).is_absolute():
            interpreter = Path(shebang[0])
            if (
                _PYTHON_EXECUTABLE_RE.fullmatch(
                    interpreter.name.lower(),
                ) is not None
                and interpreter.is_file()
                and os.access(interpreter, os.X_OK)
            ):
                return interpreter

    for name in ("python", "python3"):
        sibling = pytest_path.with_name(name)
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return sibling
    if explicit:
        raise BreachObservationConfigurationError(
            "explicit pytest launcher has no absolute executable Python "
            "interpreter; configure direct 'python -m pytest'"
        )
    return Path(os.path.abspath(sys.executable))


def pytest_python_prefix(
    command_prefix: Sequence[str],
    *,
    cwd: str | Path,
    explicit: bool,
) -> list[str]:
    """Resolve one parsed pytest prefix to its exact Python environment."""

    command = tuple(command_prefix)
    first_path = _resolved_executable(command[0], cwd=cwd)
    first = _basename(command[0])
    if (
        _PYTHON_EXECUTABLE_RE.fullmatch(first) is not None
        and len(command) == 3
        and command[1:] == ("-m", "pytest")
    ):
        return [str(first_path)]
    if first in _PYTEST_EXECUTABLES and len(command) == 1:
        return [str(_absolute_pytest_interpreter(
            first_path,
            explicit=explicit,
        ))]
    if (
        first in _ENVIRONMENT_LAUNCHERS
        and len(command) == 3
        and command[1] == "run"
        and _basename(command[2]) in _PYTEST_EXECUTABLES
    ):
        return [str(first_path), "run", "python"]
    raise BreachObservationConfigurationError(
        "exact pytest observation command prefix is not canonical"
    )


def resolve_python_observation_prefix(
    parameters: Mapping[str, object],
    *,
    default_test_command: Sequence[str],
    cwd: str | Path,
) -> list[str]:
    """Resolve explicit static selection or the recognized legacy fallback."""

    if BREACH_TEST_COMMAND_KEY in parameters:
        prefix = _parse_pytest_command(
            parameters[BREACH_TEST_COMMAND_KEY],
            path="per_language.python.breach_test_command",
            allow_suite_arguments=False,
        )
        validate_breach_test_command(
            parameters[BREACH_TEST_COMMAND_KEY],
            path="per_language.python.breach_test_command",
        )
        return pytest_python_prefix(prefix, cwd=cwd, explicit=True)

    legacy = parameters.get("test_command") or default_test_command
    prefix = _parse_pytest_command(
        legacy,
        path="test_command",
        allow_suite_arguments=True,
    )
    return pytest_python_prefix(prefix, cwd=cwd, explicit=False)


def preflight_python_observation_channel(
    parameters: Mapping[str, object],
    *,
    default_test_command: Sequence[str],
    cwd: str | Path,
    candidate_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout_s: float = _PYTEST_IMPORT_TIMEOUT_S,
) -> PythonObservationBinding:
    """Resolve and probe the selected exact-observation Python environment."""

    prefix = resolve_python_observation_prefix(
        parameters,
        default_test_command=default_test_command,
        cwd=cwd,
    )
    trusted_root = Path(candidate_root or cwd).resolve()
    # Reject candidate-controlled launchers before executing even the import
    # probe. Legacy environment launchers are reduced to the interpreter they
    # select; the retained binding never consults project configuration again.
    _executable_identity(prefix[0], candidate_root=trusted_root)
    probe = (
        "import json,pytest,sys;"
        "print(json.dumps({'python':sys.executable,'pytest':pytest.__file__},"
        "sort_keys=True,separators=(',',':')))"
    )
    # ``-I`` deliberately ignores ``PYTHONDONTWRITEBYTECODE``. Bind ``-B`` on
    # the command line so even the import-identity probe cannot create source
    # artifacts in the candidate checkout.
    command = [*prefix, "-B", "-I", "-c", probe]
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise BreachObservationInfrastructureError(
            "exact pytest observation environment probe timed out after "
            f"{timeout_s:g}s"
        ) from exc
    except OSError as exc:
        raise BreachObservationInfrastructureError(
            "exact pytest observation environment probe could not run: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode < 0:
        raise BreachObservationInfrastructureError(
            "exact pytest observation environment probe was terminated by "
            f"signal {-completed.returncode}"
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        detail = detail.replace("\x00", "")[-_MAX_FAILURE_DETAIL_CHARS:]
        suffix = f": {detail}" if detail else ""
        raise BreachObservationConfigurationError(
            "selected exact-observation Python environment cannot import "
            f"pytest (returncode={completed.returncode}){suffix}"
        )
    try:
        record = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BreachObservationInfrastructureError(
            "exact pytest observation environment probe returned malformed "
            "identity metadata"
        ) from exc
    if (
        not isinstance(record, dict)
        or set(record) != {"pytest", "python"}
        or not all(isinstance(value, str) and value for value in record.values())
    ):
        raise BreachObservationInfrastructureError(
            "exact pytest observation environment probe returned malformed "
            "identity metadata"
        )
    python_path = _resolved_executable(record["python"], cwd=cwd)
    executable_identity = _executable_identity(
        python_path,
        candidate_root=trusted_root,
    )
    pytest_identity = _trusted_file_identity(
        record["pytest"],
        candidate_root=trusted_root,
        executable=False,
    )
    return PythonObservationBinding(
        python_prefix=(str(python_path),),
        executable_identity=executable_identity,
        pytest_identity=pytest_identity,
        candidate_root=str(trusted_root),
    )


def breach_observation_failure(
    kind: str,
    detail: object,
) -> dict[str, str]:
    """Build one bounded typed runner-to-workflow failure envelope."""

    if kind not in BREACH_OBSERVATION_FAILURE_KINDS:
        raise ValueError(f"unsupported breach observation failure kind: {kind!r}")
    rendered = str(detail).replace("\x00", "")[:_MAX_FAILURE_DETAIL_CHARS]
    return {
        "schema_version": BREACH_OBSERVATION_FAILURE_SCHEMA,
        "kind": kind,
        "detail": rendered,
    }


def parse_breach_observation_failure(
    payload: object,
) -> dict[str, str] | None:
    """Strictly parse an optional exact-observation failure envelope."""

    if not isinstance(payload, Mapping):
        return None
    raw = payload.get(BREACH_OBSERVATION_FAILURE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "kind",
        "detail",
    }:
        raise ValueError("breach observation failure envelope is malformed")
    schema = raw.get("schema_version")
    kind = raw.get("kind")
    detail = raw.get("detail")
    if schema != BREACH_OBSERVATION_FAILURE_SCHEMA:
        raise ValueError("breach observation failure schema is unsupported")
    if kind not in BREACH_OBSERVATION_FAILURE_KINDS:
        raise ValueError("breach observation failure kind is unsupported")
    if (
        not isinstance(detail, str)
        or not detail
        or len(detail) > _MAX_FAILURE_DETAIL_CHARS
        or "\x00" in detail
    ):
        raise ValueError("breach observation failure detail is invalid")
    return {
        "schema_version": str(schema),
        "kind": str(kind),
        "detail": detail,
    }


__all__ = [
    "BREACH_OBSERVATION_FAILURE_KEY",
    "BREACH_OBSERVATION_FAILURE_KINDS",
    "BREACH_OBSERVATION_FAILURE_SCHEMA",
    "BREACH_TEST_COMMAND_KEY",
    "BreachObservationChannelError",
    "BreachObservationConfigurationError",
    "BreachObservationInfrastructureError",
    "PythonObservationBinding",
    "assert_python_observation_binding",
    "breach_observation_failure",
    "parse_breach_observation_failure",
    "preflight_python_observation_channel",
    "pytest_python_prefix",
    "resolve_python_observation_prefix",
    "validate_breach_test_command",
]
