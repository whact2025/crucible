# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Load + validate DeepSWE task records from a local deep-swe clone.

The corpus is distributed separately from Crucible. Fetch it out-of-band:

    git clone https://github.com/datacurve-ai/deep-swe.git

then point this loader at the local clone via
``load_dataset(<clone>/tasks/manifest.json)``.

Accepted schema:

- ``tasks/manifest.json`` is the exact DeepSWE 1.1 manifest shape::

    { "schema_version": "1.0",
      "dataset": "deep-swe-1-1",
      "source_dataset": "swe-bench-ultra",
      "task_count": <number of task entries>,
      "id_policy": { ... },
      "label_generation": { ... },
      "tasks": [ { ext_id, task_id, repo, repository_url, language,
                   category, original_title, display_title,
                   display_description }, ... ] }

  All objects are closed and all identity/display values are exact non-empty
  strings. Original ``ext_id`` values must match the source-declared
  ``^kh[0-9a-z]+$`` grammar and are capped at 64 characters (twice the current
  corpus's 32-character identifiers). The loader rejects duplicate JSON keys
  and identities and validates the declared task count against the entries it
  reads.

- Each task dir at ``tasks/<task_id>/`` has::

    task.toml         Exact Harbor schema 1.3 authority, including task,
                      metadata, verifier, agent, environment, and solution
                      tables. Mirrored metadata must match the manifest;
                      execution is admitted only for the corpus's isolated,
                      no-network, zero-GPU configuration.
    instruction.md    Prompt body
    solution/         solve.sh + solution.patch (gold; audit-only)
    tests/            test.sh + test.patch (verifier + hidden tests)
    environment/      Dockerfile (fallback build path)

Pure: stdlib only (json + tomllib). No network, no HF deps, no
docker calls. Mirrors ``products/swebench/dataset.py`` boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from crucible.common.authority_io import (
    AuthorityIOError,
    descriptor_mount_id,
    open_exact_regular_at,
    read_exact_regular_at,
    stat_identity,
)
from crucible.products.deepswe.secure_artifacts import RewardTestInventory


class DeepSWEInstanceError(ValueError):
    """Raised when a manifest entry, task.toml, or per-task file
    fails canonical-shape validation."""


@dataclass(frozen=True)
class DeepSWETestsTreeIdentity:
    """Deterministic identity of the exact hidden-tests tree."""

    sha256: str
    file_count: int
    directory_count: int
    total_bytes: int

    def to_payload(self) -> dict[str, int | str]:
        return {
            "schema": "crucible.deepswe.tests-tree.v1",
            "sha256": self.sha256,
            "file_count": self.file_count,
            "directory_count": self.directory_count,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class DeepSWETestsSnapshot(DeepSWETestsTreeIdentity):
    """Private read-only copy that is safe to bind-mount at ``/tests``."""

    root: Path

    @property
    def identity(self) -> DeepSWETestsTreeIdentity:
        return DeepSWETestsTreeIdentity(
            sha256=self.sha256,
            file_count=self.file_count,
            directory_count=self.directory_count,
            total_bytes=self.total_bytes,
        )


@dataclass(frozen=True)
class DeepSWEInstance:
    """One DeepSWE task.

    ``gold_solution_patch`` and ``test_patch`` are kept on the
    record for AUDIT ONLY — neither is fed to the agent during
    ``--implement``. ``gold_solution_patch`` is the maintainer-style
    fix that ``solution/solve.sh`` applies; ``test_patch`` is the
    hidden test additions ``tests/test.sh`` applies at verify time.
    The verifier shell handles its own apply; this record carries
    the strings only so postmortem and rubric-time comparisons can
    inspect them.

    ``instruction`` is the structured prompt the agent sees (the
    body of ``instruction.md``).
    """

    # From manifest.json (per-task entry):
    task_id: str
    ext_id: str
    repo: str  # owner/name short form (e.g. ``abs-lang/abs``)
    repository_url: str  # full https URL
    language: str  # lowercase: typescript|python|go|rust|javascript
    category: str  # feature_request|bugfix|enhancement
    display_title: str
    display_description: str

    # From task.toml [metadata]:
    base_commit_hash: str

    # From task.toml [environment]:
    docker_image: str
    cpus: int
    memory_mb: int
    storage_mb: int
    build_timeout_sec: float
    allow_internet: bool

    # From task.toml [verifier] / [agent]:
    verifier_timeout_sec: float
    agent_timeout_sec: float

    # From instruction.md:
    instruction: str

    # AUDIT-ONLY (never feed to agent under ``--implement``).
    gold_solution_patch: str = ""
    test_patch: str = ""

    @property
    def repo_org(self) -> str:
        return self.repo.split("/", 1)[0]

    @property
    def repo_name(self) -> str:
        return self.repo.split("/", 1)[-1]


# The benchmark corpus is execution authority, not an extensible application
# payload.  Keep every admitted mapping closed so a future corpus revision
# cannot silently acquire semantics this loader has never reviewed.
_DATASET_MANIFEST_SCHEMA_VERSION = "1.0"
_DATASET_NAME = "deep-swe-1-1"
_SOURCE_DATASET_NAME = "swe-bench-ultra"
_MANIFEST_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "dataset",
        "source_dataset",
        "task_count",
        "id_policy",
        "label_generation",
        "tasks",
    }
)
_MANIFEST_ID_POLICY_FIELDS = frozenset(
    {"format", "source", "original_id_location", "new_id_location"}
)
_MANIFEST_ID_POLICY = {
    "format": "lowercase kebab-case, filesystem-safe, <=56 chars",
    "source": (
        "wide-research AI summary of repository, original title, "
        "and instruction excerpt"
    ),
    "original_id_location": "task.toml [metadata].ext_id",
    "new_id_location": "task.toml [metadata].task_id",
}
_MANIFEST_LABEL_GENERATION_FIELDS = frozenset(
    {"tool", "model", "run_dir", "results_jsonl", "prompt_config"}
)
_MANIFEST_ENTRY_FIELDS = frozenset(
    {
        "task_id",
        "ext_id",
        "repo",
        "repository_url",
        "language",
        "category",
        "original_title",
        "display_title",
        "display_description",
    }
)
_NONEMPTY_MANIFEST_ENTRY_FIELDS = frozenset(
    {
        "task_id",
        "ext_id",
        "repo",
        "repository_url",
        "language",
        "category",
        "original_title",
        "display_title",
        "display_description",
    }
)
_TASK_TOML_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "artifacts",
        "task",
        "metadata",
        "environment",
        "verifier",
        "agent",
        "solution",
    }
)
_TASK_TOML_SCHEMA_VERSION = "1.3"
_TASK_TOML_SECTION_FIELDS = {
    "task": frozenset({"name", "description", "authors", "keywords"}),
    "metadata": frozenset(
        {
            "ext_id",
            "task_id",
            "repository_url",
            "base_commit_hash",
            "category",
            "language",
            "original_title",
            "display_title",
            "display_description",
        }
    ),
    "environment": frozenset(
        {
            "docker_image",
            "build_timeout_sec",
            "cpus",
            "memory_mb",
            "storage_mb",
            "os",
            "gpus",
            "mcp_servers",
            "env",
        }
    ),
    "verifier": frozenset(
        {"network_mode", "environment_mode", "timeout_sec", "env", "environment"}
    ),
    "agent": frozenset({"network_mode", "timeout_sec"}),
    "solution": frozenset({"env"}),
}
_TASK_TOML_VERIFIER_ENVIRONMENT_FIELDS = frozenset(
    {"build_timeout_sec", "cpus", "memory_mb", "storage_mb"}
)

# Identity fields duplicated between the manifest and task.toml must agree
# byte-for-byte.  Normalization happens only after this authority check.
_MIRRORED_TASK_FIELDS: tuple[str, ...] = (
    "task_id",
    "ext_id",
    "repository_url",
    "language",
    "category",
    "original_title",
    "display_title",
    "display_description",
)

# Task ids become filesystem components in both the loader and verifier.  The
# corpus's declared policy is a lowercase kebab-case identifier of at most 56
# characters; enforce that exact authority instead of accepting a broader
# merely-filesystem-safe superset.
_SAFE_TASK_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_MAX_TASK_ID_LENGTH = 56
# The frozen DeepSWE source schema declares ``^kh[0-9a-z]+$``.  Current
# identifiers are 32 characters; a fixed 64-character ceiling leaves one
# deliberate doubling for a compatible source revision without admitting an
# unbounded identity into manifests, image tags, lifecycle rows, or logs.
_SAFE_EXT_ID = re.compile(r"kh[0-9a-z]+\Z")
_MAX_EXT_ID_LENGTH = 64
_SUPPORTED_LANGUAGES = frozenset({"go", "javascript", "python", "rust", "typescript"})
_SUPPORTED_CATEGORIES = frozenset({"bugfix", "enhancement", "feature_request"})
_MAX_DATASET_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_TASK_TOML_BYTES = 1024 * 1024
_MAX_TASK_TEXT_BYTES = 16 * 1024 * 1024
_MAX_TEST_TREE_FILE_BYTES = 256 * 1024 * 1024


def _validate_task_id(task_id: Any) -> str:
    if (
        type(task_id) is not str
        or len(task_id) > _MAX_TASK_ID_LENGTH
        or not _SAFE_TASK_ID.fullmatch(task_id)
    ):
        raise DeepSWEInstanceError(
            "manifest task_id must be lowercase kebab-case and at most "
            f"{_MAX_TASK_ID_LENGTH} characters: {task_id!r}"
        )
    return task_id


def _validate_ext_id(ext_id: Any) -> str:
    """Parse one frozen DeepSWE source identifier without coercion."""

    if (
        type(ext_id) is not str
        or len(ext_id) > _MAX_EXT_ID_LENGTH
        or not _SAFE_EXT_ID.fullmatch(ext_id)
    ):
        raise DeepSWEInstanceError(
            "manifest ext_id must match ^kh[0-9a-z]+$ and be at most "
            f"{_MAX_EXT_ID_LENGTH} characters: {ext_id!r}"
        )
    return ext_id


def _require_closed_fields(
    raw: dict[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    observed = set(raw)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise DeepSWEInstanceError(
            f"{label} schema fields differ: missing={missing!r}, unknown={unknown!r}"
        )


def _validate_manifest_entry(raw: Any) -> dict[str, str]:
    """Return one exact, closed manifest entry."""

    if type(raw) is not dict:
        raise DeepSWEInstanceError(
            f"manifest entry must be a JSON object, got {type(raw).__name__}"
        )
    entry = raw
    _require_closed_fields(entry, _MANIFEST_ENTRY_FIELDS, label="manifest entry")
    for field in sorted(_MANIFEST_ENTRY_FIELDS):
        if type(entry[field]) is not str:
            raise DeepSWEInstanceError(
                f"manifest entry field {field!r} must be an exact string"
            )
    _validate_task_id(entry["task_id"])
    _validate_ext_id(entry["ext_id"])
    for field in sorted(_NONEMPTY_MANIFEST_ENTRY_FIELDS):
        if not entry[field]:
            raise DeepSWEInstanceError(
                f"manifest entry field {field!r} must not be empty"
            )
    if entry["language"] not in _SUPPORTED_LANGUAGES:
        raise DeepSWEInstanceError(
            "manifest entry language is outside the supported DeepSWE domain"
        )
    if entry["category"] not in _SUPPORTED_CATEGORIES:
        raise DeepSWEInstanceError(
            "manifest entry category is outside the supported DeepSWE domain"
        )
    return entry


def _reject_duplicate_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeepSWEInstanceError(f"manifest JSON object repeats key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> Any:
    raise DeepSWEInstanceError(
        f"manifest JSON contains unsupported numeric constant {value!r}"
    )


def _require_exact_string(value: Any, *, label: str, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise DeepSWEInstanceError(
            f"{label} must be an exact{' non-empty' if nonempty else ''} string"
        )
    return value


def _require_positive_integer(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise DeepSWEInstanceError(f"{label} must be an exact positive integer")
    return value


def _require_positive_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value <= 0.0:
        raise DeepSWEInstanceError(f"{label} must be an exact positive finite float")
    return value


def _require_exact_string_list(value: Any, *, label: str) -> list[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise DeepSWEInstanceError(f"{label} must be an exact string list")
    return value


def _require_exact_string_map(value: Any, *, label: str) -> dict[str, str]:
    if type(value) is not dict or any(
        type(key) is not str or type(item) is not str for key, item in value.items()
    ):
        raise DeepSWEInstanceError(f"{label} must be an exact string map")
    return value


def _resolve_manifest_path(manifest_path: Path) -> tuple[Path, Path]:
    """Return a real manifest and its canonical dataset root."""

    manifest_path = Path(manifest_path)
    try:
        manifest_stat = manifest_path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"DeepSWE manifest does not exist: {manifest_path}"
        ) from None
    if stat.S_ISLNK(manifest_stat.st_mode) or not stat.S_ISREG(manifest_stat.st_mode):
        raise DeepSWEInstanceError(
            f"DeepSWE manifest must be a real regular file: {manifest_path}"
        )
    resolved = manifest_path.resolve(strict=True)
    return resolved, resolved.parent.resolve(strict=True)


def _resolve_dataset_member(
    path: Path,
    *,
    dataset_root: Path,
    task_id: str,
    label: str,
    directory: bool,
    required: bool,
) -> Path | None:
    """Validate one corpus path without following image-controlled links."""

    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        if not required:
            return None
        raise DeepSWEInstanceError(
            f"task {task_id!r}: missing {label} at {path}"
        ) from None
    except OSError as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot inspect {label} at {path}: {exc}"
        ) from exc

    if stat.S_ISLNK(path_stat.st_mode):
        raise DeepSWEInstanceError(
            f"task {task_id!r}: {label} must not be a symlink: {path}"
        )
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(path_stat.st_mode):
        kind = "directory" if directory else "regular file"
        raise DeepSWEInstanceError(
            f"task {task_id!r}: {label} must be a {kind}: {path}"
        )

    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(dataset_root)
    except (OSError, ValueError) as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: {label} escapes dataset root {dataset_root}: {path}"
        ) from exc
    return resolved


_TESTS_TREE_DIGEST_HEADER = b"crucible.deepswe.tests-tree.v1\0"
_SOURCE_STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


class _TestsTreeDigestBuilder:
    def __init__(self) -> None:
        self.hasher = hashlib.sha256()
        self.hasher.update(_TESTS_TREE_DIGEST_HEADER)
        self.file_count = 0
        self.directory_count = 0
        self.total_bytes = 0

    @staticmethod
    def _path_bytes(parts: tuple[bytes, ...]) -> bytes:
        return b"/".join(parts)

    def add_directory(self, parts: tuple[bytes, ...], mode: int) -> None:
        encoded = self._path_bytes(parts)
        self.hasher.update(b"D")
        self.hasher.update(len(encoded).to_bytes(8, "big"))
        self.hasher.update(encoded)
        self.hasher.update(mode.to_bytes(4, "big"))
        self.directory_count += 1

    def begin_file(
        self,
        parts: tuple[bytes, ...],
        mode: int,
        size: int,
    ) -> None:
        encoded = self._path_bytes(parts)
        self.hasher.update(b"F")
        self.hasher.update(len(encoded).to_bytes(8, "big"))
        self.hasher.update(encoded)
        self.hasher.update(mode.to_bytes(4, "big"))
        self.hasher.update(size.to_bytes(8, "big"))
        self.file_count += 1
        self.total_bytes += size

    def add_file_bytes(self, payload: bytes) -> None:
        self.hasher.update(payload)

    def finish(self) -> DeepSWETestsTreeIdentity:
        return DeepSWETestsTreeIdentity(
            sha256=self.hasher.hexdigest(),
            file_count=self.file_count,
            directory_count=self.directory_count,
            total_bytes=self.total_bytes,
        )


def _nofollow_flags(*, directory: bool = False, writable: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise DeepSWEInstanceError(
            "descriptor-safe DeepSWE test snapshots require O_NOFOLLOW"
        )
    flags = nofollow | getattr(os, "O_CLOEXEC", 0)
    flags |= os.O_WRONLY if writable else os.O_RDONLY
    if directory:
        flags |= os.O_DIRECTORY
    elif not writable:
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _stable_source_metadata(
    before: os.stat_result,
    after: os.stat_result,
) -> bool:
    return all(
        getattr(before, field) == getattr(after, field)
        for field in _SOURCE_STABLE_FIELDS
    )


def _stable_directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _snapshot_directory_mode(_source_mode: int) -> int:
    # The verifier runs as the host uid. Keep the private snapshot owner-only;
    # Docker's read-only bind adds a second independent write barrier.
    return 0o500


def _snapshot_file_mode(source_mode: int) -> int:
    # Every hidden-test payload must remain readable. Preserve executable
    # semantics without granting write access or exposing bytes to other uids.
    return 0o500 if stat.S_IMODE(source_mode) & 0o111 else 0o400


def _encoded_directory_names(directory_fd: int) -> tuple[tuple[bytes, str], ...]:
    names = tuple(os.listdir(directory_fd))
    encoded = tuple((os.fsencode(name), name) for name in names)
    if len({item[0] for item in encoded}) != len(encoded):
        raise DeepSWEInstanceError("DeepSWE tests tree has non-unique filesystem names")
    return tuple(sorted(encoded, key=lambda item: item[0]))


def _assert_entry_stable(
    *,
    parent_fd: int,
    name: str,
    opened_fd: int,
    before: os.stat_result,
    label: str,
) -> None:
    opened_after = os.fstat(opened_fd)
    try:
        entry_after = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise DeepSWEInstanceError(
            f"DeepSWE tests source changed while copying {label}: {exc}"
        ) from exc
    if (
        not _stable_source_metadata(before, opened_after)
        or not _stable_source_metadata(opened_after, entry_after)
        or not _same_inode(opened_after, entry_after)
    ):
        raise DeepSWEInstanceError(
            f"DeepSWE tests source changed while copying {label}"
        )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("DeepSWE tests snapshot write made no progress")
        view = view[written:]


def _walk_tests_tree(
    source_fd: int,
    *,
    destination_fd: int | None,
    builder: _TestsTreeDigestBuilder,
    relative_parts: tuple[bytes, ...] = (),
    require_test_script: bool = False,
) -> None:
    directory_before = os.fstat(source_fd)
    if not stat.S_ISDIR(directory_before.st_mode):
        raise DeepSWEInstanceError("DeepSWE tests source is not a directory")
    destination_mode = _snapshot_directory_mode(directory_before.st_mode)
    builder.add_directory(relative_parts, destination_mode)
    names_before = _encoded_directory_names(source_fd)
    if require_test_script:
        by_name = {encoded: name for encoded, name in names_before}
        test_script_name = by_name.get(b"test.sh")
        if test_script_name is None:
            raise DeepSWEInstanceError(
                "DeepSWE tests snapshot requires a regular tests/test.sh"
            )
        test_script = os.stat(
            test_script_name,
            dir_fd=source_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(test_script.st_mode):
            raise DeepSWEInstanceError(
                "DeepSWE tests snapshot requires a regular tests/test.sh"
            )

    for encoded_name, name in names_before:
        label = os.fsdecode(b"/".join((*relative_parts, encoded_name)))
        try:
            entry_before = os.stat(
                name,
                dir_fd=source_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise DeepSWEInstanceError(
                f"cannot inspect DeepSWE test path {label!r}: {exc}"
            ) from exc
        if stat.S_ISLNK(entry_before.st_mode):
            raise DeepSWEInstanceError(
                f"DeepSWE tests tree contains a symlink: {label!r}"
            )

        if stat.S_ISDIR(entry_before.st_mode):
            try:
                child_source_fd = os.open(
                    name,
                    _nofollow_flags(directory=True),
                    dir_fd=source_fd,
                )
            except OSError as exc:
                raise DeepSWEInstanceError(
                    f"cannot open DeepSWE test directory {label!r}: {exc}"
                ) from exc
            child_destination_fd: int | None = None
            try:
                opened_before = os.fstat(child_source_fd)
                if not stat.S_ISDIR(
                    opened_before.st_mode
                ) or not _stable_source_metadata(entry_before, opened_before):
                    raise DeepSWEInstanceError(
                        f"DeepSWE tests source changed before copying {label!r}"
                    )
                if destination_fd is not None:
                    os.mkdir(name, mode=0o700, dir_fd=destination_fd)
                    child_destination_fd = os.open(
                        name,
                        _nofollow_flags(directory=True),
                        dir_fd=destination_fd,
                    )
                _walk_tests_tree(
                    child_source_fd,
                    destination_fd=child_destination_fd,
                    builder=builder,
                    relative_parts=(*relative_parts, encoded_name),
                )
                _assert_entry_stable(
                    parent_fd=source_fd,
                    name=name,
                    opened_fd=child_source_fd,
                    before=entry_before,
                    label=label,
                )
                if child_destination_fd is not None:
                    os.fchmod(
                        child_destination_fd,
                        _snapshot_directory_mode(entry_before.st_mode),
                    )
                    os.fsync(child_destination_fd)
            finally:
                if child_destination_fd is not None:
                    os.close(child_destination_fd)
                os.close(child_source_fd)
            continue

        if not stat.S_ISREG(entry_before.st_mode):
            raise DeepSWEInstanceError(
                f"DeepSWE tests tree contains a special file: {label!r}"
            )
        try:
            child_source_fd, source_payload, opened_before = open_exact_regular_at(
                source_fd,
                name,
                maximum=_MAX_TEST_TREE_FILE_BYTES,
            )
        except AuthorityIOError as exc:
            raise DeepSWEInstanceError(
                f"DeepSWE tests source changed while copying {label!r}"
            ) from exc
        except OSError as exc:
            raise DeepSWEInstanceError(
                f"cannot open DeepSWE test file {label!r}: {exc}"
            ) from exc
        child_destination_fd = None
        try:
            if not stat.S_ISREG(opened_before.st_mode) or not _stable_source_metadata(
                entry_before, opened_before
            ):
                raise DeepSWEInstanceError(
                    f"DeepSWE tests source changed before copying {label!r}"
                )
            destination_mode = _snapshot_file_mode(entry_before.st_mode)
            builder.begin_file(
                (*relative_parts, encoded_name),
                destination_mode,
                entry_before.st_size,
            )
            if destination_fd is not None:
                child_destination_fd = os.open(
                    name,
                    (_nofollow_flags(writable=True) | os.O_CREAT | os.O_EXCL),
                    0o600,
                    dir_fd=destination_fd,
                )
            builder.add_file_bytes(source_payload)
            if child_destination_fd is not None:
                _write_all(child_destination_fd, source_payload)
            if len(source_payload) != entry_before.st_size:
                raise DeepSWEInstanceError(
                    f"DeepSWE tests source changed while copying {label!r}"
                )
            _assert_entry_stable(
                parent_fd=source_fd,
                name=name,
                opened_fd=child_source_fd,
                before=entry_before,
                label=label,
            )
            if child_destination_fd is not None:
                os.fsync(child_destination_fd)
                os.fchmod(child_destination_fd, destination_mode)
        finally:
            if child_destination_fd is not None:
                os.close(child_destination_fd)
            os.close(child_source_fd)

    if _encoded_directory_names(source_fd) != names_before:
        raise DeepSWEInstanceError(
            "DeepSWE tests source changed while enumerating a directory"
        )
    directory_after = os.fstat(source_fd)
    if not _stable_source_metadata(directory_before, directory_after):
        raise DeepSWEInstanceError(
            "DeepSWE tests source directory changed while being copied"
        )


def _open_directory_chain(path: Path) -> tuple[list[int], list[str]]:
    if not path.is_absolute():
        raise DeepSWEInstanceError("canonical DeepSWE dataset root must be absolute")
    descriptors = [os.open("/", _nofollow_flags(directory=True))]
    names: list[str] = []
    try:
        for component in path.parts[1:]:
            entry = os.stat(
                component,
                dir_fd=descriptors[-1],
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(entry.st_mode):
                raise DeepSWEInstanceError(
                    f"DeepSWE dataset path component is not a directory: {component!r}"
                )
            child = os.open(
                component,
                _nofollow_flags(directory=True),
                dir_fd=descriptors[-1],
            )
            try:
                opened = os.fstat(child)
            except BaseException:
                os.close(child)
                raise
            if not _same_inode(entry, opened) or not stat.S_ISDIR(opened.st_mode):
                os.close(child)
                raise DeepSWEInstanceError(
                    "DeepSWE dataset directory changed while opening"
                )
            descriptors.append(child)
            names.append(component)
        return descriptors, names
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _directory_chain_is_stable(
    descriptors: list[int],
    names: list[str],
) -> bool:
    rebound: int | None = None
    try:
        for parent_fd, child_fd, name in zip(
            descriptors[:-1],
            descriptors[1:],
            names,
            strict=True,
        ):
            entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            opened = os.fstat(child_fd)
            rebound = os.open(
                name,
                _nofollow_flags(directory=True),
                dir_fd=parent_fd,
            )
            rebound_metadata = os.fstat(rebound)
            stable = (
                stat.S_ISDIR(entry.st_mode)
                and _stable_directory_identity(entry)
                == _stable_directory_identity(opened)
                and _stable_directory_identity(opened)
                == _stable_directory_identity(rebound_metadata)
                and _same_inode(entry, opened)
                and _same_inode(opened, rebound_metadata)
                and descriptor_mount_id(child_fd) == descriptor_mount_id(rebound)
            )
            owned = rebound
            rebound = None
            os.close(owned)
            if not stable:
                return False
        return True
    except (AuthorityIOError, OSError):
        return False
    finally:
        if rebound is not None:
            os.close(rebound)


def _append_open_directory(
    descriptors: list[int],
    names: list[str],
    component: str,
    *,
    label: str,
) -> None:
    parent_fd = descriptors[-1]
    try:
        entry = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode):
            raise DeepSWEInstanceError(f"{label} must be a real directory")
        child = os.open(
            component,
            _nofollow_flags(directory=True),
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise DeepSWEInstanceError(f"cannot open {label}: {exc}") from exc
    try:
        opened = os.fstat(child)
    except BaseException:
        os.close(child)
        raise
    if not _same_inode(entry, opened) or not stat.S_ISDIR(opened.st_mode):
        os.close(child)
        raise DeepSWEInstanceError(f"{label} changed while opening")
    descriptors.append(child)
    names.append(component)


@contextlib.contextmanager
def _open_task_tests_tree(
    manifest_path: Path,
    task_id: str,
) -> Iterator[int]:
    task_id = _validate_task_id(task_id)
    _, tasks_root = _resolve_manifest_path(manifest_path)
    descriptors, names = _open_directory_chain(tasks_root)
    try:
        _append_open_directory(
            descriptors,
            names,
            task_id,
            label=f"task {task_id!r} directory",
        )
        _append_open_directory(
            descriptors,
            names,
            "tests",
            label=f"task {task_id!r} tests directory",
        )
        yield descriptors[-1]
        if not _directory_chain_is_stable(descriptors, names):
            raise DeepSWEInstanceError(
                f"task {task_id!r} tests path changed while being read"
            )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _identity_from_open_tests_tree(tests_fd: int) -> DeepSWETestsTreeIdentity:
    builder = _TestsTreeDigestBuilder()
    _walk_tests_tree(
        tests_fd,
        destination_fd=None,
        builder=builder,
        require_test_script=True,
    )
    return builder.finish()


def _reject_duplicate_tests_config_key(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeepSWEInstanceError(
                "tests/config.json repeats a JSON object key"
            )
        result[key] = value
    return result


def _reject_nonfinite_tests_config_constant(_value: str) -> Any:
    raise DeepSWEInstanceError(
        "tests/config.json contains a non-finite numeric constant"
    )


def _test_inventory_from_open_tests_tree(
    tests_fd: int,
) -> RewardTestInventory | None:
    """Read the whitelist sizes that ``tests/config.json`` declares.

    The DeepSWE 1.1 grader reports these sizes as the totals of its reward:
    the number of distinct, whitespace-stripped, non-empty ids in each list.
    A tests tree without ``config.json`` declares no whitelist.
    """

    try:
        before = os.stat("config.json", dir_fd=tests_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise DeepSWEInstanceError("cannot inspect tests/config.json") from exc
    if stat.S_ISLNK(before.st_mode):
        raise DeepSWEInstanceError("tests/config.json must not be a symlink")
    try:
        payload, _metadata = read_exact_regular_at(
            tests_fd,
            "config.json",
            maximum=_MAX_TASK_TEXT_BYTES,
        )
    except AuthorityIOError as exc:
        raise DeepSWEInstanceError(
            "cannot safely read tests/config.json"
        ) from exc
    try:
        config = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_tests_config_key,
            parse_constant=_reject_nonfinite_tests_config_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise DeepSWEInstanceError(
            "tests/config.json is not strict JSON"
        ) from exc
    if type(config) is not dict:
        raise DeepSWEInstanceError(
            "tests/config.json must contain one JSON object"
        )
    sizes = {
        field: len(
            {
                stripped
                for node_id in _require_exact_string_list(
                    config.get(field),
                    label=f"tests/config.json {field}",
                )
                if (stripped := node_id.strip())
            }
        )
        for field in ("f2p_node_ids", "p2p_node_ids")
    }
    return RewardTestInventory(
        f2p_total=sizes["f2p_node_ids"],
        p2p_total=sizes["p2p_node_ids"],
    )


def _remove_private_snapshot(path: Path) -> None:
    if not path.exists():
        return
    for directory, subdirectories, _files in os.walk(path, followlinks=False):
        os.chmod(directory, 0o700)
        for name in subdirectories:
            child = Path(directory) / name
            if not child.is_symlink():
                os.chmod(child, 0o700)
    shutil.rmtree(path, ignore_errors=True)


def _read_text_strict(
    directory_fd: int,
    name: str,
    *,
    task_id: str,
    label: str,
) -> str:
    """Read one required task leaf through its held parent descriptor."""

    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        raise DeepSWEInstanceError(f"task {task_id!r}: missing {label}") from None
    except OSError as exc:
        raise DeepSWEInstanceError(f"task {task_id!r}: cannot inspect {label}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise DeepSWEInstanceError(f"task {task_id!r}: {label} must not be a symlink")
    try:
        payload, _metadata = read_exact_regular_at(
            directory_fd,
            name,
            maximum=_MAX_TASK_TEXT_BYTES,
        )
    except AuthorityIOError as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot safely read {label}"
        ) from exc
    return payload.decode("utf-8", errors="replace")


def _read_text_optional(
    directory_fd: int | None,
    name: str,
    *,
    task_id: str,
    label: str,
) -> str:
    """Read one optional task leaf through its held parent descriptor."""

    if directory_fd is None:
        return ""
    try:
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise DeepSWEInstanceError(f"task {task_id!r}: cannot inspect {label}") from exc
    if stat.S_ISLNK(before.st_mode):
        raise DeepSWEInstanceError(f"task {task_id!r}: {label} must not be a symlink")
    try:
        payload, _metadata = read_exact_regular_at(
            directory_fd,
            name,
            maximum=_MAX_TASK_TEXT_BYTES,
        )
    except AuthorityIOError as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot safely read {label}"
        ) from exc
    return payload.decode("utf-8", errors="replace")


def _parse_task_toml(
    task_directory_fd: int,
    *,
    task_id: str,
) -> dict[str, Any]:
    """Parse ``task.toml`` and validate the required sections /
    fields land where this loader expects."""
    try:
        before = os.stat(
            "task.toml",
            dir_fd=task_directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise DeepSWEInstanceError(f"task {task_id!r}: missing task.toml") from None
    except OSError as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot inspect task.toml"
        ) from exc
    if stat.S_ISLNK(before.st_mode):
        raise DeepSWEInstanceError(f"task {task_id!r}: task.toml must not be a symlink")
    try:
        payload, _metadata = read_exact_regular_at(
            task_directory_fd,
            "task.toml",
            maximum=_MAX_TASK_TOML_BYTES,
        )
        data = tomllib.loads(payload.decode("utf-8", errors="strict"))
    except AuthorityIOError as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot safely read task.toml"
        ) from exc
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: task.toml parse failed: {e}"
        ) from e

    if type(data) is not dict:
        raise DeepSWEInstanceError(f"task {task_id!r}: task.toml root is not a table")
    _require_closed_fields(
        data,
        _TASK_TOML_ROOT_FIELDS,
        label=f"task {task_id!r}: task.toml root",
    )
    if type(data["schema_version"]) is not str or data["schema_version"] != (
        _TASK_TOML_SCHEMA_VERSION
    ):
        raise DeepSWEInstanceError(
            f"task {task_id!r}: task.toml schema_version is unsupported"
        )
    artifacts = _require_exact_string_list(
        data["artifacts"],
        label=f"task {task_id!r}: task.toml artifacts",
    )
    if artifacts != ["/logs/artifacts/model.patch"]:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: task.toml artifacts authority is unsupported"
        )
    for section, fields in _TASK_TOML_SECTION_FIELDS.items():
        value = data[section]
        if type(value) is not dict:
            raise DeepSWEInstanceError(
                f"task {task_id!r}: task.toml [{section}] must be one table"
            )
        _require_closed_fields(
            value,
            fields,
            label=f"task {task_id!r}: task.toml [{section}]",
        )
    verifier_environment = data["verifier"]["environment"]
    if type(verifier_environment) is not dict:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: [verifier].environment must be one table"
        )
    _require_closed_fields(
        verifier_environment,
        _TASK_TOML_VERIFIER_ENVIRONMENT_FIELDS,
        label=f"task {task_id!r}: [verifier].environment",
    )
    return data


def _construct_instance(
    manifest_entry: dict[str, str],
    *,
    task_directory_fd: int,
    solution_directory_fd: int | None,
    tests_directory_fd: int | None,
) -> DeepSWEInstance:
    """Compose a DeepSWEInstance from a manifest entry + the
    per-task files at ``task_dir``."""
    task_id = manifest_entry["task_id"]
    toml_data = _parse_task_toml(
        task_directory_fd,
        task_id=task_id,
    )

    metadata = toml_data["metadata"]
    environment = toml_data["environment"]
    verifier = toml_data["verifier"]
    agent = toml_data["agent"]
    task = toml_data["task"]
    solution = toml_data["solution"]

    for field in _TASK_TOML_SECTION_FIELDS["metadata"]:
        _require_exact_string(
            metadata[field],
            label=f"task {task_id!r}: [metadata].{field}",
        )
    for field in _MIRRORED_TASK_FIELDS:
        if metadata[field] != manifest_entry[field]:
            raise DeepSWEInstanceError(
                f"task {task_id!r}: [metadata].{field} differs from manifest"
            )

    task_name = _require_exact_string(
        task["name"],
        label=f"task {task_id!r}: [task].name",
    )
    if task_name != f"datacurve/{task_id}":
        raise DeepSWEInstanceError(
            f"task {task_id!r}: [task].name differs from task authority"
        )
    _require_exact_string(
        task["description"],
        label=f"task {task_id!r}: [task].description",
        nonempty=False,
    )
    _require_exact_string_list(
        task["authors"],
        label=f"task {task_id!r}: [task].authors",
    )
    _require_exact_string_list(
        task["keywords"],
        label=f"task {task_id!r}: [task].keywords",
    )

    base_commit_hash = metadata["base_commit_hash"]
    docker_image = _require_exact_string(
        environment["docker_image"],
        label=f"task {task_id!r}: [environment].docker_image",
    )
    cpus = _require_positive_integer(
        environment["cpus"],
        label=f"task {task_id!r}: [environment].cpus",
    )
    memory_mb = _require_positive_integer(
        environment["memory_mb"],
        label=f"task {task_id!r}: [environment].memory_mb",
    )
    storage_mb = _require_positive_integer(
        environment["storage_mb"],
        label=f"task {task_id!r}: [environment].storage_mb",
    )
    build_timeout_sec = _require_positive_float(
        environment["build_timeout_sec"],
        label=f"task {task_id!r}: [environment].build_timeout_sec",
    )
    if type(environment["os"]) is not str or environment["os"] != "linux":
        raise DeepSWEInstanceError(f"task {task_id!r}: [environment].os is unsupported")
    if type(environment["gpus"]) is not int or environment["gpus"] != 0:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: [environment].gpus must be the exact integer zero"
        )
    if _require_exact_string_list(
        environment["mcp_servers"],
        label=f"task {task_id!r}: [environment].mcp_servers",
    ):
        raise DeepSWEInstanceError(
            f"task {task_id!r}: [environment].mcp_servers must be empty"
        )
    for label, value in (
        ("[environment].env", environment["env"]),
        ("[verifier].env", verifier["env"]),
        ("[solution].env", solution["env"]),
    ):
        if _require_exact_string_map(
            value,
            label=f"task {task_id!r}: {label}",
        ):
            raise DeepSWEInstanceError(f"task {task_id!r}: {label} must be empty")
    if (
        type(agent["network_mode"]) is not str
        or agent["network_mode"] != "no-network"
        or type(verifier["network_mode"]) is not str
        or verifier["network_mode"] != "no-network"
        or type(verifier["environment_mode"]) is not str
        or verifier["environment_mode"] != "separate"
    ):
        raise DeepSWEInstanceError(
            f"task {task_id!r}: network/environment isolation is unsupported"
        )
    verifier_environment = verifier["environment"]
    verifier_build_timeout_sec = _require_positive_float(
        verifier_environment["build_timeout_sec"],
        label=f"task {task_id!r}: [verifier].environment.build_timeout_sec",
    )
    verifier_cpus = _require_positive_integer(
        verifier_environment["cpus"],
        label=f"task {task_id!r}: [verifier].environment.cpus",
    )
    verifier_memory_mb = _require_positive_integer(
        verifier_environment["memory_mb"],
        label=f"task {task_id!r}: [verifier].environment.memory_mb",
    )
    verifier_storage_mb = _require_positive_integer(
        verifier_environment["storage_mb"],
        label=f"task {task_id!r}: [verifier].environment.storage_mb",
    )
    if (
        verifier_build_timeout_sec != build_timeout_sec
        or verifier_cpus != cpus
        or verifier_memory_mb != memory_mb
        or verifier_storage_mb != storage_mb
    ):
        raise DeepSWEInstanceError(
            f"task {task_id!r}: verifier resource authority differs from environment"
        )
    verifier_timeout_sec = _require_positive_float(
        verifier["timeout_sec"],
        label=f"task {task_id!r}: [verifier].timeout_sec",
    )
    agent_timeout_sec = _require_positive_float(
        agent["timeout_sec"],
        label=f"task {task_id!r}: [agent].timeout_sec",
    )

    instruction = _read_text_strict(
        task_directory_fd,
        "instruction.md",
        task_id=task_id,
        label="instruction.md",
    )

    # solution.patch + test.patch are audit-only; missing → empty.
    gold = _read_text_optional(
        solution_directory_fd,
        "solution.patch",
        task_id=task_id,
        label="solution/solution.patch",
    )
    test_patch = _read_text_optional(
        tests_directory_fd,
        "test.patch",
        task_id=task_id,
        label="tests/test.patch",
    )

    return DeepSWEInstance(
        task_id=task_id,
        ext_id=manifest_entry["ext_id"],
        repo=manifest_entry["repo"],
        repository_url=manifest_entry["repository_url"],
        language=manifest_entry["language"],
        category=manifest_entry["category"],
        display_title=manifest_entry["display_title"],
        display_description=manifest_entry["display_description"],
        base_commit_hash=base_commit_hash,
        docker_image=docker_image,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
        build_timeout_sec=build_timeout_sec,
        allow_internet=False,
        verifier_timeout_sec=verifier_timeout_sec,
        agent_timeout_sec=agent_timeout_sec,
        instruction=instruction,
        gold_solution_patch=gold,
        test_patch=test_patch,
    )


def _open_dataset_directory(
    parent_fd: int,
    name: str,
    *,
    task_id: str,
    label: str,
    required: bool,
) -> tuple[int, os.stat_result] | None:
    try:
        before = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if not required:
            return None
        raise DeepSWEInstanceError(f"task {task_id!r}: missing {label}") from None
    except OSError as exc:
        raise DeepSWEInstanceError(f"task {task_id!r}: cannot inspect {label}") from exc
    if not stat.S_ISDIR(before.st_mode):
        if stat.S_ISLNK(before.st_mode):
            raise DeepSWEInstanceError(
                f"task {task_id!r}: {label} must not be a symlink"
            )
        raise DeepSWEInstanceError(
            f"task {task_id!r}: {label} must be a real directory"
        )
    try:
        descriptor = os.open(
            name,
            _nofollow_flags(directory=True),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
    except OSError as exc:
        raise DeepSWEInstanceError(f"task {task_id!r}: cannot open {label}") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not _stable_source_metadata(before, opened)
        or not _same_inode(before, opened)
    ):
        os.close(descriptor)
        raise DeepSWEInstanceError(f"task {task_id!r}: {label} changed while opening")
    return descriptor, before


def _verify_dataset_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
    before: os.stat_result,
    *,
    task_id: str,
    label: str,
) -> None:
    rebound: int | None = None
    try:
        after_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        after_descriptor = os.fstat(descriptor)
        rebound = os.open(
            name,
            _nofollow_flags(directory=True),
            dir_fd=parent_fd,
        )
        rebound_metadata = os.fstat(rebound)
        if (
            not _stable_source_metadata(before, after_descriptor)
            or not _stable_source_metadata(after_descriptor, after_path)
            or not _stable_source_metadata(after_descriptor, rebound_metadata)
            or descriptor_mount_id(descriptor) != descriptor_mount_id(rebound)
        ):
            raise DeepSWEInstanceError(
                f"task {task_id!r}: {label} changed while being read"
            )
    except (AuthorityIOError, OSError) as exc:
        raise DeepSWEInstanceError(
            f"task {task_id!r}: cannot revalidate {label}"
        ) from exc
    finally:
        if rebound is not None:
            os.close(rebound)


@contextlib.contextmanager
def _open_dataset_read_authority(
    manifest_path: Path,
) -> Iterator[tuple[Path, int, bytes]]:
    requested = Path(os.path.abspath(Path(manifest_path)))
    resolved_manifest, tasks_root = _resolve_manifest_path(requested)
    descriptors, names = _open_directory_chain(tasks_root)
    manifest_descriptor: int | None = None
    rebound_manifest_descriptor: int | None = None
    try:
        tasks_before = os.fstat(descriptors[-1])
        try:
            manifest_descriptor, manifest_payload, manifest_metadata = (
                open_exact_regular_at(
                    descriptors[-1],
                    resolved_manifest.name,
                    maximum=_MAX_DATASET_MANIFEST_BYTES,
                    allow_empty=False,
                )
            )
        except AuthorityIOError as exc:
            raise DeepSWEInstanceError(
                f"manifest {resolved_manifest} is not one exact regular file"
            ) from exc
        manifest_mount = descriptor_mount_id(manifest_descriptor)
        yield tasks_root, descriptors[-1], manifest_payload
        if not _directory_chain_is_stable(descriptors, names):
            raise DeepSWEInstanceError(
                "DeepSWE dataset directory chain changed while being read"
            )
        tasks_after = os.fstat(descriptors[-1])
        manifest_after = os.stat(
            resolved_manifest.name,
            dir_fd=descriptors[-1],
            follow_symlinks=False,
        )
        (
            rebound_manifest_descriptor,
            rebound_manifest_payload,
            rebound_manifest_after,
        ) = open_exact_regular_at(
            descriptors[-1],
            resolved_manifest.name,
            maximum=_MAX_DATASET_MANIFEST_BYTES,
            allow_empty=False,
        )
        held_manifest_after = os.fstat(manifest_descriptor)
        resolved_after, tasks_root_after = _resolve_manifest_path(requested)
        if (
            not _stable_source_metadata(tasks_before, tasks_after)
            or rebound_manifest_payload != manifest_payload
            or stat_identity(manifest_metadata) != stat_identity(manifest_after)
            or stat_identity(manifest_metadata) != stat_identity(held_manifest_after)
            or stat_identity(manifest_metadata) != stat_identity(rebound_manifest_after)
            or manifest_mount != descriptor_mount_id(rebound_manifest_descriptor)
            or resolved_after != resolved_manifest
            or tasks_root_after != tasks_root
        ):
            raise DeepSWEInstanceError(
                "DeepSWE dataset authority changed while being read"
            )
    finally:
        if rebound_manifest_descriptor is not None:
            os.close(rebound_manifest_descriptor)
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _decode_dataset_manifest(
    manifest_path: Path,
    manifest_payload: bytes,
) -> list[dict[str, str]]:
    try:
        manifest = json.loads(
            manifest_payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_json_object,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: JSON parse failed: {exc}"
        ) from exc

    if type(manifest) is not dict:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: top-level must be a JSON object, "
            f"got {type(manifest).__name__}"
        )

    _require_closed_fields(
        manifest,
        _MANIFEST_ROOT_FIELDS,
        label=f"manifest {manifest_path} root",
    )
    if (
        type(manifest["schema_version"]) is not str
        or manifest["schema_version"] != _DATASET_MANIFEST_SCHEMA_VERSION
    ):
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: schema_version is unsupported"
        )
    if type(manifest["dataset"]) is not str or manifest["dataset"] != _DATASET_NAME:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: dataset identity is unsupported"
        )
    if (
        type(manifest["source_dataset"]) is not str
        or manifest["source_dataset"] != _SOURCE_DATASET_NAME
    ):
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: source_dataset identity is unsupported"
        )

    for field, expected_fields in (
        ("id_policy", _MANIFEST_ID_POLICY_FIELDS),
        ("label_generation", _MANIFEST_LABEL_GENERATION_FIELDS),
    ):
        value = manifest[field]
        if type(value) is not dict:
            raise DeepSWEInstanceError(
                f"manifest {manifest_path}: {field} must be one JSON object"
            )
        _require_closed_fields(
            value,
            expected_fields,
            label=f"manifest {manifest_path}: {field}",
        )
        for child_name in sorted(expected_fields):
            _require_exact_string(
                value[child_name],
                label=f"manifest {manifest_path}: {field}.{child_name}",
            )
    if manifest["id_policy"] != _MANIFEST_ID_POLICY:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: id_policy differs from the supported corpus"
        )

    tasks_raw = manifest["tasks"]
    if type(tasks_raw) is not list:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: ``tasks`` field must be a list, "
            f"got {type(tasks_raw).__name__}"
        )

    declared_count = manifest["task_count"]
    if type(declared_count) is not int or declared_count < 0:
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: task_count must be an exact nonnegative integer"
        )
    if declared_count != len(tasks_raw):
        raise DeepSWEInstanceError(
            f"manifest {manifest_path}: task_count={declared_count} "
            f"disagrees with len(tasks)={len(tasks_raw)}"
        )
    entries = [_validate_manifest_entry(raw) for raw in tasks_raw]
    for identity_field in ("task_id", "ext_id"):
        identities = [entry[identity_field] for entry in entries]
        if len(set(identities)) != len(identities):
            raise DeepSWEInstanceError(
                f"manifest {manifest_path}: {identity_field} values must be unique"
            )
    return entries


def _load_dataset_from_open_root(
    tasks_root: Path,
    tasks_root_fd: int,
    manifest_payload: bytes,
    *,
    selected_tests_task_id: str | None,
) -> tuple[tuple[DeepSWEInstance, ...], DeepSWETestsTreeIdentity | None]:
    tasks_raw = _decode_dataset_manifest(
        tasks_root / "manifest.json",
        manifest_payload,
    )
    selected_identity: DeepSWETestsTreeIdentity | None = None
    instances: list[DeepSWEInstance] = []
    for entry in tasks_raw:
        task_id = entry["task_id"]
        opened_task = _open_dataset_directory(
            tasks_root_fd,
            task_id,
            task_id=task_id,
            label="task directory",
            required=True,
        )
        assert opened_task is not None
        task_fd, task_before = opened_task
        solution: tuple[int, os.stat_result] | None = None
        tests: tuple[int, os.stat_result] | None = None
        try:
            solution = _open_dataset_directory(
                task_fd,
                "solution",
                task_id=task_id,
                label="solution directory",
                required=False,
            )
            tests = _open_dataset_directory(
                task_fd,
                "tests",
                task_id=task_id,
                label="tests directory",
                required=task_id == selected_tests_task_id,
            )
            instances.append(
                _construct_instance(
                    entry,
                    task_directory_fd=task_fd,
                    solution_directory_fd=(None if solution is None else solution[0]),
                    tests_directory_fd=None if tests is None else tests[0],
                )
            )
            if task_id == selected_tests_task_id:
                assert tests is not None
                selected_identity = _identity_from_open_tests_tree(tests[0])
        finally:
            cleanup_error: BaseException | None = None
            for child_name, opened, label in (
                ("tests", tests, "tests directory"),
                ("solution", solution, "solution directory"),
            ):
                if opened is None:
                    continue
                child_fd, child_before = opened
                try:
                    _verify_dataset_directory(
                        task_fd,
                        child_name,
                        child_fd,
                        child_before,
                        task_id=task_id,
                        label=label,
                    )
                except BaseException as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
                finally:
                    os.close(child_fd)
            try:
                _verify_dataset_directory(
                    tasks_root_fd,
                    task_id,
                    task_fd,
                    task_before,
                    task_id=task_id,
                    label="task directory",
                )
            except BaseException as exc:
                if cleanup_error is None:
                    cleanup_error = exc
            finally:
                os.close(task_fd)
            if cleanup_error is not None:
                raise cleanup_error

    if selected_tests_task_id is not None and selected_identity is None:
        raise DeepSWEInstanceError(
            f"task {selected_tests_task_id!r}: selected tests identity is unavailable"
        )
    return tuple(instances), selected_identity


def load_dataset(manifest_path: Path) -> tuple[DeepSWEInstance, ...]:
    """Load + validate the DeepSWE corpus from a local clone.

    ``manifest_path`` points at ``tasks/manifest.json`` inside the
    deep-swe clone. The loader walks sibling task directories
    (``tasks/<task_id>/``) for the per-task files.

    Raises ``FileNotFoundError`` if the manifest is missing, or
    ``DeepSWEInstanceError`` on any per-task validation failure.

    The manifest's declared ``task_count`` is cross-checked against
    ``len(tasks)``; mismatch raises (this would indicate corpus
    corruption mid-curation).
    """
    with _open_dataset_read_authority(manifest_path) as (
        tasks_root,
        tasks_root_fd,
        manifest_payload,
    ):
        instances, _selected_identity = _load_dataset_from_open_root(
            tasks_root,
            tasks_root_fd,
            manifest_payload,
            selected_tests_task_id=None,
        )
        return instances


def load_dataset_with_tests_identity(
    manifest_path: Path,
    task_id: str,
) -> tuple[tuple[DeepSWEInstance, ...], DeepSWETestsTreeIdentity, bytes]:
    """Load a dataset and selected tests identity under one held root chain."""

    task_id = _validate_task_id(task_id)
    with _open_dataset_read_authority(manifest_path) as (
        tasks_root,
        tasks_root_fd,
        manifest_payload,
    ):
        instances, selected_identity = _load_dataset_from_open_root(
            tasks_root,
            tasks_root_fd,
            manifest_payload,
            selected_tests_task_id=task_id,
        )
        assert selected_identity is not None
        return instances, selected_identity, manifest_payload


def resolve_task_tests_dir(manifest_path: Path, task_id: str) -> Path:
    """Resolve a verifier mount source beneath the canonical dataset root.

    Call this immediately before creating the verifier container.  It repeats
    the no-symlink checks from :func:`load_dataset` so workflow code never
    rebuilds a mount path from untrusted manifest text or a non-canonical
    ``manifest_path.parent``.
    """

    task_id = _validate_task_id(task_id)
    _, tasks_root = _resolve_manifest_path(manifest_path)
    task_dir = _resolve_dataset_member(
        tasks_root / task_id,
        dataset_root=tasks_root,
        task_id=task_id,
        label="task directory",
        directory=True,
        required=True,
    )
    assert task_dir is not None
    tests_dir = _resolve_dataset_member(
        task_dir / "tests",
        dataset_root=task_dir,
        task_id=task_id,
        label="tests directory",
        directory=True,
        required=True,
    )
    assert tests_dir is not None
    return tests_dir


def fingerprint_task_tests_tree(
    manifest_path: Path,
    task_id: str,
) -> DeepSWETestsTreeIdentity:
    """Hash the complete selected hidden-tests tree without following links.

    The identity covers normalized mounted modes, every relative byte-name,
    directory shape, and every regular-file byte. The source directory chain
    and each entry remain open and are checked again after traversal, so a
    concurrent replacement is rejected rather than fingerprinting a mixture.
    """

    with _open_task_tests_tree(manifest_path, task_id) as tests_fd:
        return _identity_from_open_tests_tree(tests_fd)


def load_task_test_inventory(
    manifest_path: Path,
    task_id: str,
) -> RewardTestInventory | None:
    """Read the whitelist sizes from the live tests tree, without authority.

    This is a fail-fast check before a task spends anything: a malformed
    ``tests/config.json`` is rejected here instead of after the cascade. The
    grade is bound to :func:`load_snapshot_test_inventory`, never to this
    read of a tree that can still change.
    """

    with _open_task_tests_tree(manifest_path, task_id) as tests_fd:
        return _test_inventory_from_open_tests_tree(tests_fd)


def load_snapshot_test_inventory(
    snapshot: DeepSWETestsSnapshot,
) -> RewardTestInventory | None:
    """Read the whitelist sizes from the private tests copy.

    The sizes come from the exact bytes the verifier mounts at ``/tests``, so
    they are bound to the snapshot digest that is checkpoint authority.
    ``None`` means the tests tree declares no whitelist.
    """

    if not isinstance(snapshot, DeepSWETestsSnapshot):
        raise TypeError("snapshot must be a DeepSWETestsSnapshot")
    snapshot_fd = os.open(snapshot.root, _nofollow_flags(directory=True))
    try:
        return _test_inventory_from_open_tests_tree(snapshot_fd)
    finally:
        os.close(snapshot_fd)


def snapshot_task_tests_tree(
    manifest_path: Path,
    task_id: str,
    staging_root: Path,
    *,
    expected_identity: DeepSWETestsTreeIdentity,
) -> DeepSWETestsSnapshot:
    """Create the exact private hidden-tests tree that Docker may mount.

    ``expected_identity`` should be captured into checkpoint authority before
    a task starts. A source change between that checkpoint and this immediate
    pre-verifier snapshot fails closed. The returned random directory is the
    only path the verifier should bind at ``/tests``; the live dataset path is
    never mounted.
    """

    if not isinstance(expected_identity, DeepSWETestsTreeIdentity):
        raise TypeError("expected_identity must be a DeepSWETestsTreeIdentity")
    staging_root = Path(staging_root)
    if not staging_root.is_absolute():
        raise DeepSWEInstanceError(
            "DeepSWE tests snapshot staging root must be absolute"
        )
    staging_root.mkdir(parents=True, mode=0o700, exist_ok=True)
    staging_metadata = staging_root.lstat()
    if not stat.S_ISDIR(staging_metadata.st_mode) or stat.S_ISLNK(
        staging_metadata.st_mode
    ):
        raise DeepSWEInstanceError(
            "DeepSWE tests snapshot staging root must be a real directory"
        )
    os.chmod(staging_root, 0o700)

    temporary = Path(
        tempfile.mkdtemp(
            prefix=".crucible-tests-snapshot-",
            dir=staging_root,
        )
    )
    published: Path | None = None
    try:
        destination_fd = os.open(
            temporary,
            _nofollow_flags(directory=True),
        )
        try:
            builder = _TestsTreeDigestBuilder()
            with _open_task_tests_tree(manifest_path, task_id) as tests_fd:
                _walk_tests_tree(
                    tests_fd,
                    destination_fd=destination_fd,
                    builder=builder,
                    require_test_script=True,
                )
                observed_identity = builder.finish()
            if observed_identity != expected_identity:
                raise DeepSWEInstanceError(
                    "DeepSWE tests tree changed after checkpoint identity "
                    f"(expected={expected_identity.sha256}, "
                    f"observed={observed_identity.sha256})"
                )
            os.fchmod(destination_fd, _snapshot_directory_mode(0))
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)

        # Re-read the completed copy through the same no-follow traversal.
        # This binds the returned digest to exact bytes and modes in the tree
        # that will be mounted, not merely to bytes observed from the source.
        copied_fd = os.open(
            temporary,
            _nofollow_flags(directory=True),
        )
        try:
            copied_identity = _identity_from_open_tests_tree(copied_fd)
        finally:
            os.close(copied_fd)
        if copied_identity != observed_identity:
            raise DeepSWEInstanceError(
                "private DeepSWE tests snapshot differs from its source"
            )

        suffix = temporary.name.removeprefix(".crucible-tests-snapshot-")
        published = staging_root / (f"tests-{copied_identity.sha256[:16]}-{suffix}")
        os.rename(temporary, published)
        staging_fd = os.open(
            staging_root,
            _nofollow_flags(directory=True),
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        return DeepSWETestsSnapshot(
            sha256=copied_identity.sha256,
            file_count=copied_identity.file_count,
            directory_count=copied_identity.directory_count,
            total_bytes=copied_identity.total_bytes,
            root=published,
        )
    except BaseException:
        _remove_private_snapshot(published or temporary)
        raise


def filter_instances(
    instances: Iterable[DeepSWEInstance],
    *,
    task_ids: Iterable[str] | None = None,
    repos: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    limit: int | None = None,
) -> tuple[DeepSWEInstance, ...]:
    """Apply optional filters in declaration order: task_ids → repos
    → languages → categories → limit.

    All filters are inclusive (whitelists). Empty / None means "no
    filter."

    Language filter inputs are compared case-insensitively against the
    loader's closed lowercase language domain.
    """
    out = list(instances)
    if task_ids is not None:
        wanted = set(task_ids)
        if wanted:
            out = [i for i in out if i.task_id in wanted]
    if repos is not None:
        wanted_repos = set(repos)
        if wanted_repos:
            out = [i for i in out if i.repo in wanted_repos]
    if languages is not None:
        wanted_langs = {str(l).lower() for l in languages}
        if wanted_langs:
            out = [i for i in out if i.language in wanted_langs]
    if categories is not None:
        wanted_cats = set(categories)
        if wanted_cats:
            out = [i for i in out if i.category in wanted_cats]
    if limit is not None and limit > 0:
        out = out[:limit]
    return tuple(out)


__all__ = [
    "DeepSWEInstance",
    "DeepSWEInstanceError",
    "DeepSWETestsSnapshot",
    "DeepSWETestsTreeIdentity",
    "fingerprint_task_tests_tree",
    "load_dataset",
    "load_dataset_with_tests_identity",
    "load_snapshot_test_inventory",
    "load_task_test_inventory",
    "filter_instances",
    "resolve_task_tests_dir",
    "snapshot_task_tests_tree",
]
