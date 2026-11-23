# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Descriptor-relative reads for untrusted hidden-verifier artifacts.

The verifier container can write its bind-mounted ``/logs`` tree.  Treat every
path component and leaf as attacker-controlled until it has been opened with
``O_NOFOLLOW`` and checked through the resulting file descriptor.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
import errno
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


REWARD_MAX_BYTES = 2
REWARD_JSON_MAX_BYTES = 16 * 1024

# The closed document the dataset's v1.1 grader writes. ``apply_failed`` is
# present, as the integer 1, only when the candidate patch did not apply.
_REWARD_COUNT_FIELDS = ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
_REWARD_RATIO_FIELDS = ("f2p", "p2p", "partial")
_REWARD_REQUIRED_FIELDS = frozenset(
    {"reward", *_REWARD_COUNT_FIELDS, *_REWARD_RATIO_FIELDS}
)
_REWARD_ADMITTED_FIELDS = _REWARD_REQUIRED_FIELDS | {"apply_failed"}
_REWARD_RATIO_TOLERANCE = 1e-9


class ArtifactReadStatus(StrEnum):
    OK = "ok"
    MISSING = "missing"
    UNSAFE = "unsafe"
    OVERSIZE = "oversize"
    CHANGED = "changed"
    UNREADABLE = "unreadable"


@dataclasses.dataclass(frozen=True)
class BoundedArtifactRead:
    status: ArtifactReadStatus
    data: bytes = dataclasses.field(default=b"", repr=False)
    error: str | None = None


def _is_exact_count(value: object) -> bool:
    return type(value) is int and value >= 0


@dataclasses.dataclass(frozen=True)
class RewardTestInventory:
    """Whitelist sizes a task's v1.1 reward must report as its totals."""

    f2p_total: int
    p2p_total: int

    def __post_init__(self) -> None:
        if not (
            _is_exact_count(self.f2p_total)
            and _is_exact_count(self.p2p_total)
        ):
            raise ValueError(
                "reward test inventory sizes must be non-negative integers"
            )

    def to_payload(self) -> dict[str, int]:
        return {"f2p_total": self.f2p_total, "p2p_total": self.p2p_total}


@dataclasses.dataclass(frozen=True)
class RewardCounts:
    """Per-test evidence behind one v1.1 reward.

    ``apply_failed`` records that the dataset grader could not apply the
    candidate patch and therefore ran no test.
    """

    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    apply_failed: bool = False

    def __post_init__(self) -> None:
        if not all(
            _is_exact_count(value)
            for value in (
                self.f2p_passed,
                self.f2p_total,
                self.p2p_passed,
                self.p2p_total,
            )
        ):
            raise ValueError("reward counts must be non-negative integers")
        if (
            self.f2p_passed > self.f2p_total
            or self.p2p_passed > self.p2p_total
        ):
            raise ValueError("reward counts pass more tests than they total")
        if type(self.apply_failed) is not bool:
            raise ValueError("reward apply_failed must be a boolean")
        if self.apply_failed and (self.f2p_passed or self.p2p_passed):
            raise ValueError("a failed patch application cannot pass tests")

    @property
    def earned_reward(self) -> int:
        """The binary reward the dataset's rule derives from these counts."""

        return int(
            self.f2p_total > 0
            and self.f2p_passed == self.f2p_total
            and self.p2p_passed == self.p2p_total
        )

    def to_payload(self) -> dict[str, int | bool]:
        return dataclasses.asdict(self)

    @classmethod
    def from_payload(cls, value: object) -> RewardCounts:
        names = {field.name for field in dataclasses.fields(cls)}
        if type(value) is not dict or set(value) != names:
            raise ValueError("reward counts payload must have exact fields")
        return cls(**value)


@dataclasses.dataclass(frozen=True)
class RewardArtifactRead:
    reward: float | None
    token: str
    state: str
    error: str | None = None
    counts: RewardCounts | None = None


def _flags(*, directory: bool = False, leaf: bool = False) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise OSError(errno.ENOTSUP, "O_NOFOLLOW is required")
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= os.O_DIRECTORY
    if leaf:
        # Opening a FIFO before checking its type would otherwise block.
        flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _safe_components(path: Path) -> tuple[str, ...]:
    if not path.is_absolute():
        raise ValueError("artifact root must be absolute")
    components = tuple(path.parts[1:])
    if any(
        not component
        or component in {".", ".."}
        or "/" in component
        or "\x00" in component
        for component in components
    ):
        raise ValueError("artifact root contains an unsafe component")
    return components


def _relative_components(relative: str | PurePosixPath) -> tuple[str, ...]:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts:
        raise ValueError("artifact path must be a non-empty relative path")
    components = tuple(path.parts)
    if any(
        not component
        or component in {".", ".."}
        or "/" in component
        or "\x00" in component
        for component in components
    ):
        raise ValueError("artifact path contains an unsafe component")
    return components


def _open_directory_chain(root: Path) -> tuple[list[int], list[str]]:
    descriptors = [os.open("/", _flags(directory=True))]
    names: list[str] = []
    try:
        for component in _safe_components(root):
            child = os.open(
                component,
                _flags(directory=True),
                dir_fd=descriptors[-1],
            )
            descriptors.append(child)
            names.append(component)
        return descriptors, names
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _directory_chain_is_stable(
    descriptors: list[int],
    names: list[str],
) -> bool:
    for parent_fd, child_fd, name in zip(
        descriptors[:-1],
        descriptors[1:],
        names,
        strict=True,
    ):
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        opened = os.fstat(child_fd)
        if not stat.S_ISDIR(entry.st_mode) or not _same_inode(entry, opened):
            return False
    return True


def _status_for_oserror(exc: OSError) -> ArtifactReadStatus:
    if exc.errno == errno.ENOENT:
        return ArtifactReadStatus.MISSING
    if exc.errno in {
        errno.ELOOP,
        errno.ENOTDIR,
        errno.ENXIO,
        errno.ENODEV,
        errno.EISDIR,
    }:
        return ArtifactReadStatus.UNSAFE
    return ArtifactReadStatus.UNREADABLE


def read_bounded_regular_artifact(
    root: Path,
    relative: str | PurePosixPath,
    *,
    max_bytes: int,
) -> BoundedArtifactRead:
    """Read one regular file without following any path component.

    The returned bytes are capped both through ``fstat`` and an extra-byte
    read. Metadata is checked again after reading so concurrent mutation is
    rejected instead of producing a mixed artifact.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("artifact max_bytes must be a non-negative integer")
    try:
        components = _relative_components(relative)
        directory_fds, directory_names = _open_directory_chain(Path(root))
    except ValueError as exc:
        return BoundedArtifactRead(
            ArtifactReadStatus.UNSAFE,
            error=str(exc),
        )
    except OSError as exc:
        return BoundedArtifactRead(
            _status_for_oserror(exc),
            error=f"artifact directory open failed: errno={exc.errno}",
        )

    leaf_fd: int | None = None
    try:
        for component in components[:-1]:
            child = os.open(
                component,
                _flags(directory=True),
                dir_fd=directory_fds[-1],
            )
            directory_fds.append(child)
            directory_names.append(component)
        leaf_fd = os.open(
            components[-1],
            _flags(leaf=True),
            dir_fd=directory_fds[-1],
        )
        before = os.fstat(leaf_fd)
        if not stat.S_ISREG(before.st_mode):
            return BoundedArtifactRead(
                ArtifactReadStatus.UNSAFE,
                error="artifact leaf is not a regular file",
            )
        if before.st_size > max_bytes:
            return BoundedArtifactRead(
                ArtifactReadStatus.OVERSIZE,
                error=(
                    "artifact exceeds its byte bound: "
                    f"size={before.st_size}, max={max_bytes}"
                ),
            )

        data = bytearray()
        while len(data) <= max_bytes:
            remaining = max_bytes + 1 - len(data)
            chunk = os.read(leaf_fd, min(64 * 1024, remaining))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > max_bytes:
            return BoundedArtifactRead(
                ArtifactReadStatus.OVERSIZE,
                error=f"artifact grew beyond its byte bound: max={max_bytes}",
            )
        after = os.fstat(leaf_fd)
        leaf_entry = os.stat(
            components[-1],
            dir_fd=directory_fds[-1],
            follow_symlinks=False,
        )
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            )
            or len(data) != after.st_size
            or not _same_inode(after, leaf_entry)
            or not stat.S_ISREG(leaf_entry.st_mode)
            or not _directory_chain_is_stable(directory_fds, directory_names)
        ):
            return BoundedArtifactRead(
                ArtifactReadStatus.CHANGED,
                error="artifact changed while it was being read",
            )
        return BoundedArtifactRead(ArtifactReadStatus.OK, data=bytes(data))
    except OSError as exc:
        return BoundedArtifactRead(
            _status_for_oserror(exc),
            error=f"artifact open/read failed: errno={exc.errno}",
        )
    finally:
        if leaf_fd is not None:
            os.close(leaf_fd)
        for descriptor in reversed(directory_fds):
            os.close(descriptor)


def _grade_reward_document(
    payload: dict[str, Any],
    *,
    inventory: RewardTestInventory | None,
) -> RewardArtifactRead:
    """Check one parsed ``reward.json`` against the dataset v1.1 contract.

    A document that is not the contract's exact shape is ``invalid``. A
    well-formed document whose reward, ratios, or totals contradict its own
    counts or the task's whitelist is a ``mismatch``. Neither is a grade.
    Rejection reasons name contract fields only, never content of the file.
    """

    def rejected(state: str, reason: str) -> RewardArtifactRead:
        return RewardArtifactRead(None, "", state, f"reward.json {reason}")

    fields = set(payload)
    if not _REWARD_REQUIRED_FIELDS <= fields <= _REWARD_ADMITTED_FIELDS:
        return rejected("invalid", "fields are not the exact v1.1 contract")
    reward = payload["reward"]
    if type(reward) is not int or reward not in (0, 1):
        return rejected("invalid", "reward must be the integer 0 or 1")
    if not all(_is_exact_count(payload[name]) for name in _REWARD_COUNT_FIELDS):
        return rejected("invalid", "counts must be non-negative integers")
    if not all(
        type(payload[name]) is float and 0.0 <= payload[name] <= 1.0
        for name in _REWARD_RATIO_FIELDS
    ):
        return rejected("invalid", "ratios must be floats between 0 and 1")
    apply_failed = "apply_failed" in payload
    if apply_failed and (
        type(payload["apply_failed"]) is not int or payload["apply_failed"] != 1
    ):
        return rejected("invalid", "apply_failed must be the integer 1")

    f2p_passed, f2p_total, p2p_passed, p2p_total = (
        payload[name] for name in _REWARD_COUNT_FIELDS
    )
    if f2p_passed > f2p_total or p2p_passed > p2p_total:
        return rejected("mismatch", "passes more tests than it totals")
    if apply_failed and (reward or f2p_passed or p2p_passed):
        return rejected(
            "mismatch",
            "reports a failed patch application with a reward or passed tests",
        )
    counts = RewardCounts(
        f2p_passed=f2p_passed,
        f2p_total=f2p_total,
        p2p_passed=p2p_passed,
        p2p_total=p2p_total,
        apply_failed=apply_failed,
    )
    total = f2p_total + p2p_total
    derived_ratios = {
        "f2p": f2p_passed / f2p_total if f2p_total else 0.0,
        "p2p": p2p_passed / p2p_total if p2p_total else 1.0,
        "partial": (f2p_passed + p2p_passed) / total if total else 0.0,
    }
    if not all(
        math.isclose(
            payload[name],
            derived_ratios[name],
            rel_tol=0.0,
            abs_tol=_REWARD_RATIO_TOLERANCE,
        )
        for name in _REWARD_RATIO_FIELDS
    ):
        return rejected("mismatch", "ratios disagree with its counts")
    if reward != counts.earned_reward:
        return rejected("mismatch", "reward disagrees with its counts")
    if inventory is not None and (f2p_total, p2p_total) != (
        inventory.f2p_total,
        inventory.p2p_total,
    ):
        return rejected(
            "mismatch",
            "totals are not the sizes of the task's test whitelist",
        )
    return RewardArtifactRead(float(reward), str(reward), "valid", counts=counts)


def read_reward_artifact(
    logs_dir: Path,
    *,
    inventory: RewardTestInventory | None,
) -> RewardArtifactRead:
    """Read one v1.0 text or v1.1 JSON reward with a strict grammar.

    ``inventory`` is the whitelist the task's ``tests/config.json`` declares.
    When it is given, a ``reward.json`` must report exactly its totals. A
    caller without that authority passes ``None``; the document is then
    checked against itself only. ``reward.txt`` carries no counts.
    """

    legacy = read_bounded_regular_artifact(
        logs_dir,
        PurePosixPath("verifier/reward.txt"),
        max_bytes=REWARD_MAX_BYTES,
    )
    current = read_bounded_regular_artifact(
        logs_dir,
        PurePosixPath("verifier/reward.json"),
        max_bytes=REWARD_JSON_MAX_BYTES,
    )
    legacy_present = legacy.status != ArtifactReadStatus.MISSING
    current_present = current.status != ArtifactReadStatus.MISSING
    if legacy_present and current_present:
        return RewardArtifactRead(
            None,
            "",
            "invalid",
            "multiple reward artifacts are ambiguous",
        )
    if current_present:
        if current.status != ArtifactReadStatus.OK:
            return RewardArtifactRead(None, "", "unreadable", current.error)

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            parsed: dict[str, object] = {}
            for name, value in pairs:
                if name in parsed:
                    raise ValueError("reward.json contains a duplicate field")
                parsed[name] = value
            return parsed

        try:

            def parse_finite_float(value: str) -> float:
                parsed = float(value)
                if not math.isfinite(parsed):
                    raise ValueError("reward.json contains a non-finite number")
                return parsed

            payload = json.loads(
                current.data.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
                parse_float=parse_finite_float,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"reward.json contains non-finite value {value}")
                ),
            )
        except (
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
            RecursionError,
        ) as exc:
            return RewardArtifactRead(
                None,
                "",
                "invalid",
                f"reward.json is not strict JSON: {exc}",
            )
        if not isinstance(payload, dict):
            return RewardArtifactRead(
                None,
                "",
                "invalid",
                "reward.json must contain one JSON object",
            )
        return _grade_reward_document(payload, inventory=inventory)
    if not legacy_present:
        return RewardArtifactRead(None, "", "missing", legacy.error or current.error)
    if legacy.status != ArtifactReadStatus.OK:
        return RewardArtifactRead(None, "", "unreadable", legacy.error)
    tokens = {
        b"0": (0.0, "0"),
        b"0\n": (0.0, "0"),
        b"1": (1.0, "1"),
        b"1\n": (1.0, "1"),
    }
    parsed = tokens.get(legacy.data)
    if parsed is None:
        return RewardArtifactRead(
            None,
            "",
            "invalid",
            "reward.txt did not contain an exact binary reward token",
        )
    reward, token = parsed
    return RewardArtifactRead(reward, token, "valid")


__all__ = [
    "ArtifactReadStatus",
    "BoundedArtifactRead",
    "REWARD_JSON_MAX_BYTES",
    "REWARD_MAX_BYTES",
    "RewardArtifactRead",
    "RewardCounts",
    "RewardTestInventory",
    "read_bounded_regular_artifact",
    "read_reward_artifact",
]
