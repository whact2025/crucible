# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""How one write root reads a leading path component that is its own name.

A path such as ``proj/x.py``, handed to a run whose write root is a directory
named ``proj``, can name ``x.py`` in that root or ``x.py`` below a directory
``proj`` inside it. The reading is decided once per root, from the kind of the
one entry ``<root>/<name>``, and never from the path that is being read:

- absent, or a regular file: the name is redundant and is dropped once;
- a real directory: every path is literal;
- a link or any other node: a path that runs through the name is refused.

A path that begins with the name twice is refused under every reading. A path
that is the name alone is that entry itself and is never changed.

Leaf module: standard library only.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


REDUNDANT = "redundant"   # <root>/<name> is absent or a regular file
DIRECTORY = "directory"   # <root>/<name> is a real directory
BLOCKED = "blocked"       # <root>/<name> is a link or any other node


class ScopeReadingError(RuntimeError):
    """No reading: an entry cannot be inspected, or the root has no name."""


class ScopeReadingChanged(ScopeReadingError):
    """The entry named like the root is no longer what the run decided on."""


class ScopePathError(ValueError):
    """An authored path is refused by the rule."""


def leading_name_rule(
    parts: tuple[str, ...], name: str, flag: object,
) -> tuple[str, ...] | None:
    """Apply the rule to the parts of a lexically normal relative path.

    ``flag`` is the decided reading as a tool context carries it
    (``workspace_relative_paths``): True when the name is redundant, False
    when every path is literal. Any other value is an undecided reading.
    Returns the parts of the literal path, or None when the path is refused:
    it begins with the name twice, or it begins with the name while the
    reading is undecided.
    """

    if len(parts) > 1 and parts[0] == name:
        if parts[1] == name or not isinstance(flag, bool):
            return None
        if flag:
            return tuple(parts[1:])
    return tuple(parts)


def _entry_kind(root: Path, name: str, dir_fd: int | None = None) -> str:
    """Return the kind of ``<root>/<name>`` from one look that follows no link.

    With ``dir_fd`` the look is made below that held descriptor of the root
    instead of through the root's path.
    """

    try:
        if dir_fd is None:
            metadata = os.lstat(root / name)
        else:
            metadata = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return REDUNDANT
    except OSError as exc:
        # Fail closed: without a clear answer there is no reading.
        raise ScopeReadingError(
            f"cannot inspect {name!r} below the write root {root}: {exc}"
        ) from exc
    if stat.S_ISREG(metadata.st_mode):
        return REDUNDANT
    if stat.S_ISDIR(metadata.st_mode):
        return DIRECTORY
    return BLOCKED


@dataclass(frozen=True)
class ScopeReading:
    """How one run reads a leading checkout name, decided once for one root."""

    root: Path          # the resolved write root
    name: str           # root.name
    leading_name: str   # REDUNDANT, DIRECTORY or BLOCKED

    @property
    def workspace_relative_paths(self) -> bool:
        """The reading as a tool context carries it."""

        return self.leading_name == REDUNDANT

    def resolve(self, authored: str) -> str:
        """Return the literal path an authored path names under this reading.

        Applied to its own result the rule changes nothing: a result never
        begins with the name while the name is redundant, and nothing is
        dropped under the two other readings.

        The rule reads the leading name and nothing else. It does not refuse
        an absolute path or one that holds ``..``, so a caller has to check
        apart from it that a path stays below the root.
        """

        if not isinstance(authored, str) or not authored or "\0" in authored:
            raise ScopePathError(f"scope path is not one safe string: {authored!r}")
        # A backslash is a separator on Windows only; elsewhere it is a byte
        # of a file name, as the mutation journal reads it.
        text = authored.replace("\\", "/") if os.name == "nt" else authored
        parts = PurePosixPath(text).parts
        if len(parts) > 1 and parts[0] == self.name:
            if parts[1] == self.name:
                raise ScopePathError(
                    f"scope path {authored!r} repeats the checkout name "
                    f"{self.name!r}; write it relative to {self.root}"
                )
            if self.leading_name == BLOCKED:
                raise ScopePathError(
                    f"scope path {authored!r} runs through {self.name!r}, which "
                    f"is not a directory below {self.root}"
                )
        kept = leading_name_rule(parts, self.name, self.workspace_relative_paths)
        assert kept is not None  # both refusals of the rule are raised above
        result = PurePosixPath(*kept).as_posix() if kept else "."
        if text.endswith("/") and result not in {".", "/"}:
            return result + "/"   # an authored directory entry stays one
        return result

    def resolve_all(self, authored: Iterable[str]) -> tuple[str, ...]:
        """Resolve every path; the order is kept and duplicates are dropped."""

        return tuple(dict.fromkeys(self.resolve(item) for item in authored))

    def require_literal(self, path: str) -> str:
        """Return ``path`` unchanged, or refuse it.

        For a path that workflow code made itself and is about to put into an
        authority list: it must already be the literal path, so a path that
        the rule would change or refuse is refused here. Like ``resolve`` it
        does not refuse an absolute path or one that holds ``..``.
        """

        if self.resolve(path) != _lexical(path):
            raise ScopePathError(
                f"workflow path {path!r} begins with the checkout name "
                f"{self.name!r} while no such directory exists below {self.root}"
            )
        return path

    def verify(self, *, dir_fd: int | None = None) -> None:
        """Take the same look again and raise when its answer has moved."""

        now = _entry_kind(self.root, self.name, dir_fd)
        if now != self.leading_name:
            raise ScopeReadingChanged(
                f"{self.name!r} below {self.root} was read as "
                f"{self.leading_name} and is now {now}"
            )


def _lexical(path: str) -> str:
    text = path.replace("\\", "/") if os.name == "nt" else path
    result = PurePosixPath(text).as_posix()
    return result + "/" if text.endswith("/") and result not in {".", "/"} else result


def decide_scope_reading(root: str | os.PathLike[str]) -> ScopeReading:
    """Decide the reading of ``root`` from one look at ``<root>/<name>``.

    The name is that of the resolved root, whatever spelling the run was
    given, because the isolated mutation host names its root after resolving
    links. A file-system root has no name and cannot be a write root.
    """

    resolved = Path(root).resolve(strict=True)
    if not resolved.name:
        raise ScopeReadingError("a file-system root cannot be a write root")
    return ScopeReading(
        root=resolved,
        name=resolved.name,
        leading_name=_entry_kind(resolved, resolved.name),
    )


__all__ = [
    "BLOCKED",
    "DIRECTORY",
    "REDUNDANT",
    "ScopePathError",
    "ScopeReading",
    "ScopeReadingChanged",
    "ScopeReadingError",
    "decide_scope_reading",
    "leading_name_rule",
]
