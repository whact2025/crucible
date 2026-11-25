# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""One write root reads a leading component of its own name one way.

The reading is decided from the kind of the one entry ``<root>/<name>``:
absent or a regular file, the name is redundant and is dropped once; a real
directory, every path is literal; a link or any other node, a path through
the name is refused. A doubled name is refused under every reading, and the
pure rule refuses a path that begins with the name while the reading is
undecided.
"""

from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Callable

import pytest

from crucible.common import scope_reading
from crucible.common.scope_reading import (
    BLOCKED,
    DIRECTORY,
    REDUNDANT,
    ScopePathError,
    ScopeReading,
    ScopeReadingChanged,
    ScopeReadingError,
    decide_scope_reading,
    leading_name_rule,
)


NAME = "proj"  # basename of every write root built here
REFUSED = None  # what the pure rule returns for a path it refuses

# Why a reading refuses a path: a phrase of the error it raises.
DOUBLED = "repeats the checkout name 'proj'"
THROUGH = "runs through 'proj', which is not a directory"
UNSAFE = "is not one safe string"
_REFUSALS = (DOUBLED, THROUGH, UNSAFE)

# Kind of the entry named like the root -> the reading it gives.
_ENTRY_KINDS = (
    ("absent", REDUNDANT),
    ("regular_file", REDUNDANT),
    ("directory", DIRECTORY),
    ("empty_directory", DIRECTORY),
    ("link_to_directory", BLOCKED),
    ("link_to_file", BLOCKED),
    ("dangling_link", BLOCKED),
    pytest.param("fifo", BLOCKED, marks=pytest.mark.skipif(
        not hasattr(os, "mkfifo"), reason="this platform has no FIFO",
    )),
)

# Reading, authored path, then the literal path it names or, where the path
# is refused with a ScopePathError, why.
_RESOLVE_ROWS = (
    (REDUNDANT, "x.py", "x.py"),
    (REDUNDANT, "proj/x.py", "x.py"),
    (REDUNDANT, "proj/newdir/new.py", "newdir/new.py"),
    (REDUNDANT, "./proj/x.py", "x.py"),
    (REDUNDANT, "proj/pkg/", "pkg/"),
    (REDUNDANT, "proj", "proj"),
    (REDUNDANT, "proj/", "proj/"),
    (REDUNDANT, "./", "."),
    (REDUNDANT, ".", "."),
    (REDUNDANT, "other/proj/x.py", "other/proj/x.py"),
    (REDUNDANT, "pkg/proj/proj/x.py", "pkg/proj/proj/x.py"),
    (REDUNDANT, "proj/proj", DOUBLED),
    (REDUNDANT, "proj/proj/x.py", DOUBLED),
    (REDUNDANT, "proj/./proj/x.py", DOUBLED),
    (REDUNDANT, "proj//proj/x.py", DOUBLED),
    (REDUNDANT, "", UNSAFE),
    (REDUNDANT, "a\0b", UNSAFE),
    (DIRECTORY, "x.py", "x.py"),
    (DIRECTORY, "proj/x.py", "proj/x.py"),
    (DIRECTORY, "proj/newdir/new.py", "proj/newdir/new.py"),
    (DIRECTORY, "proj/pkg/", "proj/pkg/"),
    (DIRECTORY, "proj", "proj"),
    (DIRECTORY, "./", "."),
    (DIRECTORY, "proj/proj/x.py", DOUBLED),
    (DIRECTORY, "", UNSAFE),
    (BLOCKED, "x.py", "x.py"),
    (BLOCKED, "proj", "proj"),
    (BLOCKED, "proj/mod.py", THROUGH),
    # A doubled name is refused as one, whatever the entry is.
    (BLOCKED, "proj/proj/x.py", DOUBLED),
)

# The pure rule: path, then its result when the name is redundant (True),
# when every path is literal (False), and when the reading is undecided.
_RULE_ROWS = (
    ("x.py", "x.py", "x.py", "x.py"),
    ("pkg/mod.py", "pkg/mod.py", "pkg/mod.py", "pkg/mod.py"),
    ("proj", "proj", "proj", "proj"),
    ("proj/x.py", "x.py", "proj/x.py", REFUSED),
    ("proj/pkg/mod.py", "pkg/mod.py", "proj/pkg/mod.py", REFUSED),
    ("proj/proj", REFUSED, REFUSED, REFUSED),
    ("proj/proj/x.py", REFUSED, REFUSED, REFUSED),
    ("other/proj/x.py", "other/proj/x.py", "other/proj/x.py", "other/proj/x.py"),
    ("pkg/proj/proj/x.py", "pkg/proj/proj/x.py", "pkg/proj/proj/x.py",
     "pkg/proj/proj/x.py"),
    ("projx/x.py", "projx/x.py", "projx/x.py", "projx/x.py"),
)
_UNDECIDED_FLAGS = (None, "yes", "", 1, 0)


def _root(tmp_path: Path, entry: str) -> Path:
    """Build ``<tmp>/<entry>/proj`` with one kind of entry named like it."""

    root = tmp_path / entry / NAME
    root.mkdir(parents=True)
    (root / "x.py").write_text("x\n", encoding="utf-8")
    (root / "pkg").mkdir()
    named = root / NAME
    if entry == "directory":
        named.mkdir()
        (named / "x.py").write_text("nested x\n", encoding="utf-8")
    elif entry == "empty_directory":
        named.mkdir()
    elif entry == "regular_file":
        named.write_text("a regular file\n", encoding="utf-8")
    elif entry == "link_to_directory":
        named.symlink_to("pkg", target_is_directory=True)
    elif entry == "link_to_file":
        named.symlink_to("x.py")
    elif entry == "dangling_link":
        named.symlink_to("absent")
    elif entry == "fifo":
        os.mkfifo(named)
    elif entry != "absent":
        raise ValueError(f"unknown entry kind {entry!r}")
    return root


def _reading(tmp_path: Path, kind: str) -> ScopeReading:
    root = tmp_path / NAME
    root.mkdir(exist_ok=True)
    return ScopeReading(root=root, name=NAME, leading_name=kind)


def _parts(path: str) -> tuple[str, ...]:
    return PurePosixPath(path).parts


def _joined(parts: tuple[str, ...] | None) -> str | None:
    return None if parts is None else PurePosixPath(*parts).as_posix()


def test_the_three_readings_have_their_names() -> None:
    assert (REDUNDANT, DIRECTORY, BLOCKED) == ("redundant", "directory", "blocked")


@pytest.mark.parametrize(("entry", "expected"), _ENTRY_KINDS)
def test_the_entry_named_like_the_root_decides_the_reading(
    tmp_path: Path, entry: str, expected: str,
) -> None:
    root = _root(tmp_path, entry)

    reading = decide_scope_reading(root)

    assert reading.leading_name == expected
    assert reading.name == NAME
    assert reading.root == root.resolve()
    assert reading.workspace_relative_paths is (expected == REDUNDANT)


def test_a_root_given_as_a_link_is_read_under_its_resolved_name(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path, "absent")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    reading = decide_scope_reading(alias)

    assert reading.name == NAME
    assert reading.root == root.resolve()
    assert reading.resolve("proj/x.py") == "x.py"
    assert reading.resolve("alias/x.py") == "alias/x.py"


def test_a_file_system_root_and_a_missing_root_have_no_reading(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ScopeReadingError, match="^a file-system root cannot be a write root$",
    ):
        decide_scope_reading(Path(tmp_path.anchor))
    with pytest.raises(FileNotFoundError):
        decide_scope_reading(tmp_path / "missing")


def test_an_entry_that_cannot_be_inspected_gives_no_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path, "directory")
    entry = os.fspath(root.resolve() / NAME)
    real_lstat = os.lstat

    def denied(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if os.fspath(path) == entry:  # type: ignore[call-overload]
            raise PermissionError(13, "denied")
        return real_lstat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(scope_reading.os, "lstat", denied)

    with pytest.raises(ScopeReadingError) as caught:
        decide_scope_reading(root)
    assert str(caught.value) == (
        f"cannot inspect 'proj' below the write root {root.resolve()}: "
        "[Errno 13] denied"
    )


@pytest.mark.parametrize(("kind", "authored", "expected"), _RESOLVE_ROWS)
def test_resolve_names_one_literal_path_or_refuses(
    tmp_path: Path, kind: str, authored: str, expected: str | None,
) -> None:
    reading = _reading(tmp_path, kind)

    if expected in _REFUSALS:
        with pytest.raises(ScopePathError, match=re.escape(expected)):
            reading.resolve(authored)
    else:
        assert reading.resolve(authored) == expected


@pytest.mark.parametrize("authored", [None, 7, b"proj/x.py", ("proj", "x.py")])
def test_resolve_refuses_what_is_not_one_string(
    tmp_path: Path, authored: object,
) -> None:
    with pytest.raises(ScopePathError, match="not one safe string"):
        _reading(tmp_path, REDUNDANT).resolve(authored)  # type: ignore[arg-type]


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a separator on Windows")
def test_a_backslash_is_a_byte_of_a_file_name(tmp_path: Path) -> None:
    reading = _reading(tmp_path, REDUNDANT)

    assert reading.resolve("proj\\x.py") == "proj\\x.py"
    # So a machine path that holds one is literal as it stands.
    assert reading.require_literal("tests\\t.py") == "tests\\t.py"


def test_resolve_applied_to_its_own_result_changes_nothing(tmp_path: Path) -> None:
    checked = 0
    for kind, authored, expected in _RESOLVE_ROWS:
        if expected in _REFUSALS:
            continue
        reading = _reading(tmp_path, kind)
        assert reading.resolve(reading.resolve(authored)) == expected, (kind, authored)
        checked += 1
    assert checked == 19


def test_a_result_never_begins_with_a_redundant_name(tmp_path: Path) -> None:
    reading = _reading(tmp_path, REDUNDANT)
    for kind, authored, expected in _RESOLVE_ROWS:
        if kind != REDUNDANT or expected in _REFUSALS:
            continue
        parts = _parts(reading.resolve(authored))
        assert not (len(parts) > 1 and parts[0] == NAME), authored


def test_resolve_all_keeps_the_order_and_drops_what_the_rule_makes_equal(
    tmp_path: Path,
) -> None:
    reading = _reading(tmp_path, REDUNDANT)

    assert reading.resolve_all(("proj/x.py", "x.py", "pkg/", "proj/pkg/")) == (
        "x.py", "pkg/",
    )
    with pytest.raises(ScopePathError):
        reading.resolve_all(("x.py", "proj/proj/x.py"))


def test_a_machine_path_must_already_be_literal(tmp_path: Path) -> None:
    flat = decide_scope_reading(_root(tmp_path, "absent"))
    nested = decide_scope_reading(_root(tmp_path, "directory"))
    blocked = decide_scope_reading(_root(tmp_path, "link_to_directory"))

    assert flat.require_literal("tests/test_attack.py") == "tests/test_attack.py"
    assert flat.require_literal("tests/") == "tests/"
    with pytest.raises(ScopePathError, match="begins with the checkout name"):
        flat.require_literal("proj/tests/test_attack.py")
    assert nested.require_literal("proj/tests/test_attack.py") == (
        "proj/tests/test_attack.py"
    )
    with pytest.raises(ScopePathError):
        nested.require_literal("proj/proj/t.py")
    with pytest.raises(ScopePathError):
        blocked.require_literal("proj/t.py")
    assert blocked.require_literal("t.py") == "t.py"


def test_an_error_names_the_path_the_name_and_the_root(tmp_path: Path) -> None:
    flat = decide_scope_reading(_root(tmp_path, "absent"))
    blocked = decide_scope_reading(_root(tmp_path, "link_to_directory"))

    def refused(call: Callable[[str], str], path: str) -> str:
        with pytest.raises(ScopePathError) as caught:
            call(path)
        return str(caught.value)

    assert refused(flat.resolve, "proj/proj/x.py") == (
        "scope path 'proj/proj/x.py' repeats the checkout name 'proj'; write it "
        f"relative to {flat.root}"
    )
    assert refused(blocked.resolve, "proj/mod.py") == (
        "scope path 'proj/mod.py' runs through 'proj', which is not a directory "
        f"below {blocked.root}"
    )
    assert refused(flat.resolve, "") == "scope path is not one safe string: ''"
    assert refused(flat.require_literal, "proj/tests/t.py") == (
        "workflow path 'proj/tests/t.py' begins with the checkout name 'proj' "
        f"while no such directory exists below {flat.root}"
    )
    (flat.root / NAME).mkdir()
    with pytest.raises(ScopeReadingChanged) as moved:
        flat.verify()
    assert str(moved.value) == (
        f"'proj' below {flat.root} was read as redundant and is now directory"
    )


def test_verify_raises_once_the_entry_appears(tmp_path: Path) -> None:
    reading = decide_scope_reading(_root(tmp_path, "absent"))
    reading.verify()

    (reading.root / NAME).mkdir()

    with pytest.raises(ScopeReadingChanged, match="was read as redundant"):
        reading.verify()
    assert isinstance(ScopeReadingChanged("x"), ScopeReadingError)
    decide_scope_reading(reading.root).verify()


def test_verify_raises_once_the_entry_is_gone(tmp_path: Path) -> None:
    reading = decide_scope_reading(_root(tmp_path, "directory"))
    reading.verify()

    (reading.root / NAME / "x.py").unlink()
    (reading.root / NAME).rmdir()

    with pytest.raises(ScopeReadingChanged, match="is now redundant"):
        reading.verify()


def test_a_regular_file_in_place_of_an_absent_entry_is_the_same_reading(
    tmp_path: Path,
) -> None:
    reading = decide_scope_reading(_root(tmp_path, "absent"))

    (reading.root / NAME).write_text("a regular file\n", encoding="utf-8")

    reading.verify()


@pytest.mark.skipif(
    os.name == "nt" or os.stat not in os.supports_dir_fd,
    reason="needs stat below a directory descriptor",
)
def test_verify_takes_its_look_below_a_held_root_descriptor(tmp_path: Path) -> None:
    reading = decide_scope_reading(_root(tmp_path, "absent"))
    descriptor = os.open(reading.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        reading.verify(dir_fd=descriptor)
        # The root's path now leads elsewhere; the held descriptor does not.
        moved = reading.root.with_name("moved")
        reading.root.rename(moved)
        reading.root.mkdir()
        (reading.root / NAME).mkdir()
        reading.verify(dir_fd=descriptor)
        with pytest.raises(ScopeReadingChanged):
            reading.verify()
        (moved / NAME).mkdir()
        with pytest.raises(ScopeReadingChanged):
            reading.verify(dir_fd=descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    os.name == "nt" or os.stat not in os.supports_dir_fd,
    reason="needs stat below a directory descriptor",
)
def test_the_look_below_a_held_root_descriptor_follows_no_link(tmp_path: Path) -> None:
    reading = decide_scope_reading(_root(tmp_path, "absent"))
    descriptor = os.open(reading.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Followed, this link would be the regular file it leads to, and a
        # regular file is the reading the run decided on.
        (reading.root / NAME).symlink_to("x.py")
        with pytest.raises(ScopeReadingChanged, match="is now blocked"):
            reading.verify(dir_fd=descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(("path", "redundant", "literal", "undecided"), _RULE_ROWS)
def test_the_rule_for_a_redundant_a_literal_and_an_undecided_reading(
    path: str, redundant: str | None, literal: str | None, undecided: str | None,
) -> None:
    parts = _parts(path)

    assert _joined(leading_name_rule(parts, NAME, True)) == redundant
    assert _joined(leading_name_rule(parts, NAME, False)) == literal
    for flag in _UNDECIDED_FLAGS:
        assert _joined(leading_name_rule(parts, NAME, flag)) == undecided, flag


def test_the_rule_reads_the_name_it_is_given_and_no_other() -> None:
    assert leading_name_rule(("proj", "x.py"), "other", True) == ("proj", "x.py")
    assert leading_name_rule(("other", "x.py"), "other", True) == ("x.py",)
    assert leading_name_rule((), NAME, True) == ()


def test_a_reading_resolves_as_the_rule_does_with_its_flag(tmp_path: Path) -> None:
    for kind, authored, expected in _RESOLVE_ROWS:
        if not authored or "\0" in authored:
            continue
        reading = _reading(tmp_path, kind)
        kept = leading_name_rule(
            _parts(authored), NAME, reading.workspace_relative_paths,
        )
        if expected in _REFUSALS:
            # The rule refuses a doubled name; a path through a link or
            # another node is refused by the reading, which knows the entry.
            assert (kept is None) is (expected == DOUBLED), (kind, authored)
        else:
            assert _joined(kept) == expected.rstrip("/"), (kind, authored)
