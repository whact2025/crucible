# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Path policy shared by SWE-bench planning and prediction capture.

The policy deliberately recognizes *structural* test conventions instead of
looking for the substring ``"test"``.  Substring matching silently drops
ordinary source files such as ``contest.py`` and ``latest.py``.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath


_TEST_DIRECTORY_NAMES = frozenset({
    "__tests__",
    "spec",
    "specs",
    "test",
    "test_data",
    "testdata",
    "testing",
    "tests",
})

_DELIMITED_TEST_SUFFIX = re.compile(
    r"(?:^|[_.-])(?:spec|test|tests)(?:[_.-]|$)",
    re.IGNORECASE,
)
_JAVA_TEST_CLASS = re.compile(r".+Tests?\.java$")


def is_standard_test_path(path: str) -> bool:
    """Return whether a repo-relative POSIX path follows a test convention.

    Covered conventions include ``tests/`` and ``src/test/`` directories,
    Django-style ``*_tests/`` packages, Python ``test_*.py`` / ``*_test.py``
    modules, Go ``*_test.go``, Ruby ``*_spec.rb``, JS ``*.test.ts`` /
    ``*.spec.js``, and Java ``*Test.java`` classes.  A mere ``"test"``
    substring is not sufficient, so names such as ``contest.py`` remain
    source paths.

    The caller must pass a repository-relative path, without Git's ``a/`` or
    ``b/`` diff prefix.  Backslashes are normalized defensively for callers
    that obtained paths from cross-platform metadata.
    """

    normalized = path.replace("\\", "/").strip("/")
    if not normalized:
        return False
    parts = tuple(part for part in PurePosixPath(normalized).parts if part != ".")
    if not parts or any(part == ".." for part in parts):
        return False

    directory_parts = (part.casefold() for part in parts[:-1])
    if any(
        part in _TEST_DIRECTORY_NAMES or part.endswith("_tests")
        for part in directory_parts
    ):
        return True

    filename = parts[-1]
    folded = filename.casefold()
    if folded in {"conftest.py", "test.py", "tests.py"}:
        return True
    if folded.startswith(("test_", "test-", "test.")):
        return True
    if _DELIMITED_TEST_SUFFIX.search(filename):
        return True
    return bool(_JAVA_TEST_CLASS.fullmatch(filename))


__all__ = ["is_standard_test_path"]
