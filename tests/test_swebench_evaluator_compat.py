# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Version-gated defenses around upstream SWE-bench eval cleanup."""

from __future__ import annotations

import shlex
from types import SimpleNamespace

import pytest

from crucible.products.swebench.evaluator_compat import (
    UnsupportedSwebenchHarness,
    _cleanup_commands,
    _guard_test_spec,
    _require_supported_harness,
)


BASE = "a" * 40
MODIFIED_PATCH = """\
diff --git a/tests/test_existing.py b/tests/test_existing.py
--- a/tests/test_existing.py
+++ b/tests/test_existing.py
@@ -1 +1 @@
-assert False
+assert True
"""
NEW_PATCH = """\
diff --git a/tests/test_new.py b/tests/test_new.py
new file mode 100644
--- /dev/null
+++ b/tests/test_new.py
@@ -0,0 +1 @@
+assert True
"""


def _legacy_spec(test_patch: str):
    from swebench.harness.utils import get_modified_files

    paths = get_modified_files(test_patch)
    reset = f"git checkout {BASE} {' '.join(paths)}"
    return SimpleNamespace(eval_script_list=[
        "cd /testbed",
        reset,
        "git apply -v hidden.patch",
        "pytest -q",
        reset,
    ])


def _assert_no_bare_checkout(commands: list[str]) -> None:
    for command in commands:
        tokens = shlex.split(command)
        if tokens[:2] == ["git", "checkout"]:
            assert "--" in tokens[2:]
            assert tokens.index("--") < len(tokens) - 1


def test_modified_hidden_test_uses_path_scoped_checkout():
    spec = _guard_test_spec(
        _legacy_spec(MODIFIED_PATCH),
        {"base_commit": BASE, "test_patch": MODIFIED_PATCH},
    )

    resets = [c for c in spec.eval_script_list if c.startswith("git checkout")]
    assert resets == [
        f"git checkout {BASE} -- tests/test_existing.py",
        f"git checkout {BASE} -- tests/test_existing.py",
    ]
    assert not any(c.startswith("rm -f") for c in spec.eval_script_list)
    _assert_no_bare_checkout(spec.eval_script_list)


def test_new_hidden_test_uses_rm_and_never_checkout_commit():
    spec = _guard_test_spec(
        _legacy_spec(NEW_PATCH),
        {"base_commit": BASE, "test_patch": NEW_PATCH},
    )

    assert spec.eval_script_list.count("rm -f -- tests/test_new.py") == 2
    assert not any(c.startswith("git checkout") for c in spec.eval_script_list)
    _assert_no_bare_checkout(spec.eval_script_list)


def test_mixed_hidden_patch_restores_existing_and_removes_new():
    mixed = MODIFIED_PATCH + NEW_PATCH
    spec = _guard_test_spec(
        _legacy_spec(mixed),
        {"base_commit": BASE, "test_patch": mixed},
    )

    checkout = f"git checkout {BASE} -- tests/test_existing.py"
    assert spec.eval_script_list.count(checkout) == 2
    assert spec.eval_script_list.count("rm -f -- tests/test_new.py") == 2
    _assert_no_bare_checkout(spec.eval_script_list)


def test_renamed_hidden_test_restores_source_and_removes_destination():
    renamed = """\
diff --git a/tests/test_old.py b/tests/test_new_name.py
similarity index 100%
rename from tests/test_old.py
rename to tests/test_new_name.py
"""
    commands = _cleanup_commands(BASE, renamed)

    assert commands == [
        f"git checkout {BASE} -- tests/test_old.py",
        "rm -f -- tests/test_new_name.py",
    ]
    _assert_no_bare_checkout(commands)


def test_guard_fails_loudly_when_upstream_script_shape_drifts():
    spec = _legacy_spec(MODIFIED_PATCH)
    spec.eval_script_list.pop()

    with pytest.raises(UnsupportedSwebenchHarness, match="exactly two"):
        _guard_test_spec(
            spec,
            {"base_commit": BASE, "test_patch": MODIFIED_PATCH},
        )


def test_exact_version_gate_rejects_unreviewed_release(monkeypatch):
    monkeypatch.setattr(
        "crucible.products.swebench.evaluator_compat.importlib.metadata.version",
        lambda package: "4.2.0",
    )

    with pytest.raises(UnsupportedSwebenchHarness, match="exactly"):
        _require_supported_harness()
