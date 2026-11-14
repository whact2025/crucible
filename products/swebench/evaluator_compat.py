# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Guard SWE-bench 4.1.0 test-patch cleanup without forking its harness.

Upstream's Python eval-script builder derives reset paths only from a patch's
``a/`` side.  An all-new test patch therefore emits ``git checkout <commit>``
with no path arguments, which checks out the commit and erases the already
applied model patch.  This module is the subprocess entry point used by
Crucible.  It wraps ``make_test_spec`` and replaces only the two generated
test cleanup commands:

* files present at the base commit use ``git checkout <base> -- <paths>``;
* files newly introduced by the hidden test patch use ``rm -f -- <paths>``.

The wrapper is deliberately exact-version and generated-shape gated.  A
future harness release must be reviewed instead of silently receiving a
source rewrite intended for a different script layout.
"""

from __future__ import annotations

import importlib.metadata
import runpy
import shlex
from collections.abc import Callable
from typing import Any

from unidiff import PatchSet


_SUPPORTED_SWEBENCH_VERSION = "4.1.0"
_NO_TEST_FILES_RESET = 'echo "No test files to reset"'


class UnsupportedSwebenchHarness(RuntimeError):
    """Installed harness version or generated eval script is unsupported."""


def _require_supported_harness() -> None:
    installed = importlib.metadata.version("swebench")
    if installed != _SUPPORTED_SWEBENCH_VERSION:
        raise UnsupportedSwebenchHarness(
            "Crucible's SWE-bench cleanup guard supports exactly "
            f"swebench=={_SUPPORTED_SWEBENCH_VERSION}; found {installed!r}. "
            "Review the upstream eval-script shape before updating the pin."
        )


def _repo_path(raw_path: str, expected_prefix: str) -> str:
    if raw_path == "/dev/null":
        raise ValueError("/dev/null is not a repository path")
    prefix = f"{expected_prefix}/"
    if not raw_path.startswith(prefix):
        raise UnsupportedSwebenchHarness(
            f"hidden test patch path lacks expected {prefix!r} prefix"
        )
    path = raw_path[len(prefix):]
    parts = path.replace("\\", "/").split("/")
    if not path or path.startswith("/") or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise UnsupportedSwebenchHarness(
            "hidden test patch contains a non-repository-relative path"
        )
    if "\0" in path or "\n" in path or "\r" in path:
        raise UnsupportedSwebenchHarness(
            "hidden test patch contains an unsafe control character in a path"
        )
    return path


def _cleanup_path_sets(test_patch: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(base_existing, newly_added)`` paths without patch contents."""

    existing: set[str] = set()
    added: set[str] = set()
    try:
        patch_set = PatchSet(test_patch)
    except Exception as exc:  # unidiff exposes several parser exceptions
        raise UnsupportedSwebenchHarness(
            "could not parse the hidden test patch for safe cleanup"
        ) from exc

    for patched_file in patch_set:
        source = patched_file.source_file
        target = patched_file.target_file
        source_path: str | None = None
        target_path: str | None = None
        if source != "/dev/null":
            source_path = _repo_path(source, "a")
            existing.add(source_path)
        if target != "/dev/null":
            target_path = _repo_path(target, "b")
        # New files and rename destinations do not exist at base_commit and
        # therefore must be removed, never passed to checkout.
        if target_path is not None and (
            source_path is None or target_path != source_path
        ):
            added.add(target_path)
    return tuple(sorted(existing)), tuple(sorted(added))


def _cleanup_commands(base_commit: str, test_patch: str) -> list[str]:
    existing, added = _cleanup_path_sets(test_patch)
    commands: list[str] = []
    if existing:
        commands.append(
            "git checkout "
            f"{shlex.quote(base_commit)} -- "
            + " ".join(shlex.quote(path) for path in existing)
        )
    if added:
        commands.append(
            "rm -f -- " + " ".join(shlex.quote(path) for path in added)
        )
    return commands or [_NO_TEST_FILES_RESET]


def _guard_test_spec(
    test_spec: Any,
    instance: dict[str, Any],
) -> Any:
    """Replace the two upstream reset commands or reject shape drift."""

    commands = getattr(test_spec, "eval_script_list", None)
    if not isinstance(commands, list) or not all(
        isinstance(command, str) for command in commands
    ):
        raise UnsupportedSwebenchHarness(
            "SWE-bench TestSpec.eval_script_list is no longer a list[str]"
        )
    base_commit = instance.get("base_commit")
    test_patch = instance.get("test_patch")
    if not isinstance(base_commit, str) or not base_commit:
        raise UnsupportedSwebenchHarness(
            "SWE-bench instance has no string base_commit"
        )
    if not isinstance(test_patch, str):
        raise UnsupportedSwebenchHarness(
            "SWE-bench instance has no string test_patch"
        )

    # This reproduces only the legacy command's path selection so we can
    # identify its exact generated slots.  Cleanup itself uses both patch
    # endpoints through _cleanup_path_sets above.
    from swebench.harness.utils import get_modified_files

    legacy_paths = get_modified_files(test_patch)
    legacy_checkout = f"git checkout {base_commit} {' '.join(legacy_paths)}"
    legacy_candidates = {legacy_checkout, _NO_TEST_FILES_RESET}
    reset_indexes = [
        index for index, command in enumerate(commands)
        if command in legacy_candidates
    ]
    if len(reset_indexes) != 2:
        raise UnsupportedSwebenchHarness(
            "SWE-bench eval script no longer contains exactly two recognized "
            "test reset commands; refusing an unreviewed compatibility edit"
        )

    replacements = _cleanup_commands(base_commit, test_patch)
    guarded = list(commands)
    for index in reversed(reset_indexes):
        guarded[index:index + 1] = replacements

    # A checkout of a commit without an explicit ``-- <path>`` must never
    # survive the transform.  Check tokenized command lines, not substrings.
    for command in guarded:
        first_line = command.splitlines()[0] if command else ""
        try:
            tokens = shlex.split(first_line)
        except ValueError as exc:
            raise UnsupportedSwebenchHarness(
                "SWE-bench emitted an unparsable eval command"
            ) from exc
        if tokens[:2] == ["git", "checkout"] and "--" not in tokens[2:]:
            raise UnsupportedSwebenchHarness(
                "SWE-bench eval script contains a pathless or ambiguous "
                "git checkout after cleanup guarding"
            )

    test_spec.eval_script_list = guarded
    return test_spec


def _install_make_test_spec_guard() -> Callable[..., Any]:
    from swebench.harness.test_spec import test_spec as test_spec_module

    original = test_spec_module.make_test_spec

    def guarded_make_test_spec(instance: dict[str, Any], *args, **kwargs):
        return _guard_test_spec(original(instance, *args, **kwargs), instance)

    test_spec_module.make_test_spec = guarded_make_test_spec
    return original


def main() -> None:
    """Run upstream's CLI with its ``make_test_spec`` import guarded."""

    _require_supported_harness()
    from swebench.harness.test_spec import test_spec as test_spec_module

    original = _install_make_test_spec_guard()
    try:
        # Importing swebench.harness eagerly caches run_evaluation, so
        # run_module would warn and risk reusing stale globals. Execute the
        # reviewed upstream entry-point file as __main__ instead; its import
        # of make_test_spec observes the patched function above while its CLI
        # remains byte-for-byte upstream.
        from swebench.harness import run_evaluation as upstream_runner

        runner_path = getattr(upstream_runner, "__file__", None)
        if not isinstance(runner_path, str) or not runner_path:
            raise UnsupportedSwebenchHarness(
                "could not locate the SWE-bench run_evaluation entry point"
            )
        runpy.run_path(runner_path, run_name="__main__")
    finally:
        test_spec_module.make_test_spec = original


if __name__ == "__main__":
    main()


__all__ = ["UnsupportedSwebenchHarness", "main"]
