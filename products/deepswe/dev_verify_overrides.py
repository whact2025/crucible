# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Crucible-owned per-project ``dev_verify`` command overrides.

The deep-swe corpus is cloned read-only (``datacurve-ai/deep-swe``), so
per-project test-command overrides live HERE, under crucible's control,
rather than as a dataset fork. The per-language defaults in
``deepswe_dev_verify._LANG_TO_COMMAND`` (bounded ``pytest`` / ``cargo test``
/ …) mis-fire on the ~10-20% of projects whose native test orchestrator
they bypass — e.g. adaptix / mashumaro drive their matrices through
``nox`` / ``tox``, which a plain ``pytest`` invocation skips.

Entries are added only when a project requires a narrower or differently
orchestrated command than the language default.

Resolution precedence: an exact ``task_id`` entry wins over a
``repo``-level entry (some overrides apply to a whole repo, some to a
single task). A resolved command is threaded into
``deepswe_metadata["dev_verify_command"]`` by the deepswe worker
(``workflows/deepswe.py::_build_run_architect_kwargs``) and honored by
``deepswe_dev_verify`` over the per-language default.

Command contract: identical to ``_LANG_TO_COMMAND`` entries — the string
runs as ``bash -c <command>`` inside the AUTHORING container, so it must
``cd /app`` itself (the wrapper prepends nothing).
"""

from __future__ import annotations

import dataclasses

from crucible.common.runners.pytest_policy import PYTEST_DEV_VERIFY_MAXFAIL_ARG


@dataclasses.dataclass(frozen=True)
class JavaScriptProbeProfile:
    """Sealed exact-test invocation for a project with non-root tooling."""

    parser_kind: str
    workdir: str
    executable: tuple[str, ...]
    setup_command: str = ""

# task_id → command. Most specific; wins over a repo-level entry.
_TASK_OVERRIDES: dict[str, str] = {
    # testem/testem (js): `npm test` runs the full mocha suite, but the
    # browser-launcher / parallel-instance / executable-hook integration
    # tests can't pass headless under --network=none. Scope to the bail
    # task's domain + deterministic core units (incl. the hidden
    # adapter_abort_tests.js) → green at baseline (114 passing). Task-level
    # because the file set is bail-specific; a launcher-domain testem task
    # would need its own subset. (calibrated 2026-06-16)
    # NOTE: this is an EXPLICIT-file list (not a glob), so it must include
    # the breach Defender's test file. breach.py mirrors the project's
    # *_tests.js convention → tests/breach_tests.js; list it here so the
    # per-round breach gate actually runs the adversarial tests. The
    # trailing file is harmless when absent (mocha skips a missing path
    # arg via the shell glob? no — list it only if present): we pass it
    # unconditionally; mocha errors on a truly-missing file, but the
    # Defender writes it before the gate runs, and a no-attack round
    # writes nothing — so guard with a shell test.
    "testem-bail-on-test-failure": (
        "cd /app && npx mocha tests/bail_on_test_failure_tests.js "
        "tests/reporter_bail_output_tests.js tests/adapter_abort_tests.js "
        "tests/client_abort_tests.js tests/server_abort_tests.js "
        "tests/config_tests.js tests/fileutils_tests.js "
        "tests/strutils_tests.js tests/process-ctl_tests.js "
        "tests/mocha_adapter_tests.js "
        "$(test -f tests/breach_tests.js && echo tests/breach_tests.js)"
    ),
}


# Exact probes cannot safely reverse-engineer arbitrary shell scripts. Profiles
# pin only the projects whose package/workspace context or runtime flags differ
# from a normal nearest-package invocation. Dynamic test file/title arguments
# are appended and shell-quoted by the runner.
_JAVASCRIPT_PROBE_PROFILES: dict[str, JavaScriptProbeProfile] = {
    "clack-async-autocomplete-options": JavaScriptProbeProfile(
        parser_kind="vitest",
        workdir="/app/packages/prompts",
        executable=("pnpm", "exec", "vitest"),
    ),
    "optique-conditional-option-dependencies": JavaScriptProbeProfile(
        parser_kind="node-tap",
        workdir="/app/packages/core",
        executable=("node", "--experimental-transform-types"),
        setup_command="pnpm exec tsdown",
    ),
    "ink-grid-box-layout": JavaScriptProbeProfile(
        parser_kind="ava-tap",
        workdir="/app",
        executable=("npx", "--no-install", "ava"),
    ),
}

# repo (``owner/name`` short form, matching ``DeepSWEInstance.repo``) →
# command. Applies to every task in the repo unless a ``task_id`` entry
# overrides it.
_REPO_OVERRIDES: dict[str, str] = {
    # --- Go (calibrated 2026-06-16; the per-language default `go test
    #     ./...` fails at BASELINE on these repos, making the gate a
    #     false-negative — see benchmark_logs/dev_verify_calib). ---
    #
    # abs-lang/abs: TestEnv + TestCommand expect CONTEXT=abs in the env
    # (the dataset's test.sh sets it; the dep-baked container does not);
    # util/TestUpdateAvailable is a version/update check; the `js` package
    # is a gopherjs build the container's build constraints exclude. Set
    # the env var, drop js, skip the update check → green at base_commit
    # (evaluator/parser/ast/etc. — the packages both abs tasks touch — run).
    "abs-lang/abs": (
        'cd /app && CONTEXT=abs go test -json '
        '$(go list ./... | grep -v "/js$") '
        "-skip TestUpdateAvailable"
    ),
    # rhysd/actionlint: the network-dependent scripts/generate-* codegen
    # tests (fetch remote YAML / detect releases) fail under
    # --network=none. Scope past scripts/ — the main package (where
    # rule_action_pinning + its hidden test live) is green at baseline.
    "rhysd/actionlint": (
        'cd /app && go test -json '
        '$(go list ./... | grep -v "/scripts/")'
    ),
    # --- TypeScript (calibrated 2026-06-16; the `npm test` default
    #     mis-fires per-project — it runs typechecks/bundle tests that
    #     fail at baseline. Each project's real unit-test runner is green). ---
    #
    # jeffijoe/awilix: `npm test` = `npm run check && jest` — the `check`
    # typecheck fails on a deliberately-broken examples/ file. Run jest
    # directly, ignoring rollup.test.ts (it imports the built lib/ bundle,
    # which needs a build step the dev_verify gate shouldn't require).
    "jeffijoe/awilix": "cd /app && npx jest --testPathIgnorePatterns rollup",
    # arktypeio/arktype: mocha via `pnpm test` (= testTyped --skipTypes).
    # Green at baseline (1693 passing); pin pnpm to avoid npm→pnpm drift.
    "arktypeio/arktype": "cd /app && pnpm test",
    # bombshell-dev/clack: pnpm workspace monorepo; `pnpm test` builds then
    # runs every package's vitest suite (packages/core + packages/prompts,
    # where the task's hidden tests live). Green at baseline.
    "bombshell-dev/clack": "cd /app && pnpm test",
    # dahlia/optique: no root package.json. The affected @optique/core
    # workspace package builds then runs Node's TAP-emitting test runner.
    "dahlia/optique": "cd /app/packages/core && pnpm test",
    # vadimdemedes/ink: npm test includes a base-red typecheck/lint phase.
    # AVA's TAP reporter is deterministic and covers the project's real tests.
    "vadimdemedes/ink": "cd /app && npx ava --tap",
    # --- Per-project default mis-fires found in the 6-default
    #     verification (calibrated 2026-06-16; defaults fail/no-op at
    #     baseline even in "validated" languages — dev_verify is
    #     per-project, not per-language). ---
    #
    # PyCQA/bandit (py): root `pytest` collects bandit's lint-SAMPLE data
    # files (bandit/core/test_properties.py etc. — code meant to be
    # analyzed, not run) → fixture-not-found errors. The real suite is
    # tests/ → 274 passed.
    "PyCQA/bandit": (
        f"cd /app && pytest tests/ {PYTEST_DEV_VERIFY_MAXFAIL_ARG} "
        "--tb=short -q"
    ),
    # noahbald/oxvg (rust): a cargo WORKSPACE — root `cargo test` collects
    # 0 tests (no-op gate). --workspace runs the member crates' tests.
    "noahbald/oxvg": "cd /app && cargo test --workspace",
    # KaTeX/KaTeX (js): `npm test` = `yarn test:lint && test:ts &&
    # test:jest`; lint/ts + a yarn-version pin (yarn@4 vs global yarn 1)
    # break it. jest alone is the unit suite → 1254 passed.
    "KaTeX/KaTeX": "cd /app && npx jest",
}


def resolve_dev_verify_command(*, task_id: str, repo: str = "") -> str | None:
    """Return the crucible-owned ``dev_verify`` command override for a
    task, or ``None`` when none is registered (the caller falls back to
    the per-language default in ``deepswe_dev_verify._LANG_TO_COMMAND``).

    An exact ``task_id`` match takes precedence over a ``repo`` match.
    """
    if task_id in _TASK_OVERRIDES:
        return _TASK_OVERRIDES[task_id]
    if repo and repo in _REPO_OVERRIDES:
        return _REPO_OVERRIDES[repo]
    return None


def resolve_javascript_probe_profile(
    *, task_id: str,
) -> JavaScriptProbeProfile | None:
    """Return a runner-owned exact-probe profile for one calibrated task."""

    return _JAVASCRIPT_PROBE_PROFILES.get(task_id)


__all__ = [
    "JavaScriptProbeProfile",
    "resolve_dev_verify_command",
    "resolve_javascript_probe_profile",
]
