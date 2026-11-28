# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Polyglot dispatch runner — invoke per-language runners in order.

An ordered list of languages is dispatched to its canonical runner. Results
are aggregated and execution short-circuits on the first failure.

Subprocess contract (per ``assay.bench``):

- Reads a JSON BenchConfig from stdin.
- Reads the workflow-injected ``parameters.languages`` (``list[str]``),
  resolved from the canonical ``constraints[kind=languages]`` value; for each
  language in order, subprocess-invokes
  ``python -m crucible.common.runners.lang.<lang>`` with a per-language
  BenchConfig derived from the input. Spec authors must not set the top-level
  runner parameter directly; only low-level runner callers construct it.
- Emits one combined JSON KPI payload on stdout.
- Exits 0 only if every language runner exited 0; exits the FIRST
  non-zero returncode encountered (subsequent languages are skipped).
- A requested ``parameters.runner_mode == "bench"`` succeeds only when every
  selected adapter echoes that same top-level mode. This preserves the outer
  workflow's fail-closed benchmark acknowledgement across the nested
  polyglot subprocess boundary.

Per-language parameter routing:

The polyglot runner accepts a ``parameters.per_language`` map where
keys are language names and values are sub-parameter dicts. Per-
language parameters merge over the base ``parameters`` with the
``languages`` / ``per_language`` keys filtered out:

.. code-block:: json

    {
      "parameters": {
        "languages": ["python", "rust"],
        "working_dir": "/repo",
        "per_language": {
          "python": {"test_args": ["-k", "fast"]},
          "rust": {
            "crate": "core",
            "cargo_features": ["vulkan-rt"],
            "cargo_test_args": ["--", "--test-threads=1"]
          }
        }
      }
    }

The aggregated ``values`` keep each language's namespaced metrics
(``python_*``, ``rust_*``, ``cpp_*``, ``dotnet_*``) plus a top-level
``tests_passed`` / ``tests_failed`` that SUM across languages. This
mirrors the behavior ``_build_verify``'s classification logic expects
(read ``values["tests_failed"]`` to detect partial test failures).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

from crucible.common.runners.breach_observation import (
    BREACH_OBSERVATION_FAILURE_KEY,
    breach_observation_failure,
    parse_breach_observation_failure,
)
from crucible.common.runners.parameters import for_language

logger = logging.getLogger(__name__)

_TRUSTED_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_TRUSTED_RUNNER_BOOTSTRAP = (
    "import runpy,sys;"
    "source_root=sys.argv.pop(1);module=sys.argv.pop(1);"
    "sys.path.insert(0,source_root);"
    "runpy.run_module(module,run_name='__main__',alter_sys=True)"
)

# Languages we know how to dispatch. Add a row when a new
# ``crucible/common/runners/lang/<lang>.py`` module lands.
_SUPPORTED_LANGUAGES: dict[str, str] = {
    "python": "crucible.common.runners.lang.python",
    "rust":   "crucible.common.runners.lang.rust",
    "cpp":    "crucible.common.runners.lang.cpp",
    "dotnet": "crucible.common.runners.lang.dotnet",
}


def _per_language_params(
    base_parameters: dict, lang: str
) -> dict:
    """Merge base parameters with the per-language override for ``lang``."""
    return for_language(base_parameters, lang)


def _run_language(
    lang: str, sub_cfg: dict
) -> tuple[
    int,
    dict,
    str,
    list[dict[str, str]],
    str | None,
    dict[str, str] | None,
]:
    """Subprocess-invoke ``python -m <lang module>`` with sub_cfg on stdin.

    Returns ``(returncode, values, raw_output_tail)``.
    """
    module = _SUPPORTED_LANGUAGES[lang]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _TRUSTED_RUNNER_BOOTSTRAP,
            str(_TRUSTED_SOURCE_ROOT),
            module,
        ],
        input=json.dumps(sub_cfg).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    raw_tail = (completed.stdout + completed.stderr).decode(
        "utf-8", errors="replace"
    )[-4_000:]
    values: dict = {}
    observations: list[dict[str, str]] = []
    reported_mode: str | None = None
    observation_failure: dict[str, str] | None = None
    breach_requested = bool(
        isinstance(sub_cfg.get("parameters"), dict)
        and sub_cfg["parameters"].get("breach_observations")
    )
    if completed.stdout:
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("values"), dict):
                values = payload["values"]
            if isinstance(payload, dict) and isinstance(
                payload.get("runner_mode"), str
            ):
                reported_mode = payload["runner_mode"]
            raw_observations = (
                payload.get("test_observations")
                if isinstance(payload, dict) else None
            )
            if isinstance(raw_observations, list):
                observations = [
                    observation
                    for observation in raw_observations
                    if isinstance(observation, dict)
                    and all(
                        isinstance(observation.get(key), str)
                        for key in ("language", "id", "status", "line")
                    )
                ]
            if (
                isinstance(payload, dict)
                and BREACH_OBSERVATION_FAILURE_KEY in payload
            ):
                try:
                    observation_failure = parse_breach_observation_failure(
                        payload
                    )
                except ValueError as exc:
                    observation_failure = breach_observation_failure(
                        "channel",
                        exc,
                    )
            if breach_requested and not isinstance(payload, dict):
                observation_failure = breach_observation_failure(
                    "channel",
                    f"{lang} breach runner payload was not a JSON object",
                )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "polyglot: %s runner stdout was not parseable JSON", lang,
            )
            if breach_requested:
                observation_failure = breach_observation_failure(
                    "channel",
                    f"{lang} breach runner payload was malformed: {exc}",
                )
    elif breach_requested:
        observation_failure = breach_observation_failure(
            "infrastructure",
            f"{lang} breach runner emitted no protocol payload",
        )
    return (
        completed.returncode,
        values,
        raw_tail,
        observations,
        reported_mode,
        observation_failure,
    )


def _aggregate(values_by_lang: dict[str, dict]) -> dict[str, float]:
    """Merge per-language ``values`` maps into a combined dict.

    - Language-namespaced keys (``python_tests_passed``, ...) are
      preserved as-is.
    - Generic aliases (``tests_passed`` / ``tests_failed`` /
      ``tests_skipped`` / ``tests_ignored``) are SUMMED across
      languages so the polyglot caller sees workspace-wide totals.
    - ``returncode`` is the worst (max) across languages.
    """
    combined: dict[str, float] = {}
    sum_keys = {"tests_passed", "tests_failed", "tests_skipped", "tests_ignored"}
    max_keys = {"returncode"}
    for _lang, values in values_by_lang.items():
        if not isinstance(values, dict):
            continue
        for k, v in values.items():
            if not isinstance(v, (int, float)):
                continue
            if k in sum_keys:
                combined[k] = combined.get(k, 0.0) + float(v)
            elif k in max_keys:
                combined[k] = max(combined.get(k, 0.0), float(v))
            else:
                # Per-language namespaced keys — pass through. If two
                # languages emit the same key (unlikely with the
                # namespacing convention), the later one wins.
                combined[k] = float(v)
    return combined


def _main(verb: str) -> int:
    cfg = json.loads(sys.stdin.read())
    parameters = (cfg.get("parameters") or {}) if isinstance(cfg, dict) else {}
    requested_mode = parameters.get("runner_mode")
    languages = parameters.get("languages") or []
    if not isinstance(languages, list) or not languages:
        sys.stdout.write(json.dumps({
            "values": {},
            "raw_output": (
                f"PolyglotRunner ({verb}): no languages declared. "
                "Declare constraints[kind=languages] in the Spec so the "
                "workflow can inject parameters.languages; do not author "
                "the top-level runner parameter directly."
            ),
        }))
        return 2  # config error — distinct from runner failure (1)

    unknown = [lang for lang in languages if lang not in _SUPPORTED_LANGUAGES]
    if unknown:
        sys.stdout.write(json.dumps({
            "values": {},
            "raw_output": (
                f"PolyglotRunner ({verb}): unknown languages {unknown!r}. "
                f"Supported: {sorted(_SUPPORTED_LANGUAGES)}"
            ),
        }))
        return 2

    if requested_mode == "bench" and "dotnet" in languages:
        sys.stdout.write(json.dumps({
            "values": {},
            "raw_output": "The dotnet adapter does not support benchmark verification.",
        }))
        return 2

    values_by_lang: dict[str, dict] = {}
    raw_chunks: list[str] = []
    test_observations: list[dict[str, str]] = []
    observation_failure: dict[str, str] | None = None
    start = time.monotonic()
    final_rc = 0

    for lang in languages:
        sub_cfg = {
            "bench_id": f"{cfg.get('bench_id', 'polyglot')}__{lang}",
            "parameters": _per_language_params(parameters, lang),
            "env": cfg.get("env") or {},
        }
        (
            rc,
            vals,
            raw,
            observations,
            reported_mode,
            language_observation_failure,
        ) = _run_language(
            lang, sub_cfg
        )
        if rc == 0 and requested_mode == "bench" and reported_mode != "bench":
            rc = 2
            raw += (
                "\nPolyglotRunner: benchmark adapter did not acknowledge "
                f"runner_mode='bench' for language {lang!r}"
            )
        values_by_lang[lang] = vals
        test_observations.extend(observations)
        if language_observation_failure is not None:
            observation_failure = language_observation_failure
        raw_chunks.append(f"=== {lang} (rc={rc}) ===\n{raw}")
        if rc != 0:
            final_rc = rc
            # Short-circuit: subsequent languages are skipped to keep
            # the aggregate signal honest. Failed-first builds a
            # consistent "did this combined verification pass" answer.
            break

    duration = time.monotonic() - start
    values = _aggregate(values_by_lang)
    values["duration_s"] = float(duration)
    values["returncode"] = float(final_rc)

    payload = {
        "values": values,
        "raw_output": "\n\n".join(raw_chunks)[-8_000:],
        "test_observations": test_observations,
    }
    if observation_failure is not None:
        payload[BREACH_OBSERVATION_FAILURE_KEY] = observation_failure
    if final_rc == 0 and isinstance(requested_mode, str):
        payload["runner_mode"] = requested_mode
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return final_rc


def verify_main() -> int:
    """Entry point for ``python -m crucible.common.runners.polyglot``."""
    return _main("verify")


# When used as a build runner: same dispatch, just labeled differently
# for logs. The per-language runners can read parameters.role to
# distinguish build vs verify if they ever need to.
def build_main() -> int:
    return _main("build")


if __name__ == "__main__":
    sys.exit(verify_main())
