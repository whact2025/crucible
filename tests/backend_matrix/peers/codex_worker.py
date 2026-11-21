# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The stage a locked Codex worker runs in front of its module, in a matrix cell.

The official-SDK route of the Codex adapter starts the locked interpreter of
``backends/codex_worker`` on the worker module. The worker hands the request
to the official SDK, and the SDK starts the vendor's command-line tool: the
runtime that its own dependency bundles, found through
``codex_cli_bin.bundled_codex_path``. A cell may not start that tool.

So the driver of the column starts the request launch of the worker on this
file, under the worker harness (:mod:`.worker_harness`), which has installed
the network guard before anything of the worker environment was imported.
The stage then

1. replaces the one name through which the SDK finds its bundled runtime by a
   function that returns the scripted peer of the cell
   (:mod:`.codex_app_server`);
2. appends one record to the journal of the cell: what it replaced, by what;
3. runs the worker module as ``__main__``, with the argument list of the
   request launch (none).

Everything else in the process is the worker's own: its request validation,
the official SDK with its generated types and its client of the app-server,
and the worker's terminal payload. The tool the SDK starts is the peer.

The stage takes its settings from one environment variable and removes it, so
the worker and the tool it starts see the environment of the original launch.
It writes nothing to standard output and, unless it cannot start, nothing to
standard error.

Standard library only at import and free of package-relative imports: the
locked interpreter runs this file by path, in isolated mode.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

# The one variable the settings travel in: a JSON object with the worker
# ``module``, the ``executable`` to put in the place of the bundled runtime
# and the ``journal`` directory of the cell.
SETTINGS_ENV = "CRUCIBLE_MATRIX_CODEX_STAGE"
# Where the official SDK looks for the runtime its dependency bundles.
RUNTIME_MODULE = "codex_cli_bin"
RUNTIME_NAME = "bundled_codex_path"
# The file of the journal directory the stage writes.
STAGED = "staged.jsonl"


class StageError(Exception):
    """The stage was started in a way it cannot run the worker under."""


def _settings() -> dict[str, str]:
    raw = os.environ.pop(SETTINGS_ENV, None)
    if raw is None:
        raise StageError(f"{SETTINGS_ENV} is not set: only the driver of the column starts me")
    try:
        settings = json.loads(raw)
    except ValueError as exc:
        raise StageError(f"{SETTINGS_ENV} is not JSON") from exc
    if (
        not isinstance(settings, dict)
        or set(settings) != {"module", "executable", "journal"}
        or not all(isinstance(value, str) and value for value in settings.values())
    ):
        raise StageError(f"{SETTINGS_ENV} does not hold the settings of a launch")
    if not os.path.isfile(settings["executable"]) or not os.path.isdir(settings["journal"]):
        raise StageError("the peer or the journal directory of the launch does not exist")
    return settings


def main() -> None:
    try:
        settings = _settings()
        import importlib  # noqa: PLC0415 - nothing of the environment before the settings

        runtime = importlib.import_module(RUNTIME_MODULE)
        bundled = str(getattr(runtime, RUNTIME_NAME)())
    except (StageError, ImportError, AttributeError, OSError) as exc:
        # Nothing ran and nothing may run with the vendor's runtime in place.
        raise SystemExit(f"codex worker stage: {type(exc).__name__}: {exc}") from None
    executable = Path(settings["executable"])

    def bundled_codex_path() -> Path:
        return executable

    setattr(runtime, RUNTIME_NAME, bundled_codex_path)
    record = {
        "pid": os.getpid(),
        "module": settings["module"],
        "replaced": f"{RUNTIME_MODULE}.{RUNTIME_NAME}",
        "bundled": bundled,
        "executable": str(executable),
    }
    descriptor = os.open(
        os.path.join(settings["journal"], STAGED), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
    )
    try:
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)
    sys.argv = ["-m"]  # runpy puts the module's file first
    runpy.run_module(settings["module"], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
