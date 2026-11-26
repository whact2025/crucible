# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The process the toy adapter runs its writes in.

The toy's write host starts this module with the workspace of the cell. The
first thing the process does is to install the network guard of the matrix,
as every process the matrix starts does, before a line of the product is
imported: a reach beyond loopback is written to standard error and ends the
process, whoever swallows the exception. It then builds the writer profile of
that workspace anew, as the fixture built it, says in one line that it is
ready and under the guard, and dispatches every call it reads from standard
input: one JSON document per line, each answered with one line that holds
the result. It ends when its input is closed, and its parent may kill it at
any time.

So a write of the toy runs through the real tool and in no process of the
test: the fixture's recorder, which lives in the test process, never sees it.

A liar of the toy asks for one fault here: the process starts another, in a
session of its own, which is still there when its parent has been killed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import netguard

SCOPED = "scoped"
UNSCOPED = "unscoped"
LEAVES_A_PROCESS = "leaves_a_process"
# The first line of the process: ``{READY: {GUARDED: ..., LEFT: ...}}``.
READY = "ready"
GUARDED = "guarded"
LEFT = "left"
# The exit status of a host that reached beyond loopback.
GUARD_EXIT = 3
# How long the process a lying host leaves behind lives when nobody ends it.
_LEFT_SECONDS = 30


def guard() -> None:
    """Install the network guard; a reach beyond loopback ends this process."""

    def ended(entry: str) -> None:
        sys.stderr.write(f"the write host of the toy reached for the network: {entry}\n")
        sys.stderr.flush()
        os._exit(GUARD_EXIT)

    netguard.install(ended)


def _leave_a_process() -> int:
    """Start a process that the kill of this one does not reach; its pid."""

    left = subprocess.Popen(
        [sys.executable, "-I", "-S", "-c", f"import time; time.sleep({_LEFT_SECONDS})"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return left.pid


def _say(document: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(document) + "\n")
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    guard()
    # Nothing of the product is imported before the guard is installed.
    from . import toolsets  # noqa: PLC0415

    workspace, scope, *faults = argv
    left = _leave_a_process() if LEAVES_A_PROCESS in faults else None
    profile = toolsets.writer_profile(Path(workspace), scoped=scope == SCOPED)
    _say({READY: {GUARDED: netguard.installed(), LEFT: left}})
    for line in sys.stdin:
        if not line.strip():
            continue
        call = json.loads(line)
        result = profile.dispatch(str(call["name"]), dict(call["arguments"]))
        _say({"result": str(result)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
