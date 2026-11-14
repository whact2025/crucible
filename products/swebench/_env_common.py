# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Workspace reset helper for the mandatory SWE-bench container path.

Container setup may seed baked artifacts or briefly modify tracked setup
files. ``_revert_workspace`` restores tracked files before authoring so the
eventual prediction contains only model-produced edits.

Tests patch the consumer binding in ``instance_container`` because it imports
this helper directly.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _revert_workspace(workspace: Path) -> None:
    """Revert tracked-file changes in the host workspace.

    Runs ``git checkout -- .`` so the implementer's later diff capture
    via ``git diff HEAD`` excludes pre_install / install-time
    mutations to tracked files (e.g. pyproject.toml). Untracked
    artifacts from the editable install (egg-info, build/) are kept
    — they're excluded from ``git diff HEAD`` by being untracked and,
    on the container path, sit inside the bind-mount so the container
    sees them.

    Race note (container path): when called on a setup failure path,
    the in-container install subprocess may still be running on the
    docker daemon side (our host-side ``docker exec`` was killed by
    ``_STEP_TIMEOUT`` but the daemon child runs to completion). Pip
    writes to the conda env's site-packages inside the image overlay,
    NOT to the bind-mounted workspace, so the host ``git checkout``
    here is safe — the only host-visible writes are pre_install's
    sed patches which complete synchronously.

    Setup failure paths call the same helper before propagating, so a partial
    seed cannot be attributed to the model.
    """
    revert = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "checkout", "--", "."],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    if revert.returncode != 0:
        logger.warning(
            "swebench env_common: post-setup ``git checkout -- .`` "
            "failed (rc=%d): %s — implementer's diff capture may "
            "include stray pre_install modifications",
            revert.returncode, (revert.stderr or "")[:300],
        )
