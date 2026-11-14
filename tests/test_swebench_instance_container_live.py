# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""End-to-end live smoke for the SWE-bench Docker container path.

Exercises the full
``docker pull → docker run → pre_install replay → install → exec →
revert → teardown`` chain against the harness's astropy-12907 image
(already cached on perflab1 — see commit f17d83b).

Marked ``live_docker`` so the default ``uv run pytest`` doesn't trigger
docker pulls in CI. Invoke explicitly:

    uv run pytest tests/test_swebench_instance_container_live.py -m live_docker -s

Skips automatically when ``docker_available()`` is False so the file is
safe to leave in the suite on machines without docker (laptops, CI).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.products.swebench.instance_container import (
    CONDA_ACTIVATE_PREFIX,
    docker_available,
    resolve_container_spec,
    setup_instance_container,
    teardown_instance_container,
)


pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="docker daemon not reachable; live smoke needs docker",
)


# astropy-12907 was the first instance the f17d83b → ae1bdd3 smoke
# ran against; the image is cached locally so this test runs in
# seconds (no cold pull). If the image gets evicted, the test will
# do a ~4 GB pull on first run.
_ASTROPY_INSTANCE = SimpleNamespace(
    instance_id="astropy__astropy-12907",
    repo="astropy/astropy",
    version="4.3",
    base_commit="d16bfe05a744909de4b27f5875fe0d4ed41ce607",
)


def _clone_at_base_commit(repo: str, base_commit: str, dest: Path) -> None:
    """Shallow-clone a github repo at ``base_commit`` into ``dest``.

    Uses ``--filter=blob:none`` so the working tree is materialized
    lazily — the smoke only needs the tree at base_commit, not the
    full history. Produces a regular clone (``.git`` is a directory).
    """
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--no-checkout",
         f"https://github.com/{repo}.git", str(dest)],
        check=True, capture_output=True, timeout=180,
    )
    subprocess.run(
        ["git", "checkout", base_commit],
        cwd=str(dest), check=True, capture_output=True, timeout=60,
    )


def _is_clean(workspace: Path) -> bool:
    """True iff ``git diff HEAD`` returns no tracked-file changes.

    Untracked artifacts (egg-info from the editable install, build/
    dirs) are explicitly fine — they aren't reported by ``git diff
    HEAD`` and don't pollute the implementer's diff capture.
    """
    diff = subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(workspace), capture_output=True, text=True, timeout=30,
    )
    return diff.returncode == 0 and not diff.stdout.strip()


@pytest.mark.live_docker
def test_live_container_lifecycle_clean_workspace_after_setup():
    """Full chain: clone → resolve_container_spec → setup_instance
    _container → docker exec smoke → teardown_instance_container.

    The image's baked extension artifacts must be seeded into the mounted
    checkout without network or a fresh package install, while tracked source
    remains clean for implementer diff attribution.
    """
    spec = resolve_container_spec(_ASTROPY_INSTANCE)
    assert spec is not None, "astropy@4.3 must be in MAP_REPO_VERSION_TO_SPECS"
    assert "astropy_1776_astropy-12907" in spec.image
    assert spec.pre_install, "astropy@4.3 has a pyproject.toml sed pre_install"

    workspace = Path(tempfile.mkdtemp(prefix="crucible_smoke_astropy_"))
    container_id = None
    try:
        _clone_at_base_commit(
            _ASTROPY_INSTANCE.repo,
            _ASTROPY_INSTANCE.base_commit,
            workspace,
        )
        # Sanity: pre_install will mutate pyproject.toml; confirm it
        # exists in the tracked tree before we touch it.
        pyproject = workspace / "pyproject.toml"
        assert pyproject.exists()

        container_id = setup_instance_container(
            workspace,
            f"crucible-smoke-{_ASTROPY_INSTANCE.instance_id}".replace("/", "_"),
            spec,
        )
        assert container_id, "setup_instance_container returned no id"

        # 1. Conda env activates + astropy imports — this is the minimum
        #    signal that the image's baked extension artifacts survived the
        #    offline workspace seed. Note: ``__version__``
        #    is computed by setuptools_scm from the git description
        #    at base_commit (typically ``5.1.dev...`` for instance
        #    12907's commit) — NOT the SWE-bench dataset's
        #    ``version: "4.3"`` field which is the release-branch
        #    marker. Just check it imports + the file lives under
        #    the bind-mount path.
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c",
             f"{CONDA_ACTIVATE_PREFIX}&& python -c "
             "'import astropy; print(astropy.__version__); print(astropy.__file__)'"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"in-container python import astropy failed: "
            f"rc={result.returncode}\nstderr={result.stderr}"
        )
        stdout = result.stdout.strip().splitlines()
        assert len(stdout) >= 2, f"expected version + path lines, got {stdout!r}"
        version_line, path_line = stdout[0], stdout[1]
        assert version_line, "empty version line"
        assert path_line.startswith("/testbed/"), (
            f"astropy.__file__ must resolve under the bind-mount; "
            f"got {path_line!r} — install may have leaked to the "
            f"conda env's site-packages instead of the editable mount"
        )

        # 2. The bind-mount geometry is right: a host-side file
        #    creation is visible inside the container at /testbed.
        sentinel = workspace / ".smoke_sentinel"
        sentinel.write_text("from host")
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c",
             "cat /testbed/.smoke_sentinel"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert result.stdout == "from host"

        # 3. CRITICAL: workspace must be clean post-setup. The
        #    pre_install replay mutated pyproject.toml (setuptools
        #    pin); _revert_workspace must have undone it. Without
        #    this, the implementer's diff capture would carry the
        #    harness's pre_install changes as if the model wrote them.
        assert _is_clean(workspace), (
            "workspace is dirty after setup_instance_container; "
            "_revert_workspace did not run / failed silently. "
            f"diff:\n{subprocess.run(['git', 'diff', 'HEAD'], cwd=str(workspace), capture_output=True, text=True).stdout[:2000]}"
        )

    finally:
        if container_id:
            teardown_instance_container(container_id)
        shutil.rmtree(workspace, ignore_errors=True)


@pytest.mark.live_docker
def test_live_teardown_stops_running_container():
    """Teardown idempotency under the live path: a running container
    gets stopped + auto-removed (we run with --rm)."""
    spec = resolve_container_spec(_ASTROPY_INSTANCE)
    assert spec is not None

    workspace = Path(tempfile.mkdtemp(prefix="crucible_smoke_teardown_"))
    container_id = None
    try:
        _clone_at_base_commit(
            _ASTROPY_INSTANCE.repo,
            _ASTROPY_INSTANCE.base_commit,
            workspace,
        )
        container_id = setup_instance_container(
            workspace,
            f"crucible-smoke-teardown-{_ASTROPY_INSTANCE.instance_id}".replace("/", "_"),
            spec,
        )
        # Confirm it's running.
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True, text=True,
        )
        assert inspect.stdout.strip() == "true"

        teardown_instance_container(container_id)

        # Teardown stops the container; ``--rm`` removes it. The
        # remove is async on the docker daemon side — an inspect
        # right after ``docker stop`` can race and see the stopped-
        # but-not-yet-removed state. Either of these signals satisfies
        # the teardown contract:
        #   (a) inspect fails (container fully removed)
        #   (b) inspect succeeds with State.Running=false (stopped,
        #       --rm cleanup pending on the daemon)
        # Both are evidence the container is no longer doing work and
        # won't block the next setup.
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True, text=True,
        )
        gone = inspect.returncode != 0
        stopped = (
            inspect.returncode == 0 and inspect.stdout.strip() == "false"
        )
        assert gone or stopped, (
            f"container should be stopped (and removed by --rm) after "
            f"teardown — got rc={inspect.returncode} "
            f"stdout={inspect.stdout!r}"
        )
        # Double teardown must not raise.
        teardown_instance_container(container_id)
        # None must not raise.
        teardown_instance_container(None)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
