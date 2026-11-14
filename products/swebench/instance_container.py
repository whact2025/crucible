# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Mandatory per-instance Docker container for SWE-bench cascades.

Uses the SWE-bench harness's pre-built per-instance images and matches its
verification environment exactly — no Python
version drift, no dep-pin resolution failures, no pip vs uv subtleties.

Image format
------------
The harness ships per-instance images at::

    swebench/sweb.eval.x86_64.<translated_instance_id>:latest

where ``__`` in the instance_id is replaced by ``_1776_`` (URL-safe
substitution). Each image carries:

- ``/testbed/`` — the repo at base_commit with the spec's pre_install
  already applied (sed patches, etc.)
- ``/opt/miniconda3/`` with a conda env named ``testbed`` that has
  the spec's pinned ``pip_packages`` + the package installed in
  editable mode

Lifecycle per instance
----------------------
1. ``docker pull swebench/sweb.eval.<arch>.<iid>:latest`` (cached
   across runs after first pull).
2. ``docker run -d --rm -v <host_workspace>:/testbed:rw`` — bind-
   mounts the cascade's worktree OVER the baked workspace so the
   implementer's edits land at the same path conda env expects.
   The conda env at /opt/miniconda3 is unaffected by the mount.
3. After mount, do ``pip install -e .[test]`` once inside the
   container to register the editable install against the now-
   live-mounted workspace.
4. Each pytest call runs as ``docker exec <id> bash -c '...source
   /opt/miniconda3/bin/activate && conda activate testbed && cd
   /testbed && pytest ...'``.
5. ``docker stop`` on exit (the ``--rm`` flag auto-removes the
   stopped container).

Operational properties
----------------------
+ Matches the harness environment exactly.
+ Per-instance dep set is pre-baked — no install-time resolution.
+ Per-instance isolation is enforced by the kernel.
- Each image is ~3-4 GB after layer dedup (the harness's images
  include conda + the per-instance dep set). A 500-Verified sweep
  needs ~1.5-2 TB unless ``docker system prune`` runs between
  batches. See ``check_disk_headroom`` for the entry-time refuse.
- Docker daemon dependency (already required for the eval harness).
- Bind-mount filesystem semantics may differ on macOS / WSL2 —
  Linux native is the production target.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from crucible.common.benchmark_runtime import (
    ImmutableImageError,
    label_args,
    reconcile_orphaned_containers,
    resolve_immutable_image,
)

from ._env_common import _revert_workspace

logger = logging.getLogger(__name__)


# Per-subprocess timeout for container setup steps. Image pull on a
# cold cache can take ~30-60s; container start is sub-second. Cap at
# 10 min so a stalled pull doesn't deadlock the worker.
_STEP_TIMEOUT = 600

# Underscore substitution in instance_id when forming the image tag.
# The harness uses ``_1776_`` (year of US independence — internal joke).
_INSTANCE_ID_SUBSTITUTION = ("__", "_1776_")

# Canonical shell prefix that activates the harness's conda env before
# any command runs. The harness's eval.sh uses this exact form; we
# match it so build / verify subprocesses see the same Python +
# site-packages. Centralized so a hypothetical upstream layout change
# is a one-line edit instead of three.
CONDA_ACTIVATE_PREFIX = (
    "source /opt/miniconda3/bin/activate "
    "&& conda activate testbed "
    "&& cd /testbed "
)

# Workspace reset is shared with the environment helpers.


# Free-disk threshold below which the worker refuses to start a Docker
# sweep. The harness images are ~3-4 GB per instance after layer
# dedup (the earlier 1 GB estimate undercounted layer overhead).
# 10 GB headroom gives room for ~2-3 instances before the worker
# starts to ENOSPC. Operators can override via env var.
_MIN_FREE_DISK_GB = 10.0
_DEFAULT_CPUS = 2.0
_DEFAULT_MEMORY_MB = 8192
_DEFAULT_PIDS_LIMIT = 512
_TMPFS_SIZE_MB = 4096


class InstanceContainerSetupError(RuntimeError):
    """Raised when container setup fails (image pull, run, install).
    Live SWE-bench callers fail closed; host execution is forbidden."""


@dataclass(frozen=True)
class InstanceContainerSpec:
    """Resolved per-instance container spec.

    Carries the image tag + the harness's pre_install / install
    commands. Both run inside the container against the bind-mounted
    /testbed so the package gets re-registered against the host
    workspace's view (the image's baked install targeted the image's
    /testbed which the mount masks).

    ``pre_install`` is critical: the harness's image build runs these
    commands (typically ``sed`` patches to pyproject.toml pinning
    setuptools / setuptools_scm to versions that have ``dep_util``
    etc.) before the install. When we bind-mount over /testbed, we
    lose those mutations and the install step's PEP-517 build
    isolation picks a newer setuptools that breaks. Replaying
    pre_install on the host workspace before install reproduces the
    harness's image-build sequence.
    """

    image: str
    install_command: str
    pre_install: tuple[str, ...] = ()


def _docker_arch() -> str:
    """Map ``uname -m`` to the harness's image-tag architecture token.

    The harness publishes ``x86_64`` and ``arm64`` images. Anything
    else (i686, aarch64 on non-Apple-Silicon flavors, etc.) is
    unsupported — caller rejects container-backed build/verify.
    """
    m = platform.machine()
    if m in ("x86_64", "amd64"):
        return "x86_64"
    if m in ("arm64", "aarch64"):
        return "arm64"
    return m  # let caller see the unsupported value


def _translate_instance_id(instance_id: str) -> str:
    """Apply the harness's ``__`` -> ``_1776_`` substitution.

    ``astropy__astropy-12907`` -> ``astropy_1776_astropy-12907``.
    """
    src, dst = _INSTANCE_ID_SUBSTITUTION
    return instance_id.replace(src, dst)


def resolve_container_spec(instance) -> InstanceContainerSpec | None:
    """Look up the per-instance container spec.

    Returns ``None`` when:
    - The architecture isn't supported by the harness's published
      images.
    - The swebench package's ``MAP_REPO_VERSION_TO_SPECS`` doesn't
      know this repo/version (rare; would have failed in the venv
      path too).

    Live workflow callers treat ``None`` as an unsupported instance and fail
    closed rather than executing the checkout on the host.
    """
    arch = _docker_arch()
    if arch not in ("x86_64", "arm64"):
        logger.warning(
            "instance_container: unsupported architecture %r — "
            "harness images are x86_64 / arm64 only",
            arch,
        )
        return None

    try:
        from swebench.harness.constants import (  # noqa: PLC0415
            MAP_REPO_VERSION_TO_SPECS,
        )
    except ImportError:
        return None

    repo_specs = MAP_REPO_VERSION_TO_SPECS.get(instance.repo)
    if not repo_specs:
        return None
    raw = repo_specs.get(instance.version)
    if not raw:
        return None
    install = str(raw.get("install") or "")
    if not install:
        return None

    image = (
        f"swebench/sweb.eval.{arch}."
        f"{_translate_instance_id(instance.instance_id)}:latest"
    )
    pre_install = tuple(str(c) for c in (raw.get("pre_install") or ()))
    return InstanceContainerSpec(
        image=image,
        install_command=install,
        pre_install=pre_install,
    )


def _run_docker(
    args: list[str],
    *,
    step_name: str,
    timeout: int = _STEP_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a docker CLI command + raise InstanceContainerSetupError on
    non-zero exit. ``errors='replace'`` keeps non-UTF-8 docker stderr
    (corrupted manifest blobs, progress-bar control codes) from
    crashing the wrapper with UnicodeDecodeError before our error-
    formatting code runs."""
    pretty = " ".join(args)
    logger.info("instance_container[%s]: docker %s", step_name, pretty[:200])
    try:
        result = subprocess.run(
            ["docker", *args],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise InstanceContainerSetupError(
            f"instance_container[{step_name}]: timed out after {timeout}s"
        ) from exc
    if result.returncode != 0:
        stderr = (result.stderr or "")[-2_000:]
        raise InstanceContainerSetupError(
            f"instance_container[{step_name}]: rc={result.returncode}\n"
            f"cmd: docker {pretty[:200]}\n"
            f"stderr: {stderr}"
        )
    return result


def _disk_check_path() -> str:
    """Resolve the path whose filesystem the headroom check measures.

    Image data lands on the filesystem backing Docker's image store,
    which is NOT always ``/`` — operators relocate it (containerd
    ``root`` / Docker ``data-root``) onto a roomier volume. Measuring
    ``/`` then reports headroom for the wrong filesystem.

    Resolution order:
    1. ``CRUCIBLE_SWEBENCH_DISK_CHECK_PATH`` env override (explicit
       operator intent — e.g. ``/data`` after a data-root move).
    2. ``docker info``'s ``DockerRootDir`` (best-effort; one cheap
       subprocess). Note this reports Docker's own root, which can
       still diverge from containerd's ``root`` under the containerd
       image store — hence the env override takes precedence.
    3. ``/`` as the conservative default.
    """
    override = os.environ.get("CRUCIBLE_SWEBENCH_DISK_CHECK_PATH", "").strip()
    if override:
        return override
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        root = result.stdout.strip()
        if root and os.path.isdir(root):
            return root
    except (subprocess.SubprocessError, OSError):
        pass
    return "/"


def check_disk_headroom(min_gb: float = _MIN_FREE_DISK_GB) -> None:
    """Refuse to start a Docker-mode sweep when free disk is below
    ``min_gb``. Per-instance images are ~3-4 GB after layer dedup;
    running a sweep below the threshold ENOSPCs partway through and
    leaves the worker in a confused state.

    The filesystem measured is resolved by ``_disk_check_path`` (the
    image store's volume, not blindly ``/``) so the check stays
    accurate after an operator relocates the Docker/containerd root.

    Operators who know what they're doing (e.g. running a single
    instance against an already-cached image) can suppress via
    ``CRUCIBLE_SWEBENCH_SKIP_DISK_CHECK=1`` (also accepts
    ``true``/``yes``, case-insensitive). Setting it to ``0`` /
    ``false`` keeps the check ON, which matches operator intuition —
    bare presence-of-env-var truthiness would silently skip on
    ``=0``.
    """
    raw = os.environ.get("CRUCIBLE_SWEBENCH_SKIP_DISK_CHECK", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return
    path = _disk_check_path()
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_gb:
        raise InstanceContainerSetupError(
            f"instance_container: free disk {free_gb:.1f} GB on {path} is "
            f"below the {min_gb:.0f} GB threshold for Docker-mode sweeps. "
            f"Each harness image is ~3-4 GB; a sweep below this floor "
            f"will ENOSPC partway through. Reclaim with ``docker "
            f"system prune -a -f``, or set "
            f"CRUCIBLE_SWEBENCH_SKIP_DISK_CHECK=1 to override."
        )


def pull_image(image: str) -> str:
    """Pull the image if not already in the local cache. Idempotent —
    if the image is present, docker pull is a fast no-op (just hits
    the registry's manifest endpoint to confirm)."""
    _run_docker(
        ["pull", image],
        step_name="pull",
        timeout=900,  # 15 min — cold pull on a slow connection
    )
    try:
        return resolve_immutable_image(image, run_docker=_run_docker)
    except ImmutableImageError as exc:
        raise InstanceContainerSetupError(str(exc)) from exc


def setup_instance_container(
    workspace: Path,
    container_name: str,
    spec: InstanceContainerSpec,
    *,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Materialize a per-instance container. Returns the container id.

    Steps:

    1. ``docker pull`` the spec's image (cached after first call).
    2. Copy the image's already-built ``/testbed`` artifacts into the host
       checkout using a short, offline container. Tracked files are reset;
       compiled/untracked artifacts remain.
    3. ``docker run -d --rm`` with the standalone, one-commit workspace
       mounted at ``/testbed``. Linked worktrees are rejected because they
       require exposing an external Git object database.
    Raises ``InstanceContainerSetupError`` on any step failure. The
    caller is responsible for ``teardown_instance_container`` in a
    finally block; we use ``--rm`` so a stopped container auto-removes.
    """
    workspace = Path(workspace).resolve()
    if not workspace.is_dir():
        raise InstanceContainerSetupError(
            f"instance_container: workspace {workspace} not a directory"
        )
    if not (workspace / ".git").is_dir():
        raise InstanceContainerSetupError(
            "instance_container: authoring workspace must have a "
            "standalone .git directory; linked worktrees are forbidden"
        )

    # 1. Pull (cached on subsequent calls).
    immutable_image = pull_image(spec.image)
    container_labels = dict(labels or {})
    container_labels["io.crucible.image-digest"] = immutable_image

    # 2. Seed the bind-mounted checkout with the benchmark image's baked
    # build artifacts. The image is the immutable dependency capsule; a fresh
    # editable install would execute project build hooks and require networked
    # build dependencies, both forbidden in the authoring sandbox. Exclude
    # .git so clone/worktree metadata remains host-owned, run as the host user,
    # and reset tracked files so only compiled/untracked artifacts survive.
    _run_docker(
        [
            "run", "--rm", "--network", "none",
            *label_args(container_labels),
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            # RLIMIT_NPROC is per real host UID, not per container. On shared
            # workers, setting it to this value can reject PID 1 before the
            # container starts because unrelated host-UID processes count
            # against it. The container's dedicated pids cgroup enforces the
            # intended boundary without that cross-workload coupling.
            "--pids-limit", str(_DEFAULT_PIDS_LIMIT),
            "--cpus", str(_DEFAULT_CPUS),
            "--memory", f"{_DEFAULT_MEMORY_MB}m",
            "--memory-swap", f"{_DEFAULT_MEMORY_MB}m",
            "--ulimit", "nofile=4096:4096",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "HOME=/tmp/home",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--tmpfs", f"/tmp:rw,exec,nosuid,nodev,size={_TMPFS_SIZE_MB}m",
            "-v", f"{workspace}:/host:rw",
            immutable_image,
            "bash", "-eu", "-o", "pipefail", "-c",
            "cd /testbed && tar --exclude=.git -cf - . "
            "| tar --no-same-owner -xf - -C /host",
        ],
        step_name="seed baked workspace",
        timeout=_STEP_TIMEOUT,
    )
    _revert_workspace(workspace)

    # 3. Run detached with only the standalone, single-commit workspace.
    result = _run_docker(
        [
            "run", "-d", "--rm",
            "--name", container_name,
            *label_args(container_labels),
            # Benchmark code is untrusted. Docker does not inherit the
            # host environment by default; keep that credential boundary
            # intact and also remove network exfiltration during setup,
            # build, and verification. The image is pulled before this
            # point and carries its dependency environment already.
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            # The candidate remains the configured non-root uid with
            # no-new-privileges. These capabilities are retained only for the
            # root-owned contained-exec supervisor, which drops its child back
            # to Config.User and terminates escaped descendants.
            "--cap-add", "SETUID",
            "--cap-add", "SETGID",
            "--cap-add", "KILL",
            "--security-opt", "no-new-privileges",
            # See the seed-container note above: use the per-container pids
            # cgroup, never host-UID-wide RLIMIT_NPROC.
            "--pids-limit", str(_DEFAULT_PIDS_LIMIT),
            "--cpus", str(_DEFAULT_CPUS),
            "--memory", f"{_DEFAULT_MEMORY_MB}m",
            "--memory-swap", f"{_DEFAULT_MEMORY_MB}m",
            "--ulimit", "nofile=4096:4096",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--env", "HOME=/tmp/home",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--tmpfs", f"/tmp:rw,exec,nosuid,nodev,size={_TMPFS_SIZE_MB}m",
            "-v", f"{workspace}:/testbed:rw",
            "-v", f"{workspace / '.git'}:/testbed/.git:ro",
            "--entrypoint", "/bin/sleep",
            immutable_image,
            "infinity",
        ],
        step_name="run",
    )
    container_id = result.stdout.strip()
    if not container_id:
        raise InstanceContainerSetupError(
            "instance_container[run]: docker reported no container id"
        )

    try:
        # The seed reset above guarantees the implementer sees no tracked setup
        # mutations. A final reset is cheap defense against an image entrypoint
        # touching the bind mount during startup.
        _revert_workspace(workspace)

        # The container runs as the host UID:GID, so bind-mount writes are
        # host-owned without a privileged post-start chown.
    except Exception:
        try:
            _run_docker(
                ["stop", container_id],
                step_name="stop (setup rollback)",
                timeout=30,
            )
        except InstanceContainerSetupError:
            pass
        raise

    logger.info(
        "instance_container: ready at container %s (image=%s)",
        container_id[:12], immutable_image,
    )
    return container_id


def container_immutable_image(container_id: str) -> str:
    """Read the exact repository digest stamped on a managed container."""

    result = _run_docker(
        [
            "inspect", "--format",
            "{{index .Config.Labels \"io.crucible.image-digest\"}}",
            container_id,
        ],
        step_name="inspect image digest",
        timeout=30,
    )
    digest = (result.stdout or "").strip()
    if "@sha256:" not in digest:
        raise InstanceContainerSetupError(
            f"container {container_id[:12]} has no immutable image digest"
        )
    return digest


def reconcile_orphans(scope: str) -> tuple[str, ...]:
    """Reclaim managed containers whose owner died without cleanup."""

    return reconcile_orphaned_containers(scope=scope, run_docker=_run_docker)


def teardown_instance_container(
    container_id: str | None,
    image: str | None = None,
) -> None:
    """Best-effort container cleanup. Idempotent — safe to call when
    setup failed mid-flight or twice in a row.

    Authoring containers run as the host UID:GID, so bind-mounted files are
    never root-owned and teardown needs no privileged repair step.

    When ``image`` is provided, removes the per-instance image from
    the local layer cache after stopping the container. The harness
    images are ~3-4 GB each; without this cleanup a 500-Verified
    sweep accumulates ~1.5-2 TB of dead image data (the ``--rm`` flag
    only frees the container's write layer, not the cached image).
    Sweep peak disk becomes ``parallel_workers × image_size`` instead
    of ``total_instances × image_size``.

    Operators iterating on a single instance can keep images via
    ``CRUCIBLE_SWEBENCH_KEEP_IMAGES=1`` (mirrors the
    ``CRUCIBLE_SWEBENCH_SKIP_DISK_CHECK`` opt-out convention) to skip
    the re-pull cost on subsequent runs. The cleanup is best-effort —
    a still-running container against the same image (concurrent
    worker) makes ``docker image rm`` fail with "image is in use",
    which we log and swallow rather than failing the cascade.
    """
    if not container_id:
        return
    # The container runs as the host UID:GID, so no privileged chown is
    # required before teardown. Stop it; --rm means stop = removal.
    try:
        _run_docker(
            ["stop", container_id],
            step_name="stop",
            timeout=30,
        )
    except InstanceContainerSetupError as exc:
        logger.warning(
            "instance_container: teardown of %s failed: %s — "
            "container may leak (docker ps -a to inspect)",
            container_id[:12], exc,
        )
    # 3. Remove the per-instance image to keep disk bounded. Skipped
    # when the operator opted out via env var, or when no image tag
    # was threaded through (caller didn't supply one).
    if image:
        _remove_image_if_unpinned(image)


def _remove_image_if_unpinned(image: str) -> None:
    """Best-effort ``docker image rm``. Honors
    ``CRUCIBLE_SWEBENCH_KEEP_IMAGES=1`` to skip removal so operators
    iterating on a single instance don't pay the re-pull cost."""
    raw = os.environ.get("CRUCIBLE_SWEBENCH_KEEP_IMAGES", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return
    try:
        _run_docker(
            ["image", "rm", image],
            step_name="image_rm",
            timeout=30,
        )
    except InstanceContainerSetupError as exc:
        # Common case: a concurrent worker is still running against
        # the same image, so Docker refuses removal with "image is in
        # use by stopped container" or "image is referenced in
        # multiple repositories". Either way the cascade continues;
        # a periodic ``docker image prune -f`` outside the workflow
        # mops up later.
        logger.info(
            "instance_container: image rm %s skipped: %s",
            image, str(exc).splitlines()[-1][:200],
        )


def docker_available() -> bool:
    """Quick probe — is the docker daemon reachable?

    Called once at SWE-bench workflow entry. False is a setup failure:
    untrusted benchmark repositories never execute on the host.
    """
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


__all__ = [
    "CONDA_ACTIVATE_PREFIX",
    "InstanceContainerSetupError",
    "InstanceContainerSpec",
    "check_disk_headroom",
    "container_immutable_image",
    "docker_available",
    "pull_image",
    "reconcile_orphans",
    "resolve_container_spec",
    "setup_instance_container",
    "teardown_instance_container",
]
