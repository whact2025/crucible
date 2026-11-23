# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-task Docker container lifecycle for DeepSWE cascades.

Pulls the pre-baked image named by each task record, runs it with
``--network=none`` to honor the upstream air-gap contract,
optionally bind-mounts a host workspace at ``/app`` (for the
cascade's implementer-edits pattern) and a host logs dir at
``/logs`` (so the verifier shell's reward.txt is visible on the
host).

Image format
------------
DeepSWE's prebuilt images are dep-baked at image-build time — there
is NO install replay step. This is the key shape difference vs.
``products/swebench/instance_container.py`` where the harness's
images require dataset-side install after bind-mount. DeepSWE therefore uses
its own smaller task record while sharing benchmark-neutral lifecycle helpers.

Lifecycle per task
------------------
1. ``docker pull <image>`` (cached across runs after first pull).
2. ``docker run -d --rm --network=none [-v <ws>:/app:rw]
   [-v <logs>:/logs:rw] [--cpus=N] [--memory=Mm] <image>
   sleep infinity`` — sleep PID 1 so the container stays up across
   multiple ``docker exec`` calls (same pattern as SWE-bench).
3. Caller takes container_id, dispatches ``docker exec`` for
   implementer + verifier phases.
4. ``teardown_instance_container`` stops the container; ``--rm``
   makes the container auto-remove. Optional ``docker image rm``
   reclaims the image's ~3 GB disk per the disk-headroom posture
   in the integration plan.

Air-gap contract
----------------
``--network=none`` is enforced unconditionally. Pier honors
Crucible always uses ``--network=none`` for authoring and verification. The
loader retains the task's ``allow_internet`` value for audit, but this
execution path does not grant network access when it is true.

Parity caveats
--------------
- ``--cpus`` / ``--memory`` translate from ``inst.cpus`` /
  ``inst.memory_mb``. Both are uniformly supported across Docker
  storage drivers.
- ``--storage-opt size=`` for the ``inst.storage_mb`` cap is NOT
  applied — it requires devicemapper-class storage drivers, which
  most production Docker setups lack. Per-task storage caps are
  best-effort via the host's disk-headroom guard, not container-
  level enforcement.

Cross-product boundary
----------------------
The product module owns DeepSWE-specific image and mount policy. Shared
containment, immutable-image, labeling, and orphan-reconciliation mechanics
live under :mod:`crucible.common`.
"""

from __future__ import annotations

import logging
import os
import re
import selectors
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from crucible.common.benchmark_runtime import (
    ImmutableImageError,
    label_args,
    managed_container_labels,
    reconcile_orphaned_containers,
    resolve_immutable_image,
)
from crucible.products.deepswe.git_safety import (
    ContainerGitEnvironmentError,
    GitMetadataSafetyError,
    git_argv,
    hermetic_container_git_exec_prefix,
    hermetic_container_git_shell_function,
    inspect_docker_container_environment,
    sanitize_copied_git_metadata,
    sanitized_git_env,
)
from crucible.products.deepswe.patch_limits import (
    DEFAULT_MAX_PATCH_BYTES,
    validate_patch_byte_limit,
)
from crucible.products.deepswe.rust_toolchain import (
    RUST_CARGO_BIN,
    RUST_CARGO_HOME,
    RUST_IMMUTABLE_ROOT,
    RUSTUP_HOME,
    environment_args as rust_environment_args,
    runtime_environment as rust_runtime_environment,
)

if TYPE_CHECKING:
    from .dataset import DeepSWEInstance

logger = logging.getLogger(__name__)


# Per-subprocess timeout for container lifecycle ops. Image pull on
# a cold cache can take ~30-60s; container start is sub-second;
# stop is sub-second. 15 min cap is generous; a stalled pull won't
# deadlock the worker.
_STEP_TIMEOUT_SEC = 900

# Free-disk floor below which the worker refuses to start.
# DeepSWE images are ~1-3 GB per task (smaller than SWE-bench's
# 3-4 GB harness images per the corpus shape).
_MIN_FREE_DISK_GB = 10.0
_PIDS_LIMIT = 512
_MAX_TMPFS_MB = 4096

# A complete SHA-1 or SHA-256 object id, and the shortest prefix of one that
# the dataset may pin instead (Git's own default abbreviation length).
_FULL_COMMIT_ID_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_ABBREVIATED_COMMIT_ID_RE = re.compile(r"[0-9a-f]{7,63}")


class InstanceContainerSetupError(RuntimeError):
    """Raised when container setup or teardown fails (image pull,
    docker run, docker exec). Caller logs + skips the task; the
    cascade's outer error handling treats this as setup-time
    failure."""


class RustToolchainUnavailable(InstanceContainerSetupError):
    """The immutable image's Rust state cannot serve the non-root runtime."""


ContainerRegistrationCallback = Callable[[str], None]


def _register_created_container(
    container_reference: str,
    *,
    register_container: ContainerRegistrationCallback | None,
    step_name: str,
) -> None:
    """Register daemon-side authority before any created container is used.

    Registration is part of creation, not later bookkeeping.  If the durable
    registration callback fails, force-remove the newly created container and
    surface the registration failure.  A cleanup failure is retained in the
    same terminal error so neither side of the failed-closed transition is
    lost.
    """

    if register_container is None:
        return
    try:
        register_container(container_reference)
    except Exception as registration_error:
        registration_failure = InstanceContainerSetupError(
            f"deepswe_container[{step_name}]: container registration failed "
            f"for {container_reference!r}: {registration_error}"
        )
        try:
            _run_docker(
                ["rm", "-f", container_reference],
                step_name=f"{step_name}_registration_failure_rm",
                timeout=30,
            )
        except InstanceContainerSetupError as cleanup_error:
            raise InstanceContainerSetupError(
                f"{registration_failure}; force-remove cleanup also failed: "
                f"{cleanup_error}"
            ) from cleanup_error
        raise registration_failure from registration_error


def _bootstrap_rust_toolchain(
    container_id: str,
    *,
    runtime_user: str,
) -> None:
    """Stage only root's Rust state for the container's non-root uid.

    Rust DeepSWE images are built as root and install rustup below ``/root``.
    The runtime deliberately remains non-root and the image root remains
    read-only.  Copy immutable proxy/toolchain bytes as root into the private
    tmpfs, then copy only mutable Cargo state as the runtime uid.  No root-home
    file outside ``.cargo``/``.rustup`` is observed or copied.
    """

    cargo_seed = "/tmp/crucible-rust-cargo-seed"
    immutable_script = "\n".join((
        "set -eu",
        "test -x /root/.cargo/bin/cargo",
        "test -d /root/.rustup/toolchains",
        f"/bin/rm -rf {RUST_IMMUTABLE_ROOT} {cargo_seed}",
        f"/bin/mkdir -p {RUST_IMMUTABLE_ROOT}",
        f"/bin/cp -a /root/.rustup {RUSTUP_HOME}",
        f"/bin/cp -a /root/.cargo {cargo_seed}",
        f"/bin/mv {cargo_seed}/bin {RUST_CARGO_BIN}",
        f"/bin/chmod -R a+rX,go-w {RUST_IMMUTABLE_ROOT} {cargo_seed}",
    ))
    _run_docker(
        [
            "exec", "--user", "0:0", container_id,
            "/bin/sh", "-c", immutable_script,
        ],
        step_name="rust_toolchain_stage_immutable",
    )

    mutable_script = "\n".join((
        "set -eu",
        f"/bin/rm -rf {RUST_CARGO_HOME}",
        f"/bin/mkdir -p {RUST_CARGO_HOME} /app/target",
        (
            f"/bin/cp -R --no-preserve=ownership,mode "
            f"{cargo_seed}/. {RUST_CARGO_HOME}/"
        ),
        f"/bin/chmod -R u+rwX,go-rwx {RUST_CARGO_HOME}",
    ))
    _run_docker(
        [
            "exec", "--user", runtime_user, container_id,
            "/bin/sh", "-c", mutable_script,
        ],
        step_name="rust_toolchain_stage_mutable",
    )
    _run_docker(
        [
            "exec", "--user", "0:0", container_id,
            "/bin/rm", "-rf", cargo_seed,
        ],
        step_name="rust_toolchain_stage_cleanup",
    )


def _probe_rust_toolchain(
    container_id: str,
    *,
    runtime_user: str,
) -> None:
    """Prove Cargo and rustc execute under the exact non-root runner env."""

    environment = {
        "HOME": "/tmp/home",
        "LANG": "C.UTF-8",
        **rust_runtime_environment(),
    }
    prefix = [
        "/usr/bin/env", "-i", "--",
        *(f"{key}={environment[key]}" for key in sorted(environment)),
    ]
    _run_docker(
        [
            "exec", "--user", runtime_user, container_id,
            *prefix,
            "/bin/sh", "-c",
            (
                "set -eu; "
                f'test "$(command -v cargo)" = "{RUST_CARGO_BIN}/cargo"; '
                "cargo --version >/dev/null; rustc --version >/dev/null"
            ),
        ],
        step_name="rust_toolchain_probe",
        timeout=120,
    )


def _remove_failed_setup_container(
    container_id: str,
    *,
    cause: InstanceContainerSetupError,
) -> None:
    """Remove a post-start setup failure, or surface both failures."""

    try:
        _run_docker(
            ["rm", "-f", container_id],
            step_name="setup_failure_rm",
            timeout=30,
        )
    except InstanceContainerSetupError as cleanup_error:
        raise RustToolchainUnavailable(
            f"{cause}; cleanup also failed: {cleanup_error}"
        ) from cause


def _run_docker(
    args: list[str],
    *,
    step_name: str,
    timeout: int = _STEP_TIMEOUT_SEC,
) -> subprocess.CompletedProcess:
    """Run ``docker <args>``. Non-zero exit raises
    ``InstanceContainerSetupError`` with step_name + truncated
    stderr context. ``errors="replace"`` keeps non-UTF-8 docker
    stderr (corrupt manifest blobs, progress-bar control codes)
    from crashing the wrapper before error formatting runs.

    DUPLICATED from ``products/swebench/instance_container.py``;
    lift candidate per the cross-product duplication note above.
    """
    pretty = " ".join(args)
    logger.info("deepswe_container[%s]: docker %s", step_name, pretty[:200])
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
            f"deepswe_container[{step_name}]: timed out after {timeout}s"
        ) from exc
    except FileNotFoundError as e:
        raise InstanceContainerSetupError(
            f"deepswe_container[{step_name}]: docker binary not found ({e})"
        ) from e
    if result.returncode != 0:
        stderr = (result.stderr or "")[-2_000:]
        raise InstanceContainerSetupError(
            f"deepswe_container[{step_name}]: docker exited "
            f"rc={result.returncode}; stderr: {stderr}"
        )
    return result


def docker_available() -> bool:
    """Quick probe — is the docker daemon reachable?

    Returns False on any error (missing binary, daemon down, or permission
    denied). The DeepSWE workflow treats a false result as a container-runtime
    prerequisite failure before starting task work.
    """
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _disk_check_path() -> str:
    """Resolve the path whose filesystem the headroom check measures.

    DUPLICATED from ``products/swebench/instance_container._disk_check_path``;
    distinct env-var name so swebench / deepswe overrides don't collide.

    Image data lands on the filesystem backing Docker's image store,
    which is NOT always ``/`` — operators relocate it (containerd
    ``root`` / Docker ``data-root``) onto a roomier volume. Measuring
    ``/`` then reports headroom for the wrong filesystem.

    Resolution order: ``CRUCIBLE_DEEPSWE_DISK_CHECK_PATH`` env override
    → ``docker info``'s ``DockerRootDir`` (best-effort) → ``/``.
    """
    override = os.environ.get("CRUCIBLE_DEEPSWE_DISK_CHECK_PATH", "").strip()
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
    """Refuse to start a Docker-mode DeepSWE sweep when free disk
    is below ``min_gb``. Per-task images are ~1-3 GB after layer
    dedup; running a 113-task sweep below the threshold ENOSPCs
    partway through and leaves the worker in a confused state.

    The filesystem measured is resolved by ``_disk_check_path`` (the
    image store's volume, not blindly ``/``) so the check stays
    accurate after an operator relocates the Docker/containerd root.

    Operators who know what they're doing (e.g. running a single
    task against an already-cached image) can suppress via
    ``CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK=1`` (also accepts
    ``true``/``yes``, case-insensitive). Setting to ``0`` / ``false``
    keeps the check ON, which matches operator intuition — bare
    presence-of-env-var truthiness would silently skip on ``=0``.

    Mirrors ``products/swebench/instance_container.check_disk_headroom``
    with a distinct env-var name so swebench and deepswe operator
    overrides don't collide.
    """
    raw = os.environ.get("CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return
    path = _disk_check_path()
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)
    if free_gb < min_gb:
        raise InstanceContainerSetupError(
            f"deepswe_container: free disk {free_gb:.1f} GB on {path} is "
            f"below the {min_gb:.0f} GB threshold for Docker-mode sweeps. "
            f"Each DeepSWE image is ~1-3 GB; a 113-task sweep below "
            f"this floor will ENOSPC partway through. Reclaim with "
            f"``docker system prune -a -f``, or set "
            f"CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK=1 to override."
        )


def pull_image(image: str) -> str:
    """Pull ``image`` if not already in the local cache. Idempotent —
    if the image is present and fresh, docker pull is a fast no-op
    (just hits the registry's manifest endpoint to confirm).

    DUPLICATED from ``products/swebench/instance_container.py``;
    lift candidate.
    """
    _run_docker(
        ["pull", image],
        step_name="pull",
        timeout=900,  # 15 min — cold pull on a slow connection
    )
    try:
        return resolve_immutable_image(image, run_docker=_run_docker)
    except ImmutableImageError as exc:
        raise InstanceContainerSetupError(str(exc)) from exc


def populate_workspace_from_image(
    image: str,
    host_workspace: Path,
    *,
    base_commit: str,
    image_app_path: str = "/app",
    labels: Mapping[str, str] | None = None,
    register_container: ContainerRegistrationCallback | None = None,
) -> str:
    """Extract the image's baked workspace into a host directory.

    Returns the full object id of the checkout's base commit. ``base_commit``
    is the dataset pin, which a few tasks abbreviate; the returned id is the
    one every later Git comparison binds.

    DeepSWE images ship with the repo cloned + checked out at
    ``base_commit_hash`` under ``/app``. The cascade's
    implementer-edits pattern needs that worktree visible on the
    host so the bind-mount at ``/app`` carries real content (and
    the verifier's ``git reset --soft $PIER_MODEL_BASE_COMMIT``
    has a valid repo to operate against). Without this step,
    ``setup_instance_container``'s bind-mount of an empty
    ``host_workspace`` over ``/app`` hides the image's baked
    repo and the verifier returns reward=0 uniformly.

    Mechanism: ``docker create`` a throwaway container off the
    image (does not start it), ``docker cp`` its ``/app/.`` into
    ``host_workspace/``, then ``docker rm`` the throwaway. Three
    short docker calls; the throwaway is cleaned up unconditionally.

    ``host_workspace`` should be empty before the call to avoid
    collisions; the caller is responsible for that. When supplied,
    ``register_container`` runs synchronously after ``docker create`` returns
    a nonempty reference and before ``docker cp``. A callback failure
    force-removes the throwaway and aborts setup.
    """
    if not host_workspace.is_dir():
        host_workspace.mkdir(parents=True, exist_ok=True)

    try:
        immutable_image = resolve_immutable_image(
            image, run_docker=_run_docker,
        )
    except ImmutableImageError as exc:
        raise InstanceContainerSetupError(str(exc)) from exc

    if labels is None:
        labels = managed_container_labels(
            scope=str(host_workspace.parent.resolve()),
            task_id=base_commit,
            phase="workspace-population",
        )

    # 1. Create a managed throwaway (does not start it). Labels make the
    # stopped container discoverable if the owner receives SIGKILL before the
    # finally block can remove it.
    create_result = _run_docker(
        ["create", *label_args(labels), immutable_image],
        step_name="populate_create",
        timeout=60,
    )
    throwaway_id = (create_result.stdout or "").strip()
    if not throwaway_id:
        raise InstanceContainerSetupError(
            f"deepswe_container: docker create returned empty id for {image}"
        )
    _register_created_container(
        throwaway_id,
        register_container=register_container,
        step_name="populate_create",
    )
    try:
        # 2. Copy the baked workspace's CONTENTS (note trailing
        # `/.`) into the host workspace. Trailing dot tells docker
        # cp to copy contents, not the `/app` directory itself.
        _run_docker(
            ["cp", f"{throwaway_id}:{image_app_path}/.",
             f"{host_workspace}/"],
            step_name="populate_cp",
            timeout=300,
        )
    finally:
        # 3. Remove throwaway. Best-effort — log + continue if
        # already removed somehow.
        try:
            _run_docker(
                ["rm", throwaway_id],
                step_name="populate_rm",
                timeout=30,
            )
        except InstanceContainerSetupError as e:
            logger.info(
                "deepswe_container: throwaway %s rm failed: %s — "
                "may have been removed already",
                throwaway_id[:12], e,
            )
    return _sanitize_workspace_history(host_workspace, base_commit=base_commit)


def is_full_commit_id(value: object) -> bool:
    """Whether ``value`` is one complete lowercase Git object id."""

    return (
        isinstance(value, str)
        and _FULL_COMMIT_ID_RE.fullmatch(value) is not None
    )


def _resolve_base_commit(*, base_commit: str, head: str) -> str:
    """Bind the dataset's base-commit pin to the checkout's full object id.

    The pin is either the complete id or a lowercase hexadecimal prefix of
    it. Only the observed ``HEAD`` is compared. Git is never asked to resolve
    the pin: it resolves a ref name before an abbreviated object id, so a ref
    that merely carries the pin's spelling could stand in for the commit.
    """

    if not is_full_commit_id(head):
        raise InstanceContainerSetupError(
            f"deepswe_container: image HEAD {head!r} is not a full Git "
            "object id"
        )
    if base_commit != head and not (
        isinstance(base_commit, str)
        and _ABBREVIATED_COMMIT_ID_RE.fullmatch(base_commit) is not None
        and len(base_commit) < len(head)
        and head.startswith(base_commit)
    ):
        raise InstanceContainerSetupError(
            f"deepswe_container: image HEAD {head!r} does not match "
            f"dataset base_commit {base_commit!r}"
        )
    return head


def _run_git(workspace: Path, *args: str) -> subprocess.CompletedProcess:
    """Run a local Git command while preserving the container error type."""
    result = subprocess.run(
        git_argv(*args),
        cwd=str(workspace), capture_output=True, text=True,
        env=sanitized_git_env(),
        timeout=120,
    )
    if result.returncode != 0:
        raise InstanceContainerSetupError(
            "deepswe_container: history isolation failed: git "
            f"{' '.join(args)}: {(result.stderr or '')[-1000:]}"
        )
    return result


def _sanitize_workspace_history(
    workspace: Path, *, base_commit: str,
) -> str:
    """Make the copied image checkout a one-commit, remote-free repo.

    DeepSWE images may carry refs and objects created after the dataset's
    base commit. Authoring roles have an in-container shell, so hiding refs
    in a prompt is insufficient: descendant commit objects must be removed
    before the workspace is mounted.

    Returns the full object id of that one commit.
    """
    try:
        git_dir = sanitize_copied_git_metadata(workspace)
    except GitMetadataSafetyError as exc:
        raise InstanceContainerSetupError(
            f"deepswe_container: unsafe copied Git metadata: {exc}"
        ) from exc
    head = _run_git(workspace, "rev-parse", "HEAD").stdout.strip()
    full_base_commit = _resolve_base_commit(base_commit=base_commit, head=head)
    _run_git(workspace, "checkout", "--detach", full_base_commit)
    refs = _run_git(
        workspace, "for-each-ref", "--format=%(refname)",
    ).stdout.splitlines()
    for ref in refs:
        if ref.strip():
            _run_git(workspace, "update-ref", "-d", ref.strip())
    remotes = _run_git(workspace, "remote").stdout.splitlines()
    for remote in remotes:
        if remote.strip():
            _run_git(workspace, "remote", "remove", remote.strip())
    shutil.rmtree(git_dir / "logs", ignore_errors=True)
    # ``.git/shallow`` holds complete object ids only.
    (git_dir / "shallow").write_text(
        full_base_commit + "\n", encoding="ascii",
    )
    _run_git(workspace, "reflog", "expire", "--expire=now", "--all")
    _run_git(workspace, "repack", "-Ad")
    _run_git(workspace, "prune", "--expire=now")
    count = _run_git(workspace, "rev-list", "--count", "HEAD").stdout.strip()
    if count != "1" or _run_git(workspace, "remote").stdout.strip():
        raise InstanceContainerSetupError(
            "deepswe_container: history isolation invariant failed; "
            f"commit_count={count!r}"
        )
    return full_base_commit


def _resolve_container_mount_roots(
    *,
    host_logs_dir: Path | None,
    host_workspace: Path | None,
    host_tests_dir: Path | None,
    host_solution_dir: Path | None,
    task_id: str,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """Canonicalize and validate host roots without mutating the filesystem."""

    logs = (
        Path(host_logs_dir).expanduser().resolve()
        if host_logs_dir is not None
        else None
    )
    workspace = (
        Path(host_workspace).expanduser().resolve()
        if host_workspace is not None
        else None
    )
    tests = (
        Path(host_tests_dir).expanduser().resolve()
        if host_tests_dir is not None
        else None
    )
    solution = (
        Path(host_solution_dir).expanduser().resolve()
        if host_solution_dir is not None
        else None
    )

    if logs is not None and logs.exists() and not logs.is_dir():
        raise InstanceContainerSetupError(
            f"deepswe_container: host_logs_dir {logs} is not a directory "
            f"(task {task_id!r})"
        )
    for label, root in (
        ("host_workspace", workspace),
        ("host_tests_dir", tests),
        ("host_solution_dir", solution),
    ):
        if root is not None and not root.is_dir():
            leakage = (
                "; GOLD-SOLUTION LEAKAGE is forbidden outside verifier pilots"
                if label == "host_solution_dir"
                else ""
            )
            raise InstanceContainerSetupError(
                f"deepswe_container: {label} {root} is not a directory "
                f"(task {task_id!r}){leakage}"
            )

    roots = tuple(
        (label, root)
        for label, root in (
            ("host_logs_dir", logs),
            ("host_workspace", workspace),
            ("host_tests_dir", tests),
            ("host_solution_dir", solution),
        )
        if root is not None
    )
    for index, (left_label, left) in enumerate(roots):
        for right_label, right in roots[index + 1:]:
            if (
                left == right
                or left.is_relative_to(right)
                or right.is_relative_to(left)
            ):
                raise InstanceContainerSetupError(
                    "deepswe_container: resolved bind-mount roots overlap: "
                    f"{left_label}={left} and {right_label}={right}"
                )
    return logs, workspace, tests, solution


def setup_instance_container(
    inst: "DeepSWEInstance",
    *,
    host_logs_dir: Path | None,
    host_workspace: Path | None = None,
    host_tests_dir: Path | None = None,
    host_solution_dir: Path | None = None,
    container_name: str | None = None,
    labels: Mapping[str, str] | None = None,
    register_container: ContainerRegistrationCallback | None = None,
    protect_git_metadata: bool = False,
    workspace_read_only: bool = False,
) -> str:
    """Materialize a per-task DeepSWE container. Returns the
    container id.

    Steps:

    1. ``docker pull <inst.docker_image>`` (cached after first call).
    2. ``docker run -d --rm --network=none [-v
       host_workspace:/app:rw] -v host_logs_dir:/logs:rw
       --cpus=<inst.cpus> --memory=<inst.memory_mb>m <image>
       sleep infinity``. ``sleep infinity`` keeps the container up
       across multiple ``docker exec`` calls; ``--rm`` makes a
       stopped container auto-remove.

    No install / pre_install step — DeepSWE images are dep-baked.

    Bind-mount notes (verified 2026-05-27 against
    ``adaptix-name-mapping-aliases`` upstream image): DeepSWE
    images only contain ``/app/`` (the repo cloned + checked out
    at base_commit_hash). The verifier shell (``/tests/test.sh``),
    its hidden test additions (``/tests/test.patch``), the gold
    solution (``/solution/solve.sh`` + ``/solution/solution.patch``),
    and the logs dir (``/logs/``) all live OUTSIDE the image —
    Pier bind-mounts them at runtime. Crucible reproduces the same
    bind-mounts here:

    - ``host_logs_dir`` is required for authoring and verification and omitted
      for patch renderers. When present it is bind-mounted at ``/logs:rw``;
      the verifier shell writes a v1.0 ``reward.txt`` or v1.1
      ``reward.json`` here. Crucible publishes the captured candidate at
      ``/logs/artifacts/model.patch`` for v1.1; legacy verifier scripts may
      regenerate the same bytes. The ``verifier/`` and ``artifacts/``
      subdirectories are pre-created on the host.
    - ``host_workspace`` is OPTIONAL. When set, bind-mounted at
      ``/app:rw``. The cascade passes a host worktree populated
      from the image's baked ``/app`` (via
      ``populate_workspace_from_image``); the verifier pilot
      passes None to use the image's baked ``/app`` directly.
    - ``host_tests_dir`` is OPTIONAL but REQUIRED for any verify
      step. When set, bind-mounted at ``/tests:ro``. Points at
      ``<deep-swe-clone>/tasks/<task_id>/tests/``. Skip when only
      doing a setup-smoke that doesn't run the verifier.
    - ``host_solution_dir`` is OPTIONAL and CASCADE-FORBIDDEN.
      When set, bind-mounted at ``/solution:ro``. ONLY the
      verifier pilot passes this — the cascade's implementer must
      not see the gold solution. Bind-mounting it during a
      cascade run would constitute gold-solution leakage.

    Raises ``InstanceContainerSetupError`` on any step failure.
    The caller is responsible for ``teardown_instance_container``
    in a finally block. When supplied, ``register_container`` runs
    synchronously after ``docker run`` returns a nonempty reference and before
    the readiness probe. A callback failure force-removes the container and
    aborts setup.
    """
    if not inst.docker_image:
        raise InstanceContainerSetupError(
            f"deepswe_container: task {inst.task_id!r} has empty "
            f"docker_image — won't run"
        )
    if inst.cpus <= 0 or inst.memory_mb <= 0 or inst.storage_mb <= 0:
        raise InstanceContainerSetupError(
            f"deepswe_container: task {inst.task_id!r} has invalid "
            "non-positive cpu/memory/storage limits"
        )
    (
        host_logs_dir,
        host_workspace,
        host_tests_dir,
        host_solution_dir,
    ) = _resolve_container_mount_roots(
        host_logs_dir=host_logs_dir,
        host_workspace=host_workspace,
        host_tests_dir=host_tests_dir,
        host_solution_dir=host_solution_dir,
        task_id=inst.task_id,
    )
    # Overlap rejection above must precede every filesystem write. Only then
    # materialize the host directories required by the hidden shell.
    if host_logs_dir is not None:
        (host_logs_dir / "verifier").mkdir(parents=True, exist_ok=True)
        (host_logs_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    # 1. Pull (cached on subsequent calls).
    immutable_image = pull_image(inst.docker_image)
    if labels is None:
        scope_root = host_logs_dir or host_workspace
        if scope_root is None:
            raise InstanceContainerSetupError(
                "deepswe_container: at least one host mount root is required"
            )
        try:
            scope = str(scope_root.parents[3])
        except IndexError:
            scope = str(scope_root.parent)
        labels = managed_container_labels(
            scope=scope,
            task_id=inst.task_id,
            phase="authoring" if protect_git_metadata else "verification",
        )
    container_labels = dict(labels)
    container_labels["io.crucible.image-digest"] = immutable_image

    # 2. Run detached.
    run_args: list[str] = [
        "run", "-d", "--rm",
        *label_args(container_labels),
        "--network=none",
        "--read-only",
        "--cap-drop", "ALL",
        # Ordinary benchmark commands run as Config.User and therefore have
        # no permitted/effective capabilities.  These three capabilities are
        # retained only so the root-owned dev-verify subreaper can drop its
        # child to Config.User and terminate escaped descendants.  With
        # no-new-privileges the non-root command cannot acquire them.
        "--cap-add", "SETUID",
        "--cap-add", "SETGID",
        "--cap-add", "KILL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", str(_PIDS_LIMIT),
        "--ulimit", "nofile=4096:4096",
        "--user", f"{os.getuid()}:{os.getgid()}",
        # Never inherit an image ENTRYPOINT: it would execute with mounted
        # capabilities before Crucible dispatches an explicit docker exec.
        "--entrypoint", "/bin/sleep",
        "--env", "HOME=/tmp/home",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--tmpfs", (
            "/tmp:rw,exec,nosuid,nodev,size="
            f"{min(max(inst.storage_mb // 4, 256), _MAX_TMPFS_MB)}m"
        ),
    ]
    if inst.language.strip().lower() == "rust":
        run_args.extend(rust_environment_args())
    if host_logs_dir is not None:
        run_args.extend(["-v", f"{host_logs_dir}:/logs:rw"])
    if host_workspace is not None:
        workspace_mode = "ro" if workspace_read_only else "rw"
        run_args.extend(["-v", f"{host_workspace}:/app:{workspace_mode}"])
        if protect_git_metadata:
            if workspace_read_only:
                raise InstanceContainerSetupError(
                    "deepswe_container: workspace_read_only and protected "
                    "Git overlay are mutually exclusive"
                )
            git_dir = host_workspace / ".git"
            if not git_dir.is_dir():
                raise InstanceContainerSetupError(
                    "deepswe_container: protected authoring workspace must "
                    "have a standalone .git directory"
                )
            run_args.extend(["-v", f"{git_dir}:/app/.git:ro"])
    if host_tests_dir is not None:
        run_args.extend(["-v", f"{host_tests_dir}:/tests:ro"])
    if host_solution_dir is not None:
        run_args.extend(["-v", f"{host_solution_dir}:/solution:ro"])
    if container_name:
        run_args.extend(["--name", container_name])
    # Resource caps are mandatory for untrusted benchmark execution.
    run_args.extend(["--cpus", str(inst.cpus)])
    run_args.extend(["--memory", f"{inst.memory_mb}m"])
    run_args.extend(["--memory-swap", f"{inst.memory_mb}m"])
    run_args.extend([immutable_image, "infinity"])

    result = _run_docker(run_args, step_name="run", timeout=60)
    container_id = (result.stdout or "").strip()
    if not container_id:
        raise InstanceContainerSetupError(
            f"deepswe_container: docker run returned empty container_id "
            f"(task {inst.task_id!r})"
        )
    _register_created_container(
        container_id,
        register_container=register_container,
        step_name="run",
    )
    try:
        readiness = _run_docker(
            ["inspect", "--format", "{{.State.Running}}", container_id],
            step_name="run_readiness",
            timeout=30,
        )
    except InstanceContainerSetupError as exc:
        raise InstanceContainerSetupError(
            "deepswe_container[run_readiness]: container exited or vanished "
            f"before readiness proof (task {inst.task_id!r}): {exc}"
        ) from exc
    running = (readiness.stdout or "").strip().lower()
    if running != "true":
        raise InstanceContainerSetupError(
            "deepswe_container[run_readiness]: container is not running "
            f"after docker run (state={running!r}, task {inst.task_id!r})"
        )
    if (
        inst.language.strip().lower() == "rust"
        # Patch rendering needs Git only. Avoid duplicating a large Rust
        # toolchain into tmpfs for that short, read-only container.
        and host_logs_dir is not None
    ):
        runtime_user = f"{os.getuid()}:{os.getgid()}"
        try:
            _bootstrap_rust_toolchain(
                container_id,
                runtime_user=runtime_user,
            )
            _probe_rust_toolchain(
                container_id,
                runtime_user=runtime_user,
            )
        except InstanceContainerSetupError as exc:
            _remove_failed_setup_container(container_id, cause=exc)
            raise RustToolchainUnavailable(str(exc)) from exc
    logger.info(
        "deepswe_container: ready at container %s (image=%s, task=%s)",
        container_id[:12], immutable_image, inst.task_id,
    )
    return container_id


def start_authoring_container(
    inst: "DeepSWEInstance",
    *,
    host_workspace: Path,
    host_logs_dir: Path,
    container_name: str | None = None,
    labels: Mapping[str, str] | None = None,
    register_container: ContainerRegistrationCallback | None = None,
) -> str:
    """Start the per-instance AUTHORING container for a DeepSWE cascade.

    This is the container the Implementer, Defender, and per-round
    breach ``dev_verify`` run against. By construction it mounts ONLY
    ``/app`` (the host worktree) and ``/logs`` — NEITHER ``/tests`` nor
    ``/solution``. Leakage is structural, not policy: the Implementer
    cannot ``cat /tests/test.patch`` because that path does not exist
    inside the container. The bind-mount surface, not a prompt or wrapper,
    is what keeps hidden tests and
    the gold solution out of authoring.

    ``host_workspace`` is REQUIRED here (unlike the underlying
    ``setup_instance_container`` where it is optional): an authoring
    container with no host worktree has nothing for the Implementer to
    diff against. The caller populates it via
    ``populate_workspace_from_image`` first.

    Thin wrapper over ``setup_instance_container`` that simply omits
    ``host_tests_dir`` / ``host_solution_dir`` — there is no parameter
    to pass them, so they cannot be mounted by accident.
    """
    return setup_instance_container(
        inst,
        host_logs_dir=host_logs_dir,
        host_workspace=host_workspace,
        host_tests_dir=None,
        host_solution_dir=None,
        container_name=container_name,
        labels=labels,
        register_container=register_container,
        protect_git_metadata=True,
        workspace_read_only=False,
    )


def start_verifier_container(
    inst: "DeepSWEInstance",
    *,
    host_workspace: Path,
    host_logs_dir: Path,
    host_tests_dir: Path,
    container_name: str | None = None,
    labels: Mapping[str, str] | None = None,
    register_container: ContainerRegistrationCallback | None = None,
) -> str:
    """Start the per-instance VERIFIER container for the final pass.

    Mounts ``/app``, ``/logs``, and ``/tests:ro`` — the dataset's
    hidden test suite is now visible so ``bash /tests/test.sh`` can run
    its Step 1 reset + Step 2 hidden-patch apply. ``/solution`` is
    STILL never mounted: the verifier shell grades against hidden tests
    alone, and the cascade never consults the gold solution.

    ``host_tests_dir`` is REQUIRED (no default) — a verifier container
    without ``/tests`` cannot run the dataset gate, so making it
    mandatory turns a silent misconfiguration into a TypeError at the
    call site.

    Thin wrapper over ``setup_instance_container``; ``host_solution_dir``
    is hard-wired to None and has no parameter to override it.
    """
    return setup_instance_container(
        inst,
        host_logs_dir=host_logs_dir,
        host_workspace=host_workspace,
        host_tests_dir=host_tests_dir,
        host_solution_dir=None,
        container_name=container_name,
        labels=labels,
        register_container=register_container,
        protect_git_metadata=False,
        workspace_read_only=False,
    )


def start_patch_render_container(
    inst: "DeepSWEInstance",
    *,
    host_workspace: Path,
    container_name: str | None = None,
    labels: Mapping[str, str] | None = None,
    register_container: ContainerRegistrationCallback | None = None,
) -> str:
    """Start a pipeline-only patch renderer with no hidden mounts.

    The checkout itself is mounted read-only. Rendering uses a temporary Git
    index/object store in ``/tmp``, so image Git/config behavior cannot mutate
    the exact workspace later passed to the verifier. It still mounts neither
    ``/tests`` nor ``/solution``.
    """

    return setup_instance_container(
        inst,
        host_logs_dir=None,
        host_workspace=host_workspace,
        host_tests_dir=None,
        host_solution_dir=None,
        container_name=container_name,
        labels=labels,
        register_container=register_container,
        protect_git_metadata=False,
        workspace_read_only=True,
    )


def container_immutable_image(container_id: str) -> str:
    """Read the repository digest used to create a managed container."""

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
    """Reclaim managed containers whose same-host owner process died."""

    return reconcile_orphaned_containers(scope=scope, run_docker=_run_docker)


def stop_instance_container_strict(container_id: str) -> None:
    """Stop ``container_id`` and prove it is no longer running.

    This is the security boundary between model-controlled authoring and
    candidate capture / hidden verification.  Unlike final best-effort
    cleanup, a failure here must abort the transition while the caller still
    retains the container id for a later cleanup attempt.

    Managed containers use ``--rm``, so a successful stop normally removes
    the object before the follow-up inspect.  Either an explicit
    ``State.Running=false`` or Docker's authoritative "no such object"
    response proves quiescence; every other result fails closed.
    """

    if not container_id:
        raise InstanceContainerSetupError(
            "deepswe_container[strict_stop]: empty container id"
        )
    try:
        _run_docker(
            ["stop", container_id],
            step_name="strict_stop",
            timeout=30,
        )
    except InstanceContainerSetupError as exc:
        message = str(exc).lower()
        if "no such object" in message or "no such container" in message:
            return
        raise
    try:
        result = _run_docker(
            ["inspect", "--format", "{{.State.Running}}", container_id],
            step_name="strict_stop_inspect",
            timeout=30,
        )
    except InstanceContainerSetupError as exc:
        message = str(exc).lower()
        if "no such object" in message or "no such container" in message:
            return
        raise
    running = (result.stdout or "").strip().lower()
    if running != "false":
        raise InstanceContainerSetupError(
            "deepswe_container[strict_stop_inspect]: container "
            f"{container_id[:12]} remains running (state={running!r})"
        )


def capture_model_patch_from_container(
    container_id: str,
    *,
    base_commit: str,
    max_bytes: int = DEFAULT_MAX_PATCH_BYTES,
) -> bytes:
    """Render the verifier workspace patch inside its immutable image.

    The hidden ``test.sh`` renders ``model.patch`` with the Git binary and
    configuration in the task image.  Rendering the pre-verification
    candidate in that same container avoids host/image Git-version byte
    differences while preserving an exact byte comparison at the oracle
    boundary. The live index/worktree remain read-only; temporary index and
    object data live only on the container's ``/tmp`` tmpfs.
    """

    git_wrapper = hermetic_container_git_shell_function()
    script = (
        "set -eu\n"
        f"{git_wrapper}\n"
        "tmp=$(/usr/bin/mktemp -d /tmp/deepswe-render.XXXXXX); "
        "trap '/bin/rm -rf \"$tmp\"' EXIT; "
        "objects=$(git rev-parse --path-format=absolute --git-path objects); "
        "/bin/mkdir -p \"$tmp/objects\"; "
        "export GIT_INDEX_FILE=\"$tmp/index\"; "
        "export GIT_OBJECT_DIRECTORY=\"$tmp/objects\"; "
        "export GIT_ALTERNATE_OBJECT_DIRECTORIES=\"$objects\"; "
        "export GIT_OPTIONAL_LOCKS=0; "
        "git read-tree \"$1\"; "
        "git add -A -- .; "
        "git diff --cached --binary \"$1\""
    )
    limit = validate_patch_byte_limit(max_bytes)
    try:
        inherited_environment = inspect_docker_container_environment(
            container_id
        )
        exec_prefix = hermetic_container_git_exec_prefix(
            inherited_environment
        )
    except ContainerGitEnvironmentError as exc:
        raise InstanceContainerSetupError(
            f"deepswe_container[render_model_patch]: {exc}"
        ) from exc
    args = [
        "docker",
        "exec",
        "--workdir",
        "/app",
        container_id,
        *exec_prefix,
        "/bin/sh",
        "-c",
        script,
        "deepswe-render-patch",
        base_commit,
    ]
    logger.info(
        "deepswe_container[render_model_patch]: docker exec %s",
        container_id[:12],
    )
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    patch = bytearray()
    stderr_tail = bytearray()
    deadline = time.monotonic() + 120
    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(args, 120)
            events = selector.select(timeout=min(remaining, 1.0))
            if not events:
                continue
            for key, _mask in events:
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if key.data == "stdout":
                    if len(patch) + len(chunk) > limit:
                        raise InstanceContainerSetupError(
                            "deepswe_container[render_model_patch]: output "
                            f"exceeds patch byte limit {limit}"
                        )
                    patch.extend(chunk)
                else:
                    stderr_tail.extend(chunk)
                    if len(stderr_tail) > 2_000:
                        del stderr_tail[:-2_000]
        remaining = max(0.0, deadline - time.monotonic())
        returncode = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        if process is not None:
            process.kill()
            process.wait()
        raise InstanceContainerSetupError(
            "deepswe_container[render_model_patch]: timed out after 120s"
        ) from exc
    except FileNotFoundError as exc:
        raise InstanceContainerSetupError(
            "deepswe_container[render_model_patch]: docker binary not found"
        ) from exc
    except InstanceContainerSetupError:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        selector.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
    if returncode != 0:
        stderr = bytes(stderr_tail).decode("utf-8", errors="replace")[-2_000:]
        raise InstanceContainerSetupError(
            "deepswe_container[render_model_patch]: docker exited "
            f"rc={returncode}; stderr: {stderr}"
        )
    return bytes(patch)


def teardown_instance_container(
    container_id: str | None,
    image: str | None = None,
) -> None:
    """Best-effort container cleanup. Idempotent — safe to call when
    setup failed mid-flight or twice in a row.

    Steps:

    1. ``docker stop <container_id>`` if container_id is non-None.
       ``--rm`` from setup makes the container auto-remove on stop.
    2. ``docker image rm <image>`` if image is non-None AND the image
       isn't otherwise pinned. Reclaims the ~1-3 GB per task between
       runs; without this, a 113-task sweep can fill ~100-300 GB
       even with the disk-headroom guard.

    Errors are logged at WARN, not raised — teardown failure
    shouldn't mask the original cascade outcome.
    """
    if container_id:
        try:
            _run_docker(
                ["stop", container_id],
                step_name="stop",
                timeout=30,
            )
        except InstanceContainerSetupError as e:
            logger.warning(
                "deepswe_container: stop failed for %s: %s — "
                "container may need manual cleanup",
                container_id[:12], e,
            )

    if image:
        try:
            _run_docker(
                ["image", "rm", image],
                step_name="image_rm",
                timeout=60,
            )
        except InstanceContainerSetupError as e:
            # Image may still be in use by another container, or
            # may have already been removed. Log + continue — the
            # next pull will re-cache if needed.
            logger.info(
                "deepswe_container: image_rm failed for %s: %s — "
                "image may still be referenced or already removed",
                image, e,
            )


__all__ = [
    "InstanceContainerSetupError",
    "RustToolchainUnavailable",
    "check_disk_headroom",
    "container_immutable_image",
    "docker_available",
    "is_full_commit_id",
    "populate_workspace_from_image",
    "pull_image",
    "reconcile_orphans",
    "setup_instance_container",
    "capture_model_patch_from_container",
    "start_authoring_container",
    "start_patch_render_container",
    "start_verifier_container",
    "stop_instance_container_strict",
    "teardown_instance_container",
]
