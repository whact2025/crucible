# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``crucible.products.swebench.instance_container``.

Covers the offline portions:
- arch detection / image-tag formatting
- instance-id substitution (``__`` -> ``_1776_``)
- spec resolution (known + unknown repo/version)
- docker_available probe (calls docker — needs daemon)

Does NOT exercise ``docker pull`` / ``docker run`` end-to-end; those
require network + run for ~minutes per image. The integration smoke
in workflows/swebench.py is the load-bearing E2E test.
"""

from __future__ import annotations

from types import SimpleNamespace
import json
import subprocess

import pytest

from crucible.products.swebench import instance_container as _ic
from crucible.products.swebench.instance_container import (
    InstanceContainerSetupError,
    _docker_arch,
    _remove_image_if_unpinned,
    _translate_instance_id,
    docker_available,
    resolve_container_spec,
    teardown_instance_container,
)


# ----------------------------------------------------------------------
# instance-id translation
# ----------------------------------------------------------------------


def test_translate_instance_id_double_underscore_to_1776():
    """The harness's URL-safe substitution is the canonical mapping
    from dataset instance_id to image-tag instance_id."""
    assert (
        _translate_instance_id("astropy__astropy-12907")
        == "astropy_1776_astropy-12907"
    )
    assert (
        _translate_instance_id("django__django-12345")
        == "django_1776_django-12345"
    )


def test_translate_instance_id_no_change_when_no_double_underscore():
    """Single underscores pass through unchanged."""
    assert _translate_instance_id("simple_repo-1") == "simple_repo-1"


# ----------------------------------------------------------------------
# arch normalization
# ----------------------------------------------------------------------


def test_docker_arch_returns_a_string():
    """Whatever the machine reports, _docker_arch returns a string the
    image tag can interpolate. Concrete value depends on the test host."""
    arch = _docker_arch()
    assert isinstance(arch, str)
    assert len(arch) > 0


# ----------------------------------------------------------------------
# spec resolution
# ----------------------------------------------------------------------


def test_resolve_container_spec_known_astropy():
    """astropy/astropy@5.0 is in MAP_REPO_VERSION_TO_SPECS — should
    resolve to a real image tag + the install command from the spec."""
    inst = SimpleNamespace(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        version="4.3",
    )
    spec = resolve_container_spec(inst)
    assert spec is not None
    assert spec.image.startswith("swebench/sweb.eval.")
    assert "astropy_1776_astropy-12907" in spec.image
    assert spec.image.endswith(":latest")
    # The install command is from the harness's spec mapping — pip-
    # editable shape is universal.
    assert "pip install" in spec.install_command


def test_resolve_container_spec_unknown_repo_returns_none():
    inst = SimpleNamespace(
        instance_id="x__x-1",
        repo="does-not-exist/repo",
        version="1.0",
    )
    assert resolve_container_spec(inst) is None


def test_resolve_container_spec_missing_version_returns_none():
    inst = SimpleNamespace(
        instance_id="astropy__astropy-12907",
        repo="astropy/astropy",
        version="99.99.99-fake",
    )
    assert resolve_container_spec(inst) is None


# ----------------------------------------------------------------------
# docker_available probe
# ----------------------------------------------------------------------


@pytest.mark.live_docker
@pytest.mark.skipif(not docker_available(), reason="Docker daemon unavailable")
def test_docker_available_when_daemon_running():
    """On the production target host (perflab1), docker is up.
    Asserts the probe returns True in that environment. On hosts
    without docker, this test exists to document the fallback path —
    but it would fail (the probe just reports reality). If you're
    running tests in CI without docker, skip via pytest's collection
    filter rather than weakening the assertion."""
    assert docker_available() is True


# ----------------------------------------------------------------------
# teardown idempotency
# ----------------------------------------------------------------------


def test_teardown_handles_none_id():
    """Caller may pass None when setup never ran. Must not raise."""
    teardown_instance_container(None)
    teardown_instance_container("")


def test_setup_runs_immutable_image_with_readonly_git_and_swap_cap(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    (workspace / ".git").mkdir(parents=True)
    image = "registry/repo:latest"
    digest = "registry/repo@sha256:" + "a" * 64
    spec = _ic.InstanceContainerSpec(image=image, install_command="true")
    calls: list[list[str]] = []

    def _docker(args, **kwargs):
        calls.append(args)
        if args[:2] == ["image", "inspect"]:
            stdout = json.dumps([digest])
        elif args[:3] == ["run", "-d", "--rm"]:
            stdout = "container-id\n"
        else:
            stdout = ""
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=stdout, stderr="",
        )

    monkeypatch.setattr(_ic, "_run_docker", _docker)
    monkeypatch.setattr(_ic, "_revert_workspace", lambda path: None)

    assert _ic.setup_instance_container(
        workspace,
        "managed-container",
        spec,
        labels={"io.crucible.managed": "true"},
    ) == "container-id"
    run = next(args for args in calls if args[:3] == ["run", "-d", "--rm"])
    assert digest in run
    assert ["--memory-swap", "8192m"] == run[
        run.index("--memory-swap"):run.index("--memory-swap") + 2
    ]
    assert f"{workspace / '.git'}:/testbed/.git:ro" in run
    assert "io.crucible.managed=true" in run


def test_teardown_tolerates_missing_container(monkeypatch):
    """A teardown called for an already-stopped or never-existed
    container logs a warning but doesn't raise — the cascade's
    finally block must always make progress."""
    def _missing(args, *, step_name, timeout):
        raise InstanceContainerSetupError(
            "instance_container[stop]: rc=1\n"
            "stderr: No such container: nonexistent_container_id_for_test"
        )

    monkeypatch.setattr(_ic, "_run_docker", _missing)
    teardown_instance_container("nonexistent_container_id_for_test")


# ----------------------------------------------------------------------
# Image cleanup — keep disk bounded for long sweeps
# ----------------------------------------------------------------------


def test_remove_image_skipped_when_keep_images_env_set(monkeypatch):
    """``CRUCIBLE_SWEBENCH_KEEP_IMAGES=1`` short-circuits before any
    docker call — operators iterating on a single instance don't pay
    the re-pull cost on the next run."""
    monkeypatch.setenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", "1")
    calls: list[list[str]] = []

    def _spy(args, **kwargs):
        calls.append(args)
        raise AssertionError("docker should not be invoked when opt-out set")

    monkeypatch.setattr(_ic, "_run_docker", _spy)
    _remove_image_if_unpinned("swebench/sweb.eval.x86_64.fake_1776_fake-1:latest")
    assert calls == []


@pytest.mark.parametrize("value", ["true", "YES", "On", "1"])
def test_remove_image_keep_env_accepts_truthy_variants(monkeypatch, value):
    """Mirrors ``CRUCIBLE_SWEBENCH_SKIP_DISK_CHECK``'s case-insensitive
    truthy parsing so the convention is uniform."""
    monkeypatch.setenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", value)
    monkeypatch.setattr(
        _ic, "_run_docker",
        lambda *a, **k: pytest.fail("docker should not be invoked"),
    )
    _remove_image_if_unpinned("ignored:latest")


def test_remove_image_runs_docker_image_rm_by_default(monkeypatch):
    """No opt-out set → ``docker image rm <image>`` is invoked."""
    monkeypatch.delenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", raising=False)
    calls: list[list[str]] = []

    def _spy(args, *, step_name, timeout):
        calls.append(args)
        # Mimic the success path — _run_docker returns a CompletedProcess
        # but the caller doesn't inspect it; None is fine for the spy.
        return None

    monkeypatch.setattr(_ic, "_run_docker", _spy)
    _remove_image_if_unpinned("swebench/sweb.eval.x86_64.fake_1776_fake-1:latest")
    assert calls == [["image", "rm", "swebench/sweb.eval.x86_64.fake_1776_fake-1:latest"]]


def test_remove_image_swallows_in_use_error(monkeypatch):
    """A concurrent worker against the same image makes ``docker
    image rm`` fail with ``image is in use``. The cascade continues —
    we log and move on rather than crashing the sweep."""
    monkeypatch.delenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", raising=False)

    def _spy(args, *, step_name, timeout):
        raise InstanceContainerSetupError(
            "instance_container[image_rm]: rc=1\n"
            "cmd: docker image rm fake:latest\n"
            "stderr: Error: image is in use by stopped container"
        )

    monkeypatch.setattr(_ic, "_run_docker", _spy)
    # Must not raise.
    _remove_image_if_unpinned("fake:latest")


def test_teardown_with_image_calls_image_rm(monkeypatch):
    """The full teardown path: ``container_id`` + ``image`` → stop +
    image rm. Verifies the image arg is threaded through end-to-end."""
    monkeypatch.delenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", raising=False)
    calls: list[list[str]] = []

    def _spy(args, *, step_name, timeout):
        calls.append(args)
        return None

    monkeypatch.setattr(_ic, "_run_docker", _spy)
    teardown_instance_container(
        "fake_container_id",
        image="swebench/sweb.eval.x86_64.fake_1776_fake-1:latest",
    )
    assert calls == [
        ["stop", "fake_container_id"],
        ["image", "rm", "swebench/sweb.eval.x86_64.fake_1776_fake-1:latest"],
    ]


def test_teardown_without_image_skips_image_rm(monkeypatch):
    """When no image is requested, teardown stops only the container."""
    monkeypatch.delenv("CRUCIBLE_SWEBENCH_KEEP_IMAGES", raising=False)
    calls: list[list[str]] = []

    def _spy(args, *, step_name, timeout):
        calls.append(args)
        return None

    monkeypatch.setattr(_ic, "_run_docker", _spy)
    teardown_instance_container("fake_container_id")
    assert calls == [["stop", "fake_container_id"]]


# ----------------------------------------------------------------------
# Offline baked-workspace seeding
# ----------------------------------------------------------------------


def test_setup_rejects_linked_worktree_before_pull(tmp_path, monkeypatch):
    workspace = tmp_path / "linked"
    workspace.mkdir()
    (workspace / ".git").write_text(
        "gitdir: /cache/objects/worktrees/linked\n"
    )
    pulled: list[str] = []
    monkeypatch.setattr(_ic, "pull_image", pulled.append)

    with pytest.raises(InstanceContainerSetupError, match="standalone .git"):
        _ic.setup_instance_container(
            workspace,
            "forbidden-linked-worktree",
            _ic.InstanceContainerSpec(
                image="fake/image:latest", install_command="true",
            ),
        )

    assert pulled == []


def test_setup_seeds_baked_workspace_without_network_or_install(monkeypatch):
    """The image is the offline dependency capsule.

    Setup copies its prebuilt /testbed artifacts as the host uid, excludes
    git metadata, and never executes dataset pre_install/install commands.
    """
    captured: list[tuple[str, list[str]]] = []

    def fake_run_docker(args, *, step_name, timeout=None):
        captured.append((step_name, list(args)))
        class _R:
            returncode = 0
            stdout = "fake_container_id_xyz\n"
        return _R()

    monkeypatch.setattr(_ic, "_run_docker", fake_run_docker)
    monkeypatch.setattr(_ic, "pull_image", lambda _: None)
    monkeypatch.setattr(_ic, "_revert_workspace", lambda _: None)

    spec = _ic.InstanceContainerSpec(
        image="fake/img:latest",
        install_command="python -m pip install -e .",
        pre_install=("wget https://example.invalid/dependency",),
    )
    # Tempdir as workspace.
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td)
        (ws / ".git").mkdir()
        _ic.setup_instance_container(ws, "fake-container", spec)

    seed_args = next(
        args for step, args in captured if step == "seed baked workspace"
    )
    assert seed_args[seed_args.index("--network") + 1] == "none"
    assert "--user" in seed_args
    assert not any(
        isinstance(arg, str) and arg.startswith("nproc=") for arg in seed_args
    )
    assert "tar --exclude=.git" in seed_args[-1]
    assert "--no-same-owner" in seed_args[-1]

    run_args = next(args for step, args in captured if step == "run")
    assert run_args[run_args.index("--network") + 1] == "none"
    assert "--read-only" in run_args
    assert run_args[run_args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in run_args
    assert "--pids-limit" in run_args
    assert not any(
        isinstance(arg, str) and arg.startswith("nproc=") for arg in run_args
    )
    assert "--cpus" in run_args
    assert "--memory" in run_args
    assert "--user" in run_args
    assert "--tmpfs" in run_args
    assert [
        run_args[index + 1]
        for index, value in enumerate(run_args)
        if value == "--cap-add"
    ] == ["SETUID", "SETGID", "KILL"]
    assert run_args[run_args.index("--entrypoint") + 1] == "/bin/sleep"
    assert run_args[-1] == "infinity"
    assert not any(
        step.startswith("pre_install") or step == "install"
        for step, _args in captured
    )
