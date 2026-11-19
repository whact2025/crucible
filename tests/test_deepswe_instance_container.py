# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.deepswe.instance_container``.

Hermetic — uses ``monkeypatch`` against ``subprocess.run`` to fake
docker CLI invocations, and ``tmp_path`` for the host logs /
workspace dirs. No real docker daemon required.

Coverage:
- docker_available: True / False / not-on-PATH
- check_disk_headroom: pass / fail / env-var override
- pull_image: happy / failure
- setup_instance_container: happy path, resource limits, optional
  host_workspace bind, empty image fail, empty container_id fail
- teardown_instance_container: idempotent (None args, stop+rm,
  stop fails, image_rm fails)
- _run_docker error handling: non-zero rc, TimeoutExpired,
  FileNotFoundError (docker missing)
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from crucible.products.deepswe import instance_container as ic
from crucible.products.deepswe.dataset import DeepSWEInstance


@pytest.fixture(autouse=True)
def _isolate_disk_check_override(monkeypatch):
    """CI's host-protection override must not mask disk-check unit tests."""
    monkeypatch.delenv("CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK", raising=False)
    monkeypatch.setattr(
        ic,
        "resolve_immutable_image",
        lambda image, **kwargs: (
            image.split(":", 1)[0] + "@sha256:" + "a" * 64
        ),
    )


def _instance(
    *,
    task_id: str = "task-1",
    docker_image: str = "public.ecr.aws/d3j8x8q7/swe-bench-202605:extid",
    cpus: int = 2,
    memory_mb: int = 8192,
    language: str = "python",
) -> DeepSWEInstance:
    """Minimal DeepSWEInstance for container tests."""
    return DeepSWEInstance(
        task_id=task_id,
        ext_id="extid",
        repo="ex/ex",
        repository_url="https://github.com/ex/ex",
        language=language,
        category="feature_request",
        display_title="Title",
        display_description="Desc",
        base_commit_hash="cafebabe" * 5,
        docker_image=docker_image,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=20480,
        build_timeout_sec=1800.0,
        allow_internet=False,
        verifier_timeout_sec=1800.0,
        agent_timeout_sec=5400.0,
        instruction="instruction body",
    )


def _fake_subprocess(rc: int = 0, stdout: str = "", stderr: str = ""):
    """Build a fake subprocess.run that records the docker invocations
    it sees and returns a configurable CompletedProcess."""
    calls: list[list[str]] = []

    def _fake(args, **kwargs):
        calls.append(list(args))
        effective_stdout = stdout
        if (
            len(args) >= 5
            and args[:2] == ["docker", "inspect"]
            and "{{.State.Running}}" in args
            and rc == 0
        ):
            effective_stdout = "true\n"
        return subprocess.CompletedProcess(
            args=args, returncode=rc, stdout=effective_stdout, stderr=stderr,
        )

    _fake.calls = calls  # type: ignore[attr-defined]
    return _fake


# ---------- docker_available ----------


def test_docker_available_true(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_subprocess(rc=0, stdout="25.0.0"))
    assert ic.docker_available() is True


def test_docker_available_false_on_rc(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_subprocess(rc=1))
    assert ic.docker_available() is False


def test_docker_available_false_on_missing_binary(monkeypatch):
    def _enoent(args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")
    monkeypatch.setattr(subprocess, "run", _enoent)
    assert ic.docker_available() is False


def test_docker_available_false_on_timeout(monkeypatch):
    def _timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))
    monkeypatch.setattr(subprocess, "run", _timeout)
    assert ic.docker_available() is False


# ---------- check_disk_headroom ----------


def test_check_disk_headroom_above_threshold(monkeypatch):
    """Plenty of disk → pass silently."""
    class _Usage:
        free = 100 * (1024 ** 3)  # 100 GB
    monkeypatch.setattr("crucible.products.deepswe.instance_container._disk_check_path",
                        lambda: "/")
    monkeypatch.setattr("crucible.products.deepswe.instance_container.shutil.disk_usage",
                        lambda _: _Usage())
    ic.check_disk_headroom(min_gb=10.0)  # no raise


def test_check_disk_headroom_below_threshold(monkeypatch):
    """Free disk below floor → raise InstanceContainerSetupError
    with reclaim guidance."""
    class _Usage:
        free = 5 * (1024 ** 3)  # 5 GB
    monkeypatch.setattr("crucible.products.deepswe.instance_container._disk_check_path",
                        lambda: "/")
    monkeypatch.setattr("crucible.products.deepswe.instance_container.shutil.disk_usage",
                        lambda _: _Usage())
    with pytest.raises(ic.InstanceContainerSetupError,
                       match=r"free disk 5\.0 GB on / is below"):
        ic.check_disk_headroom(min_gb=10.0)


def test_check_disk_headroom_env_var_override(monkeypatch):
    """CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK=1 silently skips the
    check even when below threshold."""
    class _Usage:
        free = 1 * (1024 ** 3)  # 1 GB — way below
    monkeypatch.setattr("crucible.products.deepswe.instance_container._disk_check_path",
                        lambda: "/")
    monkeypatch.setattr("crucible.products.deepswe.instance_container.shutil.disk_usage",
                        lambda _: _Usage())
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK", "1")
    ic.check_disk_headroom(min_gb=10.0)  # no raise


def test_check_disk_headroom_env_var_zero_keeps_check_on(monkeypatch):
    """=0 must NOT bypass the check (operator intuition trap —
    bare-presence truthiness would silently disable on =0)."""
    class _Usage:
        free = 1 * (1024 ** 3)
    monkeypatch.setattr("crucible.products.deepswe.instance_container._disk_check_path",
                        lambda: "/")
    monkeypatch.setattr("crucible.products.deepswe.instance_container.shutil.disk_usage",
                        lambda _: _Usage())
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_SKIP_DISK_CHECK", "0")
    with pytest.raises(ic.InstanceContainerSetupError):
        ic.check_disk_headroom(min_gb=10.0)


def test_disk_check_path_env_override(monkeypatch):
    """CRUCIBLE_DEEPSWE_DISK_CHECK_PATH wins over docker info / default."""
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_DISK_CHECK_PATH", "/data")
    assert ic._disk_check_path() == "/data"


def test_disk_check_path_falls_back_to_root_on_docker_failure(monkeypatch):
    """No env override + docker info failing → conservative ``/``."""
    monkeypatch.delenv("CRUCIBLE_DEEPSWE_DISK_CHECK_PATH", raising=False)

    def _boom(*a, **k):
        raise OSError("docker missing")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert ic._disk_check_path() == "/"


# ---------- pull_image ----------


def test_pull_image_happy(monkeypatch):
    fake = _fake_subprocess(rc=0)
    monkeypatch.setattr(subprocess, "run", fake)
    ic.pull_image("public.ecr.aws/d3j8x8q7/swe-bench-202605:abc")
    assert fake.calls == [
        ["docker", "pull", "public.ecr.aws/d3j8x8q7/swe-bench-202605:abc"]
    ]


def test_pull_image_failure(monkeypatch):
    fake = _fake_subprocess(rc=1, stderr="manifest unknown")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(ic.InstanceContainerSetupError,
                       match=r"deepswe_container\[pull\].*manifest unknown"):
        ic.pull_image("doesnotexist:latest")


# ---------- setup_instance_container ----------


def test_setup_happy_path(tmp_path: Path, monkeypatch):
    """Happy path: pull succeeds, run returns container_id. Verify
    the docker invocations carry --network=none, the logs bind-
    mount, and resource limits."""
    fake = _fake_subprocess(rc=0, stdout="abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)

    container_id = ic.setup_instance_container(
        _instance(),
        host_logs_dir=tmp_path / "logs",
    )
    assert container_id == "abc123"
    assert len(fake.calls) == 3  # pull + run + readiness inspect

    # First call: pull
    assert fake.calls[0][0:2] == ["docker", "pull"]
    # Second call: run
    run_args = fake.calls[1]
    assert run_args[0] == "docker"
    assert "run" in run_args
    assert "--network=none" in run_args
    assert "--read-only" in run_args
    assert run_args[run_args.index("--cap-drop") + 1] == "ALL"
    retained_capabilities = {
        run_args[index + 1]
        for index, argument in enumerate(run_args[:-1])
        if argument == "--cap-add"
    }
    assert retained_capabilities == {"SETUID", "SETGID", "KILL"}
    assert "no-new-privileges" in run_args
    assert "--pids-limit" in run_args
    assert "--ulimit" in run_args
    assert not any("nproc=" in argument for argument in run_args)
    assert "nofile=4096:4096" in run_args
    assert "--user" in run_args
    assert "--tmpfs" in run_args
    assert run_args[run_args.index("--entrypoint") + 1] == "/bin/sleep"
    assert "sleep" not in run_args
    assert run_args[-1] == "infinity"
    # Logs bind-mount
    assert any("-v" == a and ":/logs:rw" in run_args[i + 1]
               for i, a in enumerate(run_args[:-1]))
    # No /app bind-mount in this test
    assert not any(":/app:rw" in a for a in run_args)
    # Resource limits
    assert "--cpus" in run_args and "2" in run_args
    assert "--memory" in run_args and "8192m" in run_args


def test_setup_registers_run_before_readiness_via_authoring_wrapper(
    tmp_path: Path,
    monkeypatch,
):
    """The daemon reference is registered before any readiness-side work."""

    events: list[tuple[str, str]] = []

    def _fake(args, **kwargs):  # noqa: ARG001
        action = str(args[1])
        events.append(("docker", action))
        stdout = ""
        if action == "run":
            stdout = "authoring-abc123\n"
        elif action == "inspect":
            stdout = "true\n"
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    def _register(reference: str) -> None:
        events.append(("register", reference))

    monkeypatch.setattr(subprocess, "run", _fake)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    assert ic.start_authoring_container(
        _instance(),
        host_workspace=workspace,
        host_logs_dir=tmp_path / "logs",
        register_container=_register,
    ) == "authoring-abc123"

    assert events == [
        ("docker", "pull"),
        ("docker", "run"),
        ("register", "authoring-abc123"),
        ("docker", "inspect"),
    ]


def test_setup_registration_failure_force_removes_before_readiness(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[list[str]] = []

    def _fake(args, **kwargs):  # noqa: ARG001
        call = list(args)
        calls.append(call)
        stdout = "run-abc123\n" if call[1] == "run" else ""
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    def _registration_failure(_reference: str) -> None:
        raise RuntimeError("registration journal unavailable")

    monkeypatch.setattr(subprocess, "run", _fake)

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="registration journal unavailable",
    ):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=tmp_path / "logs",
            register_container=_registration_failure,
        )

    assert calls[-1] == ["docker", "rm", "-f", "run-abc123"]
    assert not any(call[1] == "inspect" for call in calls)


def test_setup_registration_and_cleanup_failures_are_both_preserved(
    tmp_path: Path,
    monkeypatch,
):
    def _fake(args, **kwargs):  # noqa: ARG001
        action = args[1]
        if action == "run":
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="run-abc123\n",
                stderr="",
            )
        if action == "rm":
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="daemon cleanup denied",
            )
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        )

    def _registration_failure(_reference: str) -> None:
        raise RuntimeError("registration write failed")

    monkeypatch.setattr(subprocess, "run", _fake)

    with pytest.raises(ic.InstanceContainerSetupError) as exc_info:
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=tmp_path / "logs",
            register_container=_registration_failure,
        )

    message = str(exc_info.value)
    assert "registration write failed" in message
    assert "force-remove cleanup also failed" in message
    assert "daemon cleanup denied" in message
    cleanup_error = exc_info.value.__cause__
    assert isinstance(cleanup_error, ic.InstanceContainerSetupError)
    assert isinstance(cleanup_error.__context__, RuntimeError)
    assert str(cleanup_error.__context__) == "registration write failed"


def test_setup_with_host_workspace(tmp_path: Path, monkeypatch):
    """``host_workspace`` set → /app bind-mount appears in docker
    run args."""
    fake = _fake_subprocess(rc=0, stdout="abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()
    logs = tmp_path / "logs"

    ic.setup_instance_container(
        _instance(),
        host_logs_dir=logs,
        host_workspace=ws,
    )
    run_args = fake.calls[1]
    assert any(":/app:rw" in a for a in run_args)
    assert any(":/logs:rw" in a for a in run_args)
    assert "--memory-swap" in run_args
    assert "io.crucible.managed=true" in run_args


def test_rust_setup_stages_nonroot_toolchain_and_probes_exact_environment(
    tmp_path: Path,
    monkeypatch,
):
    fake = _fake_subprocess(rc=0, stdout="abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)

    container_id = ic.setup_instance_container(
        _instance(language="rust"),
        host_logs_dir=tmp_path / "logs",
    )

    assert container_id == "abc123"
    run_args = fake.calls[1]
    assert "CARGO_HOME=/tmp/crucible-cargo-home" in run_args
    assert "RUSTUP_HOME=/tmp/crucible-rust-immutable/rustup" in run_args
    assert "CARGO_TARGET_DIR=/app/target" in run_args
    assert any(
        argument.startswith(
            "PATH=/tmp/crucible-rust-immutable/cargo-bin:"
        )
        for argument in run_args
    )
    assert run_args[run_args.index("--user") + 1] == (
        f"{ic.os.getuid()}:{ic.os.getgid()}"
    )

    immutable = next(
        call for call in fake.calls
        if "test -x /root/.cargo/bin/cargo" in call[-1]
    )
    assert immutable[1:4] == ["exec", "--user", "0:0"]
    assert "/root/.cargo" in immutable[-1]
    assert "/root/.rustup" in immutable[-1]
    assert "/root/.ssh" not in immutable[-1]
    mutable = next(
        call for call in fake.calls
        if "--no-preserve=ownership,mode" in call[-1]
    )
    runtime_user = f"{ic.os.getuid()}:{ic.os.getgid()}"
    assert mutable[1:4] == ["exec", "--user", runtime_user]
    probe = next(
        call for call in fake.calls if "cargo --version" in call[-1]
    )
    assert probe[1:4] == ["exec", "--user", runtime_user]
    assert "CARGO_HOME=/tmp/crucible-cargo-home" in probe
    assert "CARGO_TARGET_DIR=/app/target" in probe


def test_rust_toolchain_probe_failure_removes_started_container(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[list[str]] = []

    def fake(args, **kwargs):  # noqa: ARG001
        call = list(args)
        calls.append(call)
        if (
            call[:2] == ["docker", "inspect"]
            and "{{.State.Running}}" in call
        ):
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="true\n", stderr="",
            )
        if "cargo --version" in call[-1]:
            return subprocess.CompletedProcess(
                args=args,
                returncode=127,
                stdout="",
                stderr="cargo: command not found",
            )
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="abc123\n", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake)

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match=r"rust_toolchain_probe.*cargo: command not found",
    ):
        ic.setup_instance_container(
            _instance(language="rust"),
            host_logs_dir=tmp_path / "logs",
        )

    assert ["docker", "rm", "-f", "abc123"] in calls


def test_rust_patch_renderer_skips_large_toolchain_stage(
    tmp_path: Path,
    monkeypatch,
):
    fake = _fake_subprocess(rc=0, stdout="abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert ic.start_patch_render_container(
        _instance(language="rust"),
        host_workspace=workspace,
    ) == "abc123"

    assert not any(
        "test -x /root/.cargo/bin/cargo" in call[-1]
        for call in fake.calls
    )


def test_setup_creates_missing_logs_dir(tmp_path: Path, monkeypatch):
    """host_logs_dir auto-creates if missing (idempotent setup
    against partial prior state). Also pre-creates the verifier/
    and artifacts/ subdirs that test.sh expects."""
    fake = _fake_subprocess(rc=0, stdout="abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    logs = tmp_path / "logs-new"
    assert not logs.exists()
    ic.setup_instance_container(_instance(), host_logs_dir=logs)
    assert logs.is_dir()
    # Step 5 (`echo 1 > /logs/verifier/reward.txt`) needs this:
    assert (logs / "verifier").is_dir()
    # Step 0 also pre-creates inside the container, but we pre-create
    # on host too in case the bind-mount lands before mkdir.
    assert (logs / "artifacts").is_dir()


def test_setup_with_host_tests_dir(tmp_path: Path, monkeypatch):
    """host_tests_dir → -v <path>:/tests:ro in docker run args.
    Required for any verify step — the verifier shell
    (/tests/test.sh) lives in the deep-swe clone, NOT in the
    image."""
    fake = _fake_subprocess(rc=0, stdout="abc\n")
    monkeypatch.setattr(subprocess, "run", fake)
    tests = tmp_path / "tests"
    tests.mkdir()
    ic.setup_instance_container(
        _instance(), host_logs_dir=tmp_path / "logs",
        host_tests_dir=tests,
    )
    run_args = fake.calls[1]
    assert any(f"{tests}:/tests:ro" in a for a in run_args)
    # No /solution mount in this test
    assert not any(":/solution:" in a for a in run_args)


def test_setup_with_host_solution_dir(tmp_path: Path, monkeypatch):
    """host_solution_dir → -v <path>:/solution:ro. ONLY used by
    the verifier pilot — passing this in cascade mode would be
    gold-solution leakage. The setup function accepts it but
    callers are responsible for the leakage check."""
    fake = _fake_subprocess(rc=0, stdout="abc\n")
    monkeypatch.setattr(subprocess, "run", fake)
    sol = tmp_path / "solution"
    sol.mkdir()
    ic.setup_instance_container(
        _instance(), host_logs_dir=tmp_path / "logs",
        host_solution_dir=sol,
    )
    run_args = fake.calls[1]
    assert any(f"{sol}:/solution:ro" in a for a in run_args)


def test_setup_host_tests_dir_missing_fails(tmp_path: Path):
    """host_tests_dir pointing at a non-directory fails fast with
    a message naming the expected source (deep-swe clone path)."""
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="host_tests_dir.*not a directory"):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=tmp_path / "logs",
            host_tests_dir=tmp_path / "nonexistent",
        )


def test_setup_host_solution_dir_missing_mentions_leakage(tmp_path: Path):
    """host_solution_dir non-directory error message must call out
    the gold-solution-leakage risk so the operator notices if they
    accidentally passed it in cascade mode."""
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="GOLD-SOLUTION LEAKAGE"):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=tmp_path / "logs",
            host_solution_dir=tmp_path / "nonexistent",
        )


def test_setup_with_container_name(tmp_path: Path, monkeypatch):
    """Explicit container_name → --name in docker run."""
    fake = _fake_subprocess(rc=0, stdout="abc\n")
    monkeypatch.setattr(subprocess, "run", fake)
    ic.setup_instance_container(
        _instance(), host_logs_dir=tmp_path / "logs",
        container_name="my-task",
    )
    run_args = fake.calls[1]
    assert "--name" in run_args and "my-task" in run_args


def test_setup_empty_image_fails(tmp_path: Path):
    """DeepSWEInstance with empty docker_image fails fast (would
    be a dataset-load-time bug, but defense in depth here)."""
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="empty docker_image"):
        ic.setup_instance_container(
            _instance(docker_image=""),
            host_logs_dir=tmp_path / "logs",
        )


def test_setup_empty_container_id_fails(tmp_path: Path, monkeypatch):
    """docker run returned empty stdout — fail fast rather than
    silently propagate an empty id downstream."""
    monkeypatch.setattr(subprocess, "run", _fake_subprocess(rc=0, stdout="\n"))
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="empty container_id"):
        ic.setup_instance_container(
            _instance(), host_logs_dir=tmp_path / "logs",
        )


def test_setup_pull_failure(tmp_path: Path, monkeypatch):
    """Pull failure surfaces from _run_docker as
    InstanceContainerSetupError; container is never started."""
    monkeypatch.setattr(subprocess, "run", _fake_subprocess(rc=1, stderr="bad image"))
    with pytest.raises(ic.InstanceContainerSetupError,
                       match=r"\[pull\]"):
        ic.setup_instance_container(
            _instance(), host_logs_dir=tmp_path / "logs",
        )


def test_setup_run_failure(tmp_path: Path, monkeypatch):
    """Pull succeeds but run fails — surfaces cleanly."""
    calls = [0]
    def _fake(args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            # pull succeeds
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        # run fails
        return subprocess.CompletedProcess(
            args=args, returncode=125, stdout="",
            stderr="image specifies platform that does not match")
    monkeypatch.setattr(subprocess, "run", _fake)
    with pytest.raises(ic.InstanceContainerSetupError,
                       match=r"\[run\].*125"):
        ic.setup_instance_container(
            _instance(), host_logs_dir=tmp_path / "logs",
        )


@pytest.mark.parametrize(
    ("inspect_rc", "inspect_stdout", "inspect_stderr", "message"),
    [
        (0, "false\n", "", "not running"),
        (1, "", "Error: No such container: exited", "exited or vanished"),
    ],
)
def test_setup_fails_when_pid1_exits_before_readiness(
    tmp_path: Path,
    monkeypatch,
    inspect_rc: int,
    inspect_stdout: str,
    inspect_stderr: str,
    message: str,
):
    calls: list[list[str]] = []

    def _fake(args, **kwargs):
        calls.append(list(args))
        if args[1] == "pull":
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[1] == "run":
            return subprocess.CompletedProcess(
                args, 0, stdout="short-lived-cid\n", stderr=""
            )
        return subprocess.CompletedProcess(
            args,
            inspect_rc,
            stdout=inspect_stdout,
            stderr=inspect_stderr,
        )

    monkeypatch.setattr(subprocess, "run", _fake)

    with pytest.raises(ic.InstanceContainerSetupError, match=message):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=tmp_path / "logs",
        )

    assert calls[-1] == [
        "docker", "inspect", "--format", "{{.State.Running}}",
        "short-lived-cid",
    ]


def test_setup_rejects_nonpositive_resource_caps(tmp_path: Path, monkeypatch):
    """Untrusted execution may never inherit host-wide resources."""
    fake = _fake_subprocess(rc=0, stdout="abc\n")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(
        ic.InstanceContainerSetupError, match="non-positive",
    ):
        ic.setup_instance_container(
            _instance(cpus=0, memory_mb=0),
            host_logs_dir=tmp_path / "logs",
        )


def test_setup_rejects_nested_mount_roots_before_creating_logs(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    logs = workspace / "attacker-logs"

    def _unexpected_docker(*args, **kwargs):
        raise AssertionError("overlap must fail before docker or log writes")

    monkeypatch.setattr(subprocess, "run", _unexpected_docker)

    with pytest.raises(ic.InstanceContainerSetupError, match="roots overlap"):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=logs,
            host_workspace=workspace,
        )

    assert not logs.exists()


def test_setup_rejects_resolved_alias_overlap_before_log_subdirs(
    tmp_path: Path,
    monkeypatch,
):
    shared = tmp_path / "shared"
    shared.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(shared, target_is_directory=True)

    def _unexpected_docker(*args, **kwargs):
        raise AssertionError("resolved overlap must fail before docker")

    monkeypatch.setattr(subprocess, "run", _unexpected_docker)

    with pytest.raises(ic.InstanceContainerSetupError, match="roots overlap"):
        ic.setup_instance_container(
            _instance(),
            host_logs_dir=alias,
            host_tests_dir=shared,
        )

    assert not (shared / "verifier").exists()
    assert not (shared / "artifacts").exists()


# ---------- start_authoring_container / start_verifier_container ----------


def test_start_authoring_container_no_tests_no_solution(tmp_path: Path, monkeypatch):
    """The authoring container mounts /app + /logs but NEVER /tests or
    /solution — leakage is structural, not policy. Verify the docker run
    args carry neither mount."""
    fake = _fake_subprocess(rc=0, stdout="auth123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()

    cid = ic.start_authoring_container(
        _instance(),
        host_workspace=ws,
        host_logs_dir=tmp_path / "logs",
    )
    assert cid == "auth123"
    run_args = fake.calls[1]
    assert any(":/app:rw" in a for a in run_args)
    assert any(":/app/.git:ro" in a for a in run_args)
    assert any(":/logs:rw" in a for a in run_args)
    # The whole point: no hidden tests, no gold solution.
    assert not any(":/tests:" in a for a in run_args)
    assert not any(":/solution:" in a for a in run_args)


def test_start_authoring_container_passes_name(tmp_path: Path, monkeypatch):
    fake = _fake_subprocess(rc=0, stdout="auth123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / ".git").mkdir()
    ic.start_authoring_container(
        _instance(),
        host_workspace=ws,
        host_logs_dir=tmp_path / "logs",
        container_name="auth-task-1",
    )
    run_args = fake.calls[1]
    assert "--name" in run_args and "auth-task-1" in run_args


def test_start_verifier_container_mounts_tests_not_solution(tmp_path: Path, monkeypatch):
    """The verifier container mounts /app + /logs + /tests:ro so
    test.sh can run, but STILL never mounts /solution."""
    fake = _fake_subprocess(rc=0, stdout="verify123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    ws = tmp_path / "ws"
    ws.mkdir()
    tests = tmp_path / "tests"
    tests.mkdir()

    cid = ic.start_verifier_container(
        _instance(),
        host_workspace=ws,
        host_logs_dir=tmp_path / "logs",
        host_tests_dir=tests,
    )
    assert cid == "verify123"
    run_args = fake.calls[1]
    assert any(":/app:rw" in a for a in run_args)
    assert any(f"{tests}:/tests:ro" in a for a in run_args)
    assert not any(":/solution:" in a for a in run_args)


def test_start_verifier_container_requires_tests_dir():
    """``host_tests_dir`` is a required kwarg — omitting it is a
    TypeError at the call site, not a silent verify-without-tests."""
    with pytest.raises(TypeError):
        ic.start_verifier_container(  # type: ignore[call-arg]
            _instance(),
            host_workspace=Path("/tmp/ws"),
            host_logs_dir=Path("/tmp/logs"),
        )


def test_patch_renderer_cannot_mutate_live_workspace_or_git_index(
    tmp_path: Path,
    monkeypatch,
):
    """Renderer confinement is structural even for hostile Git config.

    ``/app`` is a read-only bind and the render command uses only an isolated
    index/object store in ``/tmp``. No hidden or solution mount is present.
    """

    fake = _fake_subprocess(rc=0, stdout="render123\n")
    monkeypatch.setattr(subprocess, "run", fake)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    cid = ic.start_patch_render_container(
        _instance(),
        host_workspace=workspace,
    )

    assert cid == "render123"
    run_args = fake.calls[1]
    assert any(f"{workspace.resolve()}:/app:ro" == arg for arg in run_args)
    assert not any("/app/.git:" in arg for arg in run_args)
    assert not any("/app:rw" in arg for arg in run_args)
    assert not any("/logs:" in arg for arg in run_args)
    assert not any("/tests:" in arg or "/solution:" in arg for arg in run_args)


# ---------- teardown_instance_container ----------


def test_strict_stop_proves_auto_removed_container(monkeypatch):
    calls: list[list[str]] = []

    def _fake(args, **kwargs):
        calls.append(list(args))
        if args[1] == "stop":
            return subprocess.CompletedProcess(args, 0, stdout="abc123\n", stderr="")
        return subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr="Error: No such object: abc123",
        )

    monkeypatch.setattr(subprocess, "run", _fake)

    ic.stop_instance_container_strict("abc123")

    assert calls == [
        ["docker", "stop", "abc123"],
        ["docker", "inspect", "--format", "{{.State.Running}}", "abc123"],
    ]


def test_strict_stop_accepts_container_already_auto_removed(monkeypatch):
    fake = _fake_subprocess(
        rc=1,
        stderr="Error response from daemon: No such container: abc123",
    )
    monkeypatch.setattr(subprocess, "run", fake)

    ic.stop_instance_container_strict("abc123")

    assert fake.calls == [["docker", "stop", "abc123"]]


def test_strict_stop_fails_if_container_remains_running(monkeypatch):
    calls = [0]

    def _fake(args, **kwargs):
        calls[0] += 1
        stdout = "abc123\n" if calls[0] == 1 else "true\n"
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", _fake)

    with pytest.raises(ic.InstanceContainerSetupError, match="remains running"):
        ic.stop_instance_container_strict("abc123")


def test_capture_model_patch_uses_container_git_bytes(monkeypatch):
    observed: dict = {}
    real_popen = subprocess.Popen

    def _fake(args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return real_popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'diff --git a/x b/x\\n')",
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", _fake)
    monkeypatch.setattr(
        ic,
        "inspect_docker_container_environment",
        lambda _container_id: (
            "PATH=/image/bin:/usr/bin:/bin",
            "PROJECT_MODE=cuda",
            "GIT_CONFIG_COUNT=1",
            "GIT_CONFIG_KEY_0=alias.diff",
            "GIT_CONFIG_VALUE_0=!touch /app/owned",
            "BASH_ENV=/app/evil",
            "BASH_FUNC_git%%=() { touch /app/function-owned; }",
        ),
    )

    patch = ic.capture_model_patch_from_container(
        "abc123",
        base_commit="1" * 40,
    )

    assert patch == b"diff --git a/x b/x\n"
    assert observed["args"][:5] == [
        "docker", "exec", "--workdir", "/app", "abc123",
    ]
    assert observed["args"][-1] == "1" * 40
    args = observed["args"]
    assert args[5:8] == ["/usr/bin/env", "-i", "--"]
    assert "PROJECT_MODE=cuda" in args
    assert "HOME=/tmp/home" in args
    assert not any(argument.startswith("GIT_") for argument in args[8:])
    assert not any("BASH_ENV" in arg or "function-owned" in arg for arg in args)
    assert not any("!touch /app/owned" in arg for arg in args)
    render_script = args[args.index("/bin/sh") + 2]
    assert "git() {" in render_script
    assert "/usr/bin/git" in render_script
    assert "--git-dir=/app/.git" in render_script
    assert "--work-tree=/app" in render_script
    assert "GIT_CONFIG_NOSYSTEM=1" in render_script
    assert "core.hooksPath=/dev/null" in render_script
    assert "diff --no-ext-diff --no-textconv" in render_script
    assert "GIT_INDEX_FILE=\"$tmp/index\"" in render_script
    assert "GIT_OBJECT_DIRECTORY=\"$tmp/objects\"" in render_script
    assert "git read-tree \"$1\"" in render_script
    assert "git reset" not in render_script
    assert observed["kwargs"]["stdout"] is subprocess.PIPE
    assert observed["kwargs"]["stderr"] is subprocess.PIPE


def test_capture_model_patch_kills_oversize_stream_before_return(monkeypatch):
    real_popen = subprocess.Popen

    def _fake(_args, **kwargs):
        return real_popen(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 65536)",
            ],
            **kwargs,
        )

    monkeypatch.setattr(subprocess, "Popen", _fake)
    monkeypatch.setattr(
        ic,
        "inspect_docker_container_environment",
        lambda _container_id: ("PATH=/usr/bin:/bin",),
    )

    with pytest.raises(ic.InstanceContainerSetupError, match="patch byte limit 8"):
        ic.capture_model_patch_from_container(
            "abc123",
            base_commit="1" * 40,
            max_bytes=8,
        )


def test_teardown_noop_on_none():
    """No container_id, no image → no-op (no subprocess calls).
    Defensive: caller's finally block may invoke teardown after a
    setup that never got to docker run."""
    # If subprocess.run were called here, it'd raise FileNotFoundError
    # because docker isn't on the test environment's PATH. The fact
    # that this test passes proves teardown short-circuited.
    ic.teardown_instance_container(None, None)


def test_teardown_stops_container(monkeypatch):
    """container_id only → docker stop only."""
    fake = _fake_subprocess(rc=0)
    monkeypatch.setattr(subprocess, "run", fake)
    ic.teardown_instance_container("abc123")
    assert fake.calls == [["docker", "stop", "abc123"]]


def test_teardown_stops_and_image_rm(monkeypatch):
    """Both container_id + image → stop then image rm in that order."""
    fake = _fake_subprocess(rc=0)
    monkeypatch.setattr(subprocess, "run", fake)
    ic.teardown_instance_container("abc123", "myimage:latest")
    assert fake.calls == [
        ["docker", "stop", "abc123"],
        ["docker", "image", "rm", "myimage:latest"],
    ]


def test_teardown_stop_failure_logged_not_raised(monkeypatch, caplog):
    """``docker stop`` failure is logged at WARN, NOT raised.
    Teardown failure shouldn't mask the original cascade outcome."""
    fake = _fake_subprocess(rc=1, stderr="no such container")
    monkeypatch.setattr(subprocess, "run", fake)
    with caplog.at_level("WARNING"):
        ic.teardown_instance_container("abc123")
    assert "stop failed" in caplog.text


def test_teardown_image_rm_failure_logged(monkeypatch, caplog):
    """``docker image rm`` failure (e.g. image still in use) is
    logged at INFO and continues — common during concurrent sweeps."""
    calls = [0]
    def _fake(args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="",
            stderr="image is being used by running container")
    monkeypatch.setattr(subprocess, "run", _fake)
    with caplog.at_level("INFO"):
        ic.teardown_instance_container("abc123", "img:latest")
    assert "image_rm failed" in caplog.text


# ---------- populate_workspace_from_image ----------


def test_sanitize_workspace_history_prunes_descendants(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Crucible Test"],
        cwd=workspace, check=True,
    )
    source = workspace / "module.py"
    source.write_text("value = 'base'\n")
    subprocess.run(["git", "add", "module.py"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    source.write_text("value = 'gold'\n")
    subprocess.run(
        ["git", "commit", "-qam", "descendant fix"],
        cwd=workspace,
        check=True,
    )
    gold = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    subprocess.run(
        ["git", "checkout", "--detach", base], cwd=workspace, check=True,
        capture_output=True,
    )

    ic._sanitize_workspace_history(workspace, base_commit=base)

    assert subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=workspace,
        text=True,
    ).strip() == "1"
    assert subprocess.run(
        ["git", "cat-file", "-e", gold], cwd=workspace,
        capture_output=True,
    ).returncode != 0
    assert subprocess.check_output(
        ["git", "remote"], cwd=workspace, text=True,
    ).strip() == ""


def _checkout_with_descendant(tmp_path: Path) -> tuple[Path, str, str]:
    """Return a checkout detached at its base that still holds a descendant."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Crucible Test"],
        cwd=workspace, check=True,
    )
    source = workspace / "module.py"
    source.write_text("value = 'base'\n")
    subprocess.run(["git", "add", "module.py"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    source.write_text("value = 'gold'\n")
    subprocess.run(
        ["git", "commit", "-qam", "descendant fix"],
        cwd=workspace,
        check=True,
    )
    descendant = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    subprocess.run(
        ["git", "checkout", "--detach", base], cwd=workspace, check=True,
        capture_output=True,
    )
    return workspace, base, descendant


@pytest.mark.parametrize("length", (7, 39))
def test_sanitize_workspace_history_resolves_abbreviated_base_commit(
    tmp_path: Path,
    length: int,
):
    workspace, base, descendant = _checkout_with_descendant(tmp_path)

    resolved = ic._sanitize_workspace_history(
        workspace, base_commit=base[:length],
    )

    assert resolved == base
    assert (workspace / ".git" / "shallow").read_text() == base + "\n"
    assert subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip() == base
    assert subprocess.check_output(
        ["git", "rev-list", "--count", "HEAD"], cwd=workspace, text=True,
    ).strip() == "1"
    assert subprocess.run(
        ["git", "cat-file", "-e", descendant], cwd=workspace,
        capture_output=True,
    ).returncode != 0
    assert subprocess.check_output(
        ["git", "remote"], cwd=workspace, text=True,
    ).strip() == ""


def test_sanitize_workspace_history_returns_full_base_commit_unchanged(
    tmp_path: Path,
):
    workspace, base, _descendant = _checkout_with_descendant(tmp_path)

    assert ic._sanitize_workspace_history(workspace, base_commit=base) == base


def test_sanitize_workspace_history_rejects_abbreviation_of_another_commit(
    tmp_path: Path,
):
    workspace, base, descendant = _checkout_with_descendant(tmp_path)
    assert descendant[:7] != base[:7]

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="does not match dataset base_commit",
    ):
        ic._sanitize_workspace_history(workspace, base_commit=descendant[:7])


def test_sanitize_workspace_history_rejects_ref_named_like_the_pin(
    tmp_path: Path,
):
    """A ref name is never a commit id, even when it points at HEAD."""

    workspace, base, _descendant = _checkout_with_descendant(tmp_path)
    pin = "0123abc" if not base.startswith("0123abc") else "abc0123"
    subprocess.run(["git", "tag", pin, base], cwd=workspace, check=True)
    assert subprocess.check_output(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{pin}"],
        cwd=workspace,
        text=True,
    ).strip() == base

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="does not match dataset base_commit",
    ):
        ic._sanitize_workspace_history(workspace, base_commit=pin)


def _abbreviation_with_hex_letter(commit: str) -> str:
    """Return the shortest abbreviation of 12 or more characters with a letter.

    A commit id changes with the commit time and may start with digits only,
    which upper-casing leaves unchanged.
    """

    for length in range(12, len(commit)):
        if commit[:length] != commit[:length].upper():
            return commit[:length]
    raise AssertionError("commit id abbreviation holds no hexadecimal letter")


@pytest.mark.parametrize(
    ("commit", "expected"),
    (
        ("0123456789abcdef0123456789abcdef01234567", "0123456789ab"),
        (
            "012345678901234567890123456789abcdef0123",
            "012345678901234567890123456789a",
        ),
    ),
    ids=("letter_within_twelve", "twelve_leading_digits"),
)
def test_abbreviation_with_hex_letter_extends_past_leading_digits(
    commit: str,
    expected: str,
):
    abbreviation = _abbreviation_with_hex_letter(commit)

    assert abbreviation == expected
    assert abbreviation.upper() != abbreviation


@pytest.mark.parametrize(
    "malformed",
    ("six_characters", "upper_case", "not_hex", "longer_than_head", "empty"),
)
def test_sanitize_workspace_history_rejects_malformed_base_commit(
    tmp_path: Path,
    monkeypatch,
    malformed: str,
):
    workspace, base, _descendant = _checkout_with_descendant(tmp_path)
    lettered = _abbreviation_with_hex_letter(base)
    pin = {
        "six_characters": base[:6],
        "upper_case": lettered.upper(),
        "not_hex": base[:7] + "g",
        "longer_than_head": base + "0",
        "empty": "",
    }[malformed]
    if malformed == "upper_case":
        assert pin != lettered
        assert pin.lower() == lettered
    actual_run_git = ic._run_git
    calls: list[tuple[str, ...]] = []

    def _recording_run_git(path: Path, *args: str):
        calls.append(args)
        return actual_run_git(path, *args)

    monkeypatch.setattr(ic, "_run_git", _recording_run_git)

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="does not match dataset base_commit",
    ):
        ic._sanitize_workspace_history(workspace, base_commit=pin)

    # The pin is judged against the observed HEAD only; Git is never asked to
    # resolve a string that is not an exact prefix of it.
    assert calls == [("rev-parse", "HEAD")]


def test_sanitize_replaces_local_config_before_first_host_git(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Crucible Test"],
        cwd=workspace,
        check=True,
    )
    (workspace / "module.py").write_text("value = 'base'\n")
    subprocess.run(["git", "add", "module.py"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workspace, check=True)
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=workspace, text=True,
    ).strip()
    subprocess.run(
        ["git", "config", "core.fsmonitor", "evil-fsmonitor"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(
        ["git", "config", "filter.crucible.smudge", "evil-smudge"],
        cwd=workspace,
        check=True,
    )

    actual_run_git = ic._run_git
    calls: list[tuple[str, ...]] = []

    def _guarded_run_git(path: Path, *args: str):
        config = (path / ".git" / "config").read_text()
        assert "evil-fsmonitor" not in config
        assert "evil-smudge" not in config
        assert "fsmonitor = false" in config
        calls.append(args)
        return actual_run_git(path, *args)

    monkeypatch.setattr(ic, "_run_git", _guarded_run_git)

    ic._sanitize_workspace_history(workspace, base_commit=base)

    assert calls[0] == ("rev-parse", "HEAD")


@pytest.mark.parametrize("linked_name", ["config", "HEAD"])
def test_sanitize_rejects_git_metadata_symlinks_before_host_git(
    tmp_path: Path,
    monkeypatch,
    linked_name: str,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    outside = tmp_path / f"outside-{linked_name}"
    outside.write_text("outside remains untouched\n")
    target = workspace / ".git" / linked_name
    target.unlink()
    target.symlink_to(outside)

    def _unexpected_git(*args, **kwargs):
        raise AssertionError("host Git ran before metadata validation")

    monkeypatch.setattr(ic, "_run_git", _unexpected_git)

    with pytest.raises(ic.InstanceContainerSetupError, match="symlink"):
        ic._sanitize_workspace_history(workspace, base_commit="0" * 40)
    assert outside.read_text() == "outside remains untouched\n"


@pytest.mark.parametrize("git_marker_kind", ["symlink", "file"])
def test_sanitize_requires_real_standalone_git_directory(
    tmp_path: Path,
    monkeypatch,
    git_marker_kind: str,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = workspace / ".git"
    if git_marker_kind == "symlink":
        outside_git = tmp_path / "outside-git"
        outside_git.mkdir()
        (outside_git / "HEAD").write_text("ref: refs/heads/main\n")
        marker.symlink_to(outside_git, target_is_directory=True)
    else:
        marker.write_text("gitdir: ../outside-git\n")

    def _unexpected_git(*args, **kwargs):
        raise AssertionError("host Git ran before .git validation")

    monkeypatch.setattr(ic, "_run_git", _unexpected_git)

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="real standalone directory",
    ):
        ic._sanitize_workspace_history(workspace, base_commit="0" * 40)


def test_sanitize_rejects_external_object_store_before_host_git(
    tmp_path: Path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    alternates = workspace / ".git" / "objects" / "info" / "alternates"
    alternates.write_text(str(tmp_path / "outside-objects") + "\n")

    def _unexpected_git(*args, **kwargs):
        raise AssertionError("host Git ran before alternates validation")

    monkeypatch.setattr(ic, "_run_git", _unexpected_git)

    with pytest.raises(ic.InstanceContainerSetupError, match="redirects Git storage"):
        ic._sanitize_workspace_history(workspace, base_commit="0" * 40)


def test_run_git_strips_ambient_git_execution_configuration(
    tmp_path: Path,
    monkeypatch,
):
    observed: dict[str, Any] = {}
    monkeypatch.setenv("GIT_DIR", "/attacker/repo")
    monkeypatch.setenv("GIT_EXTERNAL_DIFF", "/attacker/diff")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/attacker/system-config")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")

    def _fake(args, **kwargs):
        observed["args"] = list(args)
        observed["env"] = dict(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake)

    result = ic._run_git(tmp_path, "rev-parse", "HEAD")

    assert result.stdout == "ok\n"
    assert observed["args"][0] == "git"
    assert "core.hooksPath=/dev/null" in observed["args"]
    assert "core.fsmonitor=false" in observed["args"]
    env = observed["env"]
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_ATTR_NOSYSTEM"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_DIR" not in env
    assert "GIT_EXTERNAL_DIFF" not in env
    assert "GIT_CONFIG_SYSTEM" not in env
    assert "GIT_CONFIG_COUNT" not in env


def test_populate_workspace_happy_path(tmp_path: Path, monkeypatch):
    """docker create → docker cp → docker rm sequence with all three
    docker calls succeeding. Workspace dir auto-creates if missing."""
    fake = _fake_subprocess(rc=0, stdout="throwaway_abc123\n")
    monkeypatch.setattr(subprocess, "run", fake)

    ws = tmp_path / "ws-new"  # does not exist yet
    assert not ws.exists()
    monkeypatch.setattr(ic, "_sanitize_workspace_history", lambda *a, **k: None)
    ic.populate_workspace_from_image(
        "myimage:latest", ws, base_commit="cafebabe" * 5,
    )
    assert ws.is_dir()

    # Exactly three docker invocations in the right order.
    assert len(fake.calls) == 3
    assert fake.calls[0][:2] == ["docker", "create"]
    assert "--label" in fake.calls[0]
    assert "io.crucible.managed=true" in fake.calls[0]
    assert fake.calls[0][-1] == "myimage@sha256:" + "a" * 64
    assert fake.calls[1] == [
        "docker", "cp", "throwaway_abc123:/app/.", f"{ws}/",
    ]
    assert fake.calls[2] == ["docker", "rm", "throwaway_abc123"]


def test_populate_workspace_returns_resolved_base_commit(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_subprocess(rc=0, stdout="throwaway_abc123\n"),
    )
    observed: dict[str, Any] = {}

    def _sanitize(workspace: Path, *, base_commit: str) -> str:
        observed["workspace"] = workspace
        observed["base_commit"] = base_commit
        return "f" * 40

    monkeypatch.setattr(ic, "_sanitize_workspace_history", _sanitize)
    workspace = tmp_path / "workspace"

    resolved = ic.populate_workspace_from_image(
        "myimage:latest", workspace, base_commit="fffffff",
    )

    assert resolved == "f" * 40
    assert observed == {"workspace": workspace, "base_commit": "fffffff"}


def test_populate_registers_create_before_copy(tmp_path: Path, monkeypatch):
    events: list[tuple[str, str]] = []

    def _fake(args, **kwargs):  # noqa: ARG001
        action = str(args[1])
        events.append(("docker", action))
        stdout = "throwaway-abc123\n" if action == "create" else ""
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    def _register(reference: str) -> None:
        events.append(("register", reference))

    monkeypatch.setattr(subprocess, "run", _fake)
    monkeypatch.setattr(ic, "_sanitize_workspace_history", lambda *a, **k: None)

    ic.populate_workspace_from_image(
        "img:tag",
        tmp_path / "workspace",
        base_commit="cafebabe" * 5,
        register_container=_register,
    )

    assert events == [
        ("docker", "create"),
        ("register", "throwaway-abc123"),
        ("docker", "cp"),
        ("docker", "rm"),
    ]


def test_populate_registration_failure_force_removes_without_copy(
    tmp_path: Path,
    monkeypatch,
):
    calls: list[list[str]] = []

    def _fake(args, **kwargs):  # noqa: ARG001
        call = list(args)
        calls.append(call)
        stdout = "throwaway-abc123\n" if call[1] == "create" else ""
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    def _registration_failure(_reference: str) -> None:
        raise RuntimeError("registration fsync failed")

    monkeypatch.setattr(subprocess, "run", _fake)

    with pytest.raises(
        ic.InstanceContainerSetupError,
        match="registration fsync failed",
    ):
        ic.populate_workspace_from_image(
            "img:tag",
            tmp_path / "workspace",
            base_commit="cafebabe" * 5,
            register_container=_registration_failure,
        )

    assert len(calls) == 2
    assert calls[0][:2] == ["docker", "create"]
    assert calls[1] == ["docker", "rm", "-f", "throwaway-abc123"]
    assert not any(call[1] == "cp" for call in calls)


def test_populate_workspace_custom_image_path(tmp_path: Path, monkeypatch):
    """``image_app_path`` overrides the default ``/app``."""
    fake = _fake_subprocess(rc=0, stdout="throwaway_xyz\n")
    monkeypatch.setattr(subprocess, "run", fake)

    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(ic, "_sanitize_workspace_history", lambda *a, **k: None)
    ic.populate_workspace_from_image(
        "img:tag", ws, base_commit="cafebabe" * 5,
        image_app_path="/testbed",
    )
    assert fake.calls[1] == [
        "docker", "cp", "throwaway_xyz:/testbed/.", f"{ws}/",
    ]


def test_populate_workspace_create_empty_id_fails(tmp_path: Path, monkeypatch):
    """``docker create`` returned an empty container id — fail
    fast rather than passing an empty string to ``docker cp``."""
    monkeypatch.setattr(subprocess, "run", _fake_subprocess(rc=0, stdout="\n"))
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="docker create returned empty id"):
        ic.populate_workspace_from_image(
            "img:tag", tmp_path / "ws", base_commit="cafebabe" * 5,
        )


def test_populate_workspace_cp_failure_still_cleans_throwaway(
    tmp_path: Path, monkeypatch,
):
    """If ``docker cp`` fails, the throwaway container must STILL
    be ``docker rm``'d. Defends against zombie throwaways
    accumulating on a long sweep."""
    rms_seen: list[list[str]] = []
    cp_call_index = [0]

    def _fake(args, **kwargs):
        cp_call_index[0] += 1
        if cp_call_index[0] == 1:
            # create succeeds
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="t_abc\n", stderr="",
            )
        if cp_call_index[0] == 2:
            # cp fails
            return subprocess.CompletedProcess(
                args=args, returncode=1, stdout="",
                stderr="permission denied",
            )
        # rm (cleanup) — observe the call
        rms_seen.append(list(args))
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="", stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake)
    with pytest.raises(ic.InstanceContainerSetupError, match=r"populate_cp"):
        ic.populate_workspace_from_image(
            "img:tag", tmp_path / "ws", base_commit="cafebabe" * 5,
        )
    # The throwaway rm DID fire even though cp raised.
    assert rms_seen == [["docker", "rm", "t_abc"]]


def test_populate_workspace_rm_failure_logged_not_raised(
    tmp_path: Path, monkeypatch, caplog,
):
    """If the final ``docker rm`` fails (e.g., the throwaway was
    already removed by a parallel reaper), the function should log
    + continue. The cp succeeded; the cleanup is best-effort."""
    cp_call_index = [0]

    def _fake(args, **kwargs):
        cp_call_index[0] += 1
        if cp_call_index[0] == 1:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="t_abc\n", stderr="",
            )
        if cp_call_index[0] == 2:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr="",
            )
        # rm fails
        return subprocess.CompletedProcess(
            args=args, returncode=1, stdout="",
            stderr="No such container: t_abc",
        )

    monkeypatch.setattr(subprocess, "run", _fake)
    monkeypatch.setattr(ic, "_sanitize_workspace_history", lambda *a, **k: None)
    with caplog.at_level("INFO"):
        ic.populate_workspace_from_image(
            "img:tag", tmp_path / "ws", base_commit="cafebabe" * 5,
        )
    assert "throwaway" in caplog.text and "rm failed" in caplog.text


# ---------- _run_docker error paths ----------


def test_run_docker_timeout_raises(monkeypatch):
    def _timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=5)
    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="timed out after"):
        ic._run_docker(["pull", "foo"], step_name="pull", timeout=5)


def test_run_docker_missing_binary_raises(monkeypatch):
    def _enoent(args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'docker'")
    monkeypatch.setattr(subprocess, "run", _enoent)
    with pytest.raises(ic.InstanceContainerSetupError,
                       match="docker binary not found"):
        ic._run_docker(["ps"], step_name="ps")
