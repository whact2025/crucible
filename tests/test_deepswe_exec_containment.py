# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Process-tree containment for DeepSWE development verification."""

from __future__ import annotations

import errno
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from crucible.common.runners import container_exec as containment


_CANONICAL_ID = "a" * 64


def _inspect_payload(
    *,
    user: str = "1000:1000",
    phase: str = "authoring",
    security_option: str = "no-new-privileges",
    mounts: list[dict] | None = None,
    host_config_updates: dict | None = None,
    config_updates: dict | None = None,
) -> str:
    if mounts is None:
        mounts = [
            {
                "Destination": "/app", "Source": "/host/app",
                "RW": True, "Type": "bind",
            },
            {
                "Destination": "/app/.git", "Source": "/host/app/.git",
                "RW": False, "Type": "bind",
            },
            {
                "Destination": "/logs", "Source": "/host/logs",
                "RW": True, "Type": "bind",
            },
        ]
    host_config = {
        "CapDrop": ["ALL"],
        "CapAdd": ["CAP_KILL", "CAP_SETGID", "CAP_SETUID"],
        "SecurityOpt": [security_option],
        "NetworkMode": "none",
        "ReadonlyRootfs": True,
        "Privileged": False,
        "PidMode": "",
        "IpcMode": "private",
        "UTSMode": "",
        "UsernsMode": "",
        "CgroupnsMode": "private",
        "Devices": [],
        "DeviceRequests": None,
        "PidsLimit": 512,
        "Tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=256m"},
    }
    host_config.update(host_config_updates or {})
    config = {
        "User": user,
        "Entrypoint": ["/bin/sleep"],
        "Cmd": ["infinity"],
        "Labels": {
            "io.crucible.managed": "true",
            "io.crucible.phase": phase,
        },
    }
    config.update(config_updates or {})
    return json.dumps([
        {
            "Id": _CANONICAL_ID,
            "State": {"Running": True},
            "Config": config,
            "HostConfig": host_config,
            "Mounts": mounts,
        },
    ])


@pytest.mark.parametrize("phase", ["authoring", "baseline"])
@pytest.mark.parametrize(
    "security_option",
    [
        "no-new-privileges",
        "no-new-privileges:true",
        "no-new-privileges=true",
    ],
)
def test_source_validation_accepts_only_nonroot_managed_authoring(
    monkeypatch, phase, security_option,
):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0,
            stdout=_inspect_payload(
                phase=phase, security_option=security_option,
            ),
            stderr="",
        ),
    )

    assert containment._validate_source_container(
        "candidate", deadline=time.monotonic() + 1,
    ) == (
        1000,
        1000,
        _CANONICAL_ID,
        Path("/host/app"),
        Path("/host/logs"),
    )


def test_source_validation_accepts_swebench_profile_without_logs(monkeypatch):
    runner_profile = containment.ContainerExecProfile(
        name="swebench",
        workspace_destination="/testbed",
        git_destination="/testbed/.git",
        logs_destination=None,
        allowed_phases=frozenset({"authoring"}),
        helper_cache_env="CRUCIBLE_SWEBENCH_HELPER_DIR",
    )
    mounts = [
        {
            "Destination": "/testbed",
            "Source": "/host/testbed",
            "RW": True,
            "Type": "bind",
        },
        {
            "Destination": "/testbed/.git",
            "Source": "/host/testbed/.git",
            "RW": False,
            "Type": "bind",
        },
    ]
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=_inspect_payload(mounts=mounts, phase="authoring"),
            stderr="",
        ),
    )

    assert containment._validate_source_container(
        "candidate",
        deadline=time.monotonic() + 1,
        profile=runner_profile,
    ) == (
        1000,
        1000,
        _CANONICAL_ID,
        Path("/host/testbed"),
        None,
    )


@pytest.mark.parametrize("user", ["", "root", "0:0", "1000", "1000:0"])
def test_source_validation_rejects_root_or_ambiguous_user(monkeypatch, user):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=_inspect_payload(user=user), stderr="",
        ),
    )

    with pytest.raises(containment.ExecContainmentError, match="non-root"):
        containment._validate_source_container(
            "candidate", deadline=time.monotonic() + 1,
        )


@pytest.mark.parametrize(
    "config_updates",
    [
        {"Entrypoint": None},
        {"Entrypoint": ["/bin/sh"]},
        {"Cmd": ["-c", "service"]},
    ],
)
def test_source_validation_rejects_non_sleep_entrypoint(
    monkeypatch, config_updates,
):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0,
            stdout=_inspect_payload(config_updates=config_updates),
            stderr="",
        ),
    )

    with pytest.raises(containment.ExecContainmentError, match="sleep/infinity"):
        containment._validate_source_container(
            "candidate", deadline=time.monotonic() + 1,
        )


@pytest.mark.parametrize("hidden", ["/tests", "/tests/private", "/solution"])
def test_source_validation_rejects_hidden_capability_mounts(
    monkeypatch, hidden,
):
    mounts = [
        {"Destination": "/app", "Source": "/host/app", "RW": True},
        {
            "Destination": "/app/.git", "Source": "/host/app/.git",
            "RW": False,
        },
        {"Destination": hidden, "Source": "/hidden", "RW": False},
    ]
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=_inspect_payload(mounts=mounts), stderr="",
        ),
    )

    with pytest.raises(containment.ExecContainmentError, match="hidden-test"):
        containment._validate_source_container(
            "candidate", deadline=time.monotonic() + 1,
        )


@pytest.mark.parametrize(
    "host_config_updates",
    [
        {"Privileged": True},
        {"PidMode": "host"},
        {"IpcMode": "host"},
        {"UsernsMode": "host"},
        {"CgroupnsMode": "host"},
        {"SecurityOpt": ["no-new-privileges=false"]},
        {
            "SecurityOpt": [
                "no-new-privileges=true", "seccomp=unconfined",
            ],
        },
        {
            "CapAdd": [
                "CAP_KILL", "CAP_SETGID", "CAP_SETUID", "CAP_SYS_ADMIN",
            ],
        },
        {"CapDrop": ["ALL", "NET_RAW"]},
        {"Devices": [{"PathOnHost": "/dev/kvm"}]},
        {"DeviceRequests": [{"Driver": "nvidia"}]},
        {"PidsLimit": 0},
        {"PidsLimit": 5000},
        {"Tmpfs": {"/tmp": "rw,exec,suid,dev,size=256m"}},
        {
            "Tmpfs": {
                "/tmp": "rw,exec,nosuid,nodev,suid,dev,size=256m",
            },
        },
        {"Tmpfs": {"/tmp": "rw,exec,nosuid,nodev,noexec,size=256m"}},
        {"Tmpfs": {"/tmp": "rw,exec,nosuid,nodev,size=8192m"}},
        {
            "Tmpfs": {
                "/tmp": "rw,exec,nosuid,nodev,size=256m,size=512m",
            },
        },
    ],
)
def test_source_validation_rejects_security_profile_expansion(
    monkeypatch, host_config_updates,
):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0,
            stdout=_inspect_payload(host_config_updates=host_config_updates),
            stderr="",
        ),
    )

    with pytest.raises(containment.ExecContainmentError, match="security profile"):
        containment._validate_source_container(
            "candidate", deadline=time.monotonic() + 1,
        )


@pytest.mark.parametrize(
    "mounts",
    [
        [
            {
                "Destination": "/app", "Source": "/host/app",
                "RW": True, "Type": "bind",
            },
            {
                "Destination": "/app/.git", "Source": "/host/app/.git",
                "RW": False, "Type": "bind",
            },
        ],
        [
            {
                "Destination": "/app", "Source": "/host/app",
                "RW": True, "Type": "bind",
            },
            {
                "Destination": "/app/.git", "Source": "/host/app/.git",
                "RW": False, "Type": "bind",
            },
            {
                "Destination": "/logs", "Source": "/host/logs",
                "RW": True, "Type": "bind",
            },
            {
                "Destination": "/var/run/docker.sock",
                "Source": "/var/run/docker.sock", "RW": True,
                "Type": "bind",
            },
        ],
        [
            {
                "Destination": "/app", "Source": "/host/app",
                "RW": True, "Type": "bind",
            },
            {
                "Destination": "/app/.git", "Source": "/host/app/.git",
                "RW": False, "Type": "bind",
            },
            {
                "Destination": "/logs", "Source": "/",
                "RW": True, "Type": "bind",
            },
        ],
    ],
)
def test_source_validation_rejects_missing_or_extra_mounts(monkeypatch, mounts):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout=_inspect_payload(mounts=mounts), stderr="",
        ),
    )

    with pytest.raises(containment.ExecContainmentError, match="mount"):
        containment._validate_source_container(
            "candidate", deadline=time.monotonic() + 1,
        )


def test_stage_supervisor_recreates_private_dir_without_root_shell(
    monkeypatch, tmp_path,
):
    helper = tmp_path / "supervisor"
    helper.write_bytes(b"static-helper")
    helper.chmod(0o500)
    docker_calls: list[list[str]] = []
    streamed: dict = {}

    def fake_docker(args, **_kwargs):
        docker_calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def fake_stream(args, **kwargs):
        streamed["args"] = list(args)
        streamed["input"] = kwargs["input"]
        return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(containment, "_run_docker", fake_docker)
    monkeypatch.setattr(containment.subprocess, "run", fake_stream)

    containment._stage_supervisor(
        "candidate", helper, deadline=time.monotonic() + 1,
    )

    assert docker_calls == [
        [
            "exec", "--user", "0:0", "candidate",
            "/bin/rm", "-rf", containment._TRUSTED_DIRECTORY,
        ],
        [
            "exec", "--user", "0:0", "candidate",
            "/bin/mkdir", "-m", "0700", containment._TRUSTED_DIRECTORY,
        ],
        [
            "exec", "--user", "0:0", "candidate",
            "/bin/chmod", "0500", containment._TRUSTED_BINARY,
        ],
    ]
    assert streamed["args"] == [
        "docker", "exec", "-i", "--user", "0:0", "candidate",
        "/bin/dd", f"of={containment._TRUSTED_BINARY}", "bs=1048576",
    ]
    assert streamed["input"] == b"static-helper"
    assert "/bin/sh" not in streamed["args"]


def test_helper_cache_lock_expires_on_shared_deadline(monkeypatch):
    clock = [0.0]

    class FakeLock:
        @staticmethod
        def fileno():
            return 9

    def unavailable(*_args):
        raise BlockingIOError(errno.EWOULDBLOCK, "busy")

    monkeypatch.setattr(containment.fcntl, "flock", unavailable)
    monkeypatch.setattr(containment.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        containment.time,
        "sleep",
        lambda duration: clock.__setitem__(0, clock[0] + duration),
    )

    with pytest.raises(containment.ExecContainmentError) as caught:
        containment._acquire_helper_lock(FakeLock(), deadline=0.1)
    assert caught.value.returncode == 124
    assert clock[0] == pytest.approx(0.1)


def test_helper_cache_rejects_writable_mount_overlap_and_symlinks(
    monkeypatch, tmp_path,
):
    app = tmp_path / "app"
    logs = tmp_path / "logs"
    app.mkdir()
    logs.mkdir()
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_HELPER_DIR", str(app / "helpers"))
    with pytest.raises(containment.ExecContainmentError, match="overlaps"):
        containment._validated_helper_cache_directory(
            app_source=app, logs_source=logs,
        )

    target = tmp_path / "helper"
    target.write_bytes(b"helper")
    target.chmod(0o500)
    alias = tmp_path / "helper-alias"
    alias.symlink_to(target)
    assert not containment._valid_cached_helper(alias)
    with pytest.raises(containment.ExecContainmentError, match="symlink"):
        containment._read_helper_nofollow(alias)


def _mock_contained_invocation(
    monkeypatch,
    tmp_path: Path,
    *,
    result_rc: int,
    attestation: str = "valid",
):
    helper = tmp_path / "supervisor"
    helper.write_bytes(b"helper")
    helper.chmod(0o500)
    cache_directory = tmp_path / "helper-cache"
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSWE_HELPER_DIR", str(cache_directory),
    )
    docker_calls: list[list[str]] = []
    stage_calls: list[tuple] = []

    monkeypatch.setattr(
        containment, "_validate_source_container",
        lambda *_args, **_kwargs: (
            1234,
            2345,
            _CANONICAL_ID,
            tmp_path / "app",
            tmp_path / "logs",
        ),
    )
    monkeypatch.setattr(
        containment, "_ensure_supervisor", lambda **_kwargs: helper,
    )

    def fake_docker(args, **kwargs):
        docker_calls.append(list(args))
        if containment._TRUSTED_BINARY in args:
            assert containment._poison_path(
                cache_directory, _CANONICAL_ID,
            ).is_file()
            return subprocess.CompletedProcess(
                args, result_rc, stdout="visible output", stderr="diagnostic",
            )
        if "/bin/cat" in args:
            nonce = args[-1].removeprefix(containment._COMPLETION_PREFIX)
            if attestation == "missing":
                return subprocess.CompletedProcess(
                    args, 1, stdout="", stderr="missing",
                )
            observed_rc = result_rc + 1 if attestation == "mismatch" else result_rc
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{nonce} {observed_rc}\n", stderr="",
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(containment, "_run_docker", fake_docker)
    monkeypatch.setattr(
        containment,
        "_stage_supervisor",
        lambda *args, **kwargs: stage_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        containment, "_cleanup_staged_helper", lambda *_args, **_kwargs: "",
    )
    return docker_calls, stage_calls


def test_mutable_container_name_is_pinned_to_inspected_full_id(
    monkeypatch, tmp_path,
):
    helper = tmp_path / "supervisor"
    helper.write_bytes(b"helper")
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSWE_HELPER_DIR", str(tmp_path / "helper-cache"),
    )
    observed: dict[str, object] = {"docker": []}

    def fake_docker(args, **_kwargs):
        observed["docker"].append(list(args))
        if args[:2] == ["inspect", "--"]:
            assert args[2] == "mutable-name"
            return subprocess.CompletedProcess(
                args, 0, stdout=_inspect_payload(), stderr="",
            )
        if "/bin/cat" in args:
            nonce = args[-1].removeprefix(containment._COMPLETION_PREFIX)
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{nonce} 0\n", stderr="",
            )
        assert _CANONICAL_ID in args
        assert "mutable-name" not in args
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(containment, "_run_docker", fake_docker)
    monkeypatch.setattr(
        containment, "_ensure_supervisor", lambda **_kwargs: helper,
    )
    monkeypatch.setattr(
        containment,
        "_stage_supervisor",
        lambda container_id, *_args, **_kwargs: observed.update(
            stage=container_id,
        ),
    )
    monkeypatch.setattr(
        containment,
        "_cleanup_staged_helper",
        lambda container_id, **_kwargs: (
            observed.update(cleanup=container_id) or ""
        ),
    )

    assert containment.run_contained_docker_exec(
        "mutable-name", "printf ok", timeout=10,
    ) == (0, "ok", "")
    assert observed["stage"] == _CANONICAL_ID
    assert observed["cleanup"] == _CANONICAL_ID


def test_command_runs_only_below_root_supervisor(monkeypatch, tmp_path):
    docker_calls, stage_calls = _mock_contained_invocation(
        monkeypatch, tmp_path, result_rc=0,
    )

    result = containment.run_contained_docker_exec(
        "candidate", "cd /app && pytest -q", timeout=10, cleanup_timeout=2,
    )

    assert result == (0, "visible output", "diagnostic")
    assert len(stage_calls) == 1
    command = next(
        call for call in docker_calls if containment._TRUSTED_BINARY in call
    )
    assert command[:4] == ["exec", "--user", "0:0", _CANONICAL_ID]
    assert command[4] == containment._TRUSTED_BINARY
    assert command[5:7] == ["1234", "2345"]
    assert command[-3:] == ["/bin/bash", "-c", "cd /app && pytest -q"]
    assert not containment._poison_path(
        tmp_path / "helper-cache", _CANONICAL_ID,
    ).exists()


def test_supervisor_failure_quarantines_source(monkeypatch, tmp_path):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=126)
    quarantined: list[tuple[str, float]] = []
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda container, *, deadline: (
            quarantined.append((container, deadline)) or (True, "force-removed")
        ),
    )

    rc, _stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert rc == 126
    assert len(quarantined) == 1
    assert quarantined[0][0] == _CANONICAL_ID
    assert quarantined[0][1] > time.monotonic()
    assert "force-removed" in stderr


def test_host_deadline_quarantines_started_command(monkeypatch, tmp_path):
    helper = tmp_path / "supervisor"
    helper.write_bytes(b"helper")
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSWE_HELPER_DIR", str(tmp_path / "helper-cache"),
    )
    monkeypatch.setattr(
        containment, "_validate_source_container",
        lambda *_args, **_kwargs: (
            1000,
            1000,
            _CANONICAL_ID,
            tmp_path / "app",
            tmp_path / "logs",
        ),
    )
    monkeypatch.setattr(
        containment, "_ensure_supervisor", lambda **_kwargs: helper,
    )
    monkeypatch.setattr(containment, "_stage_supervisor", lambda *_a, **_k: None)
    monkeypatch.setattr(
        containment, "_cleanup_staged_helper", lambda *_a, **_k: "",
    )
    calls = [0]

    def fake_docker(args, **_kwargs):
        calls[0] += 1
        if containment._TRUSTED_BINARY in args:
            raise containment.ExecContainmentError(
                "timed out", returncode=124,
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    quarantined: list[str] = []
    monkeypatch.setattr(containment, "_run_docker", fake_docker)
    monkeypatch.setattr(
        containment, "_quarantine_source",
        lambda container, **_kwargs: (
            quarantined.append(container) or (True, "force-removed")
        ),
    )

    rc, _stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert rc == containment.CONTAINMENT_UNUSABLE_RC
    assert quarantined == [_CANONICAL_ID]
    assert "force-removed" in stderr


def test_transport_rc125_without_attestation_is_poisoned_and_not_reused(
    monkeypatch, tmp_path,
):
    _docker_calls, stage_calls = _mock_contained_invocation(
        monkeypatch,
        tmp_path,
        result_rc=125,
        attestation="missing",
    )
    quarantined: list[str] = []
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda container, **_kwargs: (
            quarantined.append(container) or (False, "daemon unavailable")
        ),
    )

    first = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )
    second = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert first[0] == containment.CONTAINMENT_UNUSABLE_RC
    assert "completion attestation is missing" in first[2]
    assert "quarantine_FAILED" in first[2]
    assert second[0] == containment.CONTAINMENT_UNUSABLE_RC
    assert "durably unusable" in second[2]
    assert len(stage_calls) == 1
    assert quarantined == [_CANONICAL_ID, _CANONICAL_ID]


def test_mismatched_completion_attestation_forces_quarantine(
    monkeypatch, tmp_path,
):
    _mock_contained_invocation(
        monkeypatch, tmp_path, result_rc=0, attestation="mismatch",
    )
    quarantined: list[str] = []
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda container, **_kwargs: (
            quarantined.append(container) or (True, "force-removed")
        ),
    )

    result = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert result[0] == containment.CONTAINMENT_UNUSABLE_RC
    assert "did not match" in result[2]
    assert quarantined == [_CANONICAL_ID]


def test_attested_supervisor_rc125_remains_safe_descendant_failure(
    monkeypatch, tmp_path,
):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=125)
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda *_args, **_kwargs: pytest.fail("safe rc125 was quarantined"),
    )

    result = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert result == (125, "visible output", "diagnostic")


def test_cleanup_failure_cannot_report_gate_success(monkeypatch, tmp_path):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=0)
    monkeypatch.setattr(
        containment,
        "_cleanup_staged_helper",
        lambda *_args, **_kwargs: "cleanup failed",
    )

    rc, _stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert rc == 125
    assert "cleanup failed" in stderr


@pytest.mark.parametrize(
    "failure",
    ["inspect_fails", "not_running", "staging_fails"],
)
def test_docker_failure_before_the_command_starts_reports_not_started(
    monkeypatch, tmp_path, failure,
):
    docker_calls: list[list[str]] = []
    marker_armed_at_staging: list[bool] = []
    cache_directory = tmp_path / "helper-cache"
    if failure == "inspect_fails":
        monkeypatch.setenv(
            "CRUCIBLE_DEEPSWE_HELPER_DIR", str(cache_directory),
        )

        def failing_docker(args, **_kwargs):
            docker_calls.append(list(args[1:]))
            return subprocess.CompletedProcess(
                args, 1, stdout="", stderr="Error: No such object: candidate",
            )

        monkeypatch.setattr(containment.subprocess, "run", failing_docker)
        expected = "docker inspect -- failed rc=1"
    elif failure == "not_running":
        monkeypatch.setenv(
            "CRUCIBLE_DEEPSWE_HELPER_DIR", str(cache_directory),
        )
        inspected = json.loads(_inspect_payload())
        inspected[0]["State"]["Running"] = False

        def not_running(args, **_kwargs):
            docker_calls.append(list(args[1:]))
            return subprocess.CompletedProcess(
                args, 0, stdout=json.dumps(inspected), stderr="",
            )

        monkeypatch.setattr(containment.subprocess, "run", not_running)
        expected = "source container is not running"
    else:
        docker_calls, _stage_calls = _mock_contained_invocation(
            monkeypatch, tmp_path, result_rc=0,
        )

        def staging_fails(*_args, **_kwargs):
            marker_armed_at_staging.append(
                containment._poison_path(
                    cache_directory, _CANONICAL_ID,
                ).is_file()
            )
            raise containment.ExecContainmentError(
                "static exec-supervisor staging failed rc=1: no space left"
            )

        monkeypatch.setattr(containment, "_stage_supervisor", staging_fails)
        expected = "static exec-supervisor staging failed rc=1"
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda *_args, **_kwargs: pytest.fail(
            "a command that never started must not quarantine its source"
        ),
    )

    rc, stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "cd /app && pytest -q", timeout=10, cleanup_timeout=2,
    )

    assert rc == containment.CONTAINMENT_NOT_STARTED_RC
    assert stdout == ""
    assert expected in stderr
    assert not any(
        containment._TRUSTED_BINARY in call for call in docker_calls
    )
    if failure in {"inspect_fails", "not_running"}:
        assert docker_calls == [["inspect", "--", "candidate"]]
    if failure == "staging_fails":
        # The in-flight marker was armed when staging ran and is withdrawn
        # after its failure: nothing ran, so the source stays usable for the
        # next command.
        assert marker_armed_at_staging == [True]
        assert not containment._poison_path(
            cache_directory, _CANONICAL_ID,
        ).exists()


def test_invalid_invocation_reports_not_started(monkeypatch):
    monkeypatch.setattr(
        containment,
        "_run_docker",
        lambda *_args, **_kwargs: pytest.fail(
            "an invalid invocation must not reach Docker"
        ),
    )

    assert containment.run_contained_docker_exec(
        "not a container id", "tests", timeout=10,
    ) == (
        containment.CONTAINMENT_NOT_STARTED_RC,
        "",
        "invalid source container id",
    )
    assert containment.run_contained_docker_exec(
        "candidate", "", timeout=10,
    ) == (
        containment.CONTAINMENT_NOT_STARTED_RC,
        "",
        "contained command must be non-empty",
    )


def test_attested_status_equal_to_the_not_started_code_is_unusable(
    monkeypatch, tmp_path,
):
    _mock_contained_invocation(
        monkeypatch,
        tmp_path,
        result_rc=containment.CONTAINMENT_NOT_STARTED_RC,
    )
    quarantined: list[str] = []
    monkeypatch.setattr(
        containment,
        "_quarantine_source",
        lambda container, **_kwargs: (
            quarantined.append(container) or (True, "force-removed")
        ),
    )

    rc, _stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    # A command that ran can never be reported as one that did not.
    assert rc == containment.CONTAINMENT_UNUSABLE_RC
    assert quarantined == [_CANONICAL_ID]
    assert "force-removed" in stderr


def test_not_started_code_is_outside_every_started_outcome():
    not_started = containment.CONTAINMENT_NOT_STARTED_RC

    # The supervisor reports a command's own status only up to 119; the
    # threshold lives in its C source, so pin that text with the constant.
    assert (
        "child_status <= 119 ? child_status : EXIT_CHILD_LARGE_STATUS"
        in containment._SUPERVISOR_SOURCE
    )
    assert "#define EXIT_CHILD_LARGE_STATUS 123\n" in containment._SUPERVISOR_SOURCE
    assert 119 < not_started < 128
    assert not_started not in {
        containment._SUPERVISOR_SIGNAL_RC,
        123,
        containment._SUPERVISOR_TIMEOUT_RC,
        containment._SUPERVISOR_DESCENDANT_RC,
        containment._SUPERVISOR_FAILURE_RC,
        containment.CONTAINMENT_UNUSABLE_RC,
        127,
    }
    assert f" {not_started}\n" not in "".join(
        line + "\n"
        for line in containment._SUPERVISOR_SOURCE.splitlines()
        if line.startswith("#define EXIT_")
    )


@pytest.mark.parametrize("returncode", [124, 127])
def test_timeout_and_missing_docker_before_start_keep_their_codes(
    monkeypatch, tmp_path, returncode,
):
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSWE_HELPER_DIR", str(tmp_path / "helper-cache"),
    )

    def unavailable(*_args, **_kwargs):
        raise containment.ExecContainmentError(
            "docker did not answer", returncode=returncode,
        )

    monkeypatch.setattr(
        containment, "_validate_source_container", unavailable,
    )

    assert containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    ) == (returncode, "", "docker did not answer")


def test_success_path_shares_one_absolute_cleanup_deadline(
    monkeypatch,
    tmp_path,
):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=0)
    observed: dict[str, float] = {}
    fake_docker = containment._run_docker

    def recording_docker(args, **kwargs):
        if containment._TRUSTED_BINARY in args:
            observed["supervisor"] = kwargs["deadline"]
        elif "/bin/cat" in args:
            observed["attestation"] = kwargs["deadline"]
        return fake_docker(args, **kwargs)

    def cleanup(_container_id, *, deadline):
        observed["helper_cleanup"] = deadline
        return ""

    monkeypatch.setattr(containment, "_run_docker", recording_docker)
    monkeypatch.setattr(containment, "_cleanup_staged_helper", cleanup)

    result = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert result == (0, "visible output", "diagnostic")
    assert observed.keys() == {
        "supervisor", "attestation", "helper_cleanup",
    }
    assert len(set(observed.values())) == 1


def test_supervisor_receives_absolute_monotonic_deadlines(
    monkeypatch,
    tmp_path,
):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=0)
    fake_docker = containment._run_docker
    supervisor_args: list[str] = []
    monkeypatch.setattr(containment.time, "monotonic", lambda: 100.0)

    def recording_docker(args, **kwargs):
        if containment._TRUSTED_BINARY in args:
            supervisor_args[:] = args
            assert kwargs["deadline"] == 112.0
        return fake_docker(args, **kwargs)

    monkeypatch.setattr(containment, "_run_docker", recording_docker)

    result = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert result == (0, "visible output", "diagnostic")
    assert supervisor_args[7:9] == ["110000", "111000"]


def test_failure_path_shares_cleanup_deadline_with_quarantine(
    monkeypatch,
    tmp_path,
):
    _mock_contained_invocation(monkeypatch, tmp_path, result_rc=126)
    observed: dict[str, float] = {}
    fake_docker = containment._run_docker

    def recording_docker(args, **kwargs):
        if containment._TRUSTED_BINARY in args:
            observed["supervisor"] = kwargs["deadline"]
        elif "/bin/cat" in args:
            observed["attestation"] = kwargs["deadline"]
        return fake_docker(args, **kwargs)

    def quarantine(_container_id, *, deadline):
        observed["quarantine"] = deadline
        return True, "force-removed"

    monkeypatch.setattr(containment, "_run_docker", recording_docker)
    monkeypatch.setattr(containment, "_quarantine_source", quarantine)

    rc, _stdout, stderr = containment.run_contained_docker_exec(
        "candidate", "tests", timeout=10, cleanup_timeout=2,
    )

    assert rc == containment.CONTAINMENT_UNUSABLE_RC
    assert "force-removed" in stderr
    assert observed.keys() == {"supervisor", "attestation", "quarantine"}
    assert len(set(observed.values())) == 1


def test_expired_cleanup_deadline_starts_no_new_docker_operation(
    monkeypatch,
):
    monkeypatch.setattr(containment.time, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        containment.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "expired cleanup started a Docker operation"
        ),
    )

    cleanup_error = containment._cleanup_staged_helper(
        _CANONICAL_ID, deadline=10.0,
    )
    quarantined, quarantine_error = containment._quarantine_source(
        _CANONICAL_ID, deadline=10.0,
    )

    assert "deadline expired" in cleanup_error
    assert not quarantined
    assert "deadline expired" in quarantine_error


@pytest.mark.live_docker
def test_live_docker_detached_writer_is_reaped_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A real setsid descendant cannot mutate /app after the gate returns."""

    if os.getuid() == 0:
        pytest.skip("containment requires a non-root configured runtime user")
    available = subprocess.run(
        ["docker", "version"], capture_output=True, timeout=10, check=False,
    )
    if available.returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    alpine = subprocess.run(
        ["docker", "image", "inspect", "alpine:latest"],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if alpine.returncode != 0:
        pytest.skip("cached alpine:latest image is unavailable")

    tag = f"crucible-containment-test:{uuid.uuid4().hex}"
    dockerfile = (
        b"FROM alpine:latest\n"
        b"RUN printf '#!/bin/sh\\nexec /bin/sh \"$@\"\\n' > /bin/bash "
        b"&& chmod 0755 /bin/bash\n"
    )
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=dockerfile,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert built.returncode == 0, built.stderr.decode(errors="replace")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    git_dir = workspace / ".git"
    git_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    helper_dir = tmp_path / "helpers"
    monkeypatch.setenv("CRUCIBLE_DEEPSWE_HELPER_DIR", str(helper_dir))
    container_id = ""
    try:
        started = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--label", "io.crucible.managed=true",
                "--label", "io.crucible.phase=authoring",
                "--network=none", "--read-only",
                "--cap-drop", "ALL",
                "--cap-add", "SETUID",
                "--cap-add", "SETGID",
                "--cap-add", "KILL",
                "--security-opt", "no-new-privileges",
                "--pids-limit", "64",
                "--ulimit", "nofile=128:128",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=256m",
                "-v", f"{workspace}:/app:rw",
                "-v", f"{git_dir}:/app/.git:ro",
                "-v", f"{logs_dir}:/logs:rw",
                "--entrypoint", "/bin/sleep",
                tag, "infinity",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert started.returncode == 0, started.stderr
        container_id = started.stdout.strip()

        ordinary_caps = subprocess.run(
            [
                "docker", "exec", container_id, "/bin/sh", "-c",
                "grep -E '^Cap(Prm|Eff):' /proc/self/status",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert ordinary_caps.returncode == 0
        assert ordinary_caps.stdout.splitlines() == [
            "CapPrm:\t0000000000000000",
            "CapEff:\t0000000000000000",
        ]

        child_identity = containment.run_contained_docker_exec(
            container_id,
            (
                "id -u; id -g; id -G; "
                "grep -E '^Cap(Prm|Eff):' /proc/self/status"
            ),
            timeout=10,
            cleanup_timeout=3,
        )
        assert child_identity[0] == 0, child_identity
        assert child_identity[2] == ""
        assert child_identity[1].splitlines() == [
            str(os.getuid()),
            str(os.getgid()),
            str(os.getgid()),
            "CapPrm:\t0000000000000000",
            "CapEff:\t0000000000000000",
        ]

        # This descendant deliberately exits inside the supervisor's 10 ms
        # foreground polling interval.  The old implementation reaped its
        # zombie before the /proc census and lost the only evidence that it
        # had outlived the foreground shell, allowing an unsafe rc=0.
        short_lived = containment.run_contained_docker_exec(
            container_id,
            (
                "setsid /bin/sh -c 'sleep 0.001; "
                "echo escaped > /app/short-lived-sentinel' "
                ">/dev/null 2>&1 &"
            ),
            timeout=10,
            cleanup_timeout=3,
        )
        assert short_lived[0] == 125, short_lived
        assert "background descendants" in short_lived[2]
        (workspace / "short-lived-sentinel").unlink(missing_ok=True)

        rc, _stdout, stderr = containment.run_contained_docker_exec(
            container_id,
            (
                "setsid /bin/sh -c 'sleep 1; "
                "echo escaped > /app/sentinel' >/dev/null 2>&1 &"
            ),
            timeout=10,
            cleanup_timeout=3,
        )
        assert rc == 125
        assert "background descendants" in stderr
        time.sleep(1.5)
        assert not (workspace / "sentinel").exists()

        # A foreground command killed at its own deadline is not reported as
        # an escaped background child.  Its process is accounted for by the
        # same bounded reaping loop and the safely reusable container remains.
        timed_out = containment.run_contained_docker_exec(
            container_id, "sleep 30", timeout=1, cleanup_timeout=3,
        )
        assert timed_out[0] == 124, timed_out
        assert "command deadline expired" in timed_out[2]

        followup = containment.run_contained_docker_exec(
            container_id, "printf reusable", timeout=10, cleanup_timeout=3,
        )
        assert followup == (0, "reusable", "")
        assert subprocess.run(
            [
                "docker", "exec", container_id,
                "test", "-e", containment._TRUSTED_DIRECTORY,
            ],
            capture_output=True,
            timeout=10,
            check=False,
        ).returncode != 0

        extra = subprocess.run(
            [
                "docker", "exec", "-d", container_id,
                "/bin/sh", "-c", "sleep 30",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert extra.returncode == 0, extra.stderr
        preexisting = containment.run_contained_docker_exec(
            container_id, "printf must-not-run", timeout=10, cleanup_timeout=3,
        )
        assert preexisting[0] == containment.CONTAINMENT_UNUSABLE_RC, preexisting
        assert "pre-existing container processes" in preexisting[2]
        assert subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True,
            timeout=10,
            check=False,
        ).returncode != 0
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True, timeout=30, check=False,
            )
        subprocess.run(
            ["docker", "image", "rm", "-f", tag],
            capture_output=True, timeout=30, check=False,
        )


@pytest.mark.live_docker
def test_live_docker_swebench_profile_reaps_detached_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The shared supervisor enforces the SWE-bench /testbed topology."""

    if os.getuid() == 0:
        pytest.skip("containment requires a non-root configured runtime user")
    if subprocess.run(
        ["docker", "version"], capture_output=True, timeout=10, check=False,
    ).returncode != 0:
        pytest.skip("Docker daemon is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", "alpine:latest"],
        capture_output=True,
        timeout=10,
        check=False,
    ).returncode != 0:
        pytest.skip("cached alpine:latest image is unavailable")

    tag = f"crucible-swebench-containment-test:{uuid.uuid4().hex}"
    dockerfile = (
        b"FROM alpine:latest\n"
        b"RUN printf '#!/bin/sh\\nexec /bin/sh \"$@\"\\n' > /bin/bash "
        b"&& chmod 0755 /bin/bash\n"
    )
    built = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=dockerfile,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert built.returncode == 0, built.stderr.decode(errors="replace")

    workspace = tmp_path / "testbed"
    workspace.mkdir()
    git_dir = workspace / ".git"
    git_dir.mkdir()
    helper_dir = tmp_path / "helpers"
    monkeypatch.setenv("CRUCIBLE_SWEBENCH_HELPER_DIR", str(helper_dir))
    profile = containment.ContainerExecProfile(
        name="swebench",
        workspace_destination="/testbed",
        git_destination="/testbed/.git",
        logs_destination=None,
        allowed_phases=frozenset({"visible-verifier"}),
        helper_cache_env="CRUCIBLE_SWEBENCH_HELPER_DIR",
    )
    container_id = ""
    try:
        started = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--label", "io.crucible.managed=true",
                "--label", "io.crucible.phase=visible-verifier",
                "--network=none", "--read-only",
                "--cap-drop", "ALL",
                "--cap-add", "SETUID",
                "--cap-add", "SETGID",
                "--cap-add", "KILL",
                "--security-opt", "no-new-privileges",
                "--pids-limit", "64",
                "--ulimit", "nofile=128:128",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=256m",
                "-v", f"{workspace}:/testbed:rw",
                "-v", f"{git_dir}:/testbed/.git:ro",
                "--entrypoint", "/bin/sleep",
                tag, "infinity",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert started.returncode == 0, started.stderr
        container_id = started.stdout.strip()

        rc, _stdout, stderr = containment.run_contained_docker_exec(
            container_id,
            (
                "setsid /bin/sh -c 'sleep 1; "
                "echo escaped > /testbed/sentinel' >/dev/null 2>&1 &"
            ),
            timeout=10,
            cleanup_timeout=3,
            profile=profile,
        )
        assert rc == 125
        assert "background descendants" in stderr
        time.sleep(1.5)
        assert not (workspace / "sentinel").exists()

        assert containment.run_contained_docker_exec(
            container_id,
            "printf reusable",
            timeout=10,
            cleanup_timeout=3,
            profile=profile,
        ) == (0, "reusable", "")
    finally:
        if container_id:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                timeout=30,
                check=False,
            )
        subprocess.run(
            ["docker", "image", "rm", "-f", tag],
            capture_output=True,
            timeout=30,
            check=False,
        )
