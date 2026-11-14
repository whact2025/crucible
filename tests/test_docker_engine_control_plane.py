# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hermetic adversarial tests for the Unix Docker Engine control plane."""

from __future__ import annotations

import json
import math
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from crucible.common.docker_engine_control_plane import (
    DockerDaemonGenerationError,
    DockerEngineDeadlineError,
    DockerEngineHTTPResponse,
    DockerEngineProtocolError,
    DockerEngineQueryError,
    DockerPeerCredentials,
    LocalDockerDaemonGenerationProbe,
    ProcDockerDaemonGenerationProbe,
    SystemdSocketActivatedDockerDaemonGenerationProbe,
    UnixDockerEngineControlPlane,
)
from crucible.common.docker_settlement import (
    DOCKER_ARM_GENERATION_LABEL,
    DOCKER_RUN_GENERATION_LABEL,
    DOCKER_SETTLEMENT_LABEL,
)


_A = "a" * 64
_B = "b" * 64
_PEER = DockerPeerCredentials(pid=4242, uid=1000, gid=1000)


def _response(
    status: int,
    payload: object | bytes = b"",
    *,
    peer: DockerPeerCredentials = _PEER,
) -> DockerEngineHTTPResponse:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return DockerEngineHTTPResponse(
        status=status,
        headers=(("content-length", str(len(body))),),
        body=body,
        peer=peer,
    )


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _FakeTransport:
    def __init__(
        self,
        *responses: DockerEngineHTTPResponse | Exception,
        after_request: Callable[[], None] | None = None,
    ) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, float]] = []
        self.after_request = after_request

    def request(
        self,
        method: str,
        target: str,
        *,
        deadline_monotonic: float,
    ) -> DockerEngineHTTPResponse:
        self.calls.append((method, target, deadline_monotonic))
        if not self.responses:
            raise AssertionError("unexpected Docker request")
        result = self.responses.pop(0)
        if self.after_request is not None:
            self.after_request()
        if isinstance(result, Exception):
            raise result
        return result


def _control(
    transport: _FakeTransport,
    *,
    clock: _Clock | None = None,
    generation_probe: Callable[[DockerPeerCredentials], str] | None = None,
    socket_path: str = "/run/docker.sock",
) -> UnixDockerEngineControlPlane:
    return UnixDockerEngineControlPlane(
        socket_path,
        api_version="1.41",
        transport=transport,
        generation_probe=generation_probe or (lambda _peer: "daemon-generation"),
        monotonic=clock or _Clock(),
    )


def _version() -> dict[str, object]:
    return {
        "ApiVersion": "1.51",
        "MinAPIVersion": "1.24",
        "Version": "28.1.1",
    }


def _info(*, engine_id: str = "engine-a", root: str = "/var/lib/docker"):
    return {
        "ID": engine_id,
        "ServerVersion": "28.1.1",
        "DockerRootDir": root,
        "Driver": "overlay2",
        "CgroupDriver": "systemd",
        "CgroupVersion": "2",
        "SecurityOptions": ["name=seccomp,profile=builtin", "name=apparmor"],
        "OSType": "linux",
        "Architecture": "x86_64",
        "LiveRestoreEnabled": False,
    }


def test_inspect_maps_only_exact_http_404_to_authoritative_absence() -> None:
    transport = _FakeTransport(
        _response(404, b"not JSON and deliberately ignored"),
        _response(500, {"message": "No such container: looks absent"}),
    )
    control = _control(transport)

    assert control.inspect_container("missing", deadline_monotonic=20.0) is None
    with pytest.raises(DockerEngineQueryError, match="HTTP 500"):
        control.inspect_container("missing", deadline_monotonic=20.0)

    assert [call[0] for call in transport.calls] == ["GET", "GET"]


def test_inspect_transport_failure_never_becomes_absence() -> None:
    transport = _FakeTransport(ConnectionResetError("daemon disappeared"))
    control = _control(transport)

    with pytest.raises(DockerEngineQueryError, match="transport query failed"):
        control.inspect_container("missing", deadline_monotonic=20.0)

    assert len(transport.calls) == 1


def test_inspect_returns_full_canonical_id_and_settlement_labels_only() -> None:
    transport = _FakeTransport(
        _response(
            200,
            {
                "Id": _A,
                "Config": {
                    "Labels": {
                        DOCKER_SETTLEMENT_LABEL: "true",
                        DOCKER_RUN_GENERATION_LABEL: "run-a",
                        "untrusted.unrelated": "not retained",
                    }
                },
            },
        )
    )
    control = _control(transport)

    observed = control.inspect_container("short-or-name", deadline_monotonic=20.0)

    assert observed is not None
    assert observed.container_id == _A
    assert dict(observed.labels) == {
        DOCKER_RUN_GENERATION_LABEL: "run-a",
        DOCKER_SETTLEMENT_LABEL: "true",
    }
    assert transport.calls[0][1].endswith("/containers/short-or-name/json")


@pytest.mark.parametrize("container_id", ["abc123", "A" * 64, "a" * 63])
def test_inspect_rejects_noncanonical_id_in_success_response(
    container_id: str,
) -> None:
    control = _control(
        _FakeTransport(
            _response(
                200,
                {
                    "Id": container_id,
                    "Config": {"Labels": {}},
                },
            )
        )
    )

    with pytest.raises(DockerEngineProtocolError, match="non-canonical"):
        control.inspect_container("any", deadline_monotonic=20.0)


def test_list_sends_exact_label_filters_and_supplied_limit() -> None:
    labels = {
        DOCKER_SETTLEMENT_LABEL: "true",
        DOCKER_RUN_GENERATION_LABEL: "run-a",
        DOCKER_ARM_GENERATION_LABEL: "arm=a/b",
    }
    transport = _FakeTransport(_response(200, [{"Id": _A}, {"Id": _B}]))
    control = _control(transport)

    assert control.list_container_ids(
        labels=labels,
        limit=2,
        deadline_monotonic=20.0,
    ) == (_A, _B)

    method, target, deadline = transport.calls[0]
    split = urlsplit(target)
    query = parse_qs(split.query, keep_blank_values=True)
    assert method == "GET"
    assert split.path == "/v1.41/containers/json"
    assert deadline == 20.0
    assert query["all"] == ["1"]
    assert query["limit"] == ["2"]
    assert json.loads(query["filters"][0]) == {
        "label": [f"{key}={value}" for key, value in sorted(labels.items())],
    }


def test_list_rejects_truncated_id_and_server_limit_violation() -> None:
    short_control = _control(
        _FakeTransport(_response(200, [{"Id": "abc123"}])),
    )
    with pytest.raises(DockerEngineProtocolError, match="non-canonical"):
        short_control.list_container_ids(
            labels={},
            limit=1,
            deadline_monotonic=20.0,
        )

    over_limit = _control(
        _FakeTransport(_response(200, [{"Id": _A}, {"Id": _B}])),
    )
    with pytest.raises(DockerEngineProtocolError, match="more.*requested"):
        over_limit.list_container_ids(
            labels={},
            limit=1,
            deadline_monotonic=20.0,
        )


def test_remove_uses_only_full_id_and_is_idempotent_for_exact_404() -> None:
    transport = _FakeTransport(_response(204), _response(404))
    control = _control(transport)

    with pytest.raises(ValueError, match="full canonical"):
        control.remove_container("short", deadline_monotonic=20.0)
    assert (
        control.remove_container_with_result(_A, deadline_monotonic=20.0)
        == "REMOVED_HTTP_204"
    )
    assert (
        control.remove_container_with_result(_B, deadline_monotonic=20.0)
        == "ABSENT_HTTP_404"
    )

    assert len(transport.calls) == 2
    assert transport.calls[0][0] == "DELETE"
    assert transport.calls[0][1].startswith(f"/v1.41/containers/{_A}?")
    assert parse_qs(urlsplit(transport.calls[0][1]).query) == {
        "force": ["1"],
        "v": ["1"],
    }


@pytest.mark.parametrize("deadline", [math.nan, math.inf, -math.inf, 10.0])
def test_invalid_or_expired_deadline_rejects_before_transport(
    deadline: float,
) -> None:
    transport = _FakeTransport(_response(404))
    control = _control(transport)

    with pytest.raises(DockerEngineDeadlineError):
        control.inspect_container("missing", deadline_monotonic=deadline)

    assert transport.calls == []


def test_transport_return_after_deadline_is_not_accepted() -> None:
    clock = _Clock()
    transport = _FakeTransport(
        _response(404),
        after_request=lambda: setattr(clock, "value", 21.0),
    )
    control = _control(transport, clock=clock)

    with pytest.raises(DockerEngineDeadlineError, match="expired"):
        control.inspect_container("missing", deadline_monotonic=20.0)


def test_daemon_identity_binds_endpoint_engine_config_and_generation() -> None:
    seen_peers: list[DockerPeerCredentials] = []
    generations = iter(["daemon-1", "daemon-2", "daemon-2"])

    def probe(peer: DockerPeerCredentials) -> str:
        seen_peers.append(peer)
        return next(generations)

    responses = [
        _response(200, _version()),
        _response(200, _info()),
        _response(200, _version()),
        _response(200, _info()),
        _response(200, _version()),
        _response(200, _info(root="/srv/docker")),
    ]
    control = _control(_FakeTransport(*responses), generation_probe=probe)

    original = control.daemon_identity(deadline_monotonic=20.0)
    restarted = control.daemon_identity(deadline_monotonic=20.0)
    reconfigured = control.daemon_identity(deadline_monotonic=20.0)

    assert original.engine_id == restarted.engine_id == reconfigured.engine_id
    assert original.generation == "daemon-1"
    assert restarted.generation == reconfigured.generation == "daemon-2"
    assert original.fingerprint != restarted.fingerprint
    assert restarted.configuration_fingerprint != (
        reconfigured.configuration_fingerprint
    )
    assert seen_peers == [_PEER, _PEER, _PEER]

    same_daemon_other_endpoint = _control(
        _FakeTransport(_response(200, _version()), _response(200, _info())),
        generation_probe=lambda _peer: "daemon-1",
        socket_path="/alternate/docker.sock",
    ).daemon_identity(deadline_monotonic=20.0)
    assert original.endpoint_fingerprint != (
        same_daemon_other_endpoint.endpoint_fingerprint
    )


def test_daemon_identity_rejects_peer_drift_and_api_disagreement() -> None:
    other_peer = DockerPeerCredentials(pid=4243, uid=1000, gid=1000)
    peer_drift = _control(
        _FakeTransport(
            _response(200, _version()),
            _response(200, _info(), peer=other_peer),
        )
    )
    with pytest.raises(DockerEngineQueryError, match="peer changed"):
        peer_drift.daemon_identity(deadline_monotonic=20.0)

    incompatible = _version()
    incompatible["MinAPIVersion"] = "1.42"
    api_drift = _control(_FakeTransport(_response(200, incompatible)))
    with pytest.raises(DockerEngineProtocolError, match="outside.*range"):
        api_drift.daemon_identity(deadline_monotonic=20.0)


def _write_proc_generation(
    proc_root: Path,
    *,
    pid: int,
    starttime: int,
    comm: str = "dockerd",
    parent_pid: int = 0,
) -> None:
    process_root = proc_root / str(pid)
    process_root.mkdir(parents=True, exist_ok=True)
    # fields_after_comm[0] is field 3; index 19 is starttime field 22.
    fields = [
        "S",
        str(parent_pid),
        *("0" for _ in range(17)),
        str(starttime),
    ]
    (process_root / "stat").write_text(
        f"{pid} ({comm}) {' '.join(fields)}\n",
        encoding="ascii",
    )
    (process_root / "comm").write_text(f"{comm}\n", encoding="ascii")
    executable = process_root / "exe"
    executable.unlink(missing_ok=True)
    executable.symlink_to(f"/usr/bin/{comm}")
    boot_root = proc_root / "sys/kernel/random"
    boot_root.mkdir(parents=True, exist_ok=True)
    (boot_root / "boot_id").write_text(
        "12345678-1234-5678-9abc-1234567890ab\n",
        encoding="ascii",
    )


def test_default_proc_generation_changes_with_process_starttime(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path.resolve() / "proc"
    peer = DockerPeerCredentials(pid=77, uid=0, gid=0)
    _write_proc_generation(proc_root, pid=peer.pid, starttime=100)
    probe = ProcDockerDaemonGenerationProbe(proc_root)

    first = probe(peer)
    (proc_root / str(peer.pid) / "stat").write_text(
        f"{peer.pid} (dockerd) {' '.join(['S', *('0' for _ in range(18)), '101'])}\n",
        encoding="ascii",
    )
    second = probe(peer)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_default_proc_generation_fails_closed_when_source_or_dockerd_absent(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path.resolve() / "proc"
    peer = DockerPeerCredentials(pid=88, uid=0, gid=0)
    probe = ProcDockerDaemonGenerationProbe(proc_root)

    with pytest.raises(DockerDaemonGenerationError, match="unavailable"):
        probe(peer)

    _write_proc_generation(
        proc_root,
        pid=peer.pid,
        starttime=100,
        comm="rootlesskit",
    )
    with pytest.raises(DockerDaemonGenerationError, match="not.*dockerd"):
        probe(peer)


def test_systemd_socket_activation_generation_binds_dockerd_restart(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path.resolve() / "proc"
    pid_file = tmp_path.resolve() / "docker.pid"
    uid = os.getuid()
    peer = DockerPeerCredentials(pid=1, uid=uid, gid=os.getgid())
    _write_proc_generation(
        proc_root,
        pid=1,
        starttime=10,
        comm="systemd",
        parent_pid=0,
    )
    _write_proc_generation(
        proc_root,
        pid=77,
        starttime=100,
        parent_pid=1,
    )
    pid_file.write_text("77\n", encoding="ascii")
    pid_file.chmod(0o644)
    probe = SystemdSocketActivatedDockerDaemonGenerationProbe(
        proc_root=proc_root,
        pid_file=pid_file,
        required_uid=uid,
    )

    first = probe(peer)
    _write_proc_generation(
        proc_root,
        pid=77,
        starttime=101,
        parent_pid=1,
    )
    second = probe(peer)

    assert first.startswith("sha256:")
    assert second.startswith("sha256:")
    assert first != second


def test_systemd_socket_activation_rejects_writable_pid_authority(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path.resolve() / "proc"
    pid_file = tmp_path.resolve() / "docker.pid"
    uid = os.getuid()
    peer = DockerPeerCredentials(pid=1, uid=uid, gid=os.getgid())
    _write_proc_generation(
        proc_root,
        pid=1,
        starttime=10,
        comm="systemd",
        parent_pid=0,
    )
    pid_file.write_text("77\n", encoding="ascii")
    pid_file.chmod(0o664)
    probe = SystemdSocketActivatedDockerDaemonGenerationProbe(
        proc_root=proc_root,
        pid_file=pid_file,
        required_uid=uid,
    )

    with pytest.raises(DockerDaemonGenerationError, match="root-owned"):
        probe(peer)


def test_local_generation_probe_rejects_unknown_peer_topology(
    tmp_path: Path,
) -> None:
    proc_root = tmp_path.resolve() / "proc"
    _write_proc_generation(
        proc_root,
        pid=91,
        starttime=10,
        comm="rootlesskit",
    )
    probe = LocalDockerDaemonGenerationProbe(proc_root=proc_root)

    with pytest.raises(DockerDaemonGenerationError, match="unsupported"):
        probe(DockerPeerCredentials(pid=91, uid=os.getuid(), gid=os.getgid()))


def test_duplicate_json_keys_are_rejected() -> None:
    control = _control(
        _FakeTransport(
            _response(
                200,
                b'{"Id":"'
                + _A.encode()
                + b'","Id":"'
                + _B.encode()
                + b'","Config":{"Labels":{}}}',
            )
        )
    )

    with pytest.raises(DockerEngineProtocolError, match="duplicate key"):
        control.inspect_container("any", deadline_monotonic=20.0)


class _UnixHTTPServer:
    def __init__(self, path: Path, response: bytes | None) -> None:
        self.path = path
        self.response = response
        self.request = b""
        self.accepted = threading.Event()
        self.release = threading.Event()
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(os.fspath(path))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        connection, _address = self._listener.accept()
        with connection:
            self.accepted.set()
            while b"\r\n\r\n" not in self.request:
                chunk = connection.recv(4_096)
                if not chunk:
                    return
                self.request += chunk
            if self.response is None:
                self.release.wait(timeout=2.0)
                return
            connection.sendall(self.response)

    def __enter__(self) -> _UnixHTTPServer:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release.set()
        self._listener.close()
        self._thread.join(timeout=2.0)


def test_real_unix_http_transport_uses_engine_api_not_cli_text(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "docker.sock"
    body = b"Error: No such container!!!"
    response = (
        b"HTTP/1.1 404 Not Found\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"Connection: close\r\n\r\n"
        + body
    )
    with _UnixHTTPServer(socket_path, response) as server:
        control = UnixDockerEngineControlPlane(
            socket_path,
            generation_probe=lambda _peer: "unused",
        )
        assert (
            control.inspect_container(
                "missing/name",
                deadline_monotonic=time.monotonic() + 2.0,
            )
            is None
        )

    assert server.request.startswith(
        b"GET /v1.41/containers/missing%2Fname/json HTTP/1.1\r\n"
    )


def test_real_unix_http_transport_obeys_absolute_deadline(
    tmp_path: Path,
) -> None:
    socket_path = tmp_path / "docker.sock"
    with _UnixHTTPServer(socket_path, None) as server:
        control = UnixDockerEngineControlPlane(
            socket_path,
            generation_probe=lambda _peer: "unused",
        )
        started = time.monotonic()
        with pytest.raises(DockerEngineDeadlineError):
            control.inspect_container(
                "missing",
                deadline_monotonic=started + 0.1,
            )
        elapsed = time.monotonic() - started

    assert server.accepted.is_set()
    assert elapsed < 1.0
