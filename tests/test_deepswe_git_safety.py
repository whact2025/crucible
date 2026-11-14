# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import shlex
import subprocess

import pytest

from crucible.products.deepswe import git_safety


def test_hermetic_container_git_environment_scrubs_execution_authority():
    inherited = (
        "PATH=/image/toolchain/bin:/usr/bin:/bin",
        "PROJECT_FEATURES=sim,cuda",
        "HOME=/attacker/home",
        "GIT_DIR=/attacker/repo",
        "GIT_CONFIG_COUNT=1",
        "GIT_CONFIG_KEY_0=alias.status",
        "GIT_CONFIG_VALUE_0=!touch /app/owned",
        "BASH_ENV=/app/evil-bashrc",
        "ENV=/app/evil-shrc",
        "SHELLOPTS=xtrace:verbose",
        "BASH_FUNC_git%%=() { touch /app/function-owned; }",
        "--option-shaped=value",
    )

    env = git_safety.hermetic_container_git_environment(inherited)

    assert env["PATH"] == "/image/toolchain/bin:/usr/bin:/bin"
    assert env["PROJECT_FEATURES"] == "sim,cuda"
    assert env["HOME"] == "/tmp/home"
    assert "BASH_ENV" not in env
    assert "ENV" not in env
    assert "SHELLOPTS" not in env
    assert not any(key.startswith("BASH_FUNC_") for key in env)
    assert not any("touch /app" in value for value in env.values())
    assert not any(key.startswith("GIT_") for key in env)

    prefix = git_safety.hermetic_container_git_exec_prefix(inherited)
    assert prefix[:3] == ["/usr/bin/env", "-i", "--"]
    assert "PROJECT_FEATURES=sim,cuda" in prefix
    assert not any("evil" in argument or "function-owned" in argument
                   for argument in prefix)


def test_git_wrapper_is_safe_and_not_inherited_by_nested_project_shell():
    wrapper = git_safety.hermetic_container_git_shell_function()

    assert wrapper.startswith("git() {")
    assert "/usr/bin/git" in wrapper
    assert "--git-dir=/app/.git" in wrapper
    assert "--work-tree=/app" in wrapper
    assert "GIT_CONFIG_NOSYSTEM=1" in wrapper
    assert "GIT_CONFIG_GLOBAL=/dev/null" in wrapper
    assert "core.hooksPath=/dev/null" in wrapper
    assert "core.fsmonitor=false" in wrapper
    assert "protocol.allow=never" in wrapper
    assert "diff --no-ext-diff --no-textconv" in wrapper
    assert "export -f" not in wrapper

    nested_assertion = (
        'test "$(type -t git || true)" != function && '
        'test -z "${GIT_DIR+x}" && '
        'test -z "${GIT_WORK_TREE+x}" && '
        'test -z "${GIT_CONFIG_NOSYSTEM+x}" && '
        'test -z "${GIT_CONFIG_GLOBAL+x}"'
    )
    script = (
        f"{wrapper}\n"
        'test "$(type -t git)" = function\n'
        f"/bin/bash --noprofile --norc -c {shlex.quote(nested_assertion)}"
    )
    result = subprocess.run(
        ["/bin/bash", "--noprofile", "--norc", "-c", script],
        env=git_safety.hermetic_container_git_environment((
            "PATH=/usr/bin:/bin",
            "GIT_DIR=/attacker",
        )),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_hermetic_container_path_drops_candidate_writable_entries():
    env = git_safety.hermetic_container_git_environment((
        "PATH=:/app/bin://app/double-slash:relative:/tmp/tools:/logs/bin:/tests/bin:"
        "/solution/bin:/tmp/crucible-rust-immutable/cargo-bin:"
        "/tmp/crucible-cargo-home/bin:/opt/tool/bin:/root/.cargo/bin:/usr/bin:/bin:"
        "/app/../app/other",
    ))

    assert env["PATH"] == (
        "/tmp/crucible-rust-immutable/cargo-bin:/opt/tool/bin:"
        "/root/.cargo/bin:/usr/bin:/bin"
    )


def test_inspect_container_environment_uses_option_terminator(monkeypatch):
    observed: dict[str, object] = {}

    def _fake(args, **kwargs):
        observed["args"] = list(args)
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps(["PATH=/usr/bin:/bin", "SAFE=1"]),
            stderr="",
        )

    monkeypatch.setattr(git_safety.subprocess, "run", _fake)

    result = git_safety.inspect_docker_container_environment("--hostile-id")

    assert result == ("PATH=/usr/bin:/bin", "SAFE=1")
    assert observed["args"] == [
        "docker",
        "inspect",
        "--format",
        "{{json .Config.Env}}",
        "--",
        "--hostile-id",
    ]
    assert observed["kwargs"]["timeout"] == 30  # type: ignore[index]


@pytest.mark.parametrize("payload", ["not-json", "{}", '["NO_EQUALS"]'])
def test_inspect_container_environment_rejects_invalid_shape(
    monkeypatch,
    payload: str,
):
    monkeypatch.setattr(
        git_safety.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args,
            0,
            stdout=payload,
            stderr="",
        ),
    )

    with pytest.raises(git_safety.ContainerGitEnvironmentError):
        git_safety.inspect_docker_container_environment("abc123")
