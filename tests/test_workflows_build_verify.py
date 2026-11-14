# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Core gates cannot convert missing runner settlement into build success."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from crucible.workflows import _build_verify


@pytest.mark.parametrize("unsettled_role", ["build", "verify"])
def test_unsettled_runner_blocks_the_combined_gate(
    monkeypatch, tmp_path, unsettled_role: str,
) -> None:
    spec = SimpleNamespace(
        project_context=SimpleNamespace(
            runners={
                "build": "crucible.common.runners.polyglot",
                "verify": "crucible.common.runners.polyglot",
            },
            runner_parameters={
                "per_language": {"rust": {"project_subdir": "native/brain"}},
            },
        ),
        constraints=(SimpleNamespace(
            kind="languages", languages=("python", "rust", "cpp"),
        ),),
    )
    # The subprocess supervisor owns this opaque family; dispatch must pass
    # exactly the caller's family through each attempted phase.
    writer_family = (object(),)
    roles: list[str] = []

    def supervised_runner(
        command,
        *,
        config_json,
        settlement_callback,
        writer_epoch_members,
        **_,
    ):
        parameters = json.loads(config_json)["parameters"]
        role = parameters["role"]
        roles.append(role)
        assert writer_epoch_members is writer_family
        assert parameters["languages"] == ["python", "rust", "cpp"]
        assert parameters["per_language"]["rust"]["project_subdir"] == (
            "native/brain"
        )
        if role == unsettled_role:
            raise _build_verify._RunnerSettlementError(
                "runner terminal attestation is unavailable"
            )
        settlement_callback(
            "sha256:" + ("a" if role == "build" else "b") * 64
        )
        return subprocess.CompletedProcess(
            command, 0, stdout=b'{"values": {"returncode": 0}}', stderr=b"",
        )

    monkeypatch.setattr(
        _build_verify, "_run_runner_subprocess", supervised_runner,
    )

    with pytest.raises(
        _build_verify._RunnerSettlementError, match="terminal attestation",
    ):
        _build_verify.build_and_verify(
            spec,
            cwd=str(tmp_path),
            log_dir=tmp_path / "logs",
            writer_epoch_members=writer_family,
        )

    assert roles == (["build"] if unsettled_role == "build" else ["build", "verify"])
