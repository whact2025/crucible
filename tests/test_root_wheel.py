# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import io
from pathlib import Path
import tomllib
import zipfile

import pytest

from scripts.verify_root_wheel import (
    RootWheelError,
    inspect_root_wheel,
    verify_reproducible_root_wheels,
)

_ROOT = Path(__file__).resolve().parents[1]
_DISPATCHER = b"workflow-dispatcher\n"
_DIST_INFO = "crucible-9.9.9.dist-info"
_REQUIRED = {
    "crucible/.python-version",
    "crucible/__init__.py",
    "crucible/__main__.py",
    "crucible/backends/codex_worker/pyproject.toml",
    "crucible/backends/codex_worker/.python-version",
    "crucible/backends/codex_worker/uv.lock",
    "crucible/backends/deepseek_worker/pyproject.toml",
    "crucible/backends/deepseek_worker/.python-version",
    "crucible/backends/deepseek_worker/uv.lock",
    "crucible/backends/kimi_worker/pyproject.toml",
    "crucible/backends/kimi_worker/.python-version",
    "crucible/backends/kimi_worker/uv.lock",
    "crucible/products/deepswe/backends_codex_only.yaml",
    "crucible/pyproject.toml",
    "crucible/rubrics/_manifest.yaml",
    "crucible/tools/build/__init__.py",
    "crucible/tools/build/handlers.py",
    "crucible/uv.lock",
    "crucible/workflows/__main__.py",
    f"{_DIST_INFO}/METADATA",
    f"{_DIST_INFO}/RECORD",
    f"{_DIST_INFO}/WHEEL",
    f"{_DIST_INFO}/entry_points.txt",
}


def _wheel_payload(*, extra: tuple[str, ...] = (), missing: tuple[str, ...] = ()) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted((_REQUIRED | set(extra)) - set(missing)):
            payload = (
                _DISPATCHER
                if name == "crucible/workflows/__main__.py"
                else b"runtime\n"
            )
            archive.writestr(name, payload)
    return stream.getvalue()


def test_root_wheel_build_configuration_is_an_explicit_runtime_allowlist() -> None:
    configuration = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    wheel = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]

    assert wheel == {
        "force-include": {"tools/build": "tools/build"},
        "only-include": [
            ".python-version",
            "__init__.py",
            "__main__.py",
            "constraints.py",
            "orchestration.py",
            "pyproject.toml",
            "uv.lock",
            "backends",
            "common",
            "products",
            "rubrics",
            "rules",
            "scripts",
            "tools",
            "workflows",
        ],
        "sources": {"": "crucible"},
        "exclude": [
            "**/AGENTS.md",
            "**/CLAUDE.md",
            "**/README.md",
            "**/__pycache__/**",
            "products/**/tests/**",
        ],
    }


def test_root_wheel_inspector_accepts_the_closed_runtime_shape(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "crucible-9.9.9-py3-none-any.whl"
    wheel.write_bytes(_wheel_payload())

    evidence = inspect_root_wheel(wheel, expected_dispatcher=_DISPATCHER)

    assert evidence["wheel"] == wheel.name
    assert evidence["member_count"] == len(_REQUIRED)
    assert evidence["dist_info"] == _DIST_INFO


@pytest.mark.parametrize(
    "member, message",
    (
        ("crucible/tests/test_hidden.py", "forbidden root-wheel member"),
        ("crucible/docs/internal.md", "forbidden root-wheel member"),
        ("crucible/presentation.pptx", "unallowlisted root file"),
        ("backends/kimi.py", "flattened root-wheel member"),
        ("crucible/native/retained_authority/source.c", "forbidden root-wheel member"),
        ("crucible/analysis/scaling.py", "unallowlisted root directory"),
        ("crucible/bench/gpu.py", "unallowlisted root directory"),
        ("crucible/dashboard/app.py", "unallowlisted root directory"),
        ("crucible/./tools/hidden.py", "noncanonical root-wheel member"),
        ("crucible//tools/hidden.py", "noncanonical root-wheel member"),
        ("crucible/tools/../hidden.py", "noncanonical root-wheel member"),
        ("crucible/products/swebench/CLAUDE.md", "forbidden root-wheel member"),
        ("crucible/backends/kimi_worker/README.md", "forbidden root-wheel member"),
    ),
)
def test_root_wheel_inspector_rejects_nonruntime_members(
    tmp_path: Path,
    member: str,
    message: str,
) -> None:
    wheel = tmp_path / "crucible-9.9.9-py3-none-any.whl"
    wheel.write_bytes(_wheel_payload(extra=(member,)))

    with pytest.raises(RootWheelError, match=message):
        inspect_root_wheel(wheel, expected_dispatcher=_DISPATCHER)


@pytest.mark.parametrize("member", (
    "crucible/.python-version",
    "crucible/backends/codex_worker/.python-version",
    "crucible/backends/deepseek_worker/.python-version",
    "crucible/backends/kimi_worker/.python-version",
    "crucible/tools/build/__init__.py",
    "crucible/tools/build/handlers.py",
))
def test_root_wheel_requires_benchmark_checkpoint_python_identity(
    tmp_path: Path, member: str,
) -> None:
    wheel = tmp_path / "crucible-9.9.9-py3-none-any.whl"
    wheel.write_bytes(_wheel_payload(missing=(member,)))
    with pytest.raises(RootWheelError, match="omitted required members"):
        inspect_root_wheel(wheel, expected_dispatcher=_DISPATCHER)


def test_root_wheel_reproducibility_discovers_the_dynamic_filename(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    source = tmp_path / "source"
    first.mkdir()
    second.mkdir()
    (source / "workflows").mkdir(parents=True)
    (source / "workflows" / "__main__.py").write_bytes(_DISPATCHER)
    name = "crucible-9.9.9-py3-none-any.whl"
    payload = _wheel_payload()
    (first / name).write_bytes(payload)
    (second / name).write_bytes(payload)

    evidence = verify_reproducible_root_wheels(
        first,
        second,
        source_root=source,
    )

    assert evidence["wheel"] == name
    assert evidence["reproducible"] is True


def test_root_wheel_reproducibility_rejects_different_bytes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    source = tmp_path / "source"
    first.mkdir()
    second.mkdir()
    source.mkdir()
    name = "crucible-9.9.9-py3-none-any.whl"
    (first / name).write_bytes(_wheel_payload())
    (second / name).write_bytes(_wheel_payload(extra=("crucible/tools/extra.py",)))

    with pytest.raises(RootWheelError, match="bytes differ"):
        verify_reproducible_root_wheels(first, second, source_root=source)
