#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Verify the reproducible, importable Crucible root-wheel artifact policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
import zipfile

_PACKAGE_ROOT = "crucible"
_ALLOWED_ROOT_FILES = frozenset(
    {
        ".python-version",
        "__init__.py",
        "__main__.py",
        "constraints.py",
        "orchestration.py",
        "pyproject.toml",
        "uv.lock",
    }
)
_ALLOWED_ROOT_DIRECTORIES = frozenset(
    {
        "backends",
        "common",
        "products",
        "rubrics",
        "rules",
        "scripts",
        "tools",
        "workflows",
    }
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".claude",
        ".codex",
        ".crucible",
        ".git",
        ".github",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
        "__pycache__",
        "crucible-regression",
        "docs",
        "native",
        "tests",
    }
)
_REQUIRED_MEMBERS = frozenset(
    {
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
    }
)
_DISPATCHER_MEMBER = "crucible/workflows/__main__.py"


class RootWheelError(ValueError):
    """The built root wheel violates its closed artifact contract."""


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _single_wheel(directory: Path) -> Path:
    wheels = tuple(sorted(directory.glob("*.whl")))
    if len(wheels) != 1 or not wheels[0].is_file():
        raise RootWheelError(
            f"expected exactly one wheel in {directory}, found {len(wheels)}"
        )
    return wheels[0]


def inspect_root_wheel(
    wheel: Path,
    *,
    expected_dispatcher: bytes,
) -> dict[str, object]:
    """Return bounded evidence after enforcing the root-wheel member policy."""

    try:
        archive = zipfile.ZipFile(wheel)
    except (OSError, zipfile.BadZipFile) as exc:
        raise RootWheelError(f"root wheel is unreadable: {wheel}") from exc

    with archive:
        infos = archive.infolist()
        names = tuple(info.filename for info in infos)
        if not names:
            raise RootWheelError("root wheel is empty")
        if len(names) != len(set(names)):
            raise RootWheelError("root wheel contains duplicate members")
        if len(names) != len({name.casefold() for name in names}):
            raise RootWheelError("root wheel contains case-colliding members")

        dist_info_roots: set[str] = set()
        for info in infos:
            name = info.filename
            segments = name.split("/")
            path = PurePosixPath(name)
            if (
                not name
                or name.endswith("/")
                or path.is_absolute()
                or "\\" in name
                or any(part in {"", ".", ".."} for part in segments)
            ):
                raise RootWheelError(f"noncanonical root-wheel member: {name!r}")
            mode = info.external_attr >> 16
            file_type = stat.S_IFMT(mode)
            if file_type not in (0, stat.S_IFREG):
                raise RootWheelError(f"non-regular root-wheel member: {name!r}")
            if mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
                raise RootWheelError(f"privileged root-wheel mode: {name!r}")

            root = path.parts[0]
            if root.endswith(".dist-info"):
                if not root.startswith("crucible-"):
                    raise RootWheelError(f"foreign dist-info root: {root!r}")
                dist_info_roots.add(root)
                continue
            if root != _PACKAGE_ROOT or len(path.parts) < 2:
                raise RootWheelError(f"flattened root-wheel member: {name!r}")
            if any(part in _FORBIDDEN_PARTS for part in path.parts[1:]):
                raise RootWheelError(f"forbidden root-wheel member: {name!r}")
            top_level = path.parts[1]
            if len(path.parts) == 2:
                if top_level not in _ALLOWED_ROOT_FILES:
                    raise RootWheelError(f"unallowlisted root file: {name!r}")
            elif top_level not in _ALLOWED_ROOT_DIRECTORIES:
                raise RootWheelError(f"unallowlisted root directory: {name!r}")

        if len(dist_info_roots) != 1:
            raise RootWheelError(
                "root wheel must contain exactly one Crucible dist-info root"
            )
        dist_info = next(iter(dist_info_roots))
        required = _REQUIRED_MEMBERS | {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/entry_points.txt",
        }
        missing = sorted(required.difference(names))
        if missing:
            raise RootWheelError(
                "root wheel omitted required members: " + ", ".join(missing)
            )
        dispatcher = archive.read(_DISPATCHER_MEMBER)
        if dispatcher != expected_dispatcher:
            raise RootWheelError("root-wheel dispatcher differs from source")

        return {
            "compressed_bytes": sum(info.compress_size for info in infos),
            "dist_info": dist_info,
            "member_count": len(infos),
            "sha256": _sha256_file(wheel),
            "uncompressed_bytes": sum(info.file_size for info in infos),
            "wheel": wheel.name,
        }


def verify_reproducible_root_wheels(
    first_directory: Path,
    second_directory: Path,
    *,
    source_root: Path,
) -> dict[str, object]:
    """Discover, compare, and inspect two independently built wheels."""

    first = _single_wheel(first_directory)
    second = _single_wheel(second_directory)
    if first.name != second.name:
        raise RootWheelError("independent root-wheel filenames differ")
    first_payload = first.read_bytes()
    if first_payload != second.read_bytes():
        raise RootWheelError("independent root-wheel bytes differ")
    dispatcher_path = source_root / "workflows" / "__main__.py"
    try:
        expected_dispatcher = dispatcher_path.read_bytes()
    except OSError as exc:
        raise RootWheelError(
            f"source dispatcher is unavailable: {dispatcher_path}"
        ) from exc
    evidence = inspect_root_wheel(
        first,
        expected_dispatcher=expected_dispatcher,
    )
    evidence["reproducible"] = True
    return evidence


def verify_runtime_imports(wheel: Path) -> dict[str, object]:
    """Import every command and hash its recipe from an extracted built wheel."""
    with tempfile.TemporaryDirectory(prefix="crucible-wheel-") as directory:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(directory)
        probe = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import crucible
from crucible.common.benchmark_runtime import file_digest
from crucible.workflows.__main__ import _discover_workflows, _resolve_main
from crucible.workflows._backend_recipe import backend_recipe_identity
root = Path(crucible.__file__).resolve().parent
assert root == Path(sys.argv[1]) / 'crucible', root
for name in _discover_workflows():
    _resolve_main(name)
file_digest(root / '.python-version')
recipe = backend_recipe_identity(root, digest_file=file_digest)
print(json.dumps({'commands': sorted(_discover_workflows()), 'workers': sorted(recipe['isolated_worker_projects'])}))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", probe, directory],
            capture_output=True, text=True, timeout=60, check=False,
        )
        if result.returncode:
            raise RootWheelError(
                "built runtime cannot import its commands/recipes: " + result.stderr[-4096:]
            )
        return json.loads(result.stdout)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-directory", required=True, type=Path)
    parser.add_argument("--second-directory", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--smoke-runtime", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    evidence = verify_reproducible_root_wheels(
        args.first_directory,
        args.second_directory,
        source_root=args.source_root,
    )
    if args.smoke_runtime:
        evidence["runtime"] = verify_runtime_imports(_single_wheel(args.first_directory))
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
