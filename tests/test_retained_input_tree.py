# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Product-neutral retained input-tree transport tests."""

from __future__ import annotations

import ast
import os
from pathlib import Path, PurePosixPath

import pytest

from crucible.common.runners import retained_input_tree as retained


def test_retained_input_tree_supports_a_synthetic_non_product_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog" / "terrain_package"
    (root / "meshes").mkdir(parents=True)
    payloads = {
        "description.txt": b"sealed terrain package\n",
        "meshes/tile.bin": b"tile\x00bytes\n",
    }
    records = []
    for relative, payload in payloads.items():
        (root / relative).write_bytes(payload)
        records.append(
            {
                "path": relative,
                "sha256": retained.sha256_bytes(payload),
                "size": len(payload),
            }
        )
    records.sort(key=lambda item: os.fsencode(item["path"]))
    manifest_payload = retained.canonical_json_bytes(
        {"files": records, "schema_version": 1}
    )
    (root / "inventory.json").write_bytes(manifest_payload)
    manifest = retained.RetainedInputTreeMember(
        relative=PurePosixPath("inventory.json"),
        size=len(manifest_payload),
        sha256=retained.sha256_bytes(manifest_payload),
    )

    def load_inventory(payload: bytes) -> tuple[retained.RetainedInputTreeMember, ...]:
        decoded = retained.strict_json_object(payload)
        assert decoded["schema_version"] == 1
        return tuple(
            retained.RetainedInputTreeMember(
                relative=retained.canonical_relative_path(item["path"]),
                size=item["size"],
                sha256=item["sha256"],
            )
            for item in decoded["files"]
        )

    authority = retained.open_retained_input_tree(
        source_root=root,
        destination=PurePosixPath("/run/qualification-inputs/terrain"),
        contract_fingerprint="sha256:" + "7" * 64,
        manifest=manifest,
        inventory_loader=load_inventory,
    )
    try:
        authority.assert_stable(full=True)
        authority.binding.assert_named()
        assert authority.binding.source_root == root
        assert authority.binding.fingerprint.startswith("sha256:")
        root_metadata = root.stat(follow_symlinks=False)
        assert authority.binding.matches_root_stat_identity(
            (
                root_metadata.st_dev,
                root_metadata.st_ino,
                root_metadata.st_uid,
                root_metadata.st_gid,
                root_metadata.st_mode & 0o7777,
            )
        )
        assert {str(item.relative) for item in authority.files} == {
            "description.txt",
            "inventory.json",
            "meshes/tile.bin",
        }
    finally:
        authority.close()

    with pytest.raises(
        retained.RetainedInputTreeError,
        match="input authority is closed",
    ):
        authority.assert_stable()


def test_retained_input_tree_source_has_no_product_dependency_or_identity() -> None:
    source_path = Path(retained.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    folded = source.casefold()

    for forbidden_token in (
        "axl15",
        "forklift",
        "igo",
        "joint",
        "lidar",
        "omniverse",
        "robot",
    ):
        assert forbidden_token not in folded, source_path

    imported_modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
    assert not {
        name for name in imported_modules if name.startswith("crucible.products")
    }
