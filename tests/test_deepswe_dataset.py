# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.products.deepswe.dataset``.

Hermetic — uses ``tmp_path`` to construct a fake deep-swe corpus
shape per test. No network, no real clone, no docker. The fixture
shape matches the manifest and task schema accepted by the loader.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from crucible.products.deepswe import dataset as dataset_module
from crucible.products.deepswe.dataset import (
    DeepSWEInstance,
    DeepSWEInstanceError,
    DeepSWETestsTreeIdentity,
    filter_instances,
    fingerprint_task_tests_tree,
    load_dataset,
    load_snapshot_test_inventory,
    load_task_test_inventory,
    resolve_task_tests_dir,
    snapshot_task_tests_tree,
)
from crucible.products.deepswe.secure_artifacts import RewardTestInventory


# ---------- fixture builders ----------


def _write_task_dir(
    tasks_root: Path,
    *,
    task_id: str,
    ext_id: str = "khfixturedefault",
    repository_url: str = "https://github.com/ex/ex",
    base_commit_hash: str = "0123456789abcdef0123456789abcdef01234567",
    language: str = "python",
    category: str = "feature_request",
    instruction: str = "Default instruction body.",
    cpus: int = 2,
    memory_mb: int = 8192,
    storage_mb: int = 20480,
    agent_timeout_sec: float = 5400.0,
    verifier_timeout_sec: float = 1800.0,
    allow_internet: bool = False,
    docker_image_override: str | None = None,
    write_solution: bool = True,
    write_test_patch: bool = True,
) -> Path:
    """Construct one task directory matching the verified deep-swe
    schema. Returns the task dir path."""
    task_dir = tasks_root / task_id
    task_dir.mkdir(parents=True)

    docker_image = (
        docker_image_override
        if docker_image_override is not None
        else f"public.ecr.aws/d3j8x8q7/swe-bench-202605:{ext_id}"
    )
    task_toml = f'''\
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]

[task]
name = "datacurve/{task_id}"
description = ""
authors = []
keywords = []

[metadata]
ext_id = "{ext_id}"
task_id = "{task_id}"
display_title = "Title"
display_description = "Description"
original_title = "Original title"
repository_url = "{repository_url}"
base_commit_hash = "{base_commit_hash}"
category = "{category}"
language = "{language}"

[verifier]
network_mode = "{"no-network" if not allow_internet else "host"}"
environment_mode = "separate"
timeout_sec = {verifier_timeout_sec}
env = {{}}

[verifier.environment]
build_timeout_sec = 1800.0
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}

[agent]
network_mode = "{"no-network" if not allow_internet else "host"}"
timeout_sec = {agent_timeout_sec}

[environment]
docker_image = "{docker_image}"
build_timeout_sec = 1800.0
os = "linux"
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = {storage_mb}
gpus = 0
mcp_servers = []
env = {{}}

[solution]
env = {{}}
'''
    (task_dir / "task.toml").write_text(task_toml)
    (task_dir / "instruction.md").write_text(instruction)

    if write_solution:
        (task_dir / "solution").mkdir()
        (task_dir / "solution" / "solution.patch").write_text(
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
        )
    if write_test_patch:
        (task_dir / "tests").mkdir()
        (task_dir / "tests" / "test.patch").write_text(
            "diff --git a/test b/test\n@@ -1 +1 @@\n-a\n+b\n"
        )
    return task_dir


def _write_manifest(
    tasks_root: Path,
    entries: list[dict],
    *,
    declared_count: int | None = None,
    extra_top_level: dict | None = None,
) -> Path:
    """Construct ``tasks/manifest.json`` from supplied entries."""
    manifest: dict = {
        "schema_version": "1.0",
        "dataset": "deep-swe-1-1",
        "source_dataset": "swe-bench-ultra",
        "task_count": (declared_count if declared_count is not None else len(entries)),
        "id_policy": {
            "format": "lowercase kebab-case, filesystem-safe, <=56 chars",
            "source": (
                "wide-research AI summary of repository, original title, "
                "and instruction excerpt"
            ),
            "original_id_location": "task.toml [metadata].ext_id",
            "new_id_location": "task.toml [metadata].task_id",
        },
        "label_generation": {
            "tool": "fixture",
            "model": "fixture",
            "run_dir": "fixture",
            "results_jsonl": "fixture",
            "prompt_config": "fixture",
        },
        "tasks": entries,
    }
    if extra_top_level:
        manifest.update(extra_top_level)
    manifest_path = tasks_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def _manifest_entry(
    task_id: str,
    *,
    ext_id: str = "khfixturedefault",
    repo: str = "example/example",
    repository_url: str | None = None,
    language: str = "python",
    category: str = "feature_request",
    display_title: str = "Title",
    display_description: str = "Description",
    original_title: str = "Original title",
) -> dict:
    return {
        "ext_id": ext_id,
        "task_id": task_id,
        "display_title": display_title,
        "display_description": display_description,
        "original_title": original_title,
        "repo": repo,
        "repository_url": repository_url or f"https://github.com/{repo}",
        "language": language,
        "category": category,
    }


# ---------- happy path ----------


def test_load_dataset_two_tasks(tmp_path: Path):
    """Two-task happy path. All required fields populate; gold +
    test patches load from sibling files."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    _write_task_dir(
        tasks_root,
        task_id="abs-module-cache-flags",
        ext_id="kh75679ajj3b8dtd7se3h7z0a1833y6r",
        repository_url="https://github.com/abs-lang/abs",
        base_commit_hash="cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a",
        language="go",
        category="enhancement",
        instruction="Harden ABS module resolution and caching.",
    )
    _write_task_dir(
        tasks_root,
        task_id="textual-richlog-follow-state",
        ext_id="kh70247kxfz01g03p1dpz9fqv183g47e",
        repository_url="https://github.com/Textualize/textual",
        base_commit_hash="deadbeefcafebabe1234567890abcdef12345678",
        language="python",
        category="feature_request",
        instruction="Restore RichLog follow-mode parity.",
    )
    manifest_path = _write_manifest(
        tasks_root,
        [
            _manifest_entry(
                "abs-module-cache-flags",
                ext_id="kh75679ajj3b8dtd7se3h7z0a1833y6r",
                repo="abs-lang/abs",
                language="go",
                category="enhancement",
            ),
            _manifest_entry(
                "textual-richlog-follow-state",
                ext_id="kh70247kxfz01g03p1dpz9fqv183g47e",
                repo="Textualize/textual",
                language="python",
                category="feature_request",
            ),
        ],
    )

    instances = load_dataset(manifest_path)
    assert len(instances) == 2

    by_id = {i.task_id: i for i in instances}
    abs_inst = by_id["abs-module-cache-flags"]
    assert isinstance(abs_inst, DeepSWEInstance)
    assert abs_inst.repo == "abs-lang/abs"
    assert abs_inst.repo_org == "abs-lang"
    assert abs_inst.repo_name == "abs"
    assert abs_inst.language == "go"
    assert abs_inst.category == "enhancement"
    assert abs_inst.base_commit_hash == "cb1b3b671d0ee9fa9da9f7b02f86967953ffd10a"
    assert abs_inst.docker_image == (
        "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh75679ajj3b8dtd7se3h7z0a1833y6r"
    )
    assert abs_inst.cpus == 2
    assert abs_inst.memory_mb == 8192
    assert abs_inst.storage_mb == 20480
    assert abs_inst.allow_internet is False
    assert abs_inst.agent_timeout_sec == 5400.0
    assert abs_inst.verifier_timeout_sec == 1800.0
    assert abs_inst.instruction == "Harden ABS module resolution and caching."
    assert "diff --git" in abs_inst.gold_solution_patch
    assert "diff --git" in abs_inst.test_patch

    tx_inst = by_id["textual-richlog-follow-state"]
    assert tx_inst.repo == "Textualize/textual"
    assert tx_inst.repo_org == "Textualize"
    assert tx_inst.language == "python"


def test_load_dataset_rejects_noncanonical_language_case(tmp_path: Path):
    """Corpus authority admits only the declared lowercase language domain."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(
        tasks_root,
        task_id="t1",
        ext_id="kh1",
        language="Python",  # mixed case in task.toml
    )
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex", language="Python")],
    )
    with pytest.raises(DeepSWEInstanceError, match="language.*supported.*domain"):
        load_dataset(manifest_path)


def test_load_dataset_optional_files(tmp_path: Path):
    """Missing solution.patch / test.patch → empty strings (these
    are audit-only fields, not required for the agent's run)."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(
        tasks_root,
        task_id="t1",
        ext_id="kh1",
        write_solution=False,
        write_test_patch=False,
    )
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    instances = load_dataset(manifest_path)
    assert instances[0].gold_solution_patch == ""
    assert instances[0].test_patch == ""


# ---------- error paths ----------


def test_load_dataset_missing_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "tasks" / "manifest.json")


def test_load_dataset_manifest_not_object(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest_path = tasks_root / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]))
    with pytest.raises(DeepSWEInstanceError, match="top-level must be a JSON object"):
        load_dataset(manifest_path)


def test_load_dataset_tasks_field_missing(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest_path = tasks_root / "manifest.json"
    manifest_path = _write_manifest(tasks_root, [])
    manifest = json.loads(manifest_path.read_text())
    del manifest["tasks"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DeepSWEInstanceError, match="schema fields differ.*tasks"):
        load_dataset(manifest_path)


def test_load_dataset_task_count_mismatch(tmp_path: Path):
    """Declared task_count must match len(tasks) — drift indicates
    corpus corruption."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
        declared_count=99,
    )
    with pytest.raises(DeepSWEInstanceError, match="task_count=99 disagrees"):
        load_dataset(manifest_path)


def test_load_dataset_task_dir_missing(tmp_path: Path):
    """Manifest references a task that's not on disk."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("not-on-disk", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(DeepSWEInstanceError, match="missing task directory"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    "task_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "nested/task",
        r"nested\task",
        "-flag",
        "flag-",
        "two--hyphens",
        "Uppercase",
        "snake_case",
        "dotted.name",
        "a" * 57,
    ],
)
def test_load_dataset_rejects_unsafe_task_id_components(
    tmp_path: Path,
    task_id: str,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry(task_id, ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(DeepSWEInstanceError, match="lowercase kebab-case"):
        load_dataset(manifest_path)


def test_load_dataset_accepts_56_character_kebab_task_id(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_id = "a" * 56
    _write_task_dir(tasks_root, task_id=task_id, ext_id="kh1")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry(task_id, ext_id="kh1", repo="ex/ex")],
    )

    assert load_dataset(manifest_path)[0].task_id == task_id


@pytest.mark.parametrize(
    "ext_id",
    [
        True,
        7,
        None,
        "",
        "kh",
        "not-kh",
        "KHUPPER",
        "kh/../../x",
        "khnewline\n",
        "kh" + "a" * 63,
    ],
)
def test_validate_ext_id_rejects_noncanonical_values(ext_id: object) -> None:
    """The private parser neither coerces values nor broadens source syntax."""

    with pytest.raises(DeepSWEInstanceError, match="manifest ext_id"):
        dataset_module._validate_ext_id(ext_id)


@pytest.mark.parametrize(
    "ext_id",
    [
        True,
        7,
        None,
        "",
        "kh",
        "not-kh",
        "KHUPPER",
        "kh/../../x",
        "khnewline\n",
        "kh" + "a" * 63,
    ],
)
def test_load_dataset_rejects_noncanonical_ext_id(
    tmp_path: Path,
    ext_id: object,
) -> None:
    """Full manifest admission applies the exact same hostile-value policy."""

    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    entry = _manifest_entry("t1", ext_id="kh1", repo="ex/ex")
    entry["ext_id"] = ext_id
    manifest_path = _write_manifest(tasks_root, [entry])

    with pytest.raises(DeepSWEInstanceError, match="ext_id"):
        load_dataset(manifest_path)


def test_load_dataset_accepts_64_character_ext_id_boundary(tmp_path: Path) -> None:
    """The fixed compatibility ceiling is inclusive and loader-enforced."""

    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    ext_id = "kh" + "a" * 62
    assert len(ext_id) == dataset_module._MAX_EXT_ID_LENGTH == 64
    assert dataset_module._validate_ext_id(ext_id) == ext_id
    _write_task_dir(tasks_root, task_id="t1", ext_id=ext_id)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id=ext_id, repo="ex/ex")],
    )

    assert load_dataset(manifest_path)[0].ext_id == ext_id


def test_dataset_ancestor_rebound_close_never_retries_after_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "nested" / "deeper"
    target.mkdir(parents=True)
    initial_descriptor_count = len(os.listdir("/proc/self/fd"))
    descriptors, names = dataset_module._open_directory_chain(target)
    held_descriptor_count = len(os.listdir("/proc/self/fd"))
    real_close = os.close
    real_open = os.open
    reused_descriptor: int | None = None
    injected = False

    def close_then_reuse(descriptor: int) -> None:
        nonlocal injected, reused_descriptor
        if not injected:
            injected = True
            real_close(descriptor)
            reused_descriptor = real_open(
                os.devnull,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
            )
            assert reused_descriptor == descriptor
            raise OSError("injected rebound close settlement failure")
        real_close(descriptor)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(dataset_module.os, "close", close_then_reuse)
            assert not dataset_module._directory_chain_is_stable(
                descriptors,
                names,
            )
        assert injected
        assert reused_descriptor is not None
        os.fstat(reused_descriptor)
        assert len(os.listdir("/proc/self/fd")) == held_descriptor_count + 1
    finally:
        if reused_descriptor is not None:
            try:
                real_close(reused_descriptor)
            except OSError:
                pass
        for descriptor in reversed(descriptors):
            real_close(descriptor)
    assert len(os.listdir("/proc/self/fd")) == initial_descriptor_count


def test_load_dataset_rejects_task_directory_symlink(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    outside = tmp_path / "outside"
    _write_task_dir(outside, task_id="t1", ext_id="kh1")
    (tasks_root / "t1").symlink_to(outside / "t1", target_is_directory=True)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(DeepSWEInstanceError, match="must not be a symlink"):
        load_dataset(manifest_path)


def test_load_dataset_rejects_tests_directory_symlink(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(
        tasks_root,
        task_id="t1",
        ext_id="kh1",
        write_test_patch=False,
    )
    outside_tests = tmp_path / "outside-tests"
    outside_tests.mkdir()
    (outside_tests / "test.patch").write_text("outside")
    (task_dir / "tests").symlink_to(outside_tests, target_is_directory=True)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(
        DeepSWEInstanceError,
        match="tests directory must not be a symlink",
    ):
        load_dataset(manifest_path)


def test_load_dataset_rejects_instruction_symlink(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    outside_instruction = tmp_path / "outside-instruction.md"
    outside_instruction.write_text("do something unsafe")
    (task_dir / "instruction.md").unlink()
    (task_dir / "instruction.md").symlink_to(outside_instruction)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(
        DeepSWEInstanceError,
        match="instruction.md must not be a symlink",
    ):
        load_dataset(manifest_path)


def test_resolve_task_tests_dir_rechecks_mount_source_after_load(
    tmp_path: Path,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    load_dataset(manifest_path)

    outside_tests = tmp_path / "outside-tests"
    outside_tests.mkdir()
    original_tests = task_dir / "tests"
    for child in original_tests.iterdir():
        child.unlink()
    original_tests.rmdir()
    original_tests.symlink_to(outside_tests, target_is_directory=True)

    with pytest.raises(
        DeepSWEInstanceError,
        match="tests directory must not be a symlink",
    ):
        resolve_task_tests_dir(manifest_path, "t1")


def _add_snapshot_test_files(task_dir: Path) -> Path:
    tests_dir = task_dir / "tests"
    test_script = tests_dir / "test.sh"
    test_script.write_text("#!/bin/sh\nprintf 'ok\\n'\n")
    test_script.chmod(0o755)
    nested = tests_dir / "fixtures"
    nested.mkdir()
    (nested / "input.bin").write_bytes(b"fixture\x00bytes")
    return tests_dir


def test_snapshot_task_tests_tree_is_private_deterministic_and_detached(
    tmp_path: Path,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    identity = fingerprint_task_tests_tree(manifest_path, "t1")
    first = snapshot_task_tests_tree(
        manifest_path,
        "t1",
        tmp_path / "private-one",
        expected_identity=identity,
    )
    second = snapshot_task_tests_tree(
        manifest_path,
        "t1",
        tmp_path / "private-two",
        expected_identity=identity,
    )

    assert first.identity == identity
    assert second.identity == identity
    assert first.sha256 == second.sha256
    assert first.file_count == 3
    assert first.directory_count == 2
    assert first.total_bytes == sum(
        path.stat().st_size for path in tests_dir.rglob("*") if path.is_file()
    )
    assert first.root.parent == tmp_path / "private-one"
    assert first.root != tests_dir
    assert (first.root / "test.sh").read_text() == ("#!/bin/sh\nprintf 'ok\\n'\n")
    assert (first.root / "fixtures" / "input.bin").read_bytes() == (b"fixture\x00bytes")
    for path in (first.root, *first.root.rglob("*")):
        assert path.stat().st_mode & 0o222 == 0

    (tests_dir / "fixtures" / "input.bin").write_bytes(b"changed source")
    assert (first.root / "fixtures" / "input.bin").read_bytes() == (b"fixture\x00bytes")
    assert fingerprint_task_tests_tree(manifest_path, "t1") != identity


@pytest.mark.parametrize("linked_path", ["test.sh", "fixtures/escape"])
def test_tests_tree_identity_rejects_every_symlink(
    tmp_path: Path,
    linked_path: str,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    outside = tmp_path / "outside"
    outside.write_text("outside")
    link = tests_dir / linked_path
    if link.exists():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(DeepSWEInstanceError, match="symlink|regular tests/test.sh"):
        fingerprint_task_tests_tree(manifest_path, "t1")


def test_tests_tree_identity_rejects_special_files_without_blocking(
    tmp_path: Path,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    os.mkfifo(tests_dir / "attacker-fifo")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(DeepSWEInstanceError, match="special file"):
        fingerprint_task_tests_tree(manifest_path, "t1")


def test_snapshot_rejects_source_changed_since_checkpoint(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    identity = fingerprint_task_tests_tree(manifest_path, "t1")
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 1\n")
    staging = tmp_path / "private"

    with pytest.raises(DeepSWEInstanceError, match="changed after checkpoint"):
        snapshot_task_tests_tree(
            manifest_path,
            "t1",
            staging,
            expected_identity=identity,
        )
    assert list(staging.iterdir()) == []


def test_snapshot_rejects_file_mutation_during_copy(
    tmp_path: Path,
    monkeypatch,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    racing_file = tests_dir / "race.bin"
    racing_file.write_bytes(b"race-marker" + b"x" * (128 * 1024))
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    identity = fingerprint_task_tests_tree(manifest_path, "t1")
    original_read = dataset_module.os.read
    mutated = False

    def _racing_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, count)
        if not mutated and payload.startswith(b"race-marker"):
            mutated = True
            racing_file.write_bytes(b"replacement")
        return payload

    monkeypatch.setattr(dataset_module.os, "read", _racing_read)
    staging = tmp_path / "private"

    with pytest.raises(DeepSWEInstanceError, match="changed while copying"):
        snapshot_task_tests_tree(
            manifest_path,
            "t1",
            staging,
            expected_identity=identity,
        )
    assert mutated is True
    assert list(staging.iterdir()) == []


def test_snapshot_requires_typed_expected_identity(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    _add_snapshot_test_files(task_dir)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )

    with pytest.raises(TypeError, match="DeepSWETestsTreeIdentity"):
        snapshot_task_tests_tree(
            manifest_path,
            "t1",
            tmp_path / "private",
            expected_identity="not-an-identity",  # type: ignore[arg-type]
        )

    assert isinstance(
        fingerprint_task_tests_tree(manifest_path, "t1"),
        DeepSWETestsTreeIdentity,
    )


def test_snapshot_rejects_symlinked_private_staging_root(tmp_path: Path):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    _add_snapshot_test_files(task_dir)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    identity = fingerprint_task_tests_tree(manifest_path, "t1")
    outside = tmp_path / "outside-staging"
    outside.mkdir()
    staging_link = tmp_path / "private"
    staging_link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DeepSWEInstanceError, match="real directory"):
        snapshot_task_tests_tree(
            manifest_path,
            "t1",
            staging_link,
            expected_identity=identity,
        )
    assert list(outside.iterdir()) == []


def _snapshot_with_config(tmp_path: Path, config: bytes | None):
    """Snapshot one task whose tests tree optionally declares a whitelist."""

    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    if config is not None:
        (tests_dir / "config.json").write_bytes(config)
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    snapshot = snapshot_task_tests_tree(
        manifest_path,
        "t1",
        tmp_path / "private",
        expected_identity=fingerprint_task_tests_tree(manifest_path, "t1"),
    )
    return manifest_path, tests_dir, snapshot


def _tests_config(f2p: object, p2p: object) -> bytes:
    return json.dumps(
        {
            "base_commit": "0" * 40,
            "f2p_node_ids": f2p,
            "grade": {"format": "ctrf", "reports": []},
            "p2p_node_ids": p2p,
        }
    ).encode()


def test_snapshot_test_inventory_counts_deduplicated_stripped_ids(
    tmp_path: Path,
):
    # The dataset grader strips each id and drops blanks and repeats.
    manifest_path, _tests_dir, snapshot = _snapshot_with_config(
        tmp_path,
        _tests_config(["a", " a ", "b", "", "  "], ["x"]),
    )

    expected = RewardTestInventory(f2p_total=2, p2p_total=1)
    assert load_snapshot_test_inventory(snapshot) == expected
    assert load_task_test_inventory(manifest_path, "t1") == expected
    assert expected.to_payload() == {"f2p_total": 2, "p2p_total": 1}


def test_snapshot_test_inventory_is_absent_without_config_json(tmp_path: Path):
    manifest_path, _tests_dir, snapshot = _snapshot_with_config(tmp_path, None)

    assert load_snapshot_test_inventory(snapshot) is None
    assert load_task_test_inventory(manifest_path, "t1") is None


@pytest.mark.parametrize(
    ("config", "reason"),
    (
        (b"{not json", "not strict JSON"),
        (b'{"f2p_node_ids": ["a"], "p2p_node_ids": ["\xff"]}', "not strict JSON"),
        (b'["a"]', "one JSON object"),
        (b'{"f2p_node_ids": ["a"]}', "p2p_node_ids must be an exact string list"),
        (b'{"p2p_node_ids": ["a"]}', "f2p_node_ids must be an exact string list"),
        (_tests_config("a", ["x"]), "f2p_node_ids must be an exact string list"),
        (_tests_config(["a"], {"x": 1}), "p2p_node_ids must be an exact string list"),
        (_tests_config(["a", 7], ["x"]), "f2p_node_ids must be an exact string list"),
        (
            b'{"f2p_node_ids": ["a"], "p2p_node_ids": [], "f2p_node_ids": []}',
            "repeats a JSON object key",
        ),
        (
            b'{"f2p_node_ids": ["a"], "p2p_node_ids": [], "grade": NaN}',
            "non-finite",
        ),
        # More digits than Python converts to an integer: ``json.loads``
        # raises a plain ``ValueError`` for it, not a ``JSONDecodeError``.
        (
            b'{"f2p_node_ids": ["a"], "p2p_node_ids": [], "grade": '
            + b"1" * 5000
            + b"}",
            "not strict JSON",
        ),
    ),
    ids=(
        "not_json",
        "invalid_utf8",
        "top_level_list",
        "p2p_missing",
        "f2p_missing",
        "ids_not_a_list",
        "ids_a_mapping",
        "non_string_id",
        "duplicate_key",
        "non_finite_constant",
        "oversized_integer",
    ),
)
def test_snapshot_test_inventory_rejects_malformed_config(
    tmp_path: Path,
    config: bytes,
    reason: str,
):
    manifest_path, _tests_dir, snapshot = _snapshot_with_config(tmp_path, config)

    with pytest.raises(DeepSWEInstanceError, match=reason):
        load_snapshot_test_inventory(snapshot)
    with pytest.raises(DeepSWEInstanceError, match=reason):
        load_task_test_inventory(manifest_path, "t1")


def test_snapshot_test_inventory_reads_the_private_copy(tmp_path: Path):
    manifest_path, tests_dir, snapshot = _snapshot_with_config(
        tmp_path,
        _tests_config(["a", "b"], ["x", "y", "z"]),
    )
    (tests_dir / "config.json").write_bytes(_tests_config(["a"], []))

    assert load_snapshot_test_inventory(snapshot) == RewardTestInventory(
        f2p_total=2,
        p2p_total=3,
    )
    # The live tree is what the fail-fast check reads; it is not authority.
    assert load_task_test_inventory(manifest_path, "t1") == RewardTestInventory(
        f2p_total=1,
        p2p_total=0,
    )


def test_test_inventory_never_follows_a_config_symlink(tmp_path: Path):
    manifest_path, tests_dir, snapshot = _snapshot_with_config(tmp_path, None)
    outside = tmp_path / "outside-config.json"
    outside.write_bytes(_tests_config(["a"], ["x"]))
    (tests_dir / "config.json").symlink_to(outside)
    snapshot.root.chmod(0o700)
    (snapshot.root / "config.json").symlink_to(outside)

    with pytest.raises(DeepSWEInstanceError, match="must not be a symlink"):
        load_task_test_inventory(manifest_path, "t1")
    with pytest.raises(DeepSWEInstanceError, match="must not be a symlink"):
        load_snapshot_test_inventory(snapshot)


def _oversized_config(path: Path) -> None:
    # One byte over the read limit; the file is sparse until it is copied.
    with path.open("wb") as file:
        file.truncate(dataset_module._MAX_TASK_TEXT_BYTES + 1)


@pytest.mark.parametrize(
    "place",
    (_oversized_config, Path.mkdir),
    ids=("oversized", "directory"),
)
def test_test_inventory_rejects_a_config_it_cannot_read_safely(
    tmp_path: Path,
    place,
):
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    tests_dir = _add_snapshot_test_files(task_dir)
    place(tests_dir / "config.json")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    # The snapshot copies such a tree as it is; both loaders refuse it.
    snapshot = snapshot_task_tests_tree(
        manifest_path,
        "t1",
        tmp_path / "private",
        expected_identity=fingerprint_task_tests_tree(manifest_path, "t1"),
    )

    reason = "cannot safely read tests/config.json"
    with pytest.raises(DeepSWEInstanceError, match=reason):
        load_task_test_inventory(manifest_path, "t1")
    with pytest.raises(DeepSWEInstanceError, match=reason):
        load_snapshot_test_inventory(snapshot)


def test_load_dataset_manifest_entry_missing_field(tmp_path: Path):
    """Manifest entry without one of the required fields fails fast."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    # Manually construct an entry missing `language`.
    manifest_path = tasks_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset": "deep-swe-1-1",
                "source_dataset": "swe-bench-ultra",
                "task_count": 1,
                "id_policy": {
                    "format": "lowercase kebab-case, filesystem-safe, <=56 chars",
                    "source": (
                        "wide-research AI summary of repository, original title, "
                        "and instruction excerpt"
                    ),
                    "original_id_location": "task.toml [metadata].ext_id",
                    "new_id_location": "task.toml [metadata].task_id",
                },
                "label_generation": {
                    "tool": "fixture",
                    "model": "fixture",
                    "run_dir": "fixture",
                    "results_jsonl": "fixture",
                    "prompt_config": "fixture",
                },
                "tasks": [
                    {
                        "task_id": "t1",
                        "ext_id": "kh1",
                        "repo": "ex/ex",
                        "repository_url": "https://github.com/ex/ex",
                        "category": "feature_request",
                        "display_title": "Title",
                        "display_description": "Description",
                        "original_title": "Original title",
                        # missing: language
                    }
                ],
            }
        )
    )
    with pytest.raises(DeepSWEInstanceError, match="schema fields differ.*language"):
        load_dataset(manifest_path)


def test_load_dataset_task_toml_missing_section(tmp_path: Path):
    """Hand-write a malformed task.toml with no [environment] section."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    task_toml = (task_dir / "task.toml").read_text()
    environment_start = task_toml.index("[environment]")
    solution_start = task_toml.index("[solution]")
    (task_dir / "task.toml").write_text(
        task_toml[:environment_start] + task_toml[solution_start:]
    )
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(
        DeepSWEInstanceError, match=r"schema fields differ.*environment"
    ):
        load_dataset(manifest_path)


def test_load_dataset_task_toml_malformed(tmp_path: Path):
    """Invalid TOML surfaces as DeepSWEInstanceError, not raw
    TOMLDecodeError, with the task_id in the message."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = tasks_root / "t1"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("this is not valid TOML [")
    (task_dir / "instruction.md").write_text("body")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(DeepSWEInstanceError, match=r"t1.*task\.toml parse failed"):
        load_dataset(manifest_path)


def test_load_dataset_instruction_md_missing(tmp_path: Path):
    """task.toml present, instruction.md missing → clear error."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    (tasks_root / "t1" / "instruction.md").unlink()
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(DeepSWEInstanceError, match="missing instruction.md"):
        load_dataset(manifest_path)


@pytest.mark.timeout(2)
@pytest.mark.parametrize(
    "consumer",
    ["manifest", "task_toml", "instruction", "solution", "test"],
)
def test_dataset_consumer_rejects_regular_to_fifo_races_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consumer: str,
) -> None:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    targets = {
        "manifest": manifest_path,
        "task_toml": task_dir / "task.toml",
        "instruction": task_dir / "instruction.md",
        "solution": task_dir / "solution" / "solution.patch",
        "test": task_dir / "tests" / "test.patch",
    }
    target = targets[consumer]
    displaced = target.with_name(f"displaced-{target.name}")
    real_open = dataset_module.os.open
    swapped = False

    def racing_open(
        candidate: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        candidate_name = os.fsdecode(candidate)
        parent = (
            Path(os.readlink(f"/proc/self/fd/{dir_fd}")) if dir_fd is not None else None
        )
        if (
            not swapped
            and parent == target.parent
            and candidate_name == target.name
            and flags & os.O_ACCMODE == os.O_RDONLY
            and not flags & getattr(os, "O_DIRECTORY", 0)
        ):
            assert flags & os.O_NONBLOCK
            target.rename(displaced)
            os.mkfifo(target, mode=0o600)
            swapped = True
        return real_open(candidate, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(dataset_module.os, "open", racing_open)
    with pytest.raises(
        DeepSWEInstanceError,
        match="safely|exact regular|changed while being read",
    ):
        load_dataset(manifest_path)
    assert swapped is True


def test_load_dataset_empty_base_commit_hash(tmp_path: Path):
    """task.toml with empty [metadata].base_commit_hash fails fast."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="t1", ext_id="kh1", base_commit_hash="")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(
        DeepSWEInstanceError, match=r"\[metadata\].base_commit_hash.*non-empty"
    ):
        load_dataset(manifest_path)


def test_load_dataset_empty_docker_image(tmp_path: Path):
    """task.toml with empty [environment].docker_image fails fast."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="t1", ext_id="kh1", docker_image_override="")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    with pytest.raises(
        DeepSWEInstanceError, match=r"\[environment\].docker_image.*non-empty"
    ):
        load_dataset(manifest_path)


def _write_one_valid_task(tmp_path: Path) -> tuple[Path, Path, Path]:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    task_dir = _write_task_dir(tasks_root, task_id="t1", ext_id="kh1")
    manifest_path = _write_manifest(
        tasks_root,
        [_manifest_entry("t1", ext_id="kh1", repo="ex/ex")],
    )
    return tasks_root, task_dir, manifest_path


@pytest.mark.parametrize("schema_version", ["0.9", "2.0", 1])
def test_load_dataset_rejects_unsupported_manifest_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = schema_version
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="schema_version is unsupported"):
        load_dataset(manifest_path)


def test_load_dataset_rejects_unknown_manifest_root_field(tmp_path: Path) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["future_authority"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="unknown=.*future_authority"):
        load_dataset(manifest_path)


def test_load_dataset_rejects_unknown_manifest_entry_field(tmp_path: Path) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0]["future_authority"] = "unreviewed"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="unknown=.*future_authority"):
        load_dataset(manifest_path)


@pytest.mark.parametrize("duplicate_location", ["root", "entry"])
def test_load_dataset_rejects_duplicate_json_keys(
    tmp_path: Path,
    duplicate_location: str,
) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    payload = manifest_path.read_text()
    if duplicate_location == "root":
        payload = payload.replace(
            '"task_count": 1',
            '"task_count": 1, "task_count": 1',
            1,
        )
    else:
        payload = payload.replace(
            '"task_id": "t1"',
            '"task_id": "t1", "task_id": "t1"',
            1,
        )
    manifest_path.write_text(payload)

    with pytest.raises(DeepSWEInstanceError, match="repeats key"):
        load_dataset(manifest_path)


@pytest.mark.parametrize("task_count", [True, 1.0, "1"])
def test_load_dataset_rejects_coercible_task_count_types(
    tmp_path: Path,
    task_count: object,
) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["task_count"] = task_count
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="exact nonnegative integer"):
        load_dataset(manifest_path)


def test_load_dataset_rejects_manifest_id_policy_drift(tmp_path: Path) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["id_policy"]["format"] = "filesystem-safe"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="id_policy differs"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language", "Python"),
        ("language", "java"),
        ("category", "feature"),
        ("category", "Feature_Request"),
    ],
)
def test_load_dataset_rejects_values_outside_closed_domains(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0][field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match=rf"{field}.*supported.*domain"):
        load_dataset(manifest_path)


@pytest.mark.parametrize("identity_field", ["task_id", "ext_id"])
def test_load_dataset_rejects_duplicate_manifest_identities(
    tmp_path: Path,
    identity_field: str,
) -> None:
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    _write_task_dir(tasks_root, task_id="task-one", ext_id="khone")
    second_task_id = "task-one" if identity_field == "task_id" else "task-two"
    if second_task_id != "task-one":
        _write_task_dir(tasks_root, task_id=second_task_id, ext_id="khone")
    entries = [
        _manifest_entry("task-one", ext_id="khone", repo="ex/ex"),
        _manifest_entry(second_task_id, ext_id="khone", repo="ex/ex"),
    ]
    if identity_field == "task_id":
        entries[1]["ext_id"] = "khtwo"
    manifest_path = _write_manifest(tasks_root, entries)

    with pytest.raises(
        DeepSWEInstanceError,
        match=rf"{identity_field} values must be unique",
    ):
        load_dataset(manifest_path)


@pytest.mark.parametrize("value", [True, 7, 1.5, None])
def test_load_dataset_rejects_non_string_manifest_entry_values(
    tmp_path: Path,
    value: object,
) -> None:
    _tasks_root, _task_dir, manifest_path = _write_one_valid_task(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"][0]["repo"] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeepSWEInstanceError, match="repo.*exact string"):
        load_dataset(manifest_path)


@pytest.mark.parametrize("schema_version", ["1.0", "1.4", 1.3])
def test_load_dataset_rejects_unsupported_task_schema_versions(
    tmp_path: Path,
    schema_version: object,
) -> None:
    _tasks_root, task_dir, manifest_path = _write_one_valid_task(tmp_path)
    task_toml = (task_dir / "task.toml").read_text()
    encoded = (
        f'"{schema_version}"'
        if isinstance(schema_version, str)
        else str(schema_version)
    )
    (task_dir / "task.toml").write_text(
        task_toml.replace('schema_version = "1.3"', f"schema_version = {encoded}", 1)
    )

    with pytest.raises(DeepSWEInstanceError, match="schema_version is unsupported"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'artifacts = ["/logs/artifacts/model.patch"]',
            'future_authority = "unreviewed"\nartifacts = ["/logs/artifacts/model.patch"]',
            "unknown=.*future_authority",
        ),
        (
            "[environment]\n",
            "[environment]\nfuture_authority = 1\n",
            "unknown=.*future_authority",
        ),
        (
            "[verifier.environment]\n",
            "[verifier.environment]\nfuture_authority = 1\n",
            "unknown=.*future_authority",
        ),
    ],
)
def test_load_dataset_rejects_unknown_task_toml_fields(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    _tasks_root, task_dir, manifest_path = _write_one_valid_task(tmp_path)
    task_toml_path = task_dir / "task.toml"
    payload = task_toml_path.read_text()
    assert payload.count(old) == 1
    task_toml_path.write_text(payload.replace(old, new, 1))

    with pytest.raises(DeepSWEInstanceError, match=message):
        load_dataset(manifest_path)


def test_load_dataset_rejects_duplicate_task_toml_key(tmp_path: Path) -> None:
    _tasks_root, task_dir, manifest_path = _write_one_valid_task(tmp_path)
    task_toml_path = task_dir / "task.toml"
    payload = task_toml_path.read_text()
    task_toml_path.write_text(payload.replace("gpus = 0", "gpus = 0\ngpus = 0", 1))

    with pytest.raises(DeepSWEInstanceError, match="task.toml parse failed"):
        load_dataset(manifest_path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("cpus = 2", "cpus = true", "cpus.*exact positive integer"),
        ("memory_mb = 8192", "memory_mb = 8192.0", "memory_mb.*integer"),
        ("storage_mb = 20480", "storage_mb = 0", "storage_mb.*positive"),
        (
            "build_timeout_sec = 1800.0",
            "build_timeout_sec = 1800",
            "build_timeout_sec.*exact positive finite float",
        ),
        ("gpus = 0", "gpus = false", "gpus.*exact integer zero"),
        ("mcp_servers = []", "mcp_servers = [1]", "mcp_servers.*string list"),
        ("timeout_sec = 5400.0", "timeout_sec = true", "timeout_sec.*float"),
    ],
)
def test_load_dataset_rejects_coercible_or_out_of_range_task_values(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    _tasks_root, task_dir, manifest_path = _write_one_valid_task(tmp_path)
    task_toml_path = task_dir / "task.toml"
    payload = task_toml_path.read_text()
    assert old in payload
    task_toml_path.write_text(payload.replace(old, new, 1))

    with pytest.raises(DeepSWEInstanceError, match=message):
        load_dataset(manifest_path)


def test_load_dataset_rejects_manifest_task_metadata_disagreement(
    tmp_path: Path,
) -> None:
    _tasks_root, task_dir, manifest_path = _write_one_valid_task(tmp_path)
    task_toml_path = task_dir / "task.toml"
    task_toml_path.write_text(
        task_toml_path.read_text().replace(
            'category = "feature_request"',
            'category = "bugfix"',
            1,
        )
    )

    with pytest.raises(DeepSWEInstanceError, match="category differs from manifest"):
        load_dataset(manifest_path)


# ---------- filter_instances ----------


def _two_lang_corpus(tmp_path: Path) -> tuple[DeepSWEInstance, ...]:
    """Helper: construct a 4-task corpus across 2 languages / 2
    categories. Reused by the filter tests below."""
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    entries = []
    for i, (lang, cat, repo) in enumerate(
        [
            ("python", "feature_request", "ex/py-a"),
            ("python", "bugfix", "ex/py-b"),
            ("go", "feature_request", "ex/go-a"),
            ("go", "enhancement", "ex/go-b"),
        ]
    ):
        tid = f"task-{i}"
        eid = f"kh{i}"
        _write_task_dir(
            tasks_root,
            task_id=tid,
            ext_id=eid,
            repository_url=f"https://github.com/{repo}",
            language=lang,
            category=cat,
        )
        entries.append(
            _manifest_entry(tid, ext_id=eid, repo=repo, language=lang, category=cat)
        )
    manifest_path = _write_manifest(tasks_root, entries)
    return load_dataset(manifest_path)


def test_filter_by_task_id(tmp_path: Path):
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(insts, task_ids=["task-0", "task-2"])
    assert {i.task_id for i in filtered} == {"task-0", "task-2"}


def test_filter_by_repo(tmp_path: Path):
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(insts, repos=["ex/py-a"])
    assert len(filtered) == 1 and filtered[0].repo == "ex/py-a"


def test_filter_by_language_case_insensitive(tmp_path: Path):
    insts = _two_lang_corpus(tmp_path)
    # Mixed-case input should still match (the loader lowercases at
    # load; the filter lowercases its input set).
    filtered = filter_instances(insts, languages=["Python"])
    assert {i.language for i in filtered} == {"python"}
    assert len(filtered) == 2


def test_filter_by_category(tmp_path: Path):
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(insts, categories=["bugfix", "enhancement"])
    assert {i.category for i in filtered} == {"bugfix", "enhancement"}
    assert len(filtered) == 2


def test_filter_limit_applied_last(tmp_path: Path):
    """Filters apply in declaration order: task_ids → repos →
    languages → categories → limit. With limit=2 against the 4-task
    corpus, exactly 2 come back; the limit doesn't pre-empt other
    filters."""
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(insts, limit=2)
    assert len(filtered) == 2


def test_filter_combined(tmp_path: Path):
    """Language=python AND category=feature_request → exactly 1."""
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(
        insts,
        languages=["python"],
        categories=["feature_request"],
    )
    assert len(filtered) == 1 and filtered[0].task_id == "task-0"


def test_filter_empty_iterables_no_op(tmp_path: Path):
    """Passing an empty list (vs None) means "no filter" — same as
    None. Avoids the trap where an empty list filters to zero."""
    insts = _two_lang_corpus(tmp_path)
    filtered = filter_instances(
        insts,
        task_ids=[],
        repos=[],
        languages=[],
        categories=[],
    )
    assert len(filtered) == 4
