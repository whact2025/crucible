# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed Cargo selection at the breach command boundary."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import ProjectContext, Spec
from assay.corpus.store import Corpus

from crucible.constraints import LanguagesConstraint
from crucible.workflows import breach as breach_workflow


def _git(repo: Path, *args: str) -> None:
    import subprocess

    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _write_crate(root: Path, directory: str, package_name: str) -> None:
    crate = root / "crates" / directory
    (crate / "src").mkdir(parents=True)
    (crate / "src" / "lib.rs").write_text("pub fn value() -> u8 { 1 }\n")
    (crate / "Cargo.toml").write_text(
        f'[package]\nname = "{package_name}"\nversion = "0.1.0"\n'
        'edition = "2021"\n',
        encoding="utf-8",
    )


def _setup_rust_command(
    tmp_path: Path,
    *,
    crates: tuple[tuple[str, str], ...] = (("foo", "foo-package"),),
    diff_paths: tuple[str, ...] = ("crates/foo/src/lib.rs",),
    runner_parameters: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    members = ", ".join(f'"crates/{directory}"' for directory, _ in crates)
    (repo / "Cargo.toml").write_text(
        f"[workspace]\nmembers = [{members}]\nresolver = \"2\"\n",
        encoding="utf-8",
    )
    for directory, package_name in crates:
        _write_crate(repo, directory, package_name)
    for rendered in diff_paths:
        target = repo / rendered
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("old\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "breach@example.invalid")
    _git(repo, "config", "user.name", "Breach Safety")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    spec_dir = tmp_path / "corpus" / "rust-breach"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    corpus.put_spec(Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="rust-breach",
            repo_url="https://example.invalid/rust-breach.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners={
                "build": "crucible.common.runners.polyglot",
                "verify": "crucible.common.runners.polyglot",
            },
            runner_parameters=runner_parameters or {},
        ),
        execution=None,
        intent="test Rust breach setup",
        constraints=(LanguagesConstraint(languages=("rust",)),),
        done_criteria=(),
        plan_policy=None,
    ))
    diff = tmp_path / "change.diff"
    for rendered in diff_paths:
        target = repo / rendered
        target.write_text(
            target.read_text(encoding="utf-8") + "// changed\n",
            encoding="utf-8",
        )
    import subprocess

    diff.write_bytes(subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout)
    return repo, spec_dir, diff


def _forbid_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("breach execution started before setup rejection")

    monkeypatch.setattr(breach_workflow, "backend_pool", _forbidden)


def _bind_hermetic_cargo_metadata(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
) -> None:
    """Return the fixture workspace from Cargo without requiring a toolchain."""
    manifest = (repo / "crates/foo/Cargo.toml").resolve()
    package_id = (
        f"path+file://{manifest.parent.as_posix()}#foo-package@0.1.0"
    )
    payload = {
        "workspace_members": [package_id],
        "packages": [{
            "id": package_id,
            "name": "foo-package",
            "manifest_path": str(manifest),
        }],
    }

    def _cargo_metadata(argv, **kwargs):
        assert argv == [
            "cargo", "metadata", "--no-deps", "--format-version", "1",
        ]
        assert Path(kwargs["cwd"]).resolve() == repo.resolve()
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is False
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(
        breach_workflow,
        "subprocess",
        SimpleNamespace(run=_cargo_metadata),
    )


def _run(
    repo: Path,
    spec_dir: Path,
    diff: Path,
    *extra_args: str,
) -> int:
    return breach_workflow.main([
        "--spec", str(spec_dir),
        "--diff", str(diff),
        "--cwd", str(repo),
        *extra_args,
    ])


@pytest.mark.parametrize(
    "unsafe_test_args",
    [
        ("--test", "protocol"),
        ("--", "protocol"),
        ("--", "--skip", "attack_name"),
        ("--", "--ignored"),
    ],
)
def test_standalone_breach_rejects_test_selectors_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_test_args: tuple[str, ...],
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        runner_parameters={
            "per_language": {
                "rust": {
                    "crate": "foo-package",
                    "cargo_test_args": list(unsafe_test_args),
                },
            },
        },
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2
    assert not (repo / "crates/foo/tests/breach_round.rs").exists()


def test_tier_child_filter_retains_only_safe_serial_output_controls():
    assert breach_workflow.safe_breach_cargo_test_args((
        "--test", "protocol", "--",
        "--nocapture", "attack_name", "--skip",
        "--show-output", "--test-threads=1", "--ignored",
    )) == ("--", "--test-threads=1")


def test_spec_bound_legacy_rust_test_args_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        runner_parameters={
            "per_language": {
                "rust": {
                    "crate": "foo-package",
                    "test_args": ["--test", "unrelated_target"],
                },
            },
        },
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_recursive_rust_parameter_layer_cannot_bypass_breach_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        runner_parameters={
            "per_language": {
                "rust": {
                    "crate": "foo-package",
                    "cargo_test_args": ["--", "--test-threads=1"],
                    "per_language": {
                        "rust": {
                            "crate": "wrong-package",
                            "cargo_test_args": [
                                "--test", "unrelated_target",
                            ],
                        },
                    },
                },
            },
        },
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_explicit_crate_mismatch_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        runner_parameters={
            "per_language": {"rust": {"crate": "wrong-package"}},
        },
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_multi_crate_diff_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        crates=(("foo", "foo-package"), ("bar", "bar-package")),
        runner_parameters={
            "per_language": {"rust": {"crate": "foo-package"}},
        },
        diff_paths=("crates/foo/src/lib.rs", "crates/bar/src/lib.rs"),
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_ambiguous_virtual_workspace_target_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        diff_paths=("README.md",),
    )
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_custom_breach_test_in_other_crate_rejected_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        crates=(("foo", "foo-package"), ("bar", "bar-package")),
    )
    _forbid_execution(monkeypatch)

    assert _run(
        repo,
        spec_dir,
        diff,
        "--breach-tests=crates/bar/tests/custom.rs",
    ) == 2


def test_exact_package_name_is_proven_from_generated_test_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _spec_dir, _diff = _setup_rust_command(tmp_path)
    _bind_hermetic_cargo_metadata(monkeypatch, repo)

    assert breach_workflow._resolve_rust_breach_crate(
        cwd=str(repo),
        diff_files=("crates/foo/src/lib.rs",),
        test_file=repo / "crates/foo/tests/breach_round.rs",
        explicit_crate="foo-package",
    ) == "foo-package"


def test_excluded_same_name_package_is_not_a_workspace_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _spec_dir, _diff = _setup_rust_command(tmp_path)
    _bind_hermetic_cargo_metadata(monkeypatch, repo)
    root_manifest = repo / "Cargo.toml"
    root_manifest.write_text(
        root_manifest.read_text(encoding="utf-8")
        + 'exclude = ["excluded"]\n',
        encoding="utf-8",
    )
    excluded = repo / "excluded"
    (excluded / "src").mkdir(parents=True)
    (excluded / "src" / "lib.rs").write_text(
        "pub fn excluded() {}\n",
        encoding="utf-8",
    )
    (excluded / "Cargo.toml").write_text(
        '[package]\nname = "foo-package"\nversion = "0.1.0"\n'
        'edition = "2021"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not an exact member"):
        breach_workflow._resolve_rust_breach_crate(
            cwd=str(repo),
            diff_files=("excluded/src/lib.rs",),
            test_file=excluded / "tests/breach_round.rs",
            explicit_crate="foo-package",
        )


def test_recovery_precedes_locked_preflight_and_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        runner_parameters={
            "per_language": {"rust": {"crate": "wrong-package"}},
        },
    )
    order: list[str] = []
    original_recover = breach_workflow.recover_mutation_journals
    original_prepare = breach_workflow._prepare_breach_execution

    def _recover(*args, **kwargs):
        order.append("recover")
        return original_recover(*args, **kwargs)

    def _prepare(*args, **kwargs):
        order.append("preflight")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(breach_workflow, "recover_mutation_journals", _recover)
    monkeypatch.setattr(breach_workflow, "_prepare_breach_execution", _prepare)
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2
    assert order == ["recover", "preflight"]


@pytest.mark.parametrize("operation", ["delete", "rename"])
def test_old_path_in_second_package_prevents_single_package_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
):
    repo, spec_dir, diff = _setup_rust_command(
        tmp_path,
        crates=(("foo", "foo-package"), ("bar", "bar-package")),
        runner_parameters={
            "per_language": {"rust": {"crate": "foo-package"}},
        },
    )
    bar_source = repo / "crates/bar/src/lib.rs"
    if operation == "delete":
        bar_source.unlink()
    else:
        _git(repo, "mv", "crates/bar/src/lib.rs", "crates/foo/src/from_bar.rs")
    import subprocess

    diff.write_bytes(subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout)
    _forbid_execution(monkeypatch)

    assert _run(repo, spec_dir, diff) == 2


def test_autotests_false_requires_explicit_generated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo, _spec_dir, _diff = _setup_rust_command(tmp_path)
    _bind_hermetic_cargo_metadata(monkeypatch, repo)
    manifest = repo / "crates/foo/Cargo.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "autotests = false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="autotests=false"):
        breach_workflow._resolve_rust_breach_crate(
            cwd=str(repo),
            diff_files=("crates/foo/src/lib.rs",),
            test_file=repo / "crates/foo/tests/breach_round.rs",
            explicit_crate="foo-package",
        )

    manifest.write_text(
        manifest.read_text(encoding="utf-8")
        + "\n[[test]]\nname = \"breach_round\"\n"
        + "path = \"tests/breach_round.rs\"\n",
        encoding="utf-8",
    )
    assert breach_workflow._resolve_rust_breach_crate(
        cwd=str(repo),
        diff_files=("crates/foo/src/lib.rs",),
        test_file=repo / "crates/foo/tests/breach_round.rs",
        explicit_crate="foo-package",
    ) == "foo-package"


@pytest.mark.parametrize(
    "parameters",
    [
        {"test_args": ["--test", "other"]},
        {"cargo_test_args": ["--test", "other"]},
        {"per_language": {"rust": {"test_args": ["--test", "other"]}}},
        {"per_language": {"rust": {"role": "build"}}},
        {"working_dir": "/different/checkout"},
        {"crate": "legacy-untyped"},
    ],
)
def test_legacy_rust_parameter_bypasses_are_rejected(
    tmp_path: Path,
    parameters: dict,
):
    repo, _spec_dir, _diff = _setup_rust_command(tmp_path)
    with pytest.raises(ValueError):
        breach_workflow.validate_breach_rust_parameters(
            parameters,
            cwd=str(repo),
        )


def test_seal_breach_rust_parameters_strips_only_command_without_mutation(
    tmp_path: Path,
):
    parameters = {
        "deepswe_metadata": {
            "container_id": "container",
            "language": "rust",
            "task_id": "rust-task",
            "dev_verify_command": "cargo test --test project_suite",
            "baseline": {"container_id": "baseline-container"},
        },
        "unrelated_runtime_value": "preserved",
        "per_language": {
            "rust": {
                "crate": "runtime-crate",
                "cargo_features": ["runtime-feature"],
                "cargo_test_args": ["--test", "runtime-target"],
                "adapter_option": "preserved",
            },
            "python": {"adapter_option": "untouched"},
        },
    }
    original = copy.deepcopy(parameters)

    sealed = breach_workflow.seal_breach_rust_parameters(
        parameters,
        cwd=str(tmp_path),
        selection=breach_workflow._RustRunnerSelection(
            crate="resolved-crate",
            features=("resolved-feature",),
            test_args=("--", "--test-threads=1"),
        ),
    )

    assert parameters == original
    assert sealed["deepswe_metadata"] is not parameters["deepswe_metadata"]
    assert sealed["deepswe_metadata"] == {
        key: value
        for key, value in original["deepswe_metadata"].items()
        if key != "dev_verify_command"
    }
    assert sealed["unrelated_runtime_value"] == "preserved"
    assert sealed["per_language"]["python"] == {
        "adapter_option": "untouched",
    }
    assert sealed["per_language"]["rust"] == {
        "adapter_option": "preserved",
        "crate": "resolved-crate",
        "cargo_features": ["resolved-feature"],
        "cargo_test_args": ["--", "--test-threads=1"],
    }


def test_rust_runner_contract_matrix(tmp_path: Path):
    _repo, spec_dir, _diff = _setup_rust_command(tmp_path)
    spec = Corpus(spec_dir).latest_spec()
    assert spec is not None
    breach_workflow.validate_rust_breach_runners(spec, gate="verify")

    direct = replace(
        spec,
        project_context=replace(
            spec.project_context,
            runners={
                "build": "crucible.common.runners.lang.rust",
                "verify": "crucible.common.runners.lang.rust",
            },
        ),
    )
    breach_workflow.validate_rust_breach_runners(direct, gate="verify")

    unsupported = replace(
        spec,
        project_context=replace(
            spec.project_context,
            runners={"build": "custom.build", "verify": "custom.verify"},
        ),
    )
    with pytest.raises(ValueError, match="unsupported/missing"):
        breach_workflow.validate_rust_breach_runners(
            unsupported,
            gate="verify",
        )

    deepswe = replace(
        spec,
        project_context=replace(
            spec.project_context,
            runners={
                "dev_verify": (
                    "crucible.products.deepswe.runners.deepswe_dev_verify"
                ),
            },
        ),
    )
    breach_workflow.validate_rust_breach_runners(
        deepswe,
        gate="dev_verify",
        deepswe_metadata={"container_id": "container"},
    )
    breach_workflow.validate_rust_breach_runners(
        deepswe,
        gate="dev_verify",
        deepswe_metadata={
            "container_id": "container",
            "dev_verify_command": "cargo test --test project_suite",
        },
    )


def test_deepswe_rust_runner_preserves_named_output_and_sealed_selection(
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.products.deepswe.runners import deepswe_dev_verify

    observed: list[str] = []

    def _docker_exec(container_id: str, command: str, **_kwargs):
        assert container_id == "container"
        observed.append(command)
        if command.endswith("-- --list"):
            return 0, "test_attack_race_contract: test\n", ""
        return 0, (
            "test test_attack_race_contract ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured\n"
        ), ""

    monkeypatch.setattr(deepswe_dev_verify, "_run_docker_exec", _docker_exec)
    stdout = io.StringIO()
    parameters = breach_workflow.seal_breach_rust_parameters(
        {
            "deepswe_metadata": {
                "container_id": "container",
                "language": "rust",
                "dev_verify_command": "cargo test -p wrong-crate --test wrong",
            },
            "breach_observations": True,
            "breach_attack_ids": ["attack_race_contract"],
        },
        cwd="/workspace",
        selection=breach_workflow._RustRunnerSelection(
            crate="fabric-render-ovrtx",
            features=("ovrtx",),
            test_args=("--", "--test-threads=1"),
        ),
    )
    assert deepswe_dev_verify._run_dev_verify(parameters, stdout) == 0
    assert "cargo test -p fabric-render-ovrtx --features ovrtx" in observed[0]
    assert "-- --test-threads=1" in observed[0]
    assert "cargo test --quiet" not in observed[0]
    assert all("wrong-crate" not in command for command in observed)
    assert all(
        "cargo test" not in command
        or "cargo test -p fabric-render-ovrtx --features ovrtx" in command
        for command in observed
    )
    payload = json.loads(stdout.getvalue())
    assert "test_attack_race_contract" in payload["raw_output"]
    assert payload["test_observations"][0]["status"] == "passed"


@pytest.mark.parametrize(
    ("listed_stdout", "exact_output", "expected", "expected_calls"),
    [
        ("unrelated_test: test\n", "", "runner_error", 1),
        (
            "test_attack_race_contract: test\n",
            "test test_attack_race_contract ... ignored, SAFETY: unavailable\n",
            "not_observed",
            2,
        ),
    ],
)
def test_deepswe_rust_exact_observer_distinguishes_missing_from_ignored(
    monkeypatch: pytest.MonkeyPatch,
    listed_stdout: str,
    exact_output: str,
    expected: str,
    expected_calls: int,
):
    from crucible.products.deepswe.runners import deepswe_dev_verify

    calls: list[str] = []

    def _docker_exec(container_id: str, command: str, **_kwargs):
        assert container_id == "container"
        calls.append(command)
        if command.endswith("-- --list"):
            return 0, listed_stdout, ""
        return 0, exact_output, ""

    monkeypatch.setattr(deepswe_dev_verify, "_run_docker_exec", _docker_exec)
    observations = deepswe_dev_verify._rust_exact_observations(
        "container",
        {"breach_attack_ids": ["attack_race_contract"]},
    )

    assert len(calls) == expected_calls
    assert observations[0]["id"] == "test_attack_race_contract"
    assert observations[0]["status"] == expected


@pytest.mark.parametrize(
    ("action", "returncode", "expected"),
    [
        ("pass", 0, "passed"),
        ("fail", 1, "failed"),
        ("skip", 0, "not_observed"),
    ],
)
def test_deepswe_go_exact_observation_is_package_scoped_and_consistent(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    returncode: int,
    expected: str,
):
    from crucible.products.deepswe.runners import deepswe_dev_verify

    observed: list[str] = []

    def _docker_exec(container_id: str, command: str, **_kwargs):
        assert container_id == "container"
        observed.append(command)
        if "go list" in command:
            return 0, "example/pkg\n", ""
        event = json.dumps({
            "Action": action,
            "Package": "example/pkg",
            "Test": "TestBreachRaceA",
        })
        return returncode, event + "\n", ""

    monkeypatch.setattr(deepswe_dev_verify, "_run_docker_exec", _docker_exec)
    observations = deepswe_dev_verify._go_exact_observations(
        "container",
        {
            "breach_test_file": "pkg/breach_test.go",
            "breach_attack_ids": ["race_a"],
        },
    )
    assert "go list -f '{{.ImportPath}}' ./pkg" in observed[0]
    assert "go test -json ./pkg" in observed[1]
    assert "-run '^TestBreachRaceA$'" in observed[1]
    assert observations[0]["status"] == expected


def test_deepswe_go_missing_exact_event_is_runner_error(
    monkeypatch: pytest.MonkeyPatch,
):
    from crucible.products.deepswe.runners import deepswe_dev_verify

    def _docker_exec(  # noqa: ARG001
        container_id: str, command: str, **_kwargs,
    ):
        if "go list" in command:
            return 0, "example/pkg\n", ""
        return 0, "", ""

    monkeypatch.setattr(
        deepswe_dev_verify,
        "_run_docker_exec",
        _docker_exec,
    )
    observations = deepswe_dev_verify._go_exact_observations(
        "container",
        {
            "breach_test_file": "pkg/breach_test.go",
            "breach_attack_ids": ["race_a"],
        },
    )
    assert observations[0]["status"] == "runner_error"
