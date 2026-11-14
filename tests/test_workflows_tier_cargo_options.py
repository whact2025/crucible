# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Generic Spec-bound runner configuration at workflow boundaries."""

from __future__ import annotations

import os
import subprocess
import textwrap

import pytest
from assay import ImplementationDirective, ProjectContext, Spec
from assay.types.dag import Slice, Tier

from crucible.workflows import architect as architect_workflow
from crucible.workflows import breach as breach_workflow
from crucible.workflows import tier as tier_workflow
from crucible.workflows._build_verify import build_and_verify
from crucible.workflows.tier import (
    _final_binding_runner_parameters,
    _render_tier_breach_intent,
)


def _spec(parameters: dict[str, object] | None = None) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="fabric-usd",
            repo_url="ssh://example.invalid/fabric-usd.git",
            base_commit="0" * 40,
            branch="main",
            submodule_path="fabric-usd",
            runners={
                "build": "crucible.common.runners.polyglot",
                "verify": "crucible.common.runners.polyglot",
            },
            runner_parameters=parameters or {},
        ),
        execution=None,
        intent="test generic runner configuration",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )


def test_tier_cli_has_no_language_specific_runner_arguments(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_run_tier(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(tier_workflow, "run_tier", _fake_run_tier)
    assert tier_workflow.main([
        "--spec", str(tmp_path / "spec"),
        "--tier", "t0",
        "--cwd", str(tmp_path),
    ]) == 0
    assert "cargo_options" not in captured

    with pytest.raises(SystemExit):
        tier_workflow._parse_args([
            "--spec", str(tmp_path / "spec"),
            "--tier", "t0",
            "--verify-crate", "fabric-render-ovrtx",
        ])


def test_tier_breach_intent_contains_only_active_slice_directives() -> None:
    active = ImplementationDirective(
        slice_id="active",
        intent="establish the ABI foundation now",
        requirement_refs=("Preserve the declared ABI.",),
        files_in_scope=("active.rs",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    sibling = ImplementationDirective(
        slice_id="sibling",
        intent="validate the same active-tier boundary",
        files_in_scope=("sibling.rs",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    rendered = _render_tier_breach_intent(Tier(
        tier_id="t0",
        slices=(
            Slice(id="active", directive=active),
            Slice(id="sibling", directive=sibling),
        ),
        depends_on=(),
    ))

    assert "Active tier `t0`" in rendered
    assert "Active slice `active`" in rendered
    assert "Authoritative root requirements" in rendered
    assert "Preserve the declared ABI." in rendered
    assert rendered.index("Preserve the declared ABI.") < rendered.index(
        "establish the ABI foundation now"
    )
    assert "root requirement wins" in rendered
    assert "establish the ABI foundation now" in rendered
    assert "Active slice `sibling`" in rendered
    assert "validate the same active-tier boundary" in rendered
    assert "Requirements assigned to later tiers" not in rendered


def test_tier_smoke_uses_real_spec_bound_rust_selection(
    monkeypatch,
    tmp_path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cargo.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    spec = Spec.from_yaml(textwrap.dedent("""
        schema_version: 1
        project_context:
          project: fabric-usd
          repo_url: ssh://example.invalid/fabric-usd.git
          base_commit: abc
          branch: main
          submodule_path: fabric-usd
          runners:
            build: crucible.common.runners.polyglot
            verify: crucible.common.runners.polyglot
          runner_parameters:
            per_language:
              rust:
                crate: fabric-render-ovrtx
                cargo_features: [ovrtx]
                cargo_test_args: ["--", "--test-threads=1"]
        intent: exercise canonical Rust smoke selection
        constraints: []
        done_criteria: []
    """).lstrip())
    directive = ImplementationDirective(
        slice_id="runtime_update_protocol",
        intent="compile the protocol",
        files_in_scope=("crates/fabric-render-ovrtx/src/lib.rs",),
        creates_paths=(),
        verify_mode="build_only",
        blast_radius={},
    )
    commands: list[list[str]] = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "run_supervised_process",
        _run,
    )
    result = tier_workflow._run_spec_bound_smoke_check(
        spec,
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is True
    assert commands == [[
        "cargo",
        "check",
        "-p",
        "fabric-render-ovrtx",
        "--features",
        "ovrtx",
    ]]


def test_tier_rust_smoke_resolves_relative_path_from_child_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cargo.chmod(0o755)
    monkeypatch.setenv("PATH", "bin")
    directive = ImplementationDirective(
        slice_id="runtime_update_protocol",
        intent="compile the protocol",
        files_in_scope=("crates/runtime/src/lib.rs",),
        creates_paths=(),
        verify_mode="build_only",
        blast_radius={},
    )

    result = tier_workflow._run_spec_bound_smoke_check(
        _spec(),
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is True
    assert result.language == "rust"


def test_tier_smoke_applies_slice_local_rust_project_selection(
    monkeypatch,
    tmp_path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cargo.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    spike = tmp_path / "native" / "spikes" / "rust"
    runtime = tmp_path / "native" / "runtime-rust"
    spike.mkdir(parents=True)
    runtime.mkdir(parents=True)
    spec = _spec({
        "per_language": {
            "rust": {"project_subdir": "native/spikes/rust"},
        },
        "per_slice": {
            "runtime": {
                "per_language": {
                    "rust": {"project_subdir": "native/runtime-rust"},
                },
            },
        },
    })
    directive = ImplementationDirective(
        slice_id="runtime",
        intent="compile the selected runtime",
        files_in_scope=(),
        creates_paths=("native/runtime-rust/src/lib.rs",),
        verify_mode="tests",
        blast_radius={"max_files": 1, "languages": ["rust"]},
    )
    calls = []

    def _run(command, **kwargs):
        calls.append((list(command), kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        tier_workflow._build_verify,
        "run_supervised_process",
        _run,
    )
    result = tier_workflow._run_spec_bound_smoke_check(
        spec,
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is True
    assert calls == [(["cargo", "check", "--workspace"], str(runtime))]


def test_tier_python_smoke_isolated_from_frozen_no_cache_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "alias_resolution.py"
    source.write_text(
        "def resolve_alias(value: str) -> str:\n"
        "    return value\n",
        encoding="utf-8",
    )
    directive = ImplementationDirective(
        slice_id="alias_resolution",
        intent="compile the Python implementation",
        files_in_scope=(source.name,),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = tier_workflow._run_spec_bound_smoke_check(
        _spec(),
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is True
    assert result.language == "python"
    assert result.log_excerpt == ""
    assert os.environ["PYTHONPYCACHEPREFIX"] == os.devnull
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


def test_tier_python_smoke_soft_passes_when_optional_ruff_is_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "alias_resolution.py"
    source.write_text(
        "def resolve_alias(value: str) -> str:\n"
        "    return value\n",
        encoding="utf-8",
    )
    directive = ImplementationDirective(
        slice_id="alias_resolution",
        intent="compile the Python implementation",
        files_in_scope=(source.name,),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    missing_bin = tmp_path / "missing-bin"
    monkeypatch.setenv("PATH", str(missing_bin))

    result = tier_workflow._run_spec_bound_smoke_check(
        _spec(),
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is True
    assert result.language == "python"
    assert result.log_excerpt == ""
    assert os.environ["PATH"] == str(missing_bin)


def test_tier_python_smoke_runs_target_local_ruff_from_relative_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "unused_import.py"
    source.write_text("import os\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ruff = bin_dir / "ruff"
    ruff.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then exit 0; fi\n"
        "printf 'F401 target-local ruff executed\\n'\n"
        "exit 1\n",
        encoding="utf-8",
    )
    ruff.chmod(0o755)
    monkeypatch.setenv("PATH", "bin")
    directive = ImplementationDirective(
        slice_id="unused_import",
        intent="reject an unused import",
        files_in_scope=(source.name,),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )

    result = tier_workflow._run_spec_bound_smoke_check(
        _spec(),
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is False
    assert result.language == "python"
    assert "F401 target-local ruff executed" in result.log_excerpt


def test_tier_python_smoke_preserves_syntax_failure_under_frozen_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    source = tmp_path / "alias_resolution.py"
    source.write_text("def resolve_alias(:\n    return None\n", encoding="utf-8")
    directive = ImplementationDirective(
        slice_id="alias_resolution",
        intent="reject an invalid Python implementation",
        files_in_scope=(source.name,),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={},
    )
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = tier_workflow._run_spec_bound_smoke_check(
        _spec(),
        directive,
        cwd=str(tmp_path),
        timeout=120,
    )

    assert result.passed is False
    assert result.language == "python"
    assert "syntax" in result.log_excerpt.lower()
    assert os.environ["PYTHONPYCACHEPREFIX"] == os.devnull
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before


@pytest.mark.parametrize(
    "runner",
    [
        "crucible.common.runners.lang.rust",
        "crucible.common.runners.polyglot",
    ],
)
def test_tier_bench_mode_uses_frozen_spec_argv_without_test_args(
    runner: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    spec = Spec.from_yaml(textwrap.dedent(f"""
        schema_version: 1
        project_context:
          project: fabric-usd
          repo_url: ssh://example.invalid/fabric-usd.git
          base_commit: abc
          branch: main
          submodule_path: fabric-usd
          runners:
            build: {runner}
            verify: {runner}
          runner_parameters:
            per_language:
              rust:
                crate: fabric-render-ovrtx
                cargo_features: [ovrtx]
                cargo_test_args: ["--test", "protocol", "--", "--nocapture"]
                cargo_bench_args: ["--bench", "runtime_update"]
        intent: exercise canonical Rust benchmark selection
        constraints:
          - kind: languages
            languages: [rust]
        done_criteria: []
    """).lstrip())
    directive = ImplementationDirective(
        slice_id="runtime_update_bench",
        intent="run the benchmark gate",
        files_in_scope=("crates/fabric-render-ovrtx/src/lib.rs",),
        creates_paths=(),
        verify_mode="bench",
        blast_radius={},
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_log = tmp_path / "cargo-argv.log"
    cargo = bin_dir / "cargo"
    cargo.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$CARGO_ARGV_LOG\"\n",
        encoding="utf-8",
    )
    cargo.chmod(0o755)
    monkeypatch.setenv("CARGO_ARGV_LOG", str(argv_log))
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    outcome = build_and_verify(
        spec,
        slice_id=directive.slice_id,
        cwd=str(tmp_path),
        slice_directive=directive,
        build_timeout=30,
        verify_timeout=30,
    )

    assert outcome.classification == "success"
    assert argv_log.read_text(encoding="utf-8").splitlines() == [
        "build -p fabric-render-ovrtx --features ovrtx",
        "bench -p fabric-render-ovrtx --features ovrtx "
        "--bench runtime_update",
    ]


@pytest.mark.parametrize(
    ("parser", "base"),
    [
        (
            architect_workflow._parse_args,
            ("--spec", "SPEC"),
        ),
        (
            breach_workflow._parse_args,
            ("--spec", "SPEC", "--diff", "change.diff"),
        ),
    ],
)
def test_composed_workflow_clis_reject_language_specific_runner_arguments(
    parser,
    base,
) -> None:
    with pytest.raises(SystemExit):
        parser([*base, "--cargo-feature", "ovrtx"])


def test_final_binding_uses_spec_bound_rust_selection() -> None:
    spec = _spec({
        "per_language": {
            "rust": {
                "crate": "fabric-render-ovrtx",
                "cargo_features": ["ovrtx"],
                "cargo_test_args": ["--", "--test-threads=1"],
            },
        },
    })

    final = _final_binding_runner_parameters(
        spec,
        None,
        cwd="/workspace",
        breach_language="rust",
        breach_executed=True,
    )

    assert final == {
        "working_dir": "/workspace",
        "per_language": {
            "rust": {
                "crate": "fabric-render-ovrtx",
                "cargo_features": ["ovrtx"],
                "cargo_test_args": ["--", "--test-threads=1"],
            },
        },
    }


def test_final_binding_rejects_unsafe_spec_bound_test_selector() -> None:
    spec = _spec({
        "per_language": {
            "rust": {
                "cargo_test_args": ["--test", "only-this-target"],
            },
        },
    })

    with pytest.raises(ValueError, match="target selectors"):
        _final_binding_runner_parameters(
            spec,
            None,
            cwd="/workspace",
            breach_language="rust",
            breach_executed=True,
        )


def test_review_only_final_binding_preserves_normal_rust_selection() -> None:
    spec = _spec({
        "per_language": {
            "rust": {
                "cargo_test_args": ["--test", "protocol"],
            },
        },
    })

    assert _final_binding_runner_parameters(
        spec,
        None,
        cwd="/workspace",
        breach_language="rust",
        breach_executed=False,
    ) is None
