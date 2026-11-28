# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Check language attribution at the native Breach, Tier and Handoff boundaries."""

import json
import subprocess
from dataclasses import replace
from io import StringIO

import pytest
from assay import ImplementationDirective, Slice, Tier, TierDAG
from assay.corpus.store import Corpus

from crucible.common.policies import BreachPolicy
from crucible.common.runners import polyglot
from crucible.constraints import LanguagesConstraint
from crucible.workflows import breach, handoff, tier

from .test_workflows_handoff import _init_repo
from .test_workflows_tier_cargo_options import _spec


def _slice(name, languages, path):
    return Slice(id=name, directive=ImplementationDirective(
        slice_id=name, intent=name, files_in_scope=(path,), creates_paths=(),
        verify_mode="tests", blast_radius={"languages": languages},
    ))


@pytest.fixture
def candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    (repo / "seed.txt").write_text("scoped change\n")
    patch = tmp_path / "change.diff"
    patch.write_bytes(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo))
    base = replace(_spec({"per_slice": {
        "other-project": {"per_language": {"python": {"project_subdir": "elsewhere"}}}
    }}), constraints=(
        LanguagesConstraint(languages=("python", "rust", "dotnet")),
    ))
    dag = TierDAG(tiers=(
        Tier(tier_id="py", depends_on=(), slices=(
            _slice("py-slice", ["python"], "seed.txt"),
        )),
        Tier(tier_id="mixed", depends_on=(), slices=(
            _slice("interop", ["python", "dotnet"], "seed.txt"),
        )),
        Tier(tier_id="siblings", depends_on=(), slices=(
            _slice("shared-a", ["python"], "seed.txt"),
            _slice("shared-b", ["python"], "other.txt"),
        )),
        Tier(tier_id="divergent", depends_on=(), slices=(
            _slice("base-project", ["python"], "seed.txt"),
            _slice("other-project", ["python"], "other.txt"),
        )),
        Tier(tier_id="py-rust", depends_on=(), slices=(
            _slice("python-module", ["python"], "seed.txt"),
            _slice("rust-module", ["rust"], "other.txt"),
        )),
    ))
    corpus_dir = tmp_path / "corpus"
    corpus = Corpus(corpus_dir)
    corpus.put_spec(base)
    corpus.put_spec(base.with_tier_dag(dag))
    return repo, corpus_dir, patch


@pytest.mark.parametrize("workflow", ["breach", "tier", "handoff"])
def test_mixed_scope_refuses_exact_dotnet_assurance(candidate, workflow, monkeypatch):
    repo, corpus_dir, patch = candidate

    def forbid_backend(*args, **kwargs):
        pytest.fail("backend acquired for unsupported C# test evidence")

    monkeypatch.setattr(breach, "backend_pool", forbid_backend)
    monkeypatch.setattr(tier, "backend_pool", forbid_backend)
    common = ["--cwd", str(repo), "--spec", str(corpus_dir)]
    if workflow == "breach":
        rc = breach.main([*common, "--diff", str(patch)])
    elif workflow == "tier":
        rc = tier.run_tier(
            cwd=str(repo), spec=corpus_dir, tier_id="mixed",
            breach_policy=BreachPolicy(enabled=True),
        )
    else:
        rc = handoff.main([*common, "--slice", "interop", "--with-breach"])
    assert rc == 2
    summaries = [json.loads(path.read_text()) for path in corpus_dir.glob("runs/*/summary.json")]
    assert any("no exact per-test observation adapter for 'dotnet'" in str(s) for s in summaries)
    assert not (repo / "tests" / "test_breach.py").exists()


@pytest.mark.parametrize("selection", ["tier", "handoff-packet"])
def test_supported_slice_in_mixed_spec_binds_child_routing(candidate, selection, tmp_path):
    repo, corpus_dir, patch = candidate
    corpus = Corpus(corpus_dir)
    spec = corpus.latest_spec()
    selected = next(t for t in spec.tier_dag.tiers if t.tier_id == "py")
    directive = selected.slices[0].directive
    args = ["--cwd", str(repo), "--spec", str(corpus_dir), "--diff", str(patch)]
    if selection == "tier":
        args.extend(["--observation-tier", selected.tier_id])
    else:
        packet, digest = breach.encode_breach_authority(
            tier_id=selected.tier_id, issue_text=spec.intent,
            slice_directives=(directive,), authority_kind="selected_slices",
        )
        path = tmp_path / "scope.json"
        path.write_bytes(packet)
        args.extend(["--authority-file", str(path), "--authority-digest", digest])
    preflight = breach._prepare_breach_execution(
        args=breach._parse_args(args), cwd=str(repo), spec=spec, corpus=corpus,
        diff_artifact=patch, base_treeish="HEAD",
    )
    assert preflight.language == "python"
    assert preflight.test_file_relative == "tests/test_breach.py"
    assert preflight.execution_binding == (directive.slice_id, directive)
    assert tier._tier_breach_language(spec, selected, None) == "python"
    assert breach.resolve_breach_test_target(
        spec=spec, cwd=str(repo), diff_bytes=patch.read_bytes(), directive=directive,
    ) == preflight.test_file_relative


def test_selected_rust_and_mislabeled_csharp_paths(candidate):
    _, corpus_dir, _ = candidate
    spec = Corpus(corpus_dir).latest_spec()
    rust = _slice("r", ["rust"], "src/lib.rs").directive
    assert breach.resolve_breach_observation_language(spec, rust) == "rust"
    assert breach._breach_language_for_run(
        None, spec, directives=(rust,), diff_files=("src/Rules.CS",),
    ) == "dotnet"
    assert breach._breach_language_for_run(
        {"language": "rust"}, spec, directives=(rust,), diff_files=("src/Rules.CS",),
    ) == "rust"


@pytest.mark.parametrize("path", ["src/Rules.cs", "Warehouse.slnx", "Directory.Packages.props"])
def test_candidate_paths_are_interpreted_against_declared_sdk_contract(candidate, path):
    _, corpus_dir, _ = candidate
    selected = _slice("py", ["python"], "src").directive
    assert breach._breach_language_for_run(
        None, Corpus(corpus_dir).latest_spec(), directives=(selected,), diff_files=(path,),
    ) == "dotnet"
    assert breach._breach_language_for_run(
        None, _spec(), directives=(selected,), diff_files=(f"tests/fixtures/{path}",),
    ) == "python"


def test_homogeneous_tier_binds_only_its_project_at_all_gates(candidate):
    repo, corpus_dir, patch = candidate
    corpus = Corpus(corpus_dir)
    spec = corpus.latest_spec()
    selected = next(t for t in spec.tier_dag.tiers if t.tier_id == "siblings")
    argv = ["--spec", str(corpus_dir), "--diff", str(patch), "--observation-tier", "siblings"]
    preflight = breach._prepare_breach_execution(
        args=breach._parse_args(argv), cwd=str(repo), spec=spec, corpus=corpus,
        diff_artifact=patch, base_treeish="HEAD",
    )
    binding = tier._single_tier_execution_binding(selected, fallback_slice_id="final", spec=spec)
    assert preflight.execution_binding == binding
    assert breach._build_verify._resolve_languages(spec, binding[1]) == ("python",)


@pytest.mark.parametrize("workflow", ["tier", "breach"])
def test_divergent_slice_projects_require_separate_scoped_execution(candidate, workflow, monkeypatch):
    repo, corpus_dir, patch = candidate
    def forbid_backend(*args, **kwargs):
        pytest.fail("provider called before confirming scoped runner settings")
    monkeypatch.setattr(breach, "backend_pool", forbid_backend)
    monkeypatch.setattr(tier, "backend_pool", forbid_backend)
    if workflow == "tier":
        rc = tier.run_tier(
            cwd=str(repo), spec=corpus_dir, tier_id="divergent",
            breach_policy=BreachPolicy(enabled=True),
        )
    else:
        rc = breach.main([
            "--spec", str(corpus_dir), "--cwd", str(repo), "--diff", str(patch),
            "--observation-tier", "divergent",
        ])
    assert rc == 2
    summaries = [path.read_text() for path in corpus_dir.glob("runs/*/summary.json")]
    assert any("separate scoped handoffs or tiers" in item for item in summaries)


@pytest.mark.parametrize("tier_id", ["py-rust", "mixed", "divergent"])
def test_tier_without_exact_assurance_admits_normal_build_inputs(candidate, tier_id, monkeypatch):
    repo, corpus_dir, _ = candidate
    class ReachedOrdinaryAuthoring(Exception):
        pass

    def ordinary_boundary(*args, **kwargs):
        raise ReachedOrdinaryAuthoring()

    monkeypatch.setattr(tier, "host_route_write_root_refusal", ordinary_boundary)
    with pytest.raises(ReachedOrdinaryAuthoring):
        tier.run_tier(
            cwd=str(repo), spec=corpus_dir, tier_id=tier_id,
            breach_policy=BreachPolicy(enabled=False),
        )


def test_direct_dotnet_runner_without_language_constraint():
    spec = _spec()
    spec = replace(spec, project_context=replace(
        spec.project_context, runners={"verify": "crucible.common.runners.lang.dotnet"},
    ))
    with pytest.raises(ValueError, match="no exact per-test observation adapter for 'dotnet'"):
        breach.validate_breach_observation_runner(spec, language="python", gate="verify")


def test_mixed_dotnet_bench_is_rejected_before_dispatch(monkeypatch, capsys):
    config = {"parameters": {"languages": ["rust", "dotnet"], "runner_mode": "bench"}}
    monkeypatch.setattr(polyglot.sys, "stdin", StringIO(json.dumps(config)))
    monkeypatch.setattr(polyglot, "_run_language", lambda *args: pytest.fail("unexpected bench"))
    assert polyglot.verify_main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert "dotnet" in payload["raw_output"]
    assert "bench" in payload["raw_output"]
