# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Write workflows refuse a host-routed root that holds its own name.

Codex, Claude, DeepSeek and Kimi write through the isolated mutation host,
which reads a leading path component equal to the name of its root as the root
itself. In a root that holds a directory of that name the host would write
beside the file a directive names. Implement, Tier, Breach and Architect
therefore end as ``setup_error`` before their run opens and before any backend
exists. A root without such a directory, a regular file of that name, an
OpenAI-compatible or local route, and a ``--worktree`` root are not refused.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from assay import ProjectContext, Spec
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG

from crucible.common.policies import BreachPolicy


NAME = "proj"  # basename of every checkout root built here
_ERROR_KIND = "write_root_name_collision"
_PROCEEDED = "proceeded"
_WORKFLOWS = ("architect", "breach", "implement", "tier")

_FLAT = {
    "x.py": "ROOT_X = 1\n",
    "pkg/mod.py": "ROOT_MOD = 1\n",
}
_LAYOUTS: dict[str, dict[str, str]] = {
    # No entry named like the root.
    "flat": _FLAT,
    # The root holds same-named files beside a directory of its own name.
    "nested": {
        **_FLAT,
        f"{NAME}/x.py": "NESTED_X = 1\n",
        f"{NAME}/pkg/mod.py": "NESTED_MOD = 1\n",
        f"{NAME}/only.py": "NESTED_ONLY = 1\n",
    },
    # The usual name/name package layout: the files exist only below it.
    "nested_only": {
        "setup.py": "SETUP = 1\n",
        f"{NAME}/x.py": "NESTED_X = 1\n",
        f"{NAME}/pkg/mod.py": "NESTED_MOD = 1\n",
    },
}
# Layout and directive scope of the three write-scope cases the isolated host
# misdirects: a write to the only (nested) file lands at the root; a scope
# naming the nested file admits the root file; a deny entry below a scope
# that is the nested directory is passed.
_MISDIRECTED = (
    pytest.param("nested_only", (f"{NAME}/x.py",), id="only_nested_file"),
    pytest.param("nested", (f"{NAME}/x.py",), id="scope_names_nested_file"),
    pytest.param("nested", (NAME,), id="scope_is_nested_directory"),
)

_ROUTING_SCHEMA = "crucible.backend-routing.v1"
_ROUTES: dict[str, dict[str, Any]] = {
    "codex": {"backend": "codex", "model": "gpt-6-astra"},
    "codex_official_sdk": {
        "backend": "codex",
        "model": "gpt-6-astra",
        "extra": {"transport": "official_sdk"},
    },
    "claude": {"backend": "claude", "model": "hermetic-no-network"},
    "deepseek": {"backend": "deepseek", "model": "deepseek-flash"},
    "kimi": {"backend": "kimi", "model": "kimi-k3.1-pro"},
    "openai_compat": {
        "backend": "openai_compat",
        "model": "hermetic-no-network",
    },
    "local": {"backend": "local", "model": "hermetic-no-network"},
}
_HOST_ROUTES = ("claude", "codex", "codex_official_sdk", "deepseek", "kimi")
_IN_PROCESS_ROUTES = ("local", "openai_compat")


class _Proceeded(Exception):
    """The workflow passed every setup exit and reached its own run."""


def _proceed(*_args: Any, **_kwargs: Any) -> Any:
    raise _Proceeded()


@pytest.fixture(autouse=True)
def _backend_calls(monkeypatch) -> Any:
    """Fail any test in which a routed backend is constructed."""

    calls: list[str] = []

    def construct(role: str, *_args: Any, **_kwargs: Any) -> Any:
        calls.append(role)
        raise AssertionError(f"a backend was constructed for role {role!r}")

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role", construct,
    )
    yield calls
    assert calls == []


def _tree(tmp_path: Path, layout: str) -> Path:
    root = tmp_path / NAME
    root.mkdir()
    for relative, text in _LAYOUTS.get(layout, _FLAT).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if layout == "name_file":
        (root / NAME).write_text("a regular file\n", encoding="utf-8")
    elif layout == "name_directory_link":
        (root / NAME).symlink_to("pkg", target_is_directory=True)
    elif layout == "name_file_link":
        (root / NAME).symlink_to("x.py")
    elif layout == "name_dangling_link":
        (root / NAME).symlink_to("absent")
    elif layout == "name_empty_directory":
        (root / NAME).mkdir()
    elif layout not in _LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    return root


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "guard@test.invalid")
    _git(root, "config", "user.name", "write root guard test")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")


def _snapshot(root: Path) -> dict[str, str]:
    """Return every entry below ``root`` with its kind or content digest."""

    entries: dict[str, str] = {}
    for current, directories, files in os.walk(root):
        if ".git" in directories:
            directories.remove(".git")
        for name in (*directories, *files):
            path = Path(current) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries[relative] = "link:" + os.readlink(path)
            elif path.is_dir():
                entries[relative] = "directory"
            else:
                entries[relative] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    return entries


def _corpus(
    tmp_path: Path,
    scope: tuple[str, ...] = ("x.py",),
) -> Corpus:
    directive = ImplementationDirective(
        slice_id="s0",
        intent="edit the declared file",
        files_in_scope=scope,
        creates_paths=(),
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="write_root_guard",
            repo_url="https://example.invalid/repo.git",
            base_commit="a" * 40,
            branch=None,
            submodule_path=None,
            # Tier validates the breach observation runner before it builds
            # its tool context; this bundled runner passes that validation.
            runners={"verify": "crucible.common.runners.lang.python"},
        ),
        execution=None,
        intent="write root guard",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=(Tier(
            tier_id="t0",
            slices=(Slice(id="s0", directive=directive),),
            depends_on=(),
        ),)),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(spec)
    return corpus


def _routing(tmp_path: Path, default: str, **roles: str) -> Path:
    """Write one closed routing document and return its path."""

    path = tmp_path / f"backends-{default}-{len(roles)}.yaml"
    path.write_text(
        yaml.safe_dump({
            "schema_version": _ROUTING_SCHEMA,
            "default": _ROUTES[default],
            "roles": {role: _ROUTES[route] for role, route in roles.items()},
        }),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------------
# The four entry points. Each stub sits at the first step after the
# workflow's setup exits and before its run opens, so ``_PROCEEDED`` means
# "not refused" and a returned exit code means a setup exit ended the call.
# ----------------------------------------------------------------------


def _implement(
    corpus: Corpus,
    root: Path,
    config: Path | None,
    monkeypatch,
) -> int:
    from crucible.workflows import implement

    monkeypatch.chdir(root)
    monkeypatch.setattr(implement, "_ImplementRunBoundary", _proceed)
    argv = ["--spec", str(corpus.spec_dir), "--slice", "s0"]
    if config is not None:
        argv += ["--backend-config", str(config)]
    return implement.main(argv)


def _tier(
    corpus: Corpus,
    root: Path,
    config: Path | None,
    monkeypatch,
    *,
    body: Any = _proceed,
    **kwargs: Any,
) -> int:
    from crucible.workflows import tier

    monkeypatch.setattr(tier, "_run_tier_body", body)
    return tier.run_tier(
        spec=Path(corpus.spec_dir),
        tier_id="t0",
        cwd=str(root),
        backend_config=config,
        skip_build_verify=True,
        skip_smoke_check=True,
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        **kwargs,
    )


def _breach(
    corpus: Corpus,
    root: Path,
    config: Path | None,
    monkeypatch,
) -> int:
    from crucible.workflows import breach

    monkeypatch.setattr(breach, "recovery_workspace_mutation_lock", _proceed)
    argv = [
        "--spec", str(corpus.spec_dir),
        "--diff", str(root.parent / "candidate.diff"),
        "--cwd", str(root),
    ]
    if config is not None:
        argv += ["--backend-config", str(config)]
    return breach.main(argv)


def _architect(
    corpus: Corpus,
    root: Path,
    config: Path | None,
    monkeypatch,
    **kwargs: Any,
) -> int:
    from crucible.workflows import architect

    monkeypatch.setattr(architect, "open_run_with_capture", _proceed)
    return architect.run_architect(
        spec=Path(corpus.spec_dir),
        cwd=str(root),
        backend_config=config,
        planner_count=1,
        **kwargs,
    )


_ENTRIES = {
    "architect": _architect,
    "breach": _breach,
    "implement": _implement,
    "tier": _tier,
}


def _run(
    workflow: str,
    corpus: Corpus,
    root: Path,
    config: Path | None,
    monkeypatch,
    **kwargs: Any,
) -> int | str:
    try:
        return _ENTRIES[workflow](corpus, root, config, monkeypatch, **kwargs)
    except _Proceeded:
        return _PROCEEDED


def _run_directories(corpus: Corpus) -> list[Path]:
    runs = Path(corpus.spec_dir) / "runs"
    if not runs.is_dir():
        return []
    return sorted(path for path in runs.iterdir() if path.is_dir())


def _refusal(corpus: Corpus, workflow: str) -> str:
    """Assert the one recorded terminal is the refusal; return its message."""

    runs = _run_directories(corpus)
    assert len(runs) == 1, runs
    run = runs[0]
    assert f"_{workflow}_" in run.name
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    outcome = json.loads((run / "outcome.json").read_text(encoding="utf-8"))
    assert result["status"] == "setup_error"
    assert result["error_kind"] == _ERROR_KIND
    assert result["cost_usd"] == 0.0
    assert summary["status"] == "setup_error"
    assert summary["error_kind"] == _ERROR_KIND
    assert summary["message"] == result["error_message"]
    # A setup error is a permanent refusal, never a retried backend failure.
    assert outcome["detail_status"] == "setup_error"
    assert outcome["terminal_class"] == "permanent_failure"
    # The handled terminal is the only record: no journal, no backend dump.
    assert not (run / "mutations").exists()
    return str(result["error_message"])


def _assert_refused(
    outcome: int | str,
    corpus: Corpus,
    workflow: str,
    root: Path,
    before: dict[str, str],
) -> str:
    assert outcome == 2, f"{workflow} was not refused: {outcome!r}"
    message = _refusal(corpus, workflow)
    assert _snapshot(root) == before
    return message


def _assert_proceeded(outcome: int | str, corpus: Corpus) -> None:
    assert outcome == _PROCEEDED
    # The stubs stand before each workflow's run, so nothing was recorded.
    assert _run_directories(corpus) == []


# ----------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------


@pytest.mark.parametrize("workflow", _WORKFLOWS)
@pytest.mark.parametrize(("layout", "scope"), _MISDIRECTED)
def test_host_route_refuses_a_root_holding_its_own_name(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    layout: str,
    scope: tuple[str, ...],
) -> None:
    root = _tree(tmp_path, layout)
    corpus = _corpus(tmp_path, scope)
    before = _snapshot(root)

    outcome = _run(
        workflow, corpus, root, _routing(tmp_path, "codex"), monkeypatch,
    )

    _assert_refused(outcome, corpus, workflow, root, before)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
@pytest.mark.parametrize(
    "route", [route for route in _HOST_ROUTES if route != "codex"],
)
def test_every_isolated_host_adapter_is_refused(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    route: str,
) -> None:
    root = _tree(tmp_path, "nested_only")
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    before = _snapshot(root)

    outcome = _run(
        workflow, corpus, root, _routing(tmp_path, route), monkeypatch,
    )

    _assert_refused(outcome, corpus, workflow, root, before)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
@pytest.mark.parametrize(
    "layout", ("name_directory_link", "name_empty_directory"),
)
def test_a_link_to_a_directory_and_an_empty_directory_are_refused(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    layout: str,
) -> None:
    root = _tree(tmp_path, layout)
    corpus = _corpus(tmp_path)
    before = _snapshot(root)

    outcome = _run(
        workflow, corpus, root, _routing(tmp_path, "codex"), monkeypatch,
    )

    _assert_refused(outcome, corpus, workflow, root, before)


@pytest.mark.parametrize("workflow", _WORKFLOWS)
def test_refusal_names_the_reason_and_both_ways_out(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
) -> None:
    root = _tree(tmp_path, "nested_only")
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    before = _snapshot(root)

    outcome = _run(
        workflow, corpus, root, _routing(tmp_path, "claude"), monkeypatch,
    )

    message = _assert_refused(outcome, corpus, workflow, root, before)
    write_role = "breach_defender" if workflow == "breach" else "implementer"
    for needle in (
        str(root.resolve()),
        f"directory named like itself ({NAME!r})",
        f"{write_role} (claude)",
        "isolated mutation host",
        f"{NAME}/<path>",
        "tier or architect with --worktree",
        "another name",
    ):
        assert needle in message, (needle, message)


# ----------------------------------------------------------------------
# Roots and routes that are not refused
# ----------------------------------------------------------------------


@pytest.mark.parametrize("workflow", _WORKFLOWS)
@pytest.mark.parametrize(
    ("layout", "route"),
    (
        pytest.param("flat", "codex", id="flat_tree"),
        pytest.param("name_file", "codex", id="regular_file_of_that_name"),
        pytest.param("name_file_link", "codex", id="link_to_a_regular_file"),
        pytest.param("name_dangling_link", "codex", id="dangling_link"),
        pytest.param("nested", "openai_compat", id="openai_compat_route"),
        pytest.param("nested_only", "local", id="local_route"),
    ),
)
def test_roots_and_routes_that_cannot_collide_are_not_refused(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    layout: str,
    route: str,
) -> None:
    root = _tree(tmp_path, layout)
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))

    outcome = _run(
        workflow, corpus, root, _routing(tmp_path, route), monkeypatch,
    )

    _assert_proceeded(outcome, corpus)


# ----------------------------------------------------------------------
# The route is the one each workflow's write roles resolve through
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("workflow", "default", "roles", "kwargs", "refused"),
    (
        pytest.param(
            "implement", "openai_compat", {"implementer": "codex"}, {}, True,
            id="implement_implementer_on_host",
        ),
        pytest.param(
            "implement", "codex", {"implementer": "openai_compat"}, {}, False,
            id="implement_only_other_roles_on_host",
        ),
        pytest.param(
            "breach", "openai_compat", {"breach_defender": "claude"}, {}, True,
            id="breach_defender_on_host",
        ),
        pytest.param(
            "breach", "codex", {"breach_defender": "local"}, {}, False,
            id="breach_only_other_roles_on_host",
        ),
        pytest.param(
            "tier", "codex", {"implementer": "openai_compat"}, {}, False,
            id="tier_without_breach_ignores_the_defender_route",
        ),
        pytest.param(
            "tier", "codex", {"implementer": "openai_compat"},
            {"breach_policy": BreachPolicy(enabled=True)}, True,
            id="tier_with_breach_follows_the_defender_route",
        ),
        pytest.param(
            "tier", "codex",
            {"implementer": "openai_compat", "breach_defender": "local"},
            {"breach_policy": BreachPolicy(enabled=True)}, False,
            id="tier_with_breach_and_both_writers_in_process",
        ),
        pytest.param(
            "architect", "codex", {"implementer": "openai_compat"},
            {"breach_policy": BreachPolicy(enabled=False)}, False,
            id="architect_without_breach_ignores_the_defender_route",
        ),
        pytest.param(
            "architect", "codex", {"implementer": "openai_compat"},
            {"breach_policy": BreachPolicy(enabled=True)}, True,
            id="architect_with_breach_follows_the_defender_route",
        ),
        pytest.param(
            # Without a policy Architect resolves the one its tiers will
            # resolve. This Spec authors none, so Breach is off in both.
            "architect", "codex", {"implementer": "openai_compat"}, {}, False,
            id="architect_without_a_policy_resolves_breach_off_like_a_tier",
        ),
        pytest.param(
            "architect", "openai_compat", {"implementer": "kimi"},
            {"breach_policy": BreachPolicy(enabled=False)}, True,
            id="architect_implementer_on_host",
        ),
    ),
)
def test_refusal_follows_the_routes_of_the_roles_that_write(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
    default: str,
    roles: dict[str, str],
    kwargs: dict[str, Any],
    refused: bool,
) -> None:
    root = _tree(tmp_path, "nested")
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    before = _snapshot(root)

    outcome = _run(
        workflow,
        corpus,
        root,
        _routing(tmp_path, default, **roles),
        monkeypatch,
        **kwargs,
    )

    if refused:
        _assert_refused(outcome, corpus, workflow, root, before)
    else:
        _assert_proceeded(outcome, corpus)


def _authored_breach_corpus(tmp_path: Path, enabled: bool) -> Corpus:
    """Return a Corpus whose ``spec.yaml`` authors the breach policy."""

    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    (Path(corpus.spec_dir) / "spec.yaml").write_text(
        yaml.safe_dump({"breach_policy": {"enabled": enabled}}),
        encoding="utf-8",
    )
    return corpus


@pytest.mark.parametrize("workflow", ("architect", "tier"))
def test_the_spec_decides_the_defender_route_when_no_policy_is_passed(
    tmp_path: Path,
    monkeypatch,
    workflow: str,
) -> None:
    root = _tree(tmp_path, "nested")
    # The only writer on a host route is the Defender.
    config = _routing(tmp_path, "codex", implementer="openai_compat")

    # Without a policy argument a tier reads the Spec's authored block, and
    # Architect resolves the block its tiers will read, so the roles the
    # guard is asked about follow that block in both.
    corpus = _authored_breach_corpus(tmp_path / "spec-enables-breach", True)
    before = _snapshot(root)
    message = _assert_refused(
        _run(workflow, corpus, root, config, monkeypatch),
        corpus, workflow, root, before,
    )
    assert "Write roles breach_defender (codex) reach" in message

    corpus = _authored_breach_corpus(tmp_path / "spec-disables-breach", False)
    _assert_proceeded(
        _run(workflow, corpus, root, config, monkeypatch), corpus,
    )

    # A policy that is passed outranks the authored block, as in the run.
    corpus = _authored_breach_corpus(tmp_path / "policy-disables-breach", True)
    _assert_proceeded(
        _run(
            workflow, corpus, root, config, monkeypatch,
            breach_policy=BreachPolicy(enabled=False),
        ),
        corpus,
    )


def test_explicit_routing_outranks_the_environment_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path, "nested_only")
    host = _routing(tmp_path, "codex")
    in_process = _routing(tmp_path, "openai_compat")

    monkeypatch.setenv("CRUCIBLE_BACKEND_CONFIG", str(host))
    corpus = _corpus(tmp_path / "explicit-in-process")
    _assert_proceeded(
        _run("implement", corpus, root, in_process, monkeypatch), corpus,
    )

    monkeypatch.setenv("CRUCIBLE_BACKEND_CONFIG", str(in_process))
    corpus = _corpus(tmp_path / "explicit-host")
    before = _snapshot(root)
    _assert_refused(
        _run("implement", corpus, root, host, monkeypatch),
        corpus, "implement", root, before,
    )


def test_discovered_routing_decides_without_an_explicit_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path, "nested_only")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    monkeypatch.setattr(
        "crucible.backends.config.workspace_root", lambda: workspace,
    )

    # No routing file anywhere: the built-in route is Codex for every role.
    corpus = _corpus(tmp_path / "built-in")
    before = _snapshot(root)
    _assert_refused(
        _run("implement", corpus, root, None, monkeypatch),
        corpus, "implement", root, before,
    )

    # A Spec-local routing file is discovered exactly as the pool finds it.
    corpus = _corpus(tmp_path / "spec-local")
    spec_local = Path(corpus.spec_dir) / ".crucible" / "backends.yaml"
    spec_local.parent.mkdir()
    spec_local.write_bytes(_routing(tmp_path, "local").read_bytes())
    _assert_proceeded(
        _run("implement", corpus, root, None, monkeypatch), corpus,
    )


def test_a_bound_routing_snapshot_outranks_the_routing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.backends.config import bind_routing_snapshot, load_routing

    root = _tree(tmp_path, "nested_only")
    host = _routing(tmp_path, "codex")
    in_process = _routing(tmp_path, "openai_compat")

    corpus = _corpus(tmp_path / "snapshot-in-process")
    with bind_routing_snapshot(load_routing(config_path=in_process)):
        outcome = _run("tier", corpus, root, host, monkeypatch)
    _assert_proceeded(outcome, corpus)

    corpus = _corpus(tmp_path / "snapshot-host")
    before = _snapshot(root)
    with bind_routing_snapshot(load_routing(config_path=host)):
        outcome = _run("tier", corpus, root, in_process, monkeypatch)
    _assert_refused(outcome, corpus, "tier", root, before)


# ----------------------------------------------------------------------
# The worktree option
# ----------------------------------------------------------------------


def test_tier_worktree_writes_in_a_root_that_cannot_hold_its_own_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path, "nested_only")
    _commit(root)
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    before = _snapshot(root)
    seen: dict[str, Any] = {}

    def body(args, cwd, ctx, corpus_, spec, tier_, *, extra_parameters=None):
        write_root = Path(ctx.repo_root).resolve()
        seen["root"] = write_root
        seen["cwd"] = Path(cwd).resolve()
        seen["resolver"] = ctx.path_resolver
        seen["declared_file"] = (write_root / NAME / "x.py").is_file()
        seen["own_name"] = (write_root / write_root.name).exists()
        return 0

    outcome = _run(
        "tier", corpus, root, _routing(tmp_path, "codex"), monkeypatch,
        body=body, worktree=True,
    )

    assert outcome == 0
    assert seen["root"] == seen["cwd"] != root.resolve()
    assert seen["root"].name.startswith("crucible_worktree_")
    assert seen["resolver"] is not None
    assert seen["declared_file"] is True
    assert seen["own_name"] is False
    assert not seen["root"].exists()
    assert _run_directories(corpus) == []
    assert _snapshot(root) == before


def test_architect_worktree_is_not_refused_and_its_tiers_write_there(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._worktree import worktree_session

    root = _tree(tmp_path, "nested_only")
    _commit(root)
    corpus = _corpus(tmp_path, (f"{NAME}/x.py",))
    config = _routing(tmp_path, "codex")

    _assert_proceeded(
        _run("architect", corpus, root, config, monkeypatch, worktree=True),
        corpus,
    )

    # Architect hands every tier one resolver and its worktree as ``cwd``.
    with worktree_session(root) as resolver:
        worktree = resolver.worktree_path
        assert worktree is not None
        assert worktree.name.startswith("crucible_worktree_")
        outcome = _run(
            "tier", corpus, worktree, config, monkeypatch,
            path_resolver=resolver,
        )
    _assert_proceeded(outcome, corpus)


# ----------------------------------------------------------------------
# The predicate
# ----------------------------------------------------------------------


def _guard() -> Any:
    from crucible.workflows import _write_root_guard

    return _write_root_guard


def _arguments(corpus: Corpus, config: Path | None) -> argparse.Namespace:
    return argparse.Namespace(spec=Path(corpus.spec_dir), backend_config=config)


def test_the_recorded_error_kind_is_the_guard_s_own() -> None:
    assert _guard().WRITE_ROOT_REFUSAL_ERROR_KIND == _ERROR_KIND


def test_isolated_host_writers_are_the_adapters_that_build_the_host() -> None:
    from crucible import backends
    from crucible.backends import claude, codex, deepseek, kimi, openai_compat

    from ._source_invariants import assert_source_invariant

    adapters = {
        "claude": claude,
        "codex": codex,
        "deepseek": deepseek,
        "kimi": kimi,
        "local": openai_compat,
        "openai_compat": openai_compat,
    }
    hosted = _guard().ISOLATED_HOST_WRITER_BACKENDS

    # Every registered adapter is classified; none is assumed in-process.
    assert set(adapters) == set(backends.BACKEND_NAMES)
    assert hosted == {"claude", "codex", "deepseek", "kimi"}
    # The official Codex transport inherits the Codex mutation-host path.
    assert issubclass(backends.OfficialCodexBackend, backends.CodexBackend)
    for name, module in adapters.items():
        if name in hosted:
            assert_source_invariant(
                module,
                must_contain=("build_isolated_toolset_spec",),
                scope=f"{name} writes through the isolated mutation host",
            )
        else:
            assert_source_invariant(
                module,
                must_not_contain=("_isolated_mutation_host",),
                scope=f"{name} dispatches its writers in process",
            )


@pytest.mark.parametrize(
    ("layout", "refused"),
    (
        ("flat", False),
        ("name_file", False),
        ("name_file_link", False),
        ("name_dangling_link", False),
        ("nested", True),
        ("nested_only", True),
        ("name_empty_directory", True),
        ("name_directory_link", True),
    ),
)
def test_predicate_reads_the_disk_as_the_host_names_its_root(
    tmp_path: Path,
    layout: str,
    refused: bool,
) -> None:
    root = _tree(tmp_path, layout)
    corpus = _corpus(tmp_path)

    message = _guard().host_route_write_root_refusal(
        _arguments(corpus, _routing(tmp_path, "codex")),
        root,
        ("implementer",),
    )

    assert (message is not None) is refused


def test_predicate_uses_the_name_of_the_resolved_root(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    arguments = _arguments(corpus, _routing(tmp_path, "codex"))
    guard = _guard().host_route_write_root_refusal

    # The host names its root after resolving links, so an alias of a
    # colliding root is refused under the name the root really has.
    root = _tree(tmp_path, "nested_only")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    message = guard(arguments, alias, ("implementer",))
    assert message is not None
    assert str(root.resolve()) in message

    # A directory named like the alias is not named like the resolved root.
    plain = tmp_path / "plain"
    (plain / "other").mkdir(parents=True)
    other = tmp_path / "other"
    other.symlink_to(plain, target_is_directory=True)
    assert guard(arguments, other, ("implementer",)) is None


def test_predicate_reads_no_routing_for_a_root_that_cannot_collide(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _tree(tmp_path, "flat")
    corpus = _corpus(tmp_path)
    guard = _guard()
    monkeypatch.setattr(guard, "load_routing", _proceed)

    absent = tmp_path / "absent.yaml"
    assert guard.host_route_write_root_refusal(
        _arguments(corpus, absent), root, ("implementer",),
    ) is None


def test_predicate_reads_routing_with_the_call_the_backend_pool_makes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _backends

    root = _tree(tmp_path, "nested_only")
    corpus = _corpus(tmp_path)
    arguments = _arguments(corpus, _routing(tmp_path, "codex"))
    guard = _guard()
    loaded: dict[str, tuple[dict[str, Any], Any]] = {}

    def recording(reader: str, load: Any) -> Any:
        def load_and_record(**kwargs: Any) -> Any:
            routing = load(**kwargs)
            loaded[reader] = (kwargs, routing)
            return routing

        return load_and_record

    monkeypatch.setattr(
        guard, "load_routing", recording("guard", guard.load_routing),
    )
    monkeypatch.setattr(
        _backends, "load_routing", recording("pool", _backends.load_routing),
    )

    assert guard.host_route_write_root_refusal(
        arguments, root, ("implementer",),
    ) is not None
    with _backends.backend_pool(arguments, None) as pool:
        pooled = pool.routing

    # Same discovery arguments, and so the same routing table, as the pool
    # every guarded workflow then resolves its roles through.
    assert loaded["guard"][0] == loaded["pool"][0] == {
        "config_path": arguments.backend_config,
        "spec_dir": arguments.spec,
    }
    assert loaded["guard"][1] == loaded["pool"][1] == pooled
    assert pooled.for_role("implementer").backend == "codex"


def test_predicate_fails_closed_when_the_routing_cannot_be_read(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path, "nested_only")
    corpus = _corpus(tmp_path)

    with pytest.raises(ValueError, match="explicit backend routing config"):
        _guard().host_route_write_root_refusal(
            _arguments(corpus, tmp_path / "absent.yaml"),
            root,
            ("implementer",),
        )


def test_predicate_refuses_only_while_a_write_context_carries_the_name_drop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.tools import _isolated_mutation_host as host
    from crucible.tools._context import ToolContext

    root = _tree(tmp_path, "nested_only")
    corpus = _corpus(tmp_path)
    arguments = _arguments(corpus, _routing(tmp_path, "codex"))
    guard = _guard().host_route_write_root_refusal
    declared = f"{NAME}/x.py"

    def hosted() -> Any:
        # A context built as the workflows build theirs, as the host takes it.
        return host._snapshot_context(ToolContext(
            workspace_root=root, repo_root=root, write_scope=(declared,),
        ))

    # The drop is on: the host names the root-level twin; the guard refuses.
    snapshot = hosted()
    assert snapshot.workspace_relative_paths is True
    assert host._write_is_authorized(snapshot, declared) == "x.py"
    assert guard(arguments, root, ("implementer",)) is not None

    # The class attribute is not what a context carries. Assigning it leaves
    # the drop on in the host, so the guard has to keep refusing.
    with monkeypatch.context() as patch:
        patch.setattr(ToolContext, "workspace_relative_paths", False)
        snapshot = hosted()
        assert snapshot.workspace_relative_paths is True
        assert host._write_is_authorized(snapshot, declared) == "x.py"
        assert guard(arguments, root, ("implementer",)) is not None

    # The drop is off where a context gets it: the host names the declared
    # file, and only then does the guard stop refusing.
    construct = ToolContext.__init__

    def without_the_drop(self: Any, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("workspace_relative_paths", False)
        construct(self, *args, **kwargs)

    monkeypatch.setattr(ToolContext, "__init__", without_the_drop)
    snapshot = hosted()
    assert snapshot.workspace_relative_paths is False
    assert host._write_is_authorized(snapshot, declared) == declared
    assert guard(arguments, root, ("implementer",)) is None


# ----------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------

_WRITE_TOOLSET_BUILDERS = frozenset({
    "build_autonomous_implementer_profile",
    "build_breach_defender_profile",
    "build_exact_create_recovery_profile",
    "build_exact_create_tools",
    "build_implementer_profile",
    "build_write_tools",
})
# Every production module outside ``tools`` that constructs a ToolContext,
# with the reason no write context built there reaches the isolated host in
# a root that holds its own name.
_CONTEXT_MODULES = {
    "workflows/implement.py": "refused in main",
    "workflows/tier.py": "refused in run_tier, after any worktree rebinding",
    "workflows/breach.py": "refused in _main",
    "workflows/architect.py": "read-only planner context; refused for its tiers",
    "workflows/_writer_scheduler.py": "a fresh worktree root for every lane",
    "workflows/plan.py": "read-only source tools",
    "workflows/_readonly_scheduler.py": "validated read-only",
    "workflows/_write_root_guard.py": "read for the name-drop flag only",
}
_WRITE_TOOLSET_MODULES = {
    "workflows/breach.py",
    "workflows/implement.py",
    "workflows/tier.py",
}


def _called_names(path: Path) -> set[str]:
    """Return the name of every function or constructor ``path`` calls."""

    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def test_every_module_that_builds_a_tool_context_is_covered() -> None:
    import crucible

    package = Path(crucible.__file__).resolve().parent
    watched = ("ToolContext", *_WRITE_TOOLSET_BUILDERS)
    contexts: set[str] = set()
    writers: set[str] = set()
    unguarded: set[str] = set()
    for area in (
        "backends", "common", "products", "rubrics", "rules", "scripts",
        "workflows",
    ):
        for current, directories, files in os.walk(package / area):
            directories[:] = [
                name for name in directories
                if not name.startswith(".")
                and name not in {"__pycache__", "tests"}
            ]
            for name in files:
                path = Path(current) / name
                if path.suffix != ".py":
                    continue
                source = path.read_text(encoding="utf-8")
                if not any(word in source for word in watched):
                    continue
                called = _called_names(path)
                relative = path.relative_to(package).as_posix()
                if "ToolContext" in called:
                    contexts.add(relative)
                if called & _WRITE_TOOLSET_BUILDERS:
                    writers.add(relative)
                    if "host_route_write_root_refusal" not in called:
                        unguarded.add(relative)

    # A new site has to say how the guard covers it before it is listed.
    assert contexts == set(_CONTEXT_MODULES)
    assert writers == _WRITE_TOOLSET_MODULES
    # A module that builds a write-capable ToolSet asks the guard itself.
    assert unguarded == set()
