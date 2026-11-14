# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest
import yaml

from assay import InvocationResult, Usage

from crucible.backends.config import (
    BackendRouting,
    RoleConfig,
    resolve_for_role,
    routing_fingerprint,
)
from crucible.backends.trusted_codex import (
    effective_project_instruction_paths,
    trusted_codex_asset_paths,
    trusted_codex_context_digest,
)


_ROOT = Path(__file__).resolve().parents[1]


class _Backend:
    name = "codex"
    model = "gpt-5.6-sol"

    def invoke(self, invocation):
        del invocation
        return InvocationResult(
            final_message="",
            transcript=(),
            usage=Usage(),
            stop_reason="end_turn",
        )


def _skill_frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, raw, _body = text.split("---\n", 2)
    value = yaml.safe_load(raw)
    assert isinstance(value, dict)
    return value


def test_trusted_codex_assets_are_closed_narrow_and_parseable():
    paths = trusted_codex_asset_paths(_ROOT)
    relative = {path.relative_to(_ROOT).as_posix() for path in paths}

    assert "AGENTS.md" in relative
    assert relative >= {
        ".codex/agents/crucible-explorer.toml",
        ".codex/agents/crucible-security-reviewer.toml",
        ".codex/agents/crucible-test-triage.toml",
        ".codex/skills/validate-crucible-change/SKILL.md",
        ".codex/skills/qualify-crucible-backend/SKILL.md",
    }
    for path in paths:
        assert not path.is_symlink()
        assert path.stat().st_size <= 256 * 1024

    for path in (_ROOT / ".codex" / "agents").glob("*.toml"):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        assert {
            "name",
            "description",
            "developer_instructions",
        } <= payload.keys()
        assert payload["sandbox_mode"] == "read-only"
        assert "model" not in payload

    for path in (_ROOT / ".codex" / "skills").glob("*/SKILL.md"):
        frontmatter = _skill_frontmatter(path)
        assert set(frontmatter) == {"name", "description"}
        assert frontmatter["name"] == path.parent.name
        assert "TODO" not in path.read_text(encoding="utf-8")


def test_trusted_context_digest_binds_every_asset_and_effective_chain(tmp_path):
    checkout = tmp_path / "checkout"
    shutil.copytree(
        _ROOT / ".codex",
        checkout / ".codex",
    )
    shutil.copy2(_ROOT / "AGENTS.md", checkout / "AGENTS.md")
    nested = checkout / "products" / "example"
    nested.mkdir(parents=True)
    (checkout / "products" / "AGENTS.md").write_text(
        "product instructions\n",
        encoding="utf-8",
    )

    before = trusted_codex_context_digest(checkout, nested)
    chain = effective_project_instruction_paths(checkout, nested)
    assert [path.relative_to(checkout).as_posix() for path in chain] == [
        "AGENTS.md",
        "products/AGENTS.md",
    ]

    skill = (
        checkout
        / ".codex"
        / "skills"
        / "validate-crucible-change"
        / "SKILL.md"
    )
    skill.write_text(
        skill.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert trusted_codex_context_digest(checkout, nested) != before


def test_trusted_instruction_override_is_selected_and_symlinks_fail_closed(
    tmp_path,
):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "AGENTS.md").write_text("root\n", encoding="utf-8")
    (checkout / ".codex" / "agents").mkdir(parents=True)
    (checkout / ".codex" / "skills").mkdir(parents=True)
    nested = checkout / "nested"
    nested.mkdir()
    (nested / "AGENTS.md").write_text("ordinary\n", encoding="utf-8")
    override = nested / "AGENTS.override.md"
    override.write_text("override\n", encoding="utf-8")

    assert effective_project_instruction_paths(checkout, nested)[-1] == override

    override.unlink()
    outside = tmp_path / "outside.md"
    outside.write_text("escape\n", encoding="utf-8")
    override.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes repository root"):
        effective_project_instruction_paths(checkout, nested)


def test_trusted_codex_route_is_read_only_explicit_and_provenance_bound(
    monkeypatch,
):
    captured = {}

    def fake_resolve(name, **kwargs):
        captured.update(name=name, kwargs=kwargs)
        return _Backend()

    monkeypatch.setattr("crucible.backends.config._resolve_backend", fake_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="enforce",
            extra={"trusted_project_instructions": True},
        )
    )
    ordinary = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission="enforce",
        )
    )

    backend = resolve_for_role(
        "architect_planner",
        routing=routing,
        capability="read_only",
        native_shell=False,
        cwd=str(_ROOT),
    )

    assert isinstance(backend, _Backend)
    assert captured["kwargs"]["trusted_project_instructions"] is True
    assert routing_fingerprint(routing) != routing_fingerprint(ordinary)


@pytest.mark.parametrize(
    ("role", "capability", "native_shell", "cwd", "admission"),
    [
        (
            "implementer",
            "workspace_write",
            False,
            str(_ROOT),
            "enforce",
        ),
        (
            "architect_planner",
            "read_only",
            True,
            str(_ROOT),
            "enforce",
        ),
        ("architect_planner", "read_only", False, None, "enforce"),
        (
            "architect_planner",
            "read_only",
            False,
            str(_ROOT),
            "diagnostic",
        ),
    ],
)
def test_trusted_codex_route_rejects_non_readonly_or_ambiguous_boundaries(
    role,
    capability,
    native_shell,
    cwd,
    admission,
):
    routing = BackendRouting(
        default=RoleConfig(
            backend="codex",
            model="gpt-5.6-sol",
            feature_admission=admission,
            extra={"trusted_project_instructions": True},
        )
    )

    with pytest.raises(ValueError, match="trusted_project_instructions"):
        resolve_for_role(
            role,
            routing=routing,
            capability=capability,
            native_shell=native_shell,
            cwd=cwd,
        )
