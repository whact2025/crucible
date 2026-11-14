# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Executable contracts for Crucible's active documentation.

Operator, workflow, product, backend-worker, and contributor documentation
remain navigable and structurally valid as the implementation changes.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from urllib.parse import unquote

from crucible.backends.config import DEFAULT_ROUTING
from crucible.rubrics import available as available_rubrics
from crucible.workflows.__main__ import _discover_workflows


ROOT = Path(__file__).resolve().parents[1]

_FENCE_RE = re.compile(
    r"^```(?P<language>[^\n`]*)\n(?P<body>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
_LINK_RE = re.compile(
    r"!?(?:\[[^\]]*\])\((?P<target><[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)"
)
_MERMAID_DECLARATIONS = (
    "C4Component",
    "C4Container",
    "C4Context",
    "C4Deployment",
    "block-beta",
    "classDiagram",
    "erDiagram",
    "flowchart",
    "gantt",
    "gitGraph",
    "graph",
    "journey",
    "mindmap",
    "packet-beta",
    "pie",
    "quadrantChart",
    "requirementDiagram",
    "sankey-beta",
    "sequenceDiagram",
    "stateDiagram",
    "timeline",
    "xychart-beta",
)
_LONG_OPTION_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")


def _active_markdown() -> tuple[Path, ...]:
    """Return all maintained product and contributor Markdown."""

    candidates: set[Path] = set(ROOT.glob("*.md"))
    candidates.update((ROOT / "docs").rglob("*.md"))
    candidates.update((ROOT / ".claude" / "skills").rglob("*.md"))
    candidates.update((ROOT / "products").rglob("*.md"))
    candidates.add(ROOT / "rubrics" / "README.md")
    candidates.add(ROOT / "tests" / "TEST_BASELINE.md")

    candidates.update((ROOT / ".codex" / "skills").rglob("*.md"))
    candidates.update((ROOT / "backends").glob("*/README.md"))
    return tuple(sorted(path for path in candidates if path.is_file()))


def test_documentation_inventory_excludes_installed_worker_packages(
    tmp_path: Path, monkeypatch,
) -> None:
    worker = tmp_path / "backends" / "kimi_worker"
    worker.mkdir(parents=True)
    maintained = worker / "README.md"
    maintained.write_text("# Worker\n", encoding="utf-8")
    dependency_doc = worker / ".venv" / "lib" / "dependency" / "README.md"
    dependency_doc.parent.mkdir(parents=True)
    dependency_doc.write_text("[dependency docs](absent.md)\n", encoding="utf-8")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)

    assert _active_markdown() == (maintained,)


def _outside_fences(text: str) -> str:
    return _FENCE_RE.sub("", text)


def _local_target(source: Path, rendered: str) -> Path | None:
    target = rendered.strip("<>")
    if not target or target.startswith("#"):
        return None
    if target.startswith(("http://", "https://", "mailto:", "data:")):
        return None
    if "://" in target:
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target or any(marker in target for marker in ("{", "}")):
        return None
    if target.startswith("/"):
        # Absolute links are application routes or renderer-specific source
        # links; they are not portable filesystem references.
        return None
    return (source.parent / target).resolve()


def test_active_markdown_local_links_resolve() -> None:
    failures: list[str] = []
    for source in _active_markdown():
        text = _outside_fences(source.read_text(encoding="utf-8"))
        for match in _LINK_RE.finditer(text):
            target = _local_target(source, match.group("target"))
            if target is not None and not target.exists():
                failures.append(
                    f"{source.relative_to(ROOT)} -> "
                    f"{match.group('target')}"
                )
    assert not failures, "broken local documentation links:\n" + "\n".join(failures)


def test_active_markdown_fences_are_balanced() -> None:
    failures: list[str] = []
    for source in _active_markdown():
        lines = source.read_text(encoding="utf-8").splitlines()
        opened: tuple[int, str] | None = None
        for line_number, line in enumerate(lines, start=1):
            if not line.startswith("```"):
                continue
            if opened is None:
                opened = (line_number, line)
            else:
                opened = None
        if opened is not None:
            failures.append(
                f"{source.relative_to(ROOT)}:{opened[0]} unclosed {opened[1]!r}"
            )
    assert not failures, "unbalanced Markdown fences:\n" + "\n".join(failures)


def test_python_fences_are_syntactically_valid() -> None:
    failures: list[str] = []
    for source in _active_markdown():
        text = source.read_text(encoding="utf-8")
        for index, match in enumerate(_FENCE_RE.finditer(text), start=1):
            language = match.group("language").strip().lower()
            if language not in {"py", "python"}:
                continue
            try:
                ast.parse(textwrap.dedent(match.group("body")))
            except SyntaxError as exc:
                failures.append(
                    f"{source.relative_to(ROOT)} Python fence {index}: {exc.msg}"
                )
    assert not failures, "invalid Python examples:\n" + "\n".join(failures)


def test_mermaid_fences_have_supported_declarations() -> None:
    failures: list[str] = []
    for source in _active_markdown():
        text = source.read_text(encoding="utf-8")
        for index, match in enumerate(_FENCE_RE.finditer(text), start=1):
            if match.group("language").strip().lower() != "mermaid":
                continue
            meaningful = [
                line.strip()
                for line in match.group("body").splitlines()
                if line.strip() and not line.lstrip().startswith("%%")
            ]
            if not meaningful or not meaningful[0].startswith(
                _MERMAID_DECLARATIONS
            ):
                failures.append(
                    f"{source.relative_to(ROOT)} Mermaid fence {index} "
                    "has no recognized diagram declaration"
                )
    assert not failures, "invalid Mermaid blocks:\n" + "\n".join(failures)


def test_active_docs_do_not_use_box_drawing_as_diagrams() -> None:
    failures: list[str] = []
    box_drawing = re.compile(r"[\u2500-\u257f]")
    for source in _active_markdown():
        prose = _outside_fences(source.read_text(encoding="utf-8"))
        if box_drawing.search(prose):
            failures.append(str(source.relative_to(ROOT)))
    assert not failures, (
        "active prose uses textual box-drawing; use Mermaid for diagrams: "
        + ", ".join(failures)
    )


def test_active_docs_and_source_comments_do_not_link_removed_docs_tree() -> None:
    failures: list[str] = []
    for source in _active_markdown():
        text = source.read_text(encoding="utf-8")
        if "_docs/" in text or "crucible/crucible/" in text:
            failures.append(str(source.relative_to(ROOT)))

    source_roots = (
        "backends",
        "common",
        "products",
        "rubrics",
        "rules",
        "tools",
        "workflows",
    )
    python_sources = {ROOT / "constraints.py", ROOT / "__init__.py"}
    for directory in source_roots:
        python_sources.update(
            path for path in (ROOT / directory).rglob("*.py")
            if ".venv" not in path.parts
        )
    for source in sorted(python_sources):
        if "_docs/" in source.read_text(encoding="utf-8"):
            failures.append(str(source.relative_to(ROOT)))
    assert not failures, (
        "live documentation cites removed `_docs` or duplicate package paths:\n"
        + "\n".join(failures)
    )


def test_every_dispatched_workflow_has_a_reference_page() -> None:
    workflows = set(_discover_workflows())
    documented = {
        path.stem
        for path in (ROOT / "docs" / "workflows").glob("*.md")
        if path.name not in {"README.md", "analysis-contract.md"}
    }
    assert documented == workflows, (
        f"missing docs: {sorted(workflows - documented)}; "
        f"orphan docs: {sorted(documented - workflows)}"
    )


def test_workflow_index_links_every_reference_page() -> None:
    index = (ROOT / "docs" / "workflows" / "README.md").read_text(
        encoding="utf-8"
    )
    workflows = _discover_workflows()
    inventory = re.search(r"The (\d+) workflows", index)
    assert inventory is not None, "workflow index must state its derived inventory"
    assert int(inventory.group(1)) == len(workflows)
    for workflow in sorted(workflows):
        assert f"({workflow}.md)" in index, f"workflow index omits {workflow!r}"


def test_workflow_references_cover_every_public_long_option() -> None:
    failures: list[str] = []
    environment = {**os.environ, "COLUMNS": "240"}
    for workflow in sorted(_discover_workflows()):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "crucible.workflows",
                workflow,
                "--help",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"{workflow} --help failed:\n{result.stdout}\n{result.stderr}"
        )
        public_options = set(
            _LONG_OPTION_RE.findall(result.stdout + result.stderr)
        ) - {"--help"}
        reference = (
            ROOT / "docs" / "workflows" / f"{workflow}.md"
        ).read_text(encoding="utf-8")
        documented_options = set(_LONG_OPTION_RE.findall(reference))
        missing = sorted(public_options - documented_options)
        if missing:
            failures.append(f"{workflow}: {', '.join(missing)}")
    assert not failures, (
        "workflow reference pages omit public CLI options:\n"
        + "\n".join(failures)
    )


def test_readme_names_the_implemented_backend_default() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    default = DEFAULT_ROUTING.default
    assert default.backend and default.backend in readme
    assert default.model and default.model in readme
    assert default.effort and default.effort in readme


def test_rubric_readme_inventory_count_matches_manifest() -> None:
    readme = (ROOT / "rubrics" / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Crucible ships (\d+) active prompt bodies", readme)
    assert match is not None, "rubric README must state its derived inventory"
    assert int(match.group(1)) == len(available_rubrics(include_dormant=False))
