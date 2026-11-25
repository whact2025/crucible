# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""ToolSet fixtures of the matrix, built from the real workflow ToolSets.

``READ`` is two source tools, ``READ_BOUNDED(n)`` wraps them in the Planner's
dispatch budget and ``WRITER(k)`` is the autonomous implementer profile inside
the implementer's progress bound: the objects a workflow attaches.

Scripts name tools by slot; :data:`BINDINGS` is the one place where a slot
becomes a tool name and arguments. The fixture records every call that reaches
the innermost ToolSet by shadowing that instance's ``dispatch``. The wrappers
and the types stay the product's own, so an adapter that inspects the attached
object sees exactly what a workflow would hand it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from crucible.tools import (
    BoundedReadOnlyToolSet,
    MutatingToolProgressSnapshot,
    ProgressBoundedMutatingToolSet,
    ToolContext,
)
from crucible.tools.profiles import build_autonomous_implementer_profile
from crucible.tools.source import build_source_tools

from .script import READ_A, READ_B, WRITE

NOTES_FILE = "notes.txt"
WRITE_FILE = "result.txt"
WRITE_CONTENT = "written by the matrix writer scenario\n"
_NEEDLE = "matrix-needle"
# Long enough that one read result exceeds every context budget a scenario sets.
_NOTES = "".join(
    f"line {number:02d}: {_NEEDLE if number == 3 else 'filler'} "
    "for the matrix read fixture\n"
    for number in range(1, 13)
)


@dataclass(frozen=True)
class ToolBinding:
    """The real tool and arguments a script slot stands for."""

    name: str
    arguments: Mapping[str, Any]


BINDINGS: Mapping[str, ToolBinding] = {
    READ_A: ToolBinding("read_source", {"file_path": NOTES_FILE}),
    READ_B: ToolBinding("grep_symbol", {"pattern": _NEEDLE}),
    WRITE: ToolBinding(
        "write_source", {"file_path": WRITE_FILE, "content": WRITE_CONTENT}
    ),
}


@dataclass(frozen=True)
class ToolsetSpec:
    """Which fixture a scenario attaches."""

    kind: Literal["READ", "READ_BOUNDED", "WRITER"]
    limit: int | None = None

    @property
    def label(self) -> str:
        return self.kind if self.limit is None else f"{self.kind}({self.limit})"


READ = ToolsetSpec("READ")


def READ_BOUNDED(max_calls: int) -> ToolsetSpec:  # noqa: N802 - fixture name
    return ToolsetSpec("READ_BOUNDED", max_calls)


def WRITER(max_consecutive_read_calls: int) -> ToolsetSpec:  # noqa: N802
    return ToolsetSpec("WRITER", max_consecutive_read_calls)


@dataclass(frozen=True)
class Dispatch:
    """One call that reached the innermost ToolSet."""

    name: str
    arguments: Mapping[str, Any]
    result: str


@dataclass
class Fixture:
    """An attached ToolSet with what the cell may observe about it."""

    spec: ToolsetSpec
    toolset: Any
    workspace: Path
    log: list[Dispatch] = field(default_factory=list)
    _progress: Callable[[], MutatingToolProgressSnapshot] | None = None

    def progress(self) -> MutatingToolProgressSnapshot | None:
        """The progress wrapper's own counters, where the fixture has one."""

        return None if self._progress is None else self._progress()

    def write_applied(self) -> bool:
        """Whether the scripted write is on disk with the scripted content."""

        target = self.workspace / WRITE_FILE
        return target.is_file() and target.read_text(encoding="utf-8") == WRITE_CONTENT


def _record(inner: Any, log: list[Dispatch]) -> None:
    original = inner.dispatch

    def dispatch(name: str, arguments: dict) -> str:
        result = original(name, arguments)
        log.append(Dispatch(name=name, arguments=dict(arguments or {}), result=result))
        return result

    # An instance attribute shadows the method for this one object only.
    inner.dispatch = dispatch


def build(spec: ToolsetSpec, workspace: Path, *, scoped: bool = True) -> Fixture:
    """Build the fixture in a fresh ``workspace`` directory.

    A writer states one path scope, the file its script writes. With
    ``scoped`` false it is the same tools with no path scope stated: what a
    cell attaches is always the scoped one, and a proof builds the other to
    say which authority a digest names.
    """

    workspace.mkdir(parents=True, exist_ok=False)
    (workspace / NOTES_FILE).write_text(_NOTES, encoding="utf-8")
    fixture = Fixture(spec=spec, toolset=None, workspace=workspace)
    if spec.kind == "WRITER":
        assert spec.limit is not None
        inner = build_autonomous_implementer_profile(
            ToolContext(
                workspace_root=workspace,
                repo_root=workspace,
                write_scope=(WRITE_FILE,) if scoped else (),
            )
        )
        _record(inner, fixture.log)
        bounded = ProgressBoundedMutatingToolSet(
            inner, max_consecutive_read_calls=spec.limit
        )
        fixture.toolset = bounded
        fixture._progress = bounded.snapshot
        return fixture
    read = build_source_tools(
        ToolContext(workspace_root=workspace, repo_root=workspace)
    ).filter_to({BINDINGS[READ_A].name, BINDINGS[READ_B].name})
    _record(read, fixture.log)
    if spec.kind == "READ":
        fixture.toolset = read
        return fixture
    assert spec.limit is not None
    fixture.toolset = BoundedReadOnlyToolSet(
        read, max_calls=spec.limit, max_result_bytes=1_000_000
    )
    return fixture


def capability_of(spec: ToolsetSpec | None) -> str:
    """The role capability a scenario with this fixture asks routing for."""

    if spec is None:
        return "text"
    return "workspace_write" if spec.kind == "WRITER" else "read_only"
