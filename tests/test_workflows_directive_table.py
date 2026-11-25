# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The directive table reads every directive path of a Spec once, or refuses.

Every checkout built here is called ``proj``. It is *flat* when its root
holds no entry of that name and *nested* when it holds a directory of that
name. A path "begins with the name" when its first component is ``proj`` and
more follow; an entry that is ``proj`` alone is "the name itself". A
statement is the run's ``--spec-read-against`` value, ``write-root`` or
``parent``; a run without the option made none.

- Nested: a Spec with a path that begins with the name, or with a scope entry
  that is the name itself, is read only on a statement. Entries on disk
  refuse a statement; they select no reading.
- Flat: a leading name is dropped without a statement. A stated
  ``write-root`` drops none and is refused where a path begins with the name
  or the first component of a bound is the name. A scope entry that is the
  name itself is refused without a statement where the root holds no entry
  of that name, and under ``write-root`` where it stands in a slice beside
  an entry that does not begin with the name.
- A worktree run decides on the worktree and on the checkout it was made
  from, and goes on only when both read the Spec the same way.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from assay import (
    BlastRadiusConstraint,
    BudgetConstraint,
    ImplementationDirective,
    MutationRegion,
    ProjectContext,
    RuleContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
    ValidationRuleset,
)
from assay.primitives.validate import validate

import crucible.rules as rules
from crucible.common.scope_reading import (
    BLOCKED,
    DIRECTORY,
    REDUNDANT,
    ScopePathError,
    ScopeReadingError,
    decide_scope_reading,
)
from crucible.workflows import _directive_table as directive_table
from crucible.workflows._args import SPEC_READ_AGAINST_OPTION as OPTION
from crucible.workflows._directive_table import (
    KIND_BARE_NAME,
    KIND_BLOCKED,
    KIND_BOUND,
    KIND_CONTRADICTED,
    KIND_CONTRADICTORY,
    KIND_INVALID,
    KIND_NO_DIRECTIVE_READ,
    KIND_NO_DIRECTORY,
    KIND_NO_EVIDENCE,
    KIND_NOT_BELOW,
    KIND_NOT_FROM_TABLE,
    KIND_REPEATS,
    KIND_UNSTATED,
    KIND_WITHOUT_EFFECT,
    KIND_WORKTREE,
    KIND_WORKTREE_VOTES,
    DirectiveNotFromTable,
    DirectiveRefusal,
    DirectiveTable,
)


NAME = "proj"  # basename of every checkout built here
STATEMENTS = (None, "write-root", "parent")

# What a cell of a matrix expects: an error kind, or a reading and its basis.
STATED = "write-root, stated"
NOT_NEEDED = "write-root, not_needed"
DROPPED = "None, redundant_name"
PARENT_STATED = "parent, stated"
PARENT_DROPPED = "parent, redundant_name"

_FLAT = {"setup.py": "SETUP = 1\n", "src/a.py": "A = 1\n"}
_TREES: dict[str, dict[str, str]] = {
    # No entry named like the checkout.
    "flat": _FLAT,
    # A directory named like the checkout; ``src/a.py`` exists in both.
    "nested": {**_FLAT, "proj/src/a.py": "NESTED_A = 1\n", "proj/only.py": "ONLY = 1\n"},
    # A regular file named like the checkout.
    "name_file": {**_FLAT, "proj": "a regular file\n"},
    # The trees of the statement matrix.
    "matrix_nested": {
        "setup.py": "SETUP = 1\n",
        "rendering/b.py": "B = 1\n",
        "proj/source/a.py": "A = 1\n",
    },
    "matrix_flat": {"x.py": "X = 1\n", "pkg/mod.py": "MOD = 1\n"},
}

OWN_WORK = "An entry on disk can be the result of this Spec's own earlier work."
HOW_TO_TELL = (
    "An entry that exists is not proof: file names can coincide. The Spec's "
    "paths are relative to the directory above the checkout 'proj' only if the "
    "Spec was planned from that directory; otherwise they are relative to the "
    "write root."
)
STATE_WHICH = (
    f"State which one the Spec means with {OPTION} write-root or {OPTION} parent. "
    "A statement that an entry on disk contradicts is refused."
)
# The sentences of a nested checkout for a scope entry that is the name itself.
ENTRY_HAS_TWO_MEANINGS = (
    "1 scope entries are 'proj' itself ('proj/'): read as written that is the "
    "directory of that name, read against the directory above it is the write "
    "root, and no entry on disk can tell the two apart."
)
WRITE_ROOT_FOR_THE_ENTRY = (
    f"{OPTION} write-root is the statement for a Spec planned from this "
    "checkout, in which the entry means the directory 'proj' below the write "
    "root; a statement that an entry on disk contradicts is refused."
)
PARENT_NEVER_FOR_THE_ENTRY = (
    f"{OPTION} parent is refused for every Spec that holds an entry that is "
    "'proj' itself: read against the directory above, that entry is the write "
    "root and names no path below it. If the Spec was planned from the "
    f"directory above, no {OPTION} value makes it readable in this write root "
    "as written: correct the entry (name the paths it means, each with the "
    "leading 'proj'), or run from a root that holds every path of the Spec."
)
NO_VALUE_READS = f"No {OPTION} value reads this Spec in this checkout as it stands"
CAN_LIE_INSIDE = (
    "inside which a request below the entry can lie once it is read without "
    "the leading name"
)
SESSION_SENTENCE = (
    " A directory named 'proj' that exists in the worktree only can be this "
    "session's own earlier work."
)
DECIDE_DIFFERENTLY = "The two places decide differently."
SAME_REFUSAL = (
    "Where the difference is this session's own earlier work in the worktree "
    "(an entry it created or removed there), this refusal stays: a run without "
    "a worktree reaches the same refusal after the same tier."
)
WAYS_OUT = (
    "To remove the difference, make the two places equal (commit or remove "
    "what differs; an empty directory cannot be committed, a file below it "
    "can), or run without a worktree. A statement is accepted in a worktree "
    "only where both places take the same reading under it."
)


# --------------------------------------------------------------------------
# Trees, Specs and one decision
# --------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> str:
    # The repositories are built without the user's and the system's Git
    # configuration, so that a setting such as commit signing cannot fail
    # the commit of a test tree.
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _tree(base: Path, files: dict[str, str], *, commit: bool = True) -> Path:
    """Build ``<base>/proj`` and, unless told otherwise, commit it."""

    root = base / NAME
    root.mkdir(parents=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if commit:
        _git(root, "init", "-q")
        _git(root, "config", "user.email", "directive-table@test.invalid")
        _git(root, "config", "user.name", "directive table test")
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "base")
    return root.resolve()


def _worktree(root: Path) -> Path:
    """A detached worktree of ``root``, beside it and under another name."""

    path = root.parent / "crucible_worktree_ab12cd34"
    _git(root, "worktree", "add", "-q", "--detach", str(path), "HEAD")
    return path.resolve()


def _put(root: Path, relative: str, text: str = "NEW = 1\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="module")
def trees(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Committed checkouts and worktrees of them; no test changes one."""

    base = tmp_path_factory.mktemp("directive_table")
    built = {label: _tree(base / label, files) for label, files in _TREES.items()}
    for label in ("flat", "nested", "name_file", "matrix_nested", "matrix_flat"):
        built[f"{label}_worktree"] = _worktree(built[label])
    return built


def _directive(
    slice_id: str,
    files: tuple[str, ...],
    creates: tuple[str, ...] = (),
    regions: tuple[str, ...] = (),
) -> ImplementationDirective:
    return ImplementationDirective(
        slice_id=slice_id,
        intent="edit the declared files",
        files_in_scope=tuple(files),
        creates_paths=tuple(creates),
        verify_mode="build_only",
        blast_radius={},
        mutation_regions=tuple(MutationRegion(file=path) for path in regions),
    )


def _spec(
    files: tuple[str, ...] = (),
    creates: tuple[str, ...] = (),
    *,
    bounds: tuple[str, ...] | None = None,
    regions: tuple[str, ...] = (),
    beside: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (),
    later: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (),
) -> Spec:
    """Slice ``a`` in tier ``t0``.

    ``beside`` adds slices to ``t0``; each entry of ``later`` is one more
    tier with one slice. Both are ``(slice id, files_in_scope, creates_paths)``.
    """

    first = (
        Slice(id="a", directive=_directive("a", files, creates, regions)),
        *(Slice(id=slice_id, directive=_directive(slice_id, more_files, more_creates))
          for slice_id, more_files, more_creates in beside),
    )
    tiers = [Tier(tier_id="t0", slices=first, depends_on=())]
    for index, (slice_id, more_files, more_creates) in enumerate(later, 1):
        tiers.append(Tier(
            tier_id=f"t{index}",
            slices=(Slice(
                id=slice_id, directive=_directive(slice_id, more_files, more_creates),
            ),),
            depends_on=(f"t{index - 1}",),
        ))
    return Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project=NAME, repo_url="https://example.invalid/proj.git",
            base_commit="0" * 40, branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="a synthetic Spec",
        constraints=(
            () if bounds is None
            else (BlastRadiusConstraint(no_edits_outside=tuple(bounds)),)
        ),
        done_criteria=(),
        plan_policy=None,
        tier_dag=TierDAG(tiers=tuple(tiers)),
    )


@dataclasses.dataclass(frozen=True)
class _Outcome:
    """One call of the step: the table it gave, or the refusal it raised."""

    table: DirectiveTable | None
    refusal: DirectiveRefusal | None

    @property
    def decision(self) -> str:
        if self.refusal is not None:
            return self.refusal.kind
        assert self.table is not None
        return f"{self.table.read_against}, {self.table.basis}"

    @property
    def record(self) -> dict[str, Any]:
        if self.refusal is not None:
            return self.refusal.record
        assert self.table is not None
        return self.table.record

    @property
    def message(self) -> str:
        assert self.refusal is not None, self.decision
        return str(self.refusal)

    @property
    def votes(self) -> tuple[int, int]:
        return self.record["write_root_votes"], self.record["parent_votes"]

    def scope(self, slice_id: str = "a") -> tuple[str, ...]:
        assert self.table is not None, self.decision
        return self.table.scope(slice_id)


def _read(
    root: Path,
    spec: Spec,
    statement: str | None = None,
    *,
    source: Path | None = None,
    **more: Any,
) -> _Outcome:
    try:
        table = directive_table.build_directive_table(
            decide_scope_reading(root), spec,
            read_against=statement, worktree_source=source, **more,
        )
    except DirectiveRefusal as refusal:
        return _Outcome(None, refusal)
    return _Outcome(table, None)


def _decisions(root: Path, spec: Spec, *, source: Path | None = None) -> list[str]:
    return [_read(root, spec, statement, source=source).decision
            for statement in STATEMENTS]


def _two_meanings(root: Path, led: int, votes: str) -> str:
    """How a refusal of a nested checkout opens, for ``led`` name-led paths.

    ``votes`` is the sentence that counts and names the paths that vote.
    """

    return (
        f"{led} Spec paths begin with 'proj', and {root} holds a directory of "
        "that name, so proj/<path> can mean <path> below that directory or "
        f"<path> in the write root. Of the paths, {votes}"
    )


def _not_below(root: Path, what: str) -> str:
    """The refusal of a stated ``parent`` for one entry that is not below it."""

    return (
        f"{OPTION} parent was stated, and {what} does not begin with 'proj' "
        "followed by a path. Read against the directory above the write root "
        f"it does not lie below the write root {root}, so the run is refused "
        f"instead of reading it inside the write root. {HOW_TO_TELL} If the "
        f"Spec was planned from the directory above, no {OPTION} value makes "
        "it readable in this write root as written: correct the entry (a bound "
        "is spelled with the leading 'proj', like the paths), or run from a "
        "root that holds every path of the Spec."
    )


# --------------------------------------------------------------------------
# A nested checkout: the statement decides; entries on disk only refuse
# --------------------------------------------------------------------------

# files_in_scope, creates_paths; then the decision without a statement, under
# write-root and under parent; then the votes (as written, without the name).
_NESTED_MATRIX = {
    # Planned from the directory above the checkout.
    "above": (
        ("proj/proj/source/a.py", "proj/rendering/b.py"), ("proj/.traces/new.md",),
        (KIND_UNSTATED, KIND_CONTRADICTED, PARENT_STATED), (0, 2),
    ),
    "above_new_files_only": (
        (), ("proj/.traces/new.md",),
        (KIND_NO_EVIDENCE, STATED, PARENT_STATED), (0, 0),
    ),
    "above_root_files": (
        ("proj/setup.py",), ("proj/NOTES.md",),
        (KIND_UNSTATED, KIND_CONTRADICTED, PARENT_STATED), (0, 1),
    ),
    # Planned from the checkout.
    "checkout": (
        ("proj/source/a.py", "rendering/b.py"), ("proj/newdir/new.py",),
        (KIND_UNSTATED, STATED, KIND_CONTRADICTED), (1, 0),
    ),
    "checkout_new_files_only": (
        (), ("proj/newdir/new.py",),
        (KIND_NO_EVIDENCE, STATED, PARENT_STATED), (0, 0),
    ),
    "checkout_plain_create": (
        ("proj/source/a.py", "rendering/b.py"), (".traces/new.md",),
        (KIND_UNSTATED, STATED, KIND_CONTRADICTED), (1, 0),
    ),
    # A create entry that happens to exist below the same-named directory.
    "coincidence": (
        (), ("proj/source/a.py", "proj/.traces/new.md"),
        (KIND_UNSTATED, STATED, KIND_CONTRADICTED), (1, 0),
    ),
    # One path of each spelling.
    "both_spellings": (
        ("proj/proj/source/a.py", "proj/source/a.py"), (),
        (KIND_CONTRADICTORY, KIND_CONTRADICTORY, KIND_CONTRADICTORY), (1, 1),
    ),
}
# The matrix holds no unstated cell that reads: evidence on disk selects no
# reading.
assert all(
    expected[0] in (KIND_UNSTATED, KIND_NO_EVIDENCE, KIND_CONTRADICTORY)
    for _files, _creates, expected, _votes in _NESTED_MATRIX.values()
)


@pytest.mark.parametrize("name", sorted(_NESTED_MATRIX))
def test_nested_checkout_reads_a_name_led_spec_only_on_a_statement(
    trees: dict[str, Path], name: str,
) -> None:
    files, creates, expected, votes = _NESTED_MATRIX[name]
    spec = _spec(files, creates)

    for statement, decision in zip(STATEMENTS, expected, strict=True):
        outcome = _read(trees["matrix_nested"], spec, statement)
        assert outcome.decision == decision, (name, statement)
        assert outcome.votes == votes, (name, statement)
        assert outcome.record["entry"] == DIRECTORY
        assert outcome.record["statement"] == statement


def test_nested_statement_gives_the_literal_paths_of_its_reading(
    trees: dict[str, Path],
) -> None:
    root = trees["matrix_nested"]
    above = _spec(*_NESTED_MATRIX["above"][:2])
    checkout = _spec(*_NESTED_MATRIX["checkout"][:2])

    parent = _read(root, above, "parent")
    written = _read(root, checkout, "write-root")

    # Under ``parent`` one leading name is dropped from every path, so a
    # doubled name is the authored spelling of a path below the directory.
    assert parent.scope() == ("proj/source/a.py", "rendering/b.py", ".traces/new.md")
    assert parent.record["paths_changed"] == 3
    assert written.scope() == ("proj/source/a.py", "rendering/b.py", "proj/newdir/new.py")
    assert written.record["paths_changed"] == 0
    assert written.table is not None and written.table.resolved_spec(checkout) is checkout


def test_nested_unstated_run_with_only_write_root_evidence_is_refused(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/only.py",), ("proj/newdir/new.py",))

    refused = _read(trees["nested"], spec)
    stated = _read(trees["nested"], spec, "write-root")

    assert refused.decision == KIND_UNSTATED
    # The record keeps the votes and names no reading.
    assert refused.record == {
        "statement": None, "name": NAME, "name_source": "write_root",
        "entry": DIRECTORY, "name_led_paths": 2, "name_only_paths": 0,
        "name_led_bounds": 0, "write_root_votes": 1, "parent_votes": 0,
        "refused": KIND_UNSTATED,
    }
    # The whole message: both values are named and neither is recommended.
    assert refused.message == (
        _two_meanings(
            trees["nested"], 2,
            "1 of them exist only as written ('proj/only.py'), 0 exist only "
            "without the leading 'proj'.",
        )
        + f" {HOW_TO_TELL} {OWN_WORK} {STATE_WHICH}"
    )
    # The stated run reads the paths as written.
    assert stated.decision == STATED
    assert stated.votes == (1, 0)
    assert stated.scope() == ("proj/only.py", "proj/newdir/new.py")


def test_nested_unstated_refusal_stands_before_the_bound_and_a_doubled_name(
    tmp_path: Path, trees: dict[str, Path],
) -> None:
    outside = _spec(("proj/only.py",), bounds=("src/",))
    doubled_root = _tree(tmp_path / "doubled", {
        **_FLAT, "proj/proj/x.py": "X = 1\n",
    }, commit=False)
    doubled = _spec(("proj/proj/x.py",))

    assert _decisions(trees["nested"], outside) == [
        KIND_UNSTATED, KIND_BOUND, KIND_CONTRADICTED,
    ]
    assert _decisions(doubled_root, doubled) == [
        KIND_UNSTATED, KIND_REPEATS, KIND_CONTRADICTED,
    ]
    for root, spec in ((trees["nested"], outside), (doubled_root, doubled)):
        refused = _read(root, spec)
        assert refused.votes == (1, 0)
        assert "read_against" not in refused.record and "basis" not in refused.record


def test_nested_spec_without_a_name_led_path_needs_no_statement(
    trees: dict[str, Path],
) -> None:
    plain = _spec(("src/a.py", "setup.py"))
    bound_only = _spec(("src/a.py",), bounds=("proj/src/", "src/"))

    assert _decisions(trees["nested"], plain) == [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT]
    # Only paths make a Spec ambiguous; a bound is read as written.
    unstated = _read(trees["nested"], bound_only)
    assert unstated.decision == NOT_NEEDED
    assert unstated.table is not None
    assert unstated.table.bounds == ("proj/src/", "src/")
    assert unstated.table.carries_name is True
    assert _read(trees["nested"], bound_only, "parent").decision == KIND_WITHOUT_EFFECT


def test_no_reading_is_recorded_as_confirmed_by_the_disk(
    trees: dict[str, Path],
) -> None:
    specs = [_spec(files, creates) for files, creates, *_ in _NESTED_MATRIX.values()]
    specs += [_spec(("proj/only.py",)), _spec(("proj/",), ("proj/only.py",))]
    for root in (trees["matrix_nested"], trees["nested"]):
        for spec in specs:
            for statement in STATEMENTS:
                outcome = _read(root, spec, statement)
                # A reading is stated or not needed; a refusal records none.
                assert outcome.record.get("basis") in (None, "stated", "not_needed")


def test_contradicted_statement_says_the_unstated_run_is_refused_too(
    trees: dict[str, Path],
) -> None:
    by_write_root_vote = _read(trees["nested"], _spec(("proj/only.py",)), "parent")
    by_parent_vote = _read(trees["nested"], _spec(("proj/setup.py",)), "write-root")

    for outcome, statement, votes in (
        (by_write_root_vote, "parent",
         "1 of them exist only as written ('proj/only.py'), 0 exist only "
         "without the leading 'proj'."),
        (by_parent_vote, "write-root",
         "0 of them exist only as written, 1 exist only without the leading "
         "'proj' ('proj/setup.py')."),
    ):
        assert outcome.decision == KIND_CONTRADICTED
        assert outcome.message == (
            f"{OPTION} {statement} was stated. "
            + _two_meanings(trees["nested"], 1, votes)
            + f" {HOW_TO_TELL} {OWN_WORK} A run without the statement is refused "
            "as well."
        )
        assert "read_against" not in outcome.record
    # The sentence is true: neither Spec is read without a statement.
    assert _read(trees["nested"], _spec(("proj/only.py",))).decision == KIND_UNSTATED
    assert _read(trees["nested"], _spec(("proj/setup.py",))).decision == KIND_UNSTATED


def test_contradictory_entries_refuse_every_statement(trees: dict[str, Path]) -> None:
    spec = _spec(("proj/only.py", "proj/setup.py"))

    for statement in STATEMENTS:
        outcome = _read(trees["nested"], spec, statement)
        assert outcome.decision == KIND_CONTRADICTORY
        assert outcome.votes == (1, 1)
        # The whole message, the same under every statement: it names both
        # entries and no way to change either.
        assert outcome.message == (
            _two_meanings(
                trees["nested"], 2,
                "1 of them exist only as written ('proj/only.py'), 1 exist only "
                "without the leading 'proj' ('proj/setup.py').",
            )
            + f" {HOW_TO_TELL} The entries contradict each other. {OWN_WORK} "
            f"No {OPTION} value lifts this refusal."
        )


def test_refusal_without_evidence_and_unstated_refusal_by_a_parent_vote(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]
    nothing = _read(root, _spec((), ("proj/newdir/new.py",)))
    parent_vote = _read(root, _spec(("proj/setup.py",)))

    assert nothing.decision == KIND_NO_EVIDENCE and nothing.votes == (0, 0)
    assert nothing.message == (
        _two_meanings(
            root, 1,
            "0 of them exist only as written, 0 exist only without the leading "
            "'proj'.",
        )
        + f" {HOW_TO_TELL} {STATE_WHICH}"
    )
    assert parent_vote.decision == KIND_UNSTATED and parent_vote.votes == (0, 1)
    assert parent_vote.message == (
        _two_meanings(
            root, 1,
            "0 of them exist only as written, 1 exist only without the leading "
            "'proj' ('proj/setup.py').",
        )
        + f" {HOW_TO_TELL} {STATE_WHICH}"
    )


def test_evidence_that_cannot_be_inspected_gives_no_table(
    trees: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = trees["nested"]
    reading = decide_scope_reading(root)
    evidence = os.fspath(root / NAME / "only.py")
    real_lstat = os.lstat

    def denied(path: Any, *args: Any, **kwargs: Any) -> os.stat_result:
        if os.fspath(path) == evidence:
            raise PermissionError(13, "denied")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(directive_table.os, "lstat", denied)

    # A path that would vote and cannot be looked at ends the step: it is not
    # counted as absent, and the Spec is neither read nor refused on the rest.
    for statement in STATEMENTS:
        with pytest.raises(ScopeReadingError) as caught:
            directive_table.build_directive_table(
                reading, _spec(("proj/only.py", "proj/setup.py")),
                read_against=statement,
            )
        assert str(caught.value).startswith(
            f"cannot inspect {evidence} for the directive evidence: "
        )
        assert not isinstance(caught.value, DirectiveRefusal)
    # A Spec without a path that begins with the name takes no such look.
    assert directive_table.build_directive_table(
        reading, _spec(("src/a.py",)),
    ).basis == "not_needed"


# --------------------------------------------------------------------------
# The votes
# --------------------------------------------------------------------------

def test_a_path_that_exists_under_both_spellings_gives_no_vote(
    trees: dict[str, Path],
) -> None:
    twin = _spec(("proj/src/a.py",))
    beside_evidence = _spec(("proj/src/a.py", "proj/only.py"))

    assert _decisions(trees["nested"], twin) == [KIND_NO_EVIDENCE, STATED, PARENT_STATED]
    assert _read(trees["nested"], twin).votes == (0, 0)
    assert _read(trees["nested"], twin, "parent").scope() == ("src/a.py",)
    assert _decisions(trees["nested"], beside_evidence) == [
        KIND_UNSTATED, STATED, KIND_CONTRADICTED,
    ]


def test_votes_follow_no_link(tmp_path: Path) -> None:
    root = _tree(tmp_path / "links", {
        **_FLAT, "proj/only.py": "ONLY = 1\n", "docs/guide.md": "guide\n",
    }, commit=False)
    (root / NAME / "link.py").symlink_to("only.py")
    (root / NAME / "linked").symlink_to(root / "docs", target_is_directory=True)

    # A link as the last component exists as a link.
    assert _read(root, _spec(("proj/link.py",))).votes == (1, 0)
    # A path through a link is not a path below this tree.
    through = _read(root, _spec(("proj/linked/guide.md",)))
    assert through.decision == KIND_NO_EVIDENCE and through.votes == (0, 0)


def test_every_kind_of_entry_votes_and_each_path_votes_once(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]
    later_tier = _spec((), ("proj/new.py",), later=(("b", ("proj/only.py",), ()),))
    region = _spec(("src/a.py",), regions=("proj/setup.py",))
    repeated = _spec(
        ("proj/only.py",), ("proj/only.py",), later=(("b", ("proj/only.py",), ()),),
    )
    two_tiers = _spec(("proj/only.py",), later=(("b", ("proj/setup.py",), ()),))

    assert _read(root, later_tier).votes == (1, 0)
    assert _read(root, region).decision == KIND_UNSTATED
    assert _read(root, region).votes == (0, 1)
    materialized = _read(root, _spec(("src/a.py",)), materializes={"a": ("proj/only.py",)})
    assert materialized.decision == KIND_UNSTATED and materialized.votes == (1, 0)
    assert _read(root, repeated).votes == (1, 0)
    assert _read(root, repeated).record["name_led_paths"] == 1
    assert _decisions(root, two_tiers) == [KIND_CONTRADICTORY] * 3


def test_flat_checkout_takes_no_votes(trees: dict[str, Path]) -> None:
    for statement in (None, "parent"):
        outcome = _read(trees["flat"], _spec(("proj/src/a.py", "proj/setup.py")), statement)
        assert outcome.table is not None
        assert outcome.votes == (0, 0)


# --------------------------------------------------------------------------
# A flat checkout: the name is dropped; a stated write-root drops none
# --------------------------------------------------------------------------

def test_flat_unstated_run_drops_one_leading_name_and_records_no_reading(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/x.py",), ("proj/newdir/new.py",), bounds=("proj/", "proj/pkg/"))

    outcome = _read(trees["matrix_flat"], spec)

    assert outcome.decision == DROPPED
    assert outcome.scope() == ("x.py", "newdir/new.py")
    assert outcome.table is not None
    # A bound that is the name itself is the write root and bounds nothing.
    assert outcome.table.bounds == ("**", "pkg/")
    assert outcome.table.read_against is None
    assert outcome.record == {
        "statement": None, "name": NAME, "name_source": "write_root",
        "entry": REDUNDANT, "name_led_paths": 2, "name_only_paths": 0,
        "name_led_bounds": 2, "write_root_votes": 0, "parent_votes": 0,
        "read_against": None, "basis": "redundant_name", "paths_changed": 2,
    }
    # The Spec itself is not changed; the table hands out a copy.
    assert spec.tier_dag.tiers[0].slices[0].directive.files_in_scope == ("proj/x.py",)
    assert outcome.table.resolved_spec(spec) is not spec


_FLAT_MATRIX = {
    "redundant_name": (("proj/x.py",), ("proj/newdir/new.py",), None,
                       (DROPPED, KIND_NO_DIRECTORY, PARENT_DROPPED)),
    "plain": (("x.py",), (), None, (NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT)),
    # The refusal of a stated write-root stands before the doubled name.
    "doubled_name": (("proj/proj/x.py",), (), None,
                     (KIND_REPEATS, KIND_NO_DIRECTORY, KIND_REPEATS)),
    "bound_below_the_name": (("pkg/mod.py",), (), ("proj/pkg/",),
                             (DROPPED, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT)),
    "bound_is_the_name": (("pkg/mod.py",), (), ("proj",),
                          (DROPPED, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT)),
    "plain_bound": (("pkg/mod.py",), (), ("pkg/",),
                    (NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT)),
}


@pytest.mark.parametrize("name", sorted(_FLAT_MATRIX))
@pytest.mark.parametrize("place", ["matrix_flat", "matrix_flat_worktree"])
def test_flat_checkout_matrix(trees: dict[str, Path], name: str, place: str) -> None:
    files, creates, bounds, expected = _FLAT_MATRIX[name]
    source = trees["matrix_flat"] if place.endswith("_worktree") else None

    assert _decisions(trees[place], _spec(files, creates, bounds=bounds), source=source) == (
        list(expected)
    )


def test_flat_parent_statement_drops_the_name_it_was_stated_for(
    trees: dict[str, Path],
) -> None:
    outcome = _read(
        trees["matrix_flat"], _spec(("proj/x.py",), ("proj/newdir/new.py",)), "parent",
    )

    assert outcome.decision == PARENT_DROPPED
    assert outcome.scope() == ("x.py", "newdir/new.py")
    assert outcome.record["read_against"] == "parent"


def test_stated_write_root_is_refused_where_a_name_would_be_dropped(
    trees: dict[str, Path],
) -> None:
    root = trees["matrix_flat"]

    outcome = _read(root, _spec(("proj/x.py",), ("proj/newdir/new.py",)), "write-root")

    assert outcome.decision == KIND_NO_DIRECTORY
    assert "read_against" not in outcome.record and "basis" not in outcome.record
    assert outcome.message == (
        f"{OPTION} write-root was stated. 2 Spec paths and 0 bounds begin with "
        f"'proj' ('proj/x.py', 'proj/newdir/new.py'), and {root} holds no "
        "directory named 'proj'. Without a statement a leading 'proj' is dropped "
        "from such an entry and it is read in the write root; under write-root "
        "no leading name is dropped, so the run is refused. Two cases. (i) The "
        "Spec means a directory 'proj' below the write root: that directory has "
        "to exist before the run. (ii) The leading 'proj' only repeats the name "
        "of the checkout: then write-root is not what this Spec needs here. The "
        "Spec means such a directory only if it was written to create or change "
        "files below a directory 'proj' inside this checkout. Once that "
        "directory exists, every Spec whose paths begin with 'proj' is read as "
        "in a checkout that holds one, and a worktree run needs a committed file "
        "below it. It also changes this checkout for every run, whatever the "
        "Spec: a request for a path that begins with 'proj' is then read as "
        "written and no longer without that name, and the attempts a Tier has "
        "retained for any Spec in this checkout are no longer current for a "
        "later run."
    )
    assert "parent" not in outcome.message.replace(str(root), "<root>")


# --------------------------------------------------------------------------
# A scope entry that is the name itself, where the root holds no such entry
# --------------------------------------------------------------------------

# Slice ``a`` and what stands beside the entry; then the three decisions.
_BARE_NAME_SHAPES: dict[str, tuple[Spec, tuple[str, str, str]]] = {
    "alone": (
        _spec(("proj/",)),
        (KIND_BARE_NAME, STATED, KIND_WITHOUT_EFFECT),
    ),
    "beside_a_root_path": (
        _spec(("proj/", "setup.py")),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "beside_a_root_directory": (
        _spec(("proj/", "src/")),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "beside_a_name_led_path": (
        _spec(("proj/", "proj/src/a.py")),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_NOT_BELOW),
    ),
    "as_a_create_entry_beside_a_root_file": (
        _spec(("setup.py",), ("proj",)),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "as_a_region_file_beside_a_root_file": (
        _spec(("setup.py",), regions=("proj",)),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "alone_with_a_bound_that_is_the_name": (
        _spec(("proj/",), bounds=("proj",)),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "alone_with_a_bound_below_the_name": (
        _spec(("proj/",), bounds=("proj/sub/",)),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
    "root_path_in_another_slice_of_the_tier": (
        _spec(("proj/",), beside=(("b", ("setup.py",), ()),)),
        (KIND_BARE_NAME, STATED, KIND_WITHOUT_EFFECT),
    ),
    "root_path_in_a_later_tier": (
        _spec(("proj/",), later=(("b", ("setup.py",), ()),)),
        (KIND_BARE_NAME, STATED, KIND_WITHOUT_EFFECT),
    ),
    # A doubled name and the Spec's own bound refuse later; the entry first.
    "beside_a_doubled_name": (
        _spec(("proj/", "proj/proj/x.py")),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_NOT_BELOW),
    ),
    "beside_a_root_path_outside_the_bound": (
        _spec(("proj/", "setup.py"), bounds=("src/",)),
        (KIND_BARE_NAME, KIND_NO_DIRECTORY, KIND_WITHOUT_EFFECT),
    ),
}
# The shapes a stated write-root reads: the entry stands alone in its slices.
_READ_UNDER_WRITE_ROOT = (
    "alone", "root_path_in_another_slice_of_the_tier", "root_path_in_a_later_tier",
)
assert {
    shape for shape, (_spec_of_shape, expected) in _BARE_NAME_SHAPES.items()
    if expected[1] == STATED
} == set(_READ_UNDER_WRITE_ROOT)
# What the refusal of such an entry says in no shape. It offers ``parent`` in
# no case. It does not say that a name is dropped from the entry: the entry
# is the name itself, and the run that would drop a name is the one refused.
# It says what a request below the entry does in each place, and not that
# nothing can be written there or that the run fails closed: in a worktree
# the same request is written. And it says of no value that it lifts the
# refusal: where the Spec was planned from the directory above, none does.
_NOT_SAID_IN_BARE_NAME_REFUSAL = (
    "parent", "is dropped from", "nothing can be written", "fails closed", "lifts",
)


@pytest.mark.parametrize("shape", sorted(_BARE_NAME_SHAPES))
@pytest.mark.parametrize("place", ["flat", "flat_worktree"])
def test_bare_name_entry_without_a_directory_matrix(
    trees: dict[str, Path], shape: str, place: str,
) -> None:
    spec, expected = _BARE_NAME_SHAPES[shape]
    source = trees["flat"] if place.endswith("_worktree") else None

    assert _decisions(trees[place], spec, source=source) == list(expected)


def test_bare_name_refusal_records_the_counts_and_no_reading(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/",))

    here = _read(trees["flat"], spec)
    there = _read(trees["flat_worktree"], spec, source=trees["flat"])

    assert here.record == {
        "statement": None, "name": NAME, "name_source": "write_root",
        "entry": REDUNDANT, "name_led_paths": 0, "name_only_paths": 1,
        "name_led_bounds": 0, "write_root_votes": 0, "parent_votes": 0,
        "refused": KIND_BARE_NAME,
    }
    assert there.record == {
        **here.record, "name_source": "worktree_source", "source_entry": REDUNDANT,
        "source_write_root_votes": 0, "source_parent_votes": 0,
        "source_decision": f"refused: {KIND_BARE_NAME}",
    }
    assert here.refusal is not None and here.refusal.carries_name is True
    assert there.refusal is not None and there.refusal.carries_name is True


def test_bare_name_refusal_names_write_root_only_where_it_reads_the_spec(
    trees: dict[str, Path],
) -> None:
    for shape, (spec, expected) in _BARE_NAME_SHAPES.items():
        message = _read(trees["flat"], spec).message.replace(str(trees["flat"]), "<root>")
        for phrase in _NOT_SAID_IN_BARE_NAME_REFUSAL:
            assert phrase not in message, (shape, phrase)
        if expected[1] == STATED:
            assert f"{OPTION} write-root is the statement for that" in message, shape
            assert NO_VALUE_READS not in message, shape
        else:
            assert "write-root" not in message, shape
            assert NO_VALUE_READS in message, shape


def test_bare_name_refusal_where_write_root_reads_the_spec(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]

    message = _read(root, _spec(("proj/",))).message

    assert message == (
        "1 scope entries of the Spec are 'proj' itself ('proj/'), in 1 slices "
        f"('a'), no {OPTION} value was stated, and {root} holds no entry named "
        "'proj'. Read as written such an entry is a directory 'proj' below the "
        "write root; read against the directory above the checkout it is the "
        "write root itself; no entry on disk can tell the two apart, so the run "
        "is refused instead of reading the entry without a statement. What a "
        "request below the entry does depends on the place: in the checkout "
        "'proj' itself a request for a path that begins with 'proj' is read "
        "without that name, with and without a statement, and is not written "
        "below the entry; in a worktree of that checkout the same request is "
        "written below a new directory 'proj'. Two cases. (i) The Spec was "
        "planned from this checkout and the entry means a directory 'proj' "
        f"below the write root: {OPTION} write-root is the statement for that. "
        "(ii) The Spec was planned from the directory above the checkout and "
        f"the entry only repeats the name of the checkout. Then no {OPTION} "
        "value makes the Spec readable in this write root as written: correct "
        "the entry (name the paths it means, each with the leading 'proj'), or "
        "run from a root that holds every path of the Spec. The entry means "
        "such a directory only if the Spec was written to create or change "
        "files below a directory 'proj' inside this checkout."
    )


def test_bare_name_refusal_where_no_value_reads_the_spec(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]
    led = _read(root, _BARE_NAME_SHAPES["beside_a_name_led_path"][0]).message
    beside = _read(root, _BARE_NAME_SHAPES["beside_a_root_path"][0]).message
    bound = _read(root, _BARE_NAME_SHAPES["alone_with_a_bound_that_is_the_name"][0]).message

    assert (
        f"{NO_VALUE_READS}: 1 Spec paths and 0 bounds begin with 'proj' "
        "('proj/src/a.py'). Two cases. (i) The Spec means a directory 'proj' "
        "below the write root: that directory has to exist before the run. (ii) "
        "The name only repeats the name of the checkout, as in a Spec planned "
        "from the directory above it: then the Spec is corrected (each such "
        "entry names the paths it means, each with the leading 'proj'). The Spec "
        "means such a directory only if it was written to create or change "
        "files below a directory 'proj' inside this checkout. Once that "
        "directory exists,"
    ) in led
    assert (
        f"{NO_VALUE_READS}: slice 'a' holds such an entry beside 1 entries that "
        f"do not begin with 'proj' ('setup.py'), {CAN_LIE_INSIDE}. Two cases."
    ) in beside
    assert f"{NO_VALUE_READS}: 0 Spec paths and 1 bounds begin with 'proj' ('proj')." in bound
    for message in (led, beside, bound):
        assert message.endswith(
            "the attempts a Tier has retained for any Spec in this checkout are "
            "no longer current for a later run."
        )
    # Both grounds, and the entry beside another entry in two slices.
    both = _read(root, _spec(("proj/", "setup.py", "proj/src/a.py"))).message
    two_slices = _read(
        root, _spec(("proj/", "setup.py"), beside=(("b", ("proj/", "src/a.py"), ()),)),
    ).message
    assert (
        f"{NO_VALUE_READS}: 1 Spec paths and 0 bounds begin with 'proj' "
        "('proj/src/a.py'); slice 'a' holds such an entry beside 1 entries that "
        f"do not begin with 'proj' ('setup.py'), {CAN_LIE_INSIDE}. Two cases."
    ) in both
    assert (
        f"{NO_VALUE_READS}: slice 'a' holds such an entry beside 1 entries that "
        f"do not begin with 'proj' ('setup.py'), {CAN_LIE_INSIDE}; slice 'b' "
        "holds such an entry beside 1 entries that do not begin with 'proj' "
        f"('src/a.py'), {CAN_LIE_INSIDE}. Two cases."
    ) in two_slices


def test_bare_name_refusal_counts_the_entries_and_the_slices_it_lists(
    trees: dict[str, Path],
) -> None:
    in_two_slices = _spec(("proj/",), later=(("b", ("proj/",), ()),))
    two_spellings = _spec(("proj/", "proj"))

    first = _read(trees["flat"], in_two_slices)
    second = _read(trees["flat"], two_spellings)

    assert first.message.startswith(
        "1 scope entries of the Spec are 'proj' itself ('proj/'), in 2 slices "
        f"('a', 'b'), no {OPTION} value was stated, and "
    )
    assert first.record["name_only_paths"] == 1
    assert second.message.startswith(
        "2 scope entries of the Spec are 'proj' itself ('proj/', 'proj'), in 1 "
        f"slices ('a'), no {OPTION} value was stated, and "
    )
    assert second.record["name_only_paths"] == 2
    # A slice that does not hold the entry is not counted.
    apart = _read(trees["flat"], _spec(("proj/",), beside=(("b", ("setup.py",), ()),)))
    assert apart.message.startswith(
        "1 scope entries of the Spec are 'proj' itself ('proj/'), in 1 slices "
        f"('a'), no {OPTION} value was stated, and "
    )


def test_a_message_names_six_entries_or_slices_and_counts_the_rest(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]

    def led(count: int) -> str:
        paths = tuple(f"proj/f{index}.py" for index in range(1, count + 1))
        return _read(root, _spec(paths), "write-root").message

    six = (
        "'proj/f1.py', 'proj/f2.py', 'proj/f3.py', 'proj/f4.py', 'proj/f5.py', "
        "'proj/f6.py'"
    )
    assert led(6).startswith(
        f"{OPTION} write-root was stated. 6 Spec paths and 0 bounds begin with "
        f"'proj' ({six}), and {root} holds no directory named 'proj'."
    )
    assert led(7).startswith(
        f"{OPTION} write-root was stated. 7 Spec paths and 0 bounds begin with "
        f"'proj' ({six} and 1 more), and {root} holds no directory named 'proj'."
    )
    assert led(9).startswith(
        f"{OPTION} write-root was stated. 9 Spec paths and 0 bounds begin with "
        f"'proj' ({six} and 3 more), and {root} holds no directory named 'proj'."
    )

    # Seven slices that each hold the entry beside a path of their own.
    seven = _spec(
        ("proj/", "fa.py"),
        beside=tuple((slice_id, ("proj/", f"f{slice_id}.py"), ()) for slice_id in "bcdefg"),
    )
    unstated = _read(root, seven).message
    stated = _read(root, seven, "write-root").message
    assert unstated.startswith(
        "1 scope entries of the Spec are 'proj' itself ('proj/'), in 7 slices "
        f"('a', 'b', 'c', 'd', 'e', 'f' and 1 more), no {OPTION} value was stated, "
    )
    assert "slice 'f' holds such an entry beside 1 entries" in unstated
    assert "slice 'g' holds" not in unstated
    assert (
        "Slice 'f' holds a scope entry that is 'proj' itself ('proj/') beside 1 "
        f"entries that do not begin with 'proj' ('ff.py'). {root} holds no entry "
        "named 'proj'."
    ) in stated
    assert "Slice 'g' holds" not in stated


def test_one_bare_name_refusal_for_both_places_of_a_worktree_run(
    trees: dict[str, Path],
) -> None:
    for shape, (spec, _expected) in _BARE_NAME_SHAPES.items():
        here = _read(trees["flat"], spec)
        there = _read(trees["flat_worktree"], spec, source=trees["flat"])
        assert there.decision == KIND_BARE_NAME, shape
        assert DECIDE_DIFFERENTLY not in there.message, shape
        # The text is the checkout's, the root apart.
        assert there.message.replace(str(trees["flat_worktree"]), "<root>") == (
            here.message.replace(str(trees["flat"]), "<root>")
        ), shape


def test_bare_name_entry_reads_on_where_the_root_holds_a_regular_file(
    trees: dict[str, Path],
) -> None:
    expected = {
        "alone": [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT],
        "beside_a_root_path": [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT],
        "beside_a_name_led_path": [DROPPED, KIND_NO_DIRECTORY, KIND_NOT_BELOW],
        "as_a_create_entry_beside_a_root_file": [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT],
        "alone_with_a_bound_that_is_the_name": [DROPPED, KIND_NO_DIRECTORY,
                                                KIND_WITHOUT_EFFECT],
        "root_path_in_a_later_tier": [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT],
    }
    for shape, decisions in expected.items():
        spec = _BARE_NAME_SHAPES[shape][0]
        assert _decisions(trees["name_file"], spec) == decisions, shape
        assert _decisions(
            trees["name_file_worktree"], spec, source=trees["name_file"],
        ) == decisions, shape
    alone = _read(trees["name_file"], _BARE_NAME_SHAPES["alone"][0])
    assert alone.scope() == ("proj/",)
    assert alone.record["entry"] == REDUNDANT


def test_regular_file_of_the_name_is_named_in_the_write_root_refusal(
    trees: dict[str, Path],
) -> None:
    root = trees["name_file"]

    message = _read(root, _spec(("proj/", "proj/src/a.py")), "write-root").message

    assert (
        f"1 Spec paths and 0 bounds begin with 'proj' ('proj/src/a.py'), and {root} "
        "holds a regular file named 'proj', not a directory. Without a statement "
        "a leading 'proj' is dropped from such an entry"
    ) in message
    # Where the root holds an entry of the name, nothing is said of the
    # unstated refusal: that run reads on.
    assert "refused as well" not in message and NO_VALUE_READS not in message


def test_bare_name_entry_in_a_nested_checkout_follows_the_votes(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]
    alone = _spec(("proj/",))
    beside_root_path = _spec(("proj/", "setup.py"))
    beside_written = _spec(("proj/", "proj/only.py"))
    beside_stripped = _spec(("proj/", "proj/setup.py"))

    assert _decisions(root, alone) == [KIND_NO_EVIDENCE, STATED, KIND_WITHOUT_EFFECT]
    # In a nested checkout the entry is read as written beside a root path.
    assert _decisions(root, beside_root_path) == [
        KIND_NO_EVIDENCE, STATED, KIND_WITHOUT_EFFECT,
    ]
    assert _decisions(root, beside_written) == [KIND_UNSTATED, STATED, KIND_CONTRADICTED]
    assert _decisions(root, beside_stripped) == [
        KIND_UNSTATED, KIND_CONTRADICTED, KIND_NOT_BELOW,
    ]
    assert _read(root, alone, "write-root").scope() == ("proj/",)

    # The three messages, whole. ``parent`` is named and not offered, and the
    # sentence on a contradicted write-root is said only where a path votes
    # for the parent.
    no_votes = (
        "0 of them exist only as written, 0 exist only without the leading "
        "'proj'."
    )
    assert _read(root, alone).message == (
        f"{_two_meanings(root, 0, no_votes)} {ENTRY_HAS_TWO_MEANINGS} "
        f"{HOW_TO_TELL} {WRITE_ROOT_FOR_THE_ENTRY} {PARENT_NEVER_FOR_THE_ENTRY}"
    )
    assert _read(root, beside_written).message == (
        _two_meanings(
            root, 1,
            "1 of them exist only as written ('proj/only.py'), 0 exist only "
            "without the leading 'proj'.",
        )
        + f" {ENTRY_HAS_TWO_MEANINGS} {HOW_TO_TELL} {OWN_WORK} "
        f"{WRITE_ROOT_FOR_THE_ENTRY} {PARENT_NEVER_FOR_THE_ENTRY}"
    )
    assert _read(root, beside_stripped).message == (
        _two_meanings(
            root, 1,
            "0 of them exist only as written, 1 exist only without the leading "
            "'proj' ('proj/setup.py').",
        )
        + f" {ENTRY_HAS_TWO_MEANINGS} {HOW_TO_TELL} {WRITE_ROOT_FOR_THE_ENTRY} "
        "Here 1 paths exist only without the leading 'proj', which contradicts "
        f"write-root. {PARENT_NEVER_FOR_THE_ENTRY}"
    )
    # Under ``parent`` the entry does not lie below the write root; the
    # refusal says nothing of a checkout without the directory.
    assert _read(root, beside_stripped, "parent").message == _not_below(
        root, "the path of slice 'a' 'proj/'",
    )


# --------------------------------------------------------------------------
# A stated write-root and a slice that holds the entry beside another entry
# --------------------------------------------------------------------------

def test_write_root_refusal_for_an_entry_beside_a_root_path(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]

    outcome = _read(root, _spec(("proj/", "src/a.py")), "write-root")

    assert outcome.decision == KIND_NO_DIRECTORY
    assert outcome.record == {
        "statement": "write-root", "name": NAME, "name_source": "write_root",
        "entry": REDUNDANT, "name_led_paths": 0, "name_only_paths": 1,
        "name_led_bounds": 0, "write_root_votes": 0, "parent_votes": 0,
        "refused": KIND_NO_DIRECTORY,
    }
    assert outcome.message == (
        f"{OPTION} write-root was stated. Slice 'a' holds a scope entry that is "
        "'proj' itself ('proj/') beside 1 entries that do not begin with 'proj' "
        f"('src/a.py'). {root} holds no entry named 'proj'. In the checkout "
        "'proj' itself a request for a path below such an entry is read without "
        "the leading 'proj', with and without a statement, and can then lie "
        "inside another entry of the same slice and be written at the root of "
        "that checkout, so the run is refused instead of reading the entry as "
        "written. A run without a statement is refused as well, because 1 scope "
        "entries of the Spec are 'proj' itself ('proj/'). Two cases. (i) The "
        "Spec means a directory 'proj' below the write root: that directory has "
        "to exist before the run. (ii) The name 'proj' only repeats the name of "
        "the checkout: then write-root is not what this Spec needs here. The "
        "Spec means such a directory only if it was written to create or change "
        "files below a directory 'proj' inside this checkout. Once that "
        "directory exists, every Spec whose paths begin with 'proj' is read as "
        "in a checkout that holds one, and a worktree run needs a committed file "
        "below it. It also changes this checkout for every run, whatever the "
        "Spec: a request for a path that begins with 'proj' is then read as "
        "written and no longer without that name, and the attempts a Tier has "
        "retained for any Spec in this checkout are no longer current for a "
        f"later run. {NO_VALUE_READS}; in case (ii) the way on is a corrected Spec."
    )
    for banned in ("parent", "no leading name is dropped", "holds no directory named",
                   "0 Spec paths"):
        assert banned not in outcome.message.replace(str(root), "<root>"), banned
    # The sentence is true: the same Spec is refused without a statement.
    assert _read(root, _spec(("proj/", "src/a.py"))).decision == KIND_BARE_NAME


def test_write_root_refusal_for_an_entry_beside_both_kinds_of_path(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]

    message = _read(root, _spec(("proj/", "setup.py", "proj/src/a.py")), "write-root").message

    assert message.startswith(
        f"{OPTION} write-root was stated. 1 Spec paths and 0 bounds begin with "
        "'proj' ('proj/src/a.py'). Slice 'a' holds a scope entry that is 'proj' "
        "itself ('proj/') beside 1 entries that do not begin with 'proj' "
        f"('setup.py'). {root} holds no entry named 'proj'."
    )
    assert (
        "be written at the root of that checkout. A run without a statement is "
        "refused as well, because 1 scope entries of the Spec are 'proj' itself "
        "('proj/'); under write-root no leading name is dropped, so the run is "
        "refused. Two cases."
    ) in message
    assert "(ii) The leading 'proj' only repeats the name of the checkout" in message
    assert message.endswith(
        f"{NO_VALUE_READS}; in case (ii) the way on is a corrected Spec."
    )
    assert message.count("holds no entry named 'proj'") == 1
    for banned in ("parent", "holds no directory named",
                   "instead of reading the entry as written"):
        assert banned not in message.replace(str(root), "<root>"), banned


def test_write_root_refusal_names_each_slice_and_each_entry_once(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]
    two_slices = _spec(("proj/", "setup.py"), beside=(("b", ("proj/", "src/a.py"), ()),))
    named_twice = _spec(("proj/", "setup.py"), ("proj/", "setup.py"))
    bound_only = _spec(("proj/", "setup.py"), bounds=("proj/sub/", "setup.py"))

    assert _read(root, two_slices, "write-root").message.startswith(
        f"{OPTION} write-root was stated. Slice 'a' holds a scope entry that is "
        "'proj' itself ('proj/') beside 1 entries that do not begin with 'proj' "
        "('setup.py'). Slice 'b' holds a scope entry that is 'proj' itself "
        "('proj/') beside 1 entries that do not begin with 'proj' ('src/a.py'). "
        f"{root} holds no entry named 'proj'. In the checkout 'proj' itself "
    )
    assert _read(root, named_twice, "write-root").message.startswith(
        f"{OPTION} write-root was stated. Slice 'a' holds a scope entry that is "
        "'proj' itself ('proj/') beside 1 entries that do not begin with 'proj' "
        f"('setup.py'). {root} holds no entry named 'proj'. "
    )
    # A bound alone carries the name: it is counted before the slice.
    assert _read(root, bound_only, "write-root").message.startswith(
        f"{OPTION} write-root was stated. 0 Spec paths and 1 bounds begin with "
        "'proj' ('proj/sub/'). Slice 'a' holds a scope entry that is 'proj' "
        "itself ('proj/') beside 1 entries that do not begin with 'proj' "
        f"('setup.py'). {root} holds no entry named 'proj'. "
    )


def test_write_root_refusal_beside_a_name_led_path_says_the_unstated_run_is_refused(
    trees: dict[str, Path],
) -> None:
    message = _read(trees["flat"], _spec(("proj/", "proj/src/a.py")), "write-root").message

    assert (
        f"and {trees['flat']} holds no directory named 'proj'. A run without a "
        "statement is refused as well, because 1 scope entries of the Spec are "
        "'proj' itself ('proj/'); under write-root no leading name is dropped, "
        "so the run is refused. Two cases."
    ) in message
    assert "is dropped from such an entry" not in message
    assert message.endswith(
        f"{NO_VALUE_READS}; in case (ii) the way on is a corrected Spec."
    )


def test_write_root_reads_the_entry_where_it_stands_alone_in_its_slices(
    trees: dict[str, Path],
) -> None:
    alone = _read(trees["flat"], _spec(("proj/",)), "write-root")
    apart = _read(
        trees["flat"], _spec(("proj/",), beside=(("b", ("setup.py",), ()),)), "write-root",
    )
    all_led = _read(trees["flat"], _spec(("proj/", "proj/src/a.py")), "write-root")

    assert alone.decision == STATED and alone.scope() == ("proj/",)
    assert alone.table is not None and alone.table.carries_name is True
    assert apart.decision == STATED
    assert apart.scope("a") == ("proj/",) and apart.scope("b") == ("setup.py",)
    # Beside a path that begins with the name only, the refusal has the kind a
    # name-led path has without the entry, and its message names no slice.
    assert all_led.decision == KIND_NO_DIRECTORY
    assert "Slice 'a' holds a scope entry" not in all_led.message


def test_write_root_refusal_stands_before_the_specs_own_bound(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/", "setup.py"), bounds=("src/",))
    alone_outside = _spec(("proj/",), bounds=("src/",))

    assert _read(trees["flat"], spec, "write-root").decision == KIND_NO_DIRECTORY
    # Alone in its slice the entry is read, and the bound then refuses.
    assert _decisions(trees["flat"], alone_outside) == [
        KIND_BARE_NAME, KIND_BOUND, KIND_WITHOUT_EFFECT,
    ]


# --------------------------------------------------------------------------
# Refusals that rest on the text or on one entry
# --------------------------------------------------------------------------

@pytest.mark.parametrize(("path", "message"), [
    ("", "scope path is not one safe string: ''"),
    ("a\0b", "scope path is not one safe string: 'a\\x00b'"),
    ("/abs/x.py", "scope path '/abs/x.py' is absolute or leaves the write root"),
    ("../up.py", "scope path '../up.py' is absolute or leaves the write root"),
    ("src/../../up.py",
     "scope path 'src/../../up.py' is absolute or leaves the write root"),
], ids=["empty", "nul_byte", "absolute", "leaves_the_root", "leaves_it_below_a_directory"])
def test_an_unsafe_path_is_refused_before_anything_is_decided(
    trees: dict[str, Path], path: str, message: str,
) -> None:
    for spec in (_spec(("src/a.py", path)), _spec(("src/a.py",), (path,)),
                 _spec(("src/a.py",), regions=(path,))):
        for place, source in (("nested", None), ("flat_worktree", trees["flat"])):
            outcome = _read(trees[place], spec, "parent", source=source)
            assert outcome.decision == KIND_INVALID
            assert outcome.message == message
            assert outcome.refusal is not None and outcome.refusal.carries_name is None
            assert "name_led_paths" not in outcome.record
    bound = _read(trees["flat"], _spec(("src/a.py",), bounds=("../up/",)))
    assert bound.decision == KIND_INVALID


def test_what_is_not_one_string_is_refused_as_unsafe(trees: dict[str, Path]) -> None:
    outcome = _read(trees["flat"], _spec(("src/a.py", 7)))  # type: ignore[arg-type]

    assert outcome.decision == KIND_INVALID
    assert outcome.message == "scope path is not one safe string: 7"


@pytest.mark.skipif(os.name == "nt", reason="a backslash is a separator on Windows")
def test_a_backslash_is_a_byte_of_a_file_name(trees: dict[str, Path]) -> None:
    outcome = _read(trees["flat"], _spec(("proj\\x.py",)))

    # The path does not begin with the name, so nothing is dropped from it.
    assert outcome.decision == NOT_NEEDED
    assert outcome.scope() == ("proj\\x.py",)
    assert outcome.table is not None and outcome.table.carries_name is False


def test_a_link_named_like_the_checkout_blocks_a_path_through_it(tmp_path: Path) -> None:
    root = _tree(tmp_path / "link", _FLAT, commit=False)
    (root / NAME).symlink_to("src", target_is_directory=True)
    through = _spec(("proj/a.py",))

    for statement in STATEMENTS:
        outcome = _read(root, through, statement)
        assert outcome.decision == KIND_BLOCKED
        assert outcome.message == (
            f"1 Spec paths begin with 'proj' ('proj/a.py'), and 'proj' below {root} "
            "is not a directory. No such path can be written."
        )
        assert outcome.record["entry"] == BLOCKED
    assert _decisions(root, _spec(("src/a.py",))) == [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT]
    # The entry itself is not a path through it.
    assert _decisions(root, _spec(("proj",))) == [NOT_NEEDED, STATED, KIND_WITHOUT_EFFECT]


def test_parent_statement_that_changes_no_path_is_refused(trees: dict[str, Path]) -> None:
    here = _read(trees["flat"], _spec(("src/a.py",), bounds=("proj/src/",)), "parent")
    there = _read(
        trees["flat_worktree"], _spec(("src/a.py",)), "parent", source=trees["flat"],
    )

    assert here.decision == there.decision == KIND_WITHOUT_EFFECT
    assert here.message == (
        f"{OPTION} parent was stated, and no path of the Spec begins with the name "
        "of this write root, 'proj'. The statement would change no path, so the "
        "run is refused instead of reading the paths against the write root."
    )
    assert "the name of the checkout this worktree was made from, 'proj'" in there.message


def test_under_parent_every_path_and_bound_begins_with_the_name(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]
    beside = _read(root, _spec(("proj/setup.py", "docs/x.md")), "parent")
    bound = _read(root, _spec(("proj/setup.py",), bounds=("proj/", "docs/")), "parent")
    bare_bound = _read(root, _spec(("proj/setup.py",), bounds=("proj",)), "parent")

    assert beside.decision == bound.decision == KIND_NOT_BELOW
    assert beside.message == (
        f"{OPTION} parent was stated, and the path of slice 'a' 'docs/x.md' does "
        "not begin with 'proj' followed by a path. Read against the directory "
        f"above the write root it does not lie below the write root {root}, so "
        "the run is refused instead of reading it inside the write root. An "
        "entry that exists is not proof: file names can coincide. The Spec's "
        "paths are relative to the directory above the checkout 'proj' only if "
        "the Spec was planned from that directory; otherwise they are relative "
        "to the write root. If the Spec was planned from the directory above, "
        f"no {OPTION} value makes it readable in this write root as written: "
        "correct the entry (a bound is spelled with the leading 'proj', like "
        "the paths), or run from a root that holds every path of the Spec."
    )
    assert beside.message == _not_below(root, "the path of slice 'a' 'docs/x.md'")
    assert bound.message == _not_below(
        root, "the bound of the Spec's no_edits_outside 'docs/'",
    )
    # A refusal raised after a reading was taken says which reading it was.
    assert beside.record["read_against"] == "parent" and beside.record["basis"] == "stated"
    # A bound that is the name itself is the write root and bounds nothing.
    assert bare_bound.decision == PARENT_STATED
    assert bare_bound.table is not None and bare_bound.table.bounds == ("**",)
    assert bare_bound.scope() == ("setup.py",)


def test_not_below_refusal_says_what_the_unstated_run_does_in_a_flat_checkout(
    trees: dict[str, Path],
) -> None:
    plain = _read(trees["flat"], _spec(("proj/src/a.py", "docs/x.md")), "parent")
    with_entry = _read(trees["flat"], _spec(("proj/", "proj/src/a.py")), "parent")
    nested = _read(trees["nested"], _spec(("proj/setup.py", "docs/x.md")), "parent")

    assert plain.message == (
        _not_below(trees["flat"], "the path of slice 'a' 'docs/x.md'")
        + " This checkout holds no directory 'proj': a run without a statement is "
        "not refused here and reads the entry inside the write root, which is "
        "not the file a Spec planned from the directory above means."
    )
    assert with_entry.message == (
        _not_below(trees["flat"], "the path of slice 'a' 'proj/'")
        + " This checkout holds no entry named 'proj', and 1 scope entries of the "
        "Spec are 'proj' itself: a run without a statement is refused here as "
        "well, and the way on is to correct the entry."
    )
    assert nested.message == _not_below(
        trees["nested"], "the path of slice 'a' 'docs/x.md'",
    )
    # Both sentences are true of the unstated run.
    assert _read(trees["flat"], _spec(("proj/src/a.py", "docs/x.md"))).decision == DROPPED
    assert _read(trees["flat"], _spec(("proj/", "proj/src/a.py"))).decision == KIND_BARE_NAME


def test_a_literal_path_that_begins_with_the_name_twice_is_refused(
    trees: dict[str, Path],
) -> None:
    flat = _read(trees["flat"], _spec(("proj/proj/x.py",)))
    written = _read(trees["nested"], _spec(("proj/proj/x.py",)), "write-root")
    dropped_once = _read(trees["nested"], _spec(("proj/proj/x.py",)), "parent")
    still_doubled = _read(trees["nested"], _spec(("proj/proj/proj/x.py",)), "parent")

    assert flat.decision == written.decision == KIND_REPEATS
    assert flat.message == (
        f"scope path 'proj/proj/x.py' repeats the name 'proj'; write it relative "
        f"to {trees['flat']}"
    )
    assert written.message == (
        f"scope path 'proj/proj/x.py' repeats the name 'proj'; write it relative "
        f"to {trees['nested']}"
    )
    # Read against the parent, a doubled name is the spelling of one name.
    assert dropped_once.decision == PARENT_STATED
    assert dropped_once.scope() == ("proj/x.py",)
    assert still_doubled.decision == KIND_REPEATS
    assert still_doubled.message == (
        "scope path 'proj/proj/proj/x.py' still begins with 'proj' twice after "
        "one leading name is dropped; no reader can carry it"
    )
    # What is left after the drop is refused only where it begins with the
    # name twice: the name alone and a name further down are read.
    assert _read(trees["nested"], _spec(("proj/proj",)), "parent").scope() == ("proj",)
    assert _read(
        trees["nested"], _spec(("proj/src/proj/x.py",)), "parent",
    ).scope() == ("src/proj/x.py",)


def test_a_literal_scope_path_outside_the_literal_bound_is_refused(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]
    outside = _read(root, _spec(("proj/only.py",), bounds=("only.py", "src/")), "write-root")
    inside = _read(root, _spec(("proj/only.py",), bounds=("proj/",)), "write-root")
    nothing_bounds = [
        _read(root, _spec(("proj/only.py",), bounds=bounds), "write-root").decision
        for bounds in (("**",), ("",), ("docs/", "**"), ())
    ]

    assert outside.decision == KIND_BOUND
    assert outside.message == (
        "slice 'a': 1 scope paths lie outside the Spec's no_edits_outside as this "
        "run reads both ('proj/only.py'). The bounds were read: 'only.py' as "
        "'only.py', 'src/' as 'src/' ('**' bounds nothing; a bound that is the "
        "bare name 'proj' is the directory of that name when the paths are read "
        "as written, and the write root itself when a leading name is dropped)."
    )
    assert outside.record["read_against"] == "write-root"
    assert inside.decision == STATED
    assert inside.table is not None and inside.table.bounds == ("proj/",)
    assert nothing_bounds == [STATED] * 4
    # A create entry is bounded like a scope file; under ``parent`` the bound
    # is read with the paths.
    assert _read(
        root, _spec((), ("proj/new.py",), bounds=("src/",)), "write-root",
    ).decision == KIND_BOUND
    assert _read(
        root, _spec(("proj/setup.py",), bounds=("proj/setup.py",)), "parent",
    ).decision == PARENT_STATED


def test_a_bound_is_a_directory_and_not_a_prefix_of_a_name(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]

    assert _read(root, _spec(("srcx/a.py",), bounds=("src",))).decision == KIND_BOUND
    assert _read(root, _spec(("src.py",), bounds=("src/",))).decision == KIND_BOUND
    # The bound itself and what lies below it are inside.
    for path in ("src/a.py", "src", "src/"):
        for bound in ("src", "src/"):
            assert _read(root, _spec((path,), bounds=(bound,))).decision == NOT_NEEDED


def test_the_bound_is_held_against_scope_files_and_create_entries_only(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]
    spec = _spec(("src/a.py",), bounds=("src/",), regions=("setup.py",))

    outcome = _read(root, spec, materializes={"a": ("out.md",)})

    # A region file and a materialized output are read with the paths and
    # are not held against the bound: like the ``blast_radius`` rule of the
    # validation gate, the bound takes the scope files and the create
    # entries of a slice.
    assert outcome.decision == NOT_NEEDED
    table = outcome.table
    assert table is not None and table.bounds == ("src/",)
    assert table.scope("a") == ("src/a.py",)
    assert [region.file for region in table.directive("a").mutation_regions] == ["setup.py"]
    assert table.slice("a").materializes_paths == ("out.md",)
    # Either path as a create entry lies outside the bound.
    for path in ("setup.py", "out.md"):
        refused = _read(root, _spec(("src/a.py",), (path,), bounds=("src/",)))
        assert refused.decision == KIND_BOUND
        assert refused.message.startswith(
            "slice 'a': 1 scope paths lie outside the Spec's no_edits_outside "
            f"as this run reads both ({path!r})."
        )


def test_what_bounds_nothing_is_read_under_a_stated_parent(
    trees: dict[str, Path],
) -> None:
    root = trees["nested"]

    for bounds, literal in (
        (("**",), ("**",)),
        (("",), ("**",)),
        # A bound that is the name itself is the write root once the name is
        # dropped.
        (("proj/", "**"), ("**", "**")),
    ):
        outcome = _read(root, _spec(("proj/setup.py",), bounds=bounds), "parent")
        assert outcome.decision == PARENT_STATED, bounds
        assert outcome.table is not None and outcome.table.bounds == literal
        assert outcome.scope() == ("setup.py",)
    # Any other bound has to begin with the name, like a path.
    assert _read(
        root, _spec(("proj/setup.py",), bounds=("docs/", "**")), "parent",
    ).decision == KIND_NOT_BELOW


def test_the_bound_is_the_first_constraint_that_holds_one(
    trees: dict[str, Path],
) -> None:
    root = trees["flat"]
    spec = _spec(("setup.py",))
    budget = BudgetConstraint(usd_limit=1.0)

    def constrained(*constraints: Any) -> Spec:
        return dataclasses.replace(spec, constraints=tuple(constraints))

    # A constraint of another kind is passed over; of two bounds the first
    # is the one read.
    refused = _read(root, constrained(
        budget,
        BlastRadiusConstraint(no_edits_outside=("src/",)),
        BlastRadiusConstraint(no_edits_outside=("**",)),
    ))
    assert refused.decision == KIND_BOUND
    read = _read(root, constrained(
        budget,
        BlastRadiusConstraint(no_edits_outside=("**",)),
        BlastRadiusConstraint(no_edits_outside=("src/",)),
    ))
    assert read.decision == NOT_NEEDED
    assert read.table is not None and read.table.bounds == ("**",)
    # Without such a constraint the table holds no bounds at all.
    for unbounded in (spec, constrained(budget)):
        table = _read(root, unbounded).table
        assert table is not None and table.bounds is None


def test_the_step_refuses_a_statement_that_is_no_value_and_a_spec_without_a_dag(
    trees: dict[str, Path],
) -> None:
    reading = decide_scope_reading(trees["flat"])

    with pytest.raises(
        ValueError,
        match=f"^{OPTION} must be one of \\('write-root', 'parent'\\) or absent$",
    ):
        directive_table.build_directive_table(
            reading, _spec(("src/a.py",)), read_against="write_root",
        )
    with pytest.raises(ValueError, match="^a directive table needs a tier_dag$"):
        directive_table.build_directive_table(
            reading, dataclasses.replace(_spec(("src/a.py",)), tier_dag=None),
        )


# --------------------------------------------------------------------------
# A worktree run: the decision is taken in both places
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_NESTED_MATRIX))
def test_tracked_worktree_decides_as_its_checkout_and_records_both_places(
    trees: dict[str, Path], name: str,
) -> None:
    files, creates, expected, votes = _NESTED_MATRIX[name]
    spec = _spec(files, creates)

    for statement, decision in zip(STATEMENTS, expected, strict=True):
        here = _read(trees["matrix_nested"], spec, statement)
        there = _read(
            trees["matrix_nested_worktree"], spec, statement, source=trees["matrix_nested"],
        )
        assert there.decision == decision, (name, statement)
        assert there.record == {
            **here.record,
            "name_source": "worktree_source", "source_entry": DIRECTORY,
            "source_write_root_votes": votes[0], "source_parent_votes": votes[1],
            "source_decision": (
                f"refused: {decision}" if here.refusal is not None else decision
            ),
        }, (name, statement)
        if there.refusal is not None:
            assert DECIDE_DIFFERENTLY not in there.message
        else:
            assert there.table is not None and here.table is not None
            assert there.table.name == NAME
            assert there.table.reading.name == "crucible_worktree_ab12cd34"
            assert [there.scope(item.slice_id) for item in there.table.slices] == [
                here.scope(item.slice_id) for item in here.table.slices
            ]


def _made_in_the_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """A flat checkout and a worktree of it that gained a directory ``proj``."""

    root = _tree(tmp_path / "made", _FLAT)
    worktree = _worktree(root)
    _put(worktree, "proj/new.py")
    return worktree, root


def test_entry_of_two_kinds_refuses_a_spec_that_carries_the_name(tmp_path: Path) -> None:
    worktree, root = _made_in_the_worktree(tmp_path)
    led = _spec((), ("proj/newdir/new.py", "proj/other.py"))

    for statement in STATEMENTS:
        outcome = _read(worktree, led, statement, source=root)
        assert outcome.decision == KIND_WORKTREE, statement
        assert outcome.record["entry"] == DIRECTORY
        assert outcome.record["source_entry"] == REDUNDANT
        assert outcome.refusal is not None and outcome.refusal.carries_name is True
    unstated = _read(worktree, led, source=root)
    assert unstated.message == (
        "2 Spec paths and 0 bounds begin with 'proj', and the entry of that name "
        "is not of one kind in the two places, so they would be read "
        f"differently.{SESSION_SENTENCE} In the checkout this worktree was made "
        f"from, {str(root)!r}: no directory named 'proj'; 0 paths exist only as "
        "written, 0 only without the leading 'proj'; decision: name dropped, "
        "redundant_name. In the worktree: a directory named 'proj'; 0 and 0; "
        f"decision: refused: {KIND_NO_EVIDENCE}. {WAYS_OUT}"
    )
    assert unstated.record["source_decision"] == "name dropped, redundant_name"
    # The message does not say what a run without a worktree reaches.
    assert "reaches the same refusal" not in unstated.message


def test_entry_of_two_kinds_counts_every_way_a_spec_carries_the_name(
    tmp_path: Path,
) -> None:
    worktree, root = _made_in_the_worktree(tmp_path)

    entry = _read(worktree, _spec(("proj/",)), "write-root", source=root)
    bound = _read(worktree, _spec(("src/a.py",), bounds=("proj",)), source=root)
    plain = _read(worktree, _spec(("src/a.py",)), source=root)

    assert entry.decision == bound.decision == KIND_WORKTREE
    # Both places would read the entry alike under the statement, so the
    # message says only that the entry is not of one kind.
    assert entry.message.startswith(
        "0 Spec paths and 0 bounds begin with 'proj', 1 scope entries are 'proj' "
        f"itself, and the entry of that name is not of one kind in the two "
        f"places.{SESSION_SENTENCE} In the checkout"
    )
    assert "would be read differently" not in entry.message
    assert "decision: write-root, stated. In the worktree" in entry.message
    assert entry.message.endswith(f"decision: write-root, stated. {WAYS_OUT}")
    assert bound.message.startswith("0 Spec paths and 1 bounds begin with 'proj', and ")
    # Both places read this Spec, and they read its bound differently: as the
    # directory of the name in the worktree, as the write root in the checkout.
    both_read = _read(
        worktree, _spec(("src/a.py",), bounds=("proj", "src/")), source=root,
    )
    assert both_read.decision == KIND_WORKTREE
    assert both_read.message.startswith(
        "0 Spec paths and 1 bounds begin with 'proj', and the entry of that name "
        "is not of one kind in the two places, so they would be read "
        f"differently.{SESSION_SENTENCE} In the checkout"
    )
    assert both_read.message.endswith(
        "decision: name dropped, redundant_name. In the worktree: a directory "
        f"named 'proj'; 0 and 0; decision: write-root, not_needed. {WAYS_OUT}"
    )
    # A Spec that carries the name nowhere is read.
    assert plain.decision == NOT_NEEDED
    assert plain.record["source_decision"] == NOT_NEEDED


def test_a_link_of_the_name_in_the_worktree_only_is_an_entry_of_another_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _tree(tmp_path / "linked", _FLAT)
    worktree = _worktree(root)
    (worktree / NAME).symlink_to("src", target_is_directory=True)
    lookups: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        directive_table, "_commit_holds_entry", lambda *args: lookups.append(args),
    )

    outcome = _read(worktree, _spec(("proj/",)), "write-root", source=root)

    assert outcome.decision == KIND_WORKTREE
    assert outcome.record["entry"] == BLOCKED
    assert outcome.record["source_entry"] == REDUNDANT
    assert outcome.message == (
        "0 Spec paths and 0 bounds begin with 'proj', 1 scope entries are 'proj' "
        "itself, and the entry of that name is not of one kind in the two "
        f"places. In the checkout this worktree was made from, {str(root)!r}: no "
        "directory named 'proj'; 0 paths exist only as written, 0 only without "
        "the leading 'proj'; decision: write-root, stated. In the worktree: a "
        "link or another node named 'proj'; 0 and 0; decision: write-root, "
        f"stated. {WAYS_OUT}"
    )
    # Only a directory can be the session's own work on a path of the Spec:
    # for a link nothing is looked up and the sentence is not said.
    assert lookups == []


def test_own_work_sentence_needs_a_commit_without_an_entry_of_the_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    made_worktree, made_root = _made_in_the_worktree(tmp_path)
    # The commit holds the directory; the source checkout's working copy lost it.
    lost_root = _tree(tmp_path / "lost", _TREES["nested"])
    lost_worktree = _worktree(lost_root)
    shutil.rmtree(lost_root / NAME)
    # The commit holds a regular file of the name; the worktree replaced it.
    file_root = _tree(tmp_path / "file", _TREES["name_file"])
    file_worktree = _worktree(file_root)
    (file_worktree / NAME).unlink()
    _put(file_worktree, "proj/new.py")
    # The directory was made in the source checkout only.
    source_root = _tree(tmp_path / "source", _FLAT)
    source_worktree = _worktree(source_root)
    _put(source_root, "proj/x.py")

    lookups: list[tuple[Path, str]] = []
    real = directive_table._commit_holds_entry

    def counted(root: Path, name: str) -> bool | None:
        lookups.append((root, name))
        return real(root, name)

    monkeypatch.setattr(directive_table, "_commit_holds_entry", counted)
    spec = _spec(("proj/",), ("proj/new.py",))
    said = {}
    for label, worktree, root in (
        ("made in the worktree", made_worktree, made_root),
        ("lost in the source checkout", lost_worktree, lost_root),
        ("a regular file replaced", file_worktree, file_root),
        ("made in the source checkout", source_worktree, source_root),
    ):
        before = len(lookups)
        outcome = _read(worktree, spec, "write-root", source=root)
        assert outcome.decision == KIND_WORKTREE, label
        said[label] = (SESSION_SENTENCE in outcome.message, len(lookups) - before)

    assert said == {
        "made in the worktree": (True, 1),
        "lost in the source checkout": (False, 1),
        "a regular file replaced": (False, 1),
        # The worktree holds no directory there: nothing is looked up.
        "made in the source checkout": (False, 0),
    }


def test_own_work_sentence_takes_the_fact_in_memory_and_survives_no_failed_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    worktree, root = _made_in_the_worktree(tmp_path)
    spec = _spec(("proj/",))
    lookups: list[str] = []
    answers: list[bool | None] = [None]

    def lookup(_root: Path, name: str) -> bool | None:
        lookups.append(name)
        return answers[0]

    monkeypatch.setattr(directive_table, "_commit_holds_entry", lookup)
    by_fact = {
        fact: _read(worktree, spec, "write-root", source=root,
                    worktree_base_holds_name=fact)
        for fact in (False, True)
    }
    assert lookups == []
    failed = _read(worktree, spec, "write-root", source=root)
    assert lookups == ["proj"]
    answers[0] = False
    answered = _read(worktree, spec, "write-root", source=root)

    assert SESSION_SENTENCE in by_fact[False].message
    assert SESSION_SENTENCE not in by_fact[True].message
    assert SESSION_SENTENCE not in failed.message
    assert SESSION_SENTENCE in answered.message
    # Kind, record and the rest of the text do not depend on the fact.
    for outcome in (by_fact[True], failed, answered):
        assert outcome.decision == KIND_WORKTREE
        assert outcome.record == by_fact[False].record
        assert outcome.message.replace(SESSION_SENTENCE, "") == (
            by_fact[False].message.replace(SESSION_SENTENCE, "")
        )
    # A run that reads on looks nothing up.
    assert _read(worktree, _spec(("src/a.py",)), source=root).table is not None
    assert lookups == ["proj", "proj"]


def test_commit_lookup_answers_for_the_worktrees_own_repository_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    holds = _tree(tmp_path / "holds", _TREES["nested"])
    plain = _tree(tmp_path / "plain", _FLAT)
    a_file = _tree(tmp_path / "a_file", _TREES["name_file"])
    no_repository = _tree(tmp_path / "none", _FLAT, commit=False)
    # An entry whose name only begins with the name is another entry.
    near = _tree(tmp_path / "near", {**_FLAT, "projects/x.py": "X = 1\n"})
    # A worktree inside the working tree of another repository.
    outer = _tree(tmp_path / "outer", _FLAT)
    inside = outer / "tmp" / "crucible_worktree_inside"
    inside.parent.mkdir()
    _git(holds, "worktree", "add", "-q", "--detach", str(inside), "HEAD")

    lookup = directive_table._commit_holds_entry
    assert lookup(holds, NAME) is True
    assert lookup(plain, NAME) is False
    assert lookup(a_file, NAME) is True
    assert lookup(near, NAME) is False
    assert lookup(no_repository, NAME) is None
    assert lookup(inside, NAME) is True
    # Ambient Git variables do not redirect the lookup.
    monkeypatch.setenv("GIT_DIR", str(plain / ".git"))
    assert lookup(holds, NAME) is True
    monkeypatch.delenv("GIT_DIR")
    # With its link gone, Git would find the repository above and answer for
    # that one, in which the name is absent; the lookup fails instead.
    (inside / ".git").unlink()
    assert _git(inside, "rev-parse", "--show-toplevel") == str(outer)
    assert lookup(inside, NAME) is None
    # Git that cannot be started, or does not end, is a failed lookup and not
    # a commit without the entry.
    for failure in (FileNotFoundError("git"), subprocess.TimeoutExpired("git", 60)):
        def fails(*_args: Any, _failure: BaseException = failure, **_kwargs: Any) -> None:
            raise _failure

        monkeypatch.setattr(directive_table.subprocess, "run", fails)
        assert lookup(plain, NAME) is None


def test_worktrees_own_refusal_names_both_places_where_they_decide_differently(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path / "votes", _TREES["nested"])
    worktree = _worktree(root)
    (worktree / NAME / "only.py").unlink()
    spec = _spec(("proj/only.py",))

    outcome = _read(worktree, spec, source=root)

    assert outcome.decision == KIND_NO_EVIDENCE
    assert outcome.votes == (0, 0)
    assert outcome.record["source_write_root_votes"] == 1
    assert outcome.record["source_decision"] == f"refused: {KIND_UNSTATED}"
    assert outcome.message.endswith(
        f" {DECIDE_DIFFERENTLY} In the checkout this worktree was made from, "
        f"{str(root)!r}: a directory named 'proj'; 1 paths exist only as written, "
        "0 only without the leading 'proj'; decision: refused: "
        f"{KIND_UNSTATED}. In the worktree: a directory named 'proj'; 0 and 0; "
        f"decision: refused: {KIND_NO_EVIDENCE}. {WAYS_OUT} {SAME_REFUSAL}"
    )
    assert outcome.refusal is not None and outcome.refusal.carries_name is True


def test_worktree_that_would_read_on_is_refused_where_its_checkout_reads_otherwise(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path / "moved", _TREES["nested"])
    worktree = _worktree(root)
    # In the source checkout's working copy the voting file moved to the root.
    (root / NAME / "only.py").rename(root / "only.py")
    spec = _spec(("proj/only.py",))

    written = _read(worktree, spec, "write-root", source=root)
    parent = _read(worktree, spec, "parent", source=root)

    assert written.decision == KIND_WORKTREE_VOTES
    assert written.message == (
        "The Spec's paths that begin with 'proj' are read differently in the two "
        f"places. In the checkout this worktree was made from, {str(root)!r}: a "
        "directory named 'proj'; 0 paths exist only as written, 1 only without "
        f"the leading 'proj'; decision: refused: {KIND_CONTRADICTED}. In the "
        "worktree: a directory named 'proj'; 1 and 0; decision: write-root, "
        f"stated. {WAYS_OUT}"
    )
    assert written.record["read_against"] == "write-root"
    assert written.record["source_decision"] == f"refused: {KIND_CONTRADICTED}"
    assert written.refusal is not None and written.refusal.carries_name is True
    # The worktree's own refusal keeps its kind.
    assert parent.decision == KIND_CONTRADICTED
    assert DECIDE_DIFFERENTLY in parent.message and SAME_REFUSAL in parent.message
    assert parent.record["source_decision"] == PARENT_STATED


def test_regular_file_in_one_place_of_a_worktree_run_is_no_difference_of_kind(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path / "onefile", _FLAT)
    worktree = _worktree(root)
    (worktree / NAME).write_text("a regular file\n", encoding="utf-8")
    spec = _spec(("proj/",))

    unstated = _read(worktree, spec, source=root)
    stated = _read(worktree, spec, "write-root", source=root)

    # Absent and a regular file are one kind of entry; the two places still
    # decide for themselves, and the run goes on only where they agree.
    assert unstated.record["entry"] == unstated.record["source_entry"] == REDUNDANT
    assert unstated.decision == KIND_WORKTREE_VOTES
    assert unstated.record["source_decision"] == f"refused: {KIND_BARE_NAME}"
    assert stated.decision == STATED


# --------------------------------------------------------------------------
# The table
# --------------------------------------------------------------------------

def test_the_table_gives_each_slice_its_literal_paths(trees: dict[str, Path]) -> None:
    spec = _spec(
        ("proj/src/a.py", "setup.py", "proj/setup.py"), ("proj/new.py", "setup.py"),
        regions=("proj/src/a.py",),
        later=(("b", ("src/a.py",), ("proj/docs/",)),),
    )

    outcome = _read(trees["flat"], spec, materializes={"b": ("proj/out.md",)})
    table = outcome.table

    assert table is not None
    assert (table.name, table.statement, table.read_against, table.basis) == (
        NAME, None, None, "redundant_name",
    )
    assert table.reading == decide_scope_reading(trees["flat"])
    # files_in_scope then creates_paths; the order is kept, duplicates dropped.
    assert table.scope("a") == ("src/a.py", "setup.py", "new.py")
    assert table.scope("b") == ("src/a.py", "docs/")
    first, second = table.slices
    assert (first.tier_id, first.slice_id) == ("t0", "a")
    assert first.directive.files_in_scope == ("src/a.py", "setup.py", "setup.py")
    assert first.directive.creates_paths == ("new.py", "setup.py")
    assert [region.file for region in first.directive.mutation_regions] == ["src/a.py"]
    assert first.directive.mutation_regions[0].start_marker == (
        spec.tier_dag.tiers[0].slices[0].directive.mutation_regions[0].start_marker
    )
    assert first.materializes_paths == ()
    assert second.materializes_paths == ("out.md",)
    assert table.directive("b") is second.directive
    assert outcome.record["paths_changed"] == 6
    with pytest.raises(KeyError):
        table.slice("zz")


def test_the_table_hands_out_the_spec_itself_where_no_path_differs(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("src/a.py",), ("docs/new.md",), later=(("b", ("setup.py",), ()),))

    table = _read(trees["flat"], spec).table

    assert table is not None
    assert table.resolved_spec(spec) is spec
    assert table.tier(spec, "t1") is spec.tier_dag.tiers[1]
    for item in table.slices:
        assert item.directive is directive_table.authored_pair(spec, item.slice_id)[1].directive


def test_the_table_copy_of_a_spec_changes_directive_paths_and_nothing_else(
    trees: dict[str, Path],
) -> None:
    spec = _spec(
        ("proj/src/a.py",), bounds=("proj/src/", "setup.py"),
        later=(("b", ("setup.py",), ()),),
    )

    table = _read(trees["flat"], spec).table

    assert table is not None
    copy = table.resolved_spec(spec)
    assert copy is not spec
    assert copy.tier_dag.tiers[0].slices[0].directive.files_in_scope == ("src/a.py",)
    # The slice whose paths did not change keeps the Spec's own directive.
    assert copy.tier_dag.tiers[1].slices[0].directive is (
        spec.tier_dag.tiers[1].slices[0].directive
    )
    # The bound stays authored in the copy; the table holds how it was read.
    assert copy.constraints == spec.constraints
    assert table.bounds == ("src/", "setup.py")
    assert dataclasses.replace(copy, tier_dag=spec.tier_dag) == spec
    assert table.tier(spec, "t0").slices[0].directive.files_in_scope == ("src/a.py",)
    with pytest.raises(KeyError):
        table.tier(spec, "t9")
    tier, slice_ = directive_table.authored_pair(spec, "a")
    assert tier is spec.tier_dag.tiers[0] and slice_ is tier.slices[0]
    with pytest.raises(KeyError):
        directive_table.authored_pair(spec, "zz")


@pytest.mark.parametrize(("spec", "expected"), [
    (_spec(("src/a.py",)), False),
    (_spec(("proj/src/a.py",)), True),
    (_spec(("src/a.py",), ("proj",)), True),
    (_spec(("src/a.py",), regions=("proj/src/a.py",)), True),
    (_spec(("src/a.py",), bounds=("proj",)), True),
    (_spec(("src/a.py",), bounds=("proj/src/", "src/")), True),
    (_spec(("src/proj/a.py",), bounds=("src/proj",)), False),
], ids=[
    "nowhere", "a_path_begins_with_it", "a_create_entry_is_it",
    "a_region_file_begins_with_it", "a_bound_is_it", "a_bound_begins_with_it",
    "only_further_down",
])
def test_the_table_says_whether_the_spec_carries_the_name(
    trees: dict[str, Path], spec: Spec, expected: bool,
) -> None:
    # Where the root holds a regular file of the name, each of these Specs is
    # read without a statement.
    outcome = _read(trees["name_file"], spec)

    assert outcome.table is not None
    assert outcome.table.carries_name is expected


def test_a_refusal_says_whether_the_spec_carries_the_name(trees: dict[str, Path]) -> None:
    without = _read(trees["flat"], _spec(("src/a.py",)), "parent")
    with_entry = _read(trees["flat"], _spec(("proj/",)), "parent")
    with_bound = _read(trees["flat"], _spec(("src/a.py",), bounds=("proj",)), "parent")

    assert without.decision == with_entry.decision == with_bound.decision == (
        KIND_WITHOUT_EFFECT
    )
    assert without.refusal is not None and without.refusal.carries_name is False
    assert with_entry.refusal is not None and with_entry.refusal.carries_name is True
    assert with_bound.refusal is not None and with_bound.refusal.carries_name is True


def test_the_record_is_plain_json_and_belongs_to_one_call(trees: dict[str, Path]) -> None:
    spec = _spec(("proj/src/a.py",))
    table = _read(trees["flat"], spec).table
    refusal = _read(trees["flat"], spec, "write-root").refusal

    assert table is not None and refusal is not None
    assert json.loads(table.record_json) == table.record
    table.record["basis"] = "changed"
    assert table.record["basis"] == "redundant_name"
    assert json.loads(json.dumps(refusal.record)) == refusal.record
    assert isinstance(refusal, ScopePathError)
    assert dataclasses.is_dataclass(table)
    with pytest.raises(dataclasses.FrozenInstanceError):
        table.basis = "stated"  # type: ignore[misc]


# --------------------------------------------------------------------------
# The checks where a writer starts
# --------------------------------------------------------------------------

def test_a_directive_handed_to_an_implementer_carries_the_tables_paths(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/src/a.py",), ("proj/new.py",), regions=("proj/src/a.py",))
    table = _read(trees["flat"], spec).table
    assert table is not None
    own = spec.tier_dag.tiers[0].slices[0].directive

    table.require_directive("a", table.directive("a"))
    table.require_directive("a", dataclasses.replace(table.directive("a"), intent="retry"))
    for stranger in (
        own,
        dataclasses.replace(table.directive("a"), creates_paths=()),
        dataclasses.replace(table.directive("a"), mutation_regions=()),
        dataclasses.replace(table.directive("a"), files_in_scope=("src/a.py", "setup.py")),
    ):
        with pytest.raises(DirectiveNotFromTable) as caught:
            table.require_directive("a", stranger)
        assert caught.value.kind == KIND_NOT_FROM_TABLE
        assert str(caught.value) == (
            "slice 'a': the directive handed to the implementer does not carry "
            "the literal paths of this run's directive table. This is a defect "
            "of the workflow, not of the Spec."
        )
    with pytest.raises(KeyError):
        table.require_directive("zz", own)


def test_a_breach_directive_names_the_test_path_or_a_literal_path_of_the_packet(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/src/a.py",), ("proj/new.py",), later=(("b", ("setup.py",), ()),))
    table = _read(trees["flat"], spec).table
    assert table is not None
    test_path = "tests/test_attack.py"

    def breach(files: tuple[str, ...], creates: tuple[str, ...] = ()) -> Any:
        return _directive("breach-defender", files, creates)

    check = directive_table.require_breach_directive
    check(table, breach((test_path,)), test_path=test_path, packet_slice_ids=())
    check(table, breach((), (test_path,)), test_path=test_path, packet_slice_ids=())
    check(table, breach(("src/a.py",), ("new.py", test_path)), test_path=test_path,
          packet_slice_ids=("a",))
    for files, slices in (
        (("proj/src/a.py",), ("a",)),      # the packet's path as authored
        (("setup.py",), ("a",)),           # a slice the packet does not name
        (("src/a.py",), ()),               # no packet slice at all
    ):
        with pytest.raises(DirectiveNotFromTable) as caught:
            check(table, breach(files), test_path=test_path, packet_slice_ids=slices)
        assert caught.value.kind == KIND_NOT_FROM_TABLE
        assert str(caught.value) == (
            "breach directive 'breach-defender': 1 paths are neither the breach "
            "test path, nor a literal path of the slices the authority packet "
            f"names, nor a path of the diff ({files[0]!r}). This is a defect of "
            "the workflow, not of the Spec."
        )
    # The test path is a machine path and has to be literal already.
    with pytest.raises(ScopePathError, match="begins with the checkout name"):
        check(table, breach(("tests/t.py",)), test_path="proj/tests/t.py",
              packet_slice_ids=())


def test_a_breach_without_a_packet_names_the_test_path_or_a_path_of_the_diff(
    trees: dict[str, Path],
) -> None:
    reading = decide_scope_reading(trees["flat"])
    test_path = "tests/test_attack.py"
    check = directive_table.require_breach_diff_directive

    check(reading, _directive("d", ("src/a.py", test_path)), test_path=test_path,
          diff_paths=("src/a.py",))
    with pytest.raises(DirectiveNotFromTable, match="'setup.py'"):
        check(reading, _directive("d", ("src/a.py", "setup.py")), test_path=test_path,
              diff_paths=("src/a.py",))
    with pytest.raises(ScopePathError):
        check(reading, _directive("d", ()), test_path="proj/tests/t.py", diff_paths=())


@pytest.mark.parametrize("statement", ["write-root", "parent"])
def test_a_statement_on_a_breach_that_reads_no_spec_path_is_a_refusal(
    statement: str,
) -> None:
    refusal = directive_table.breach_statement_without_packet(statement)

    assert refusal.kind == KIND_NO_DIRECTIVE_READ
    assert refusal.record == {"statement": statement, "refused": KIND_NO_DIRECTIVE_READ}
    assert refusal.carries_name is None
    assert str(refusal) == (
        f"{OPTION} {statement} was stated, and this run reads no path of the "
        "Spec: without --authority-file a Breach works on the paths of Git's "
        "diff and on its own test file. The statement would change no path, so "
        "the run is refused instead of accepting it unapplied. Leave the "
        "statement out, or give the authority packet of the slices this Breach "
        "is for."
    )


# --------------------------------------------------------------------------
# The hand-over to the validation gate
# --------------------------------------------------------------------------

def test_the_resolver_hands_the_tables_copy_to_three_rules_and_no_other(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/src/a.py",))
    table = _read(trees["flat"], spec).table
    assert table is not None
    literal = table.resolved_spec(spec)
    handed: dict[str, Any] = {}

    def lookup(rule_id: str) -> Any:
        if rule_id in rules.context_aware():
            def with_context(target: Any, ctx: Any) -> tuple[()]:
                handed[rule_id] = (target, ctx)
                return ()
            return with_context

        def pure(target: Any) -> tuple[()]:
            handed[rule_id] = (target,)
            return ()
        return pure

    resolver = directive_table.gate_rule_resolver(table, spec, lookup)
    for rule_id in rules.available():
        rule = resolver(rule_id)
        if rule_id in rules.context_aware():
            rule(spec, "ctx")
        else:
            rule(spec)

    assert directive_table.TABLE_RULES == (
        "cited_path_exists", "parallel_tier_conflicts", "creates_paths",
    )
    assert literal != spec
    for rule_id in rules.available():
        if rule_id in directive_table.TABLE_RULES:
            assert handed[rule_id][0] == literal, rule_id
        else:
            assert handed[rule_id][0] is spec, rule_id
    assert handed["cited_path_exists"][1] == "ctx"


def test_the_resolver_keeps_every_rule_and_the_arity_of_the_three(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("proj/src/a.py",))
    table = _read(trees["flat"], spec).table
    assert table is not None

    resolver = directive_table.gate_rule_resolver(table, spec)

    for rule_id in rules.available():
        rule = resolver(rule_id)
        own = getattr(rules, rule_id)
        assert directive_table.crucible_rule(rule_id) is own
        if rule_id in directive_table.TABLE_RULES:
            assert rule is not own
            assert len(inspect.signature(rule).parameters) == (
                len(inspect.signature(own).parameters)
            )
        else:
            assert rule is own
    with pytest.raises(ValueError, match="unknown validation rule 'no_such_rule'"):
        resolver("no_such_rule")
    with pytest.raises(ValueError, match="unknown validation rule"):
        directive_table.crucible_rule("available")


def test_the_gate_reads_the_file_the_table_names(trees: dict[str, Path]) -> None:
    root = trees["flat"]
    spec = _spec(("proj/src/a.py",))
    table = _read(root, spec).table
    assert table is not None
    ruleset = ValidationRuleset(name="files", rule_ids=("cited_path_exists",))
    ctx = RuleContext(workspace_root=root)

    authored = validate(spec, ruleset=ruleset, ctx=ctx)
    through_table = validate(
        spec, ruleset=ruleset, ctx=ctx,
        rule_resolver=directive_table.gate_rule_resolver(table, spec),
    )

    def missing(verdict: Any) -> list[str]:
        return [finding.id for finding in verdict.findings
                if finding.id.startswith("missing_files_in_scope")]

    assert missing(authored) == ["missing_files_in_scope__a__proj/src/a.py"]
    assert authored.decision != "pass"
    assert missing(through_table) == []
    # Nothing else moves: the other findings are those of the authored Spec.
    assert [finding.id for finding in through_table.findings] == [
        finding.id for finding in authored.findings
        if not finding.id.startswith("missing_files_in_scope")
    ]
    assert through_table.decision == "pass"


def test_a_resolver_built_for_another_spec_fails_closed(trees: dict[str, Path]) -> None:
    spec = _spec(("proj/src/a.py",))
    equal_copy = _spec(("proj/src/a.py",))
    assert equal_copy == spec and equal_copy is not spec
    table = _read(trees["flat"], spec).table
    assert table is not None
    resolver = directive_table.gate_rule_resolver(table, spec)

    with pytest.raises(
        ValueError,
        match="^rule 'creates_paths' was handed another Spec than the one its "
              "directive table was built for$",
    ):
        resolver("creates_paths")(equal_copy)
    verdict = validate(
        equal_copy,
        ruleset=ValidationRuleset(name="three", rule_ids=directive_table.TABLE_RULES),
        ctx=RuleContext(workspace_root=trees["flat"]),
        rule_resolver=resolver,
    )

    assert sorted(finding.id for finding in verdict.findings) == [
        "rule_error_cited_path_exists", "rule_error_creates_paths",
        "rule_error_parallel_tier_conflicts",
    ]
    assert {finding.severity for finding in verdict.findings} == {"Critical"}
    assert verdict.decision != "pass"


def test_a_table_that_changes_no_path_changes_nothing_for_the_gate(
    trees: dict[str, Path],
) -> None:
    spec = _spec(("src/a.py",), ("docs/new.md",))
    table = _read(trees["flat"], spec).table
    assert table is not None
    seen: list[Any] = []

    def lookup(rule_id: str) -> Any:
        def pure(target: Any) -> tuple[()]:
            seen.append(target)
            return ()
        return pure

    directive_table.gate_rule_resolver(table, spec, lookup)("creates_paths")(spec)

    assert seen == [spec] and seen[0] is spec


# --------------------------------------------------------------------------
# A refusal of a plan that the same run made
# --------------------------------------------------------------------------

def test_the_text_for_a_refused_plan_made_in_the_same_run(trees: dict[str, Path]) -> None:
    refusal = _read(trees["nested"], _spec((), ("proj/newdir/new.py",))).refusal
    assert refusal is not None

    text = directive_table.plan_made_in_this_run(refusal, trees["nested"])

    assert text == (
        f"{refusal} The commands that take {OPTION} write-root or {OPTION} parent "
        "are implement, tier, breach, handoff, reconcile and architect; swebench "
        "and deepswe take neither. Two facts about this refusal: the plan was "
        f"made in this run, from {str(trees['nested'])!r}, and it is not kept; "
        f"and architect refuses {OPTION} parent on a run that plans. A statement "
        "about a plan that a run makes is given on the run that makes it (with "
        "--replan where the Spec holds an earlier plan); given on a run that "
        "reuses a retained plan, it is a statement about that retained plan."
    )


def test_every_error_kind_has_its_name() -> None:
    kinds = {
        name: value for name, value in vars(directive_table).items()
        if name.startswith("KIND_")
    }

    assert kinds == {
        "KIND_INVALID": "scope_path_invalid",
        "KIND_REPEATS": "scope_path_repeats_name",
        "KIND_BLOCKED": "scope_path_blocked",
        "KIND_BOUND": "scope_outside_bound",
        "KIND_NOT_BELOW": "scope_path_not_below_write_root",
        "KIND_UNSTATED": "spec_read_against_unstated",
        "KIND_NO_EVIDENCE": "spec_read_against_no_evidence",
        "KIND_CONTRADICTED": "spec_read_against_contradicted",
        "KIND_CONTRADICTORY": "spec_read_against_contradictory",
        "KIND_WITHOUT_EFFECT": "spec_read_against_without_effect",
        "KIND_NO_DIRECTORY": "spec_read_against_write_root_without_directory",
        "KIND_BARE_NAME": "spec_read_against_bare_name_without_directory",
        "KIND_PARENT_NEW_PLAN": "spec_read_against_parent_with_new_plan",
        "KIND_NO_DIRECTIVE_READ": "spec_read_against_no_directive_read",
        "KIND_WORKTREE": "worktree_entry_differs",
        "KIND_WORKTREE_VOTES": "worktree_votes_differ",
        "KIND_NOT_FROM_TABLE": "directive_not_from_table",
    }
    # One kind for each refusal: no two share a name.
    assert len(set(kinds.values())) == len(kinds)
    assert directive_table.RECORD_KEY == "read_against"


def test_every_message_takes_the_option_from_its_one_spelling() -> None:
    tree = ast.parse(inspect.getsource(directive_table))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]

    assert OPTION == "--spec-read-against"
    assert directive_table.OPTION is OPTION
    assert len(literals) > 100
    # No message spells the option out: each takes it from the constant.
    assert [text for text in literals if "spec-read-against" in text] == []
