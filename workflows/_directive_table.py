# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Read every directive path of a Spec once for one write root, or refuse it.

``build_directive_table`` turns the authored paths of a Spec into the literal
paths one run uses below its write root: ``files_in_scope``, ``creates_paths``
and the region files of every slice, the outputs handed in as materialized,
and the Spec's ``no_edits_outside``. The result is a ``DirectiveTable``; a
Spec that cannot be read is refused with a ``DirectiveRefusal``, which carries
an error kind and the record of what was found. Nothing of either is stored.
The record is output: the messages print its counts, and no decision is taken
from a key of it.

The name that matters is the directive-level name: the name of the write
root, or in a worktree the name of the checkout the worktree was made from.
A path or bound "begins with the name" when its first component is that name
and more components follow; an entry that is the name alone is "the name
itself". A statement is the run's ``--spec-read-against`` value,
``write-root`` or ``parent``; a run without the option made no statement.
What ``<root>/<name>`` is decides the rest:

- Absent or a regular file. Without a statement one leading name is dropped
  from every path and bound that begins with it. A scope entry that is the
  name itself is refused without a statement where the root holds no entry of
  that name. A stated ``write-root`` drops no name, so it is refused when a
  path begins with the name or the first component of a bound is the name,
  and where the root holds no entry of that name also when a slice holds a
  scope entry that is the name itself beside an entry that does not begin
  with the name.
- A link or any other node. A path that begins with the name is refused.
- A real directory. A Spec with a path that begins with the name, or with a
  scope entry that is the name itself, is read only on a statement. The
  paths that begin with the name vote: one that exists as written and not
  without its leading name votes for the write root, one that exists without
  it and not as written votes for the parent. Votes only refuse: a stated
  ``write-root`` is refused by a parent vote, a stated ``parent`` by a
  write-root vote, and both kinds of vote refuse every statement.

Under a stated ``parent`` every path has to begin with the name, and one
leading name is dropped from it; the statement is refused when no path begins
with the name. A bound is read like a path; wherever a leading name is
dropped, a bound that is the name itself is the write root and, like ``**``,
bounds nothing. A path or bound that begins with the name twice is refused;
under a stated ``parent`` in a checkout that holds the directory one name is
dropped first, and what then still begins with the name twice is refused. A
literal scope path outside the literal bounds is refused. In a run whose
write root is a worktree the decision is taken on the worktree and on the
checkout it was made from, and the run goes on only when both take the same
reading and give the same literal paths and bounds.

No workflow builds a table yet. The write workflows read directive paths
with their own rules, and ``_write_root_guard`` refuses a host-routed write
root in which those rules name different files.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

import crucible.rules as _rules
from crucible.common.scope_reading import (
    BLOCKED,
    DIRECTORY,
    REDUNDANT,
    ScopePathError,
    ScopeReading,
    ScopeReadingError,
    _entry_kind,
    decide_scope_reading,
)

from ._args import SPEC_READ_AGAINST_OPTION as OPTION
from ._args import SPEC_READ_AGAINST_PARENT as PARENT
from ._args import SPEC_READ_AGAINST_VALUES as VALUES
from ._args import SPEC_READ_AGAINST_WRITE_ROOT as WRITE_ROOT
from ._mutation_journal import _git_argv, _sanitized_git_env


# The key of the record under which the applied reading is written.
RECORD_KEY = "read_against"

# The error kinds of a refused directive.
KIND_INVALID = "scope_path_invalid"
KIND_REPEATS = "scope_path_repeats_name"
KIND_BLOCKED = "scope_path_blocked"
KIND_BOUND = "scope_outside_bound"
KIND_NOT_BELOW = "scope_path_not_below_write_root"
KIND_UNSTATED = "spec_read_against_unstated"
KIND_NO_EVIDENCE = "spec_read_against_no_evidence"
KIND_CONTRADICTED = "spec_read_against_contradicted"
KIND_CONTRADICTORY = "spec_read_against_contradictory"
KIND_WITHOUT_EFFECT = "spec_read_against_without_effect"
# A stated write-root in a checkout without a directory of its own name.
KIND_NO_DIRECTORY = "spec_read_against_write_root_without_directory"
# No statement, a scope entry that is the name itself, and no entry of that
# name in the root. Kept apart from the kind above, which needs a stated
# write-root, and from ``_no_evidence``, whose root holds a directory.
KIND_BARE_NAME = "spec_read_against_bare_name_without_directory"
# A stated parent on a run that makes the plan the statement would be about.
# The kind is named with the others; no function of this module raises it.
KIND_PARENT_NEW_PLAN = "spec_read_against_parent_with_new_plan"
# A statement on a run that reads no path of the Spec.
KIND_NO_DIRECTIVE_READ = "spec_read_against_no_directive_read"
KIND_WORKTREE = "worktree_entry_differs"
KIND_WORKTREE_VOTES = "worktree_votes_differ"
KIND_NOT_FROM_TABLE = "directive_not_from_table"

_SHOWN = 6   # entries named in a message; the counts are always complete

_ENTRY_WORDS = {
    REDUNDANT: "no directory",
    DIRECTORY: "a directory",
    BLOCKED: "a link or another node",
}


class DirectiveRefusal(ScopePathError):
    """The directive step refuses the Spec.

    ``kind`` is the error kind and ``record`` what the step found, with the
    kind under ``refused``. ``carries_name`` says whether a path of the Spec
    begins with the directive-level name or is that name, or the first
    component of a bound is that name. It is None for a refusal raised before
    the entries of the Spec were read.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        record: Mapping[str, Any],
        *,
        carries_name: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.record = dict(record, refused=kind)
        self.carries_name = carries_name


@dataclasses.dataclass(frozen=True)
class ResolvedSlice:
    """One slice of the DAG with the literal paths of its directive."""

    tier_id: str
    slice_id: str
    # The Spec's own ImplementationDirective where no path differs, otherwise
    # a copy of it that carries the literal paths.
    directive: Any
    materializes_paths: tuple[str, ...]  # literal


@dataclasses.dataclass(frozen=True)
class DirectiveTable:
    """One per run and root. Plain data; nothing of it is stored."""

    reading: ScopeReading                # the reading of the write root
                                         # (common/scope_reading.py)
    name: str                            # the directive-level name
    read_against: str | None             # WRITE_ROOT, PARENT, or None: an unstated run
                                         # that dropped a redundant name
    basis: str                           # stated | redundant_name | not_needed
    statement: str | None                # --spec-read-against as given
    slices: tuple[ResolvedSlice, ...]
    bounds: tuple[str, ...] | None       # literal; "**" bounds nothing
    carries_name: bool                   # a path or bound of the Spec begins with
                                         # ``name``, or is the name itself
    record_json: str                     # the record of the reading, as JSON

    @property
    def record(self) -> dict[str, Any]:
        return json.loads(self.record_json)

    def slice(self, slice_id: str) -> ResolvedSlice:
        for item in self.slices:
            if item.slice_id == slice_id:
                return item
        raise KeyError(slice_id)

    def directive(self, slice_id: str) -> Any:
        return self.slice(slice_id).directive

    def scope(self, slice_id: str) -> tuple[str, ...]:
        """The literal write scope of one slice: order kept, duplicates dropped."""

        directive = self.slice(slice_id).directive
        return tuple(dict.fromkeys(
            (*directive.files_in_scope, *directive.creates_paths)
        ))

    def resolved_spec(self, spec: Any) -> Any:
        """The Spec with the table's directives; the Spec itself when no path differs."""

        by_key = {(item.tier_id, item.slice_id): item.directive for item in self.slices}
        tiers = tuple(
            dataclasses.replace(tier, slices=tuple(
                dataclasses.replace(slice_, directive=by_key[(tier.tier_id, slice_.id)])
                for slice_ in tier.slices
            ))
            for tier in spec.tier_dag.tiers
        )
        copy = dataclasses.replace(
            spec, tier_dag=dataclasses.replace(spec.tier_dag, tiers=tiers),
        )
        return spec if copy == spec else copy

    def tier(self, spec: Any, tier_id: str) -> Any:
        """The Spec's tier with the table's directives."""

        for tier in self.resolved_spec(spec).tier_dag.tiers:
            if tier.tier_id == tier_id:
                return tier
        raise KeyError(tier_id)

    def require_directive(self, slice_id: str, directive: Any) -> None:
        """Refuse a directive that does not carry this slice's literal paths.

        The check for the place where a directive is handed to an
        implementer: its ``files_in_scope``, ``creates_paths`` and region
        files have to be the table's, entry by entry.
        """

        mine = self.slice(slice_id).directive
        if _paths_of(directive) != _paths_of(mine):
            raise DirectiveNotFromTable(
                f"slice {slice_id!r}: the directive handed to the implementer "
                "does not carry the literal paths of this run's directive "
                "table. This is a defect of the workflow, not of the Spec."
            )


class DirectiveNotFromTable(ValueError):
    """A directive names a path its table does not give.

    A defect of the code that built the directive, not of the Spec.
    """

    kind = KIND_NOT_FROM_TABLE


def _paths_of(directive: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(directive.files_in_scope or ()),
        tuple(directive.creates_paths or ()),
        tuple(region.file for region in (directive.mutation_regions or ())),
    )


# --------------------------------------------------------------------------
# Lexical helpers
# --------------------------------------------------------------------------

def _parts(authored: str, record: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(authored, str) or not authored or "\0" in authored:
        raise DirectiveRefusal(
            KIND_INVALID, f"scope path is not one safe string: {authored!r}", record,
        )
    text = authored.replace("\\", "/") if os.name == "nt" else authored
    pure = PurePosixPath(text)
    if pure.is_absolute() or ".." in pure.parts:
        raise DirectiveRefusal(
            KIND_INVALID,
            f"scope path {authored!r} is absolute or leaves the write root",
            record,
        )
    return pure.parts


def _text(parts: tuple[str, ...], authored: str) -> str:
    result = PurePosixPath(*parts).as_posix() if parts else "."
    if authored.endswith("/") and result not in {".", "/"}:
        return result + "/"          # an authored directory entry stays one
    return result


def _lexists_below(root: Path, parts: tuple[str, ...]) -> bool:
    """Whether ``parts`` names an entry below ``root``; no link is followed.

    A link or a file on the way is not a directory of this tree, so the path
    does not exist below the root. The last component is tested with
    ``lstat``: a link there exists as a link.
    """

    current = root
    last = len(parts) - 1
    for index, part in enumerate(parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except (FileNotFoundError, NotADirectoryError):
            return False
        except OSError as exc:     # fail closed: no vote without a clear answer
            raise ScopeReadingError(
                f"cannot inspect {current} for the directive evidence: {exc}"
            ) from exc
        if index < last and not stat.S_ISDIR(metadata.st_mode):
            return False
    return True


def _within(path: str, bounds: tuple[str, ...]) -> bool:
    for bound in bounds:
        base = bound.rstrip("/")
        if bound == "**" or base in {"", "."}:
            return True
        if path.rstrip("/") == base or path.startswith(base + "/"):
            return True
    return False


def _named(entries: list[str]) -> str:
    shown = ", ".join(repr(item) for item in entries[:_SHOWN])
    more = len(entries) - _SHOWN
    return shown + (f" and {more} more" if more > 0 else "")


# --------------------------------------------------------------------------
# Readers that take the Spec as authored, and the checks where a writer starts
# --------------------------------------------------------------------------

def authored_pair(spec: Any, slice_id: str) -> tuple[Any, Any]:
    """The Spec's own tier and slice for ``slice_id``.

    A resume identity hashes the directive of the slice it is given, so a
    reader of identities has to be handed the Spec's own pair and never a
    slice that carries the table's copy of a directive. Code that walks the
    table's slices looks the Spec's pair up here, by slice id.
    """

    for tier in spec.tier_dag.tiers:
        for slice_ in tier.slices:
            if slice_.id == slice_id:
                return tier, slice_
    raise KeyError(slice_id)


def require_breach_directive(
    table: DirectiveTable,
    directive: Any,
    *,
    test_path: str,
    packet_slice_ids: tuple[str, ...],
) -> None:
    """Refuse a Breach directive that names a path the table does not give.

    For a Breach that holds an authority packet. Breach builds every
    directive it hands to an implementer under a slice id of its own, so
    there is no table slice to compare it with. What is checked instead:
    every ``files_in_scope``, ``creates_paths`` and region file of the
    directive is the breach test path, which has to be a literal path, or a
    literal path of a table slice that the authority packet names.
    """

    allowed = {table.reading.require_literal(test_path)}
    for slice_id in packet_slice_ids:
        for group in _paths_of(table.slice(slice_id).directive):
            allowed.update(group)
    _require_breach_paths(directive, allowed)


def require_breach_diff_directive(
    reading: ScopeReading,
    directive: Any,
    *,
    test_path: str,
    diff_paths: tuple[str, ...],
) -> None:
    """The same check for a standalone Breach without an authority packet.

    Such a run reads no directive path of the Spec, so it builds no
    directive table and takes the reading of its write root
    (``common/scope_reading.py``) only. Every path of the directive is the
    breach test path or a path of Git's diff.
    """

    _require_breach_paths(
        directive, {reading.require_literal(test_path), *diff_paths},
    )


def _require_breach_paths(directive: Any, allowed: set[str]) -> None:
    strangers = [
        path for group in _paths_of(directive) for path in group
        if path not in allowed
    ]
    if strangers:
        raise DirectiveNotFromTable(
            f"breach directive {directive.slice_id!r}: {len(strangers)} paths are "
            "neither the breach test path, nor a literal path of the slices "
            "the authority packet names, nor a path of the diff "
            f"({_named(strangers)}). This is a defect of the workflow, not of "
            "the Spec."
        )


def breach_statement_without_packet(statement: str) -> DirectiveRefusal:
    """A statement on a standalone Breach that holds no authority packet.

    The run reads no path of the Spec, so the statement could not be
    applied: a statement is applied as given or the run is refused.
    """

    return DirectiveRefusal(
        KIND_NO_DIRECTIVE_READ,
        f"{OPTION} {statement} was stated, and this run reads no path of the "
        "Spec: without --authority-file a Breach works on the paths of "
        "Git's diff and on its own test file. The statement would change no "
        "path, so the run is refused instead of accepting it unapplied. Leave "
        "the statement out, or give the authority packet of the slices this "
        "Breach is for.",
        {"statement": statement, "refused": KIND_NO_DIRECTIVE_READ},
    )


# --------------------------------------------------------------------------
# The hand-over to the validation gate
# --------------------------------------------------------------------------

# The rules that decide which file a directive path names.
TABLE_RULES = ("cited_path_exists", "parallel_tier_conflicts", "creates_paths")


def crucible_rule(rule_id: str) -> Callable[..., Any]:
    """Look a validation rule up among the package's own exported functions."""

    if rule_id not in _rules.available():
        raise ValueError(f"unknown validation rule {rule_id!r}")
    return getattr(_rules, rule_id)


def gate_rule_resolver(
    table: DirectiveTable, spec: Any,
    lookup: Callable[[str], Callable[..., Any]] = crucible_rule,
) -> Callable[[str], Callable[..., Any]]:
    """The rule resolver for one validate call of one Spec.

    The three rules that decide which file a directive path names read the
    table's copy of the Spec; every other rule reads the Spec as authored.
    The wrapper refuses any other Spec than ``spec``, the one this resolver
    was made for: a raise in a rule is a Critical finding, so a resolver
    reused for another candidate fails closed. ``table`` has to be the table
    that was built for ``spec``; a table keeps no reference to the Spec it
    was built for, so that is not checked here.
    """

    literal = table.resolved_spec(spec)

    def resolver(rule_id: str) -> Callable[..., Any]:
        rule = lookup(rule_id)
        if rule_id not in TABLE_RULES:
            return rule
        positional = [
            parameter for parameter in inspect.signature(rule).parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]

        def own(target: Any) -> None:
            if target is not spec:
                raise ValueError(
                    f"rule {rule_id!r} was handed another Spec than the one its "
                    "directive table was built for"
                )

        # The wrapper keeps the arity of its rule, by which the validator
        # tells a pure rule from one that takes a context.
        if len(positional) == 2:
            def with_context(target: Any, ctx: Any) -> Any:
                own(target)
                return rule(literal, ctx)
            return with_context

        def pure(target: Any) -> Any:
            own(target)
            return rule(literal)
        return pure

    return resolver


# --------------------------------------------------------------------------
# The one function
# --------------------------------------------------------------------------

def build_directive_table(
    reading: ScopeReading,
    spec: Any,
    *,
    read_against: str | None = None,
    worktree_source: Path | str | None = None,
    materializes: Mapping[str, tuple[str, ...]] | None = None,
    worktree_base_holds_name: bool | None = None,
) -> DirectiveTable:
    """Read every directive path of ``spec`` once, for the root of ``reading``.

    ``read_against`` is the run's ``--spec-read-against`` (None, ``write-root``
    or ``parent``). ``materializes`` gives, by slice id, the outputs of an
    authoring finalization, which are read like the slice's other paths.

    ``worktree_source`` is given by a run whose write root is a worktree: the
    resolved path of the checkout the worktree was made from. Its last
    component, as it is spelled, is the directive-level name; a link to the
    checkout, or ``.``, would give another name and must not be handed in.
    The decision is then taken in both places. ``worktree_base_holds_name``
    is one fact about such a run, handed over in memory where the caller has
    it: whether the commit the worktree stands on holds an entry of the
    source checkout's name. None means that the step looks it up itself, and
    only where a refusal's message needs it. No decision, kind or record
    depends on the fact.

    A Spec that is not read raises ``DirectiveRefusal``. A ``read_against``
    that is no value and a Spec without a ``tier_dag`` raise ``ValueError``,
    an entry that cannot be inspected raises ``ScopeReadingError``, and a
    ``worktree_source`` that cannot be resolved raises ``OSError``. A
    ``DirectiveRefusal`` is itself a ``ValueError``, so a caller that tells
    the two apart catches the refusal first.
    """

    if read_against not in (None, *VALUES):
        raise ValueError(f"{OPTION} must be one of {VALUES} or absent")
    if spec.tier_dag is None:
        raise ValueError("a directive table needs a tier_dag")
    if worktree_source is None:
        return _decide(
            reading, spec, read_against, materializes,
            name=reading.name, kind=reading.leading_name, source_kind=None,
        )

    source = Path(worktree_source)
    name = source.name
    kind = _entry_kind(reading.root, name)
    source_kind = _entry_kind(source, name)

    def one(place_reading: ScopeReading, place_kind: str, other: str | None):
        try:
            return _decide(
                place_reading, spec, read_against, materializes,
                name=name, kind=place_kind, source_kind=other,
            ), None
        except DirectiveRefusal as exc:
            return None, exc

    here, here_refusal = one(reading, kind, source_kind)
    if here_refusal is not None and here_refusal.kind == KIND_INVALID:
        raise here_refusal               # the text alone; the same in both places
    there, there_refusal = one(decide_scope_reading(source), source_kind, None)

    record = dict(here_refusal.record if here_refusal is not None else here.record)
    record.pop("refused", None)
    source_record = there_refusal.record if there_refusal is not None else there.record
    record["source_write_root_votes"] = source_record.get("write_root_votes")
    record["source_parent_votes"] = source_record.get("parent_votes")
    source_decision = _decision(there, there_refusal)
    record["source_decision"] = source_decision
    here_decision = _decision(here, here_refusal)
    both = (
        f"In the checkout this worktree was made from, {str(source)!r}: "
        f"{_ENTRY_WORDS[source_kind]} named {name!r}; "
        f"{record['source_write_root_votes']} paths exist only as written, "
        f"{record['source_parent_votes']} only without the leading {name!r}; "
        f"decision: {source_decision}. In the worktree: "
        f"{_ENTRY_WORDS[kind]} named {name!r}; {record['write_root_votes']} and "
        f"{record['parent_votes']}; decision: {here_decision}."
    )
    ways = (
        "To remove the difference, make the two places equal (commit or remove "
        "what differs; an empty directory cannot be committed, a file below it "
        "can), or run without a worktree. A statement is accepted in a "
        "worktree only where both places take the same reading under it."
    )

    carries_name = (
        here_refusal.carries_name if here_refusal is not None else here.carries_name
    )
    if source_kind != kind and carries_name:
        bare = record["name_only_paths"]
        # The message says that the Spec would be read differently only where
        # it would: one place reads it and the other refuses it, or both read
        # it and their literal paths or bounds differ. Where both refuse, or
        # both read alike, the entry is merely not of one kind in the two
        # places.
        differently = (
            ", so they would be read differently"
            if (here is None) != (there is None) or (
                here is not None and there is not None
                and _literal(here) != _literal(there)
            ) else ""
        )
        # A directory of the name that exists in the worktree only can be the
        # work of the session that owns the worktree. The message says so as
        # a possibility and names no cause, and only where it can be true:
        # the commit the worktree stands on holds no entry of that name, so
        # the directory was made in the worktree after it was created. The
        # fact is handed over in memory or looked up here, on this refusal's
        # path; a lookup that fails leaves the sentence out.
        own_work = ""
        if kind == DIRECTORY and source_kind != DIRECTORY:
            base_holds = (
                worktree_base_holds_name
                if worktree_base_holds_name is not None
                else _commit_holds_entry(reading.root, name)
            )
            if base_holds is False:
                own_work = (
                    f" A directory named {name!r} that exists in the worktree "
                    "only can be this session's own earlier work."
                )
        raise DirectiveRefusal(
            KIND_WORKTREE,
            f"{record['name_led_paths']} Spec paths and {record['name_led_bounds']} "
            f"bounds begin with {name!r}"
            + (f", {bare} scope entries are {name!r} itself" if bare else "")
            + f", and the entry of that name is not of "
            f"one kind in the two places{differently}.{own_work} {both} {ways}",
            record,
            carries_name=carries_name,
        )
    if here_refusal is not None:
        message = str(here_refusal)
        if source_decision != here_decision:
            # The worktree's own refusal can rest on the session's own
            # earlier work in either direction (an entry it created, or one
            # it removed), and this function cannot tell that from a
            # difference made in the source checkout's working copy, so the
            # sentence is conditional and is said in every such refusal. It
            # is true here because the entry of the name is of one kind in
            # both places: the refusal above has ended every other run of a
            # Spec that carries the name.
            message += (
                f" The two places decide differently. {both} {ways} Where the "
                "difference is this session's own earlier work in the worktree "
                "(an entry it created or removed there), this refusal stays: a "
                "run without a worktree reaches the same refusal after the "
                "same tier."
            )
        raise DirectiveRefusal(
            here_refusal.kind, message, record, carries_name=carries_name,
        )
    assert here is not None
    if there_refusal is not None or _literal(there) != _literal(here):
        raise DirectiveRefusal(
            KIND_WORKTREE_VOTES,
            f"The Spec's paths that begin with {name!r} are read differently in "
            f"the two places. {both} {ways}",
            record,
            carries_name=carries_name,
        )
    return dataclasses.replace(here, record_json=json.dumps(record, sort_keys=True))


def _commit_holds_entry(root: Path, name: str) -> bool | None:
    """Whether the commit the worktree at ``root`` stands on holds ``name``.

    One read-only lookup: the names of the entries of the tree of ``HEAD``
    at ``root``. True or False where Git answers for the repository whose top
    is ``root``; None where the lookup fails (no Git, no repository at
    ``root``, no commit, a root that cannot be read).

    The lookup answers for the worktree's own repository only. Git looks for
    a repository in its working directory and then in every directory above
    it. A worktree whose link to its repository is gone and that lies inside
    the working tree of another repository (a directory for temporary files
    inside a checkout) would so be answered for that other repository, where
    the name is absent, and a failed lookup would read as "the commit holds
    no entry". ``GIT_CEILING_DIRECTORIES`` names the directory above
    ``root``: Git then looks at ``root`` and no further up, and the lookup
    fails there.

    The entries are listed, not probed one by one: a probe of one entry ends
    non-zero for a missing entry and for a missing repository alike, so it
    cannot tell an absent entry from a failed lookup.
    """

    top = os.path.realpath(root)
    try:
        done = subprocess.run(
            _git_argv("ls-tree", "-z", "--name-only", "HEAD"),
            cwd=top,
            env=_sanitized_git_env(GIT_CEILING_DIRECTORIES=os.path.dirname(top)),
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return os.fsencode(name) in done.stdout.split(b"\0")


def _decision(table: DirectiveTable | None, refusal: DirectiveRefusal | None) -> str:
    if refusal is not None:
        return f"refused: {refusal.kind}"
    assert table is not None
    return f"{table.read_against or 'name dropped'}, {table.basis}"


def _literal(table: DirectiveTable) -> tuple[Any, ...]:
    return (
        table.read_against,
        tuple((item.slice_id, table.scope(item.slice_id), item.materializes_paths,
               tuple(region.file for region in item.directive.mutation_regions or ()))
              for item in table.slices),
        table.bounds,
    )


def _decide(
    reading: ScopeReading,
    spec: Any,
    statement: str | None,
    materializes: Mapping[str, tuple[str, ...]] | None,
    *,
    name: str,
    kind: str,
    source_kind: str | None,
) -> DirectiveTable:
    """The decision on one tree. ``kind`` is the kind of ``<root>/<name>``."""

    root = reading.root
    in_worktree = source_kind is not None
    where = (
        f"the checkout this worktree was made from, {name!r}"
        if in_worktree else f"this write root, {name!r}"
    )
    record: dict[str, Any] = {
        "statement": statement,
        "name": name,
        "name_source": "worktree_source" if in_worktree else "write_root",
        "entry": kind,
    }
    if in_worktree:
        record["source_entry"] = source_kind

    materialized = dict(materializes or {})
    entries: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    for tier in spec.tier_dag.tiers:
        for slice_ in tier.slices:
            directive = slice_.directive
            fields = (
                ("files_in_scope", tuple(directive.files_in_scope or ())),
                ("creates_paths", tuple(directive.creates_paths or ())),
                ("mutation_regions", tuple(
                    region.file for region in (directive.mutation_regions or ())
                )),
                ("materializes_paths", tuple(materialized.get(slice_.id, ()))),
            )
            for field_name, values in fields:
                for authored in values:
                    entries.append((
                        tier.tier_id, slice_.id, field_name, authored,
                        _parts(authored, record),
                    ))

    authored_bounds: tuple[str, ...] | None = None
    for item in spec.constraints:
        if hasattr(item, "no_edits_outside"):
            authored_bounds = tuple(item.no_edits_outside or ())
            break
    bound_parts: list[tuple[str, tuple[str, ...]]] = []
    for bound in authored_bounds or ():
        if bound in {"**", ""}:
            bound_parts.append((bound, ()))
        else:
            bound_parts.append((bound, _parts(bound, record)))

    def led(parts: tuple[str, ...]) -> bool:
        return len(parts) > 1 and parts[0] == name

    led_paths = list(dict.fromkeys(
        authored for *_ignored, authored, parts in entries if led(parts)
    ))
    # A bound carries the name when its first component is the name; a bound
    # that is the name itself counts.
    led_bounds = [bound for bound, parts in bound_parts if parts and parts[0] == name]
    # A scope entry that is the name itself: read as written it is the
    # directory of that name, read against the directory above it is the write
    # root. It gives no vote (it exists under both readings or under none).
    bare_paths = list(dict.fromkeys(
        authored for *_ignored, authored, parts in entries if parts == (name,)
    ))
    record["name_led_paths"] = len(led_paths)
    record["name_only_paths"] = len(bare_paths)
    record["name_led_bounds"] = len(led_bounds)

    write_root_votes: list[str] = []
    parent_votes: list[str] = []
    if kind == DIRECTORY:
        seen: set[str] = set()
        for _tier, _slice, _field, authored, parts in entries:
            if not led(parts) or authored in seen:
                continue
            seen.add(authored)
            as_written = _lexists_below(root, parts)
            stripped = _lexists_below(root, parts[1:])
            if as_written and not stripped:
                write_root_votes.append(authored)
            if stripped and not as_written:
                parent_votes.append(authored)
    record["write_root_votes"] = len(write_root_votes)
    record["parent_votes"] = len(parent_votes)

    # The one definition of the attribute; a scope entry that is the name
    # itself counts in every checkout.
    carries_name = bool(led_paths or bare_paths or led_bounds)

    def refuse(kind_: str, message: str) -> DirectiveRefusal:
        return DirectiveRefusal(kind_, message, record, carries_name=carries_name)

    how_to_tell = (
        f"An entry that exists is not proof: file names can coincide. The "
        f"Spec's paths are relative to the directory above the checkout "
        f"{name!r} only if the Spec was planned from that directory; "
        "otherwise they are relative to the write root."
    )
    evidence = (
        f"{len(write_root_votes)} of them exist only as written"
        + (f" ({_named(write_root_votes)})" if write_root_votes else "")
        + f", {len(parent_votes)} exist only without the leading {name!r}"
        + (f" ({_named(parent_votes)})" if parent_votes else "")
        + "."
    )

    if kind == BLOCKED and led_paths:
        raise refuse(
            KIND_BLOCKED,
            f"{len(led_paths)} Spec paths begin with {name!r} ({_named(led_paths)}), "
            f"and {name!r} below {root} is not a directory. No such path can "
            "be written.",
        )

    if statement == PARENT and not led_paths:
        raise refuse(
            KIND_WITHOUT_EFFECT,
            f"{OPTION} parent was stated, and no path of the Spec begins "
            f"with the name of {where}. The statement would change no path, "
            "so the run is refused instead of reading the paths against the "
            "write root.",
        )

    # What the three refusals below read.
    #
    # ``no_entry``: the write root holds no entry of the name. ``kind`` is
    # REDUNDANT for an absent entry and for a regular file; this lookup tells
    # the two apart.
    no_entry = kind == REDUNDANT and not os.path.lexists(root / name)
    # Whether a stated write-root is refused in a checkout without a
    # directory of the name. The refusal of a stated write-root reads it,
    # and the refusal of an unstated run reads it to decide whether its
    # message may name write-root, so the two cannot drift apart. First: a
    # path that begins with the name and has more components, or a bound
    # whose first component is the name.
    write_root_refused = bool(led_paths or led_bounds)
    # (tier, slice) -> (its scope entries that are the name itself, its
    # entries that do not begin with the name).
    beside: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    # Two sentences that both refusals say of the same Spec.
    means_directory_if = (
        "The Spec means such a "
        "directory only if it was written to create or change files "
        f"below a directory {name!r} inside this checkout."
    )
    making_changes = (
        "Once that "
        f"directory exists, every Spec whose paths begin with {name!r} is "
        "read as in a checkout that holds one, and a worktree run needs "
        "a committed file below it. It also changes this checkout for "
        "every run, whatever the Spec: a request for a path that begins "
        f"with {name!r} is then read as written and no longer without "
        "that name, and the attempts a Tier has retained for any Spec "
        "in this checkout are no longer current for a later run."
    )
    # Second: where the root holds no entry of that name, a slice that holds
    # a scope entry that is the name itself beside an entry that does not
    # begin with the name (a path, a create entry, a region file or a
    # materialized path whose first component is not the name). In the
    # checkout a request below such an entry is read without the leading
    # name and can then lie inside the other entry of the slice. Alone in its
    # slices the entry is read as written, and so it is where the root holds
    # a regular file of that name: the term adds nothing there.
    for tier_id, slice_id, _field, _authored, parts in entries:
        if parts == (name,):
            beside.setdefault((tier_id, slice_id), ([], []))
    for tier_id, slice_id, _field, authored, parts in entries:
        if (tier_id, slice_id) not in beside:
            continue
        own, others = beside[(tier_id, slice_id)]
        if parts == (name,):
            if authored not in own:
                own.append(authored)
        elif parts[:1] != (name,) and authored not in others:
            others.append(authored)
    beside = {
        key: pair for key, pair in beside.items() if pair[1] and no_entry
    }
    write_root_refused = write_root_refused or bool(beside)
    # A Spec that holds a scope entry that is the name itself is not read
    # without a statement where the root holds no entry of that name. A
    # regular file of that name reads on; a link or another node is not
    # touched here. The refusal is a first decision: it stands in front of
    # the refusal of a stated write-root, of a doubled name and of the bound,
    # and where neither place of a worktree run holds an entry of that name
    # it is one refusal for both. Its message names write-root only where the
    # refusal of a stated write-root below would not refuse the Spec, and
    # offers parent in no case.
    if statement is None and bare_paths and no_entry:
        # The head counts what it lists. First the entries as the Spec
        # spells them: the count of the record (``name_only_paths``), with
        # those spellings. Then the slices that hold one, with their number.
        # (An entry spelled once in two slices is one entry in two slices.)
        in_slices = list(dict.fromkeys(
            slice_id for _tier, slice_id, _field, _authored, parts in entries
            if parts == (name,)
        ))
        message = (
            f"{len(bare_paths)} scope entries of the Spec are {name!r} itself "
            f"({_named(bare_paths)}), in {len(in_slices)} slices "
            f"({_named(in_slices)}), no {OPTION} value was stated, and {root} "
            f"holds no entry named {name!r}. Read as written such an entry is a "
            f"directory {name!r} below the write root; read against the "
            "directory above the checkout it is the write root itself; no "
            "entry on disk can tell the two apart, so the run is refused "
            "instead of reading the entry without a statement. What a "
            "request below the entry does depends on the place: in the "
            f"checkout {name!r} itself a request for a path that begins with "
            f"{name!r} is read without that name, with and without a "
            "statement, and is not written below the entry; in a worktree "
            "of that checkout the same request is written below a new "
            f"directory {name!r}. "
        )
        if not write_root_refused:
            message += (
                "Two cases. (i) The Spec was planned from this checkout and "
                f"the entry means a directory {name!r} below the write root: "
                f"{OPTION} write-root is the statement for that. (ii) The "
                "Spec was planned from the directory above the checkout and "
                "the entry only repeats the name of the checkout. Then no "
                f"{OPTION} value makes the Spec readable in this write root "
                "as written: correct the entry (name the paths it means, "
                f"each with the leading {name!r}), or run from a root that "
                "holds every path of the Spec. The entry means such a "
                "directory only if the Spec was written to create or change "
                f"files below a directory {name!r} inside this checkout."
            )
        else:
            why = []
            if led_paths or led_bounds:
                why.append(
                    f"{len(led_paths)} Spec paths and {len(led_bounds)} bounds "
                    f"begin with {name!r} ({_named([*led_paths, *led_bounds])})"
                )
            for (_tier, slice_id), (_own, others) in list(beside.items())[:_SHOWN]:
                why.append(
                    f"slice {slice_id!r} holds such an entry beside "
                    f"{len(others)} entries that do not begin with {name!r} "
                    f"({_named(others)}), inside which a request below the "
                    "entry can lie once it is read without the leading name"
                )
            message += (
                f"No {OPTION} value reads this Spec in this checkout as it "
                f"stands: {'; '.join(why)}. Two cases. (i) The Spec means a "
                f"directory {name!r} below the write root: that directory "
                "has to exist before the run. (ii) The name only repeats the "
                "name of the checkout, as in a Spec planned from the "
                "directory above it: then the Spec is corrected (each such "
                "entry names the paths it means, each with the leading "
                f"{name!r}). {means_directory_if} {making_changes}"
            )
        raise refuse(KIND_BARE_NAME, message)
    # A stated write-root drops no name. The refusal stands before the
    # doubled name, so that a doubled name under this statement reports this
    # kind.
    if kind == REDUNDANT and statement == WRITE_ROOT and write_root_refused:
        holds = (
            f"a regular file named {name!r}, not a directory"
            if os.path.lexists(root / name) else f"no directory named {name!r}"
        )
        # The message is put together from its sentences. Where the Spec
        # also holds a scope entry that is the name itself and the root holds
        # no entry of that name, an unstated run is refused as well, so the
        # message says that instead of saying that a name is dropped, and
        # ends with the sentence the unstated refusal has for the same Spec.
        carried = (
            f"{len(led_paths)} Spec paths "
            f"and {len(led_bounds)} bounds begin with {name!r} "
            f"({_named([*led_paths, *led_bounds])}), and {root} holds {holds}. "
        )
        unstated_run = (
            f"Without a statement a leading {name!r} is dropped from such an "
            "entry and it is read in the write root; under write-root no "
            "leading name is dropped, so the run is refused. "
        )
        case_two = (
            "(ii) The leading "
            f"{name!r} only repeats the name of the checkout: then write-root "
            "is not what this Spec needs here. "
        )
        added = closing = ""
        if bare_paths and no_entry:
            unstated_run = (
                "A run without a statement is refused as well, because "
                f"{len(bare_paths)} scope entries of the Spec are {name!r} "
                f"itself ({_named(bare_paths)}); under write-root no leading "
                "name is dropped, so the run is refused. "
            )
            closing = (
                f" No {OPTION} value reads this Spec in this checkout as it "
                "stands; in case (ii) the way on is a corrected Spec."
            )
        # The sentences for a slice that holds a scope entry that is the
        # name itself beside an entry that does not begin with the name.
        # They name the entry and the entries beside it, say that the root
        # holds no entry of that name and what a request below the entry
        # does in the checkout itself, worded so that the text is true in
        # both places of a worktree run. The state of the root is said once,
        # by these sentences. Where nothing of the Spec begins with the name,
        # "no leading name is dropped" is not the ground of the refusal and
        # is not said: the ground is what a request below the entry does.
        if beside:
            ground = ""
            if led_paths or led_bounds:
                carried = (
                    f"{len(led_paths)} Spec paths "
                    f"and {len(led_bounds)} bounds begin with {name!r} "
                    f"({_named([*led_paths, *led_bounds])}). "
                )
            else:
                carried = ""
                case_two = (
                    f"(ii) The name {name!r} only repeats the name of the "
                    "checkout: then write-root is not what this Spec needs "
                    "here. "
                )
                unstated_run = (
                    "A run without a statement is refused as well, because "
                    f"{len(bare_paths)} scope entries of the Spec are {name!r} "
                    f"itself ({_named(bare_paths)}). "
                )
                ground = (
                    ", so the run is refused instead of reading the entry as "
                    "written"
                )
            added = " ".join(
                f"Slice {slice_id!r} holds a scope entry that is {name!r} "
                f"itself ({_named(own)}) beside {len(others)} entries that do "
                f"not begin with {name!r} ({_named(others)})."
                for (_tier, slice_id), (own, others) in list(beside.items())[:_SHOWN]
            ) + (
                f" {root} holds no entry named {name!r}. In the checkout "
                f"{name!r} itself a request for a path below such an entry "
                f"is read without the leading {name!r}, with and without a "
                "statement, and can then lie inside another entry of the "
                f"same slice and be written at the root of that checkout{ground}. "
            )
        raise refuse(
            KIND_NO_DIRECTORY,
            f"{OPTION} write-root was stated. {carried}{added}{unstated_run}"
            "Two cases. (i) "
            f"The Spec means a directory {name!r} below the write root: that "
            f"directory has to exist before the run. {case_two}"
            f"{means_directory_if} {making_changes}{closing}",
        )

    if kind == DIRECTORY and (led_paths or bare_paths):
        ambiguous = (
            f"{len(led_paths)} Spec paths begin with {name!r}, and {root} "
            f"holds a directory of that name, so {name}/<path> can mean "
            f"<path> below that directory or <path> in the write root. Of "
            f"the paths, {evidence}"
        )
        both_arguments = (
            f"State which one the Spec means with {OPTION} write-root or "
            f"{OPTION} parent. A statement that an entry on disk "
            "contradicts is refused."
        )
        if bare_paths:
            ambiguous += (
                f" {len(bare_paths)} scope entries are {name!r} itself "
                f"({_named(bare_paths)}): read as written that is the "
                "directory of that name, read against the directory above "
                "it is the write root, and no entry on disk can tell the two "
                "apart."
            )
            # ``parent`` is refused for every Spec that holds such an entry
            # (without effect when no path begins with the name; otherwise
            # by a write-root vote, or because the entry itself does not lie
            # below the write root), so it is named and not offered.
            both_arguments = (
                f"{OPTION} write-root is the statement for a Spec planned "
                f"from this checkout, in which the entry means the directory "
                f"{name!r} below the write root; a statement that an entry on "
                "disk contradicts is refused."
            )
            if parent_votes:
                both_arguments += (
                    f" Here {len(parent_votes)} paths exist only without the "
                    f"leading {name!r}, which contradicts write-root."
                )
            both_arguments += (
                f" {OPTION} parent is refused for every Spec that holds an "
                f"entry that is {name!r} itself: read against the directory "
                "above, that entry is the write root and names no path below "
                "it. If the Spec was planned from the directory above, no "
                f"{OPTION} value makes it readable in this write root as "
                "written: correct the entry (name the paths it means, each "
                f"with the leading {name!r}), or run from a root that holds "
                "every path of the Spec."
            )
        own_work = (
            "An entry on disk can be the result of this Spec's own earlier "
            "work."
        )
        if write_root_votes and parent_votes:
            raise refuse(
                KIND_CONTRADICTORY,
                f"{ambiguous} {how_to_tell} The entries contradict each "
                f"other. {own_work} No {OPTION} value lifts this refusal.",
            )
        if statement is None and parent_votes:
            raise refuse(KIND_UNSTATED, f"{ambiguous} {how_to_tell} {both_arguments}")
        if statement is None and not write_root_votes:
            raise refuse(
                KIND_NO_EVIDENCE, f"{ambiguous} {how_to_tell} {both_arguments}",
            )
        # What is left without a statement is a write-root vote and no parent
        # vote. Evidence on disk refuses; it selects no reading and confirms
        # none, so this run is refused as well. The record keeps the votes
        # and names no reading. The message names both values and the
        # entries that voted, and recommends neither value.
        if statement is None:
            raise refuse(
                KIND_UNSTATED,
                f"{ambiguous} {how_to_tell} {own_work} {both_arguments}",
            )
        if (statement == WRITE_ROOT and parent_votes) or (
            statement == PARENT and write_root_votes
        ):
            raise refuse(
                KIND_CONTRADICTED,
                f"{OPTION} {statement} was stated. {ambiguous} {how_to_tell} "
                # True wherever this message is raised: the run without the
                # statement is refused by the same votes.
                f"{own_work} A run without the statement is refused as well.",
            )

    if kind == REDUNDANT:
        drop = True
        dropped = bool(led_paths or led_bounds)
        basis = (
            "redundant_name" if dropped
            else "stated" if statement is not None else "not_needed"
        )
        # An unstated run that dropped a name was read against neither word:
        # a stated write-root on it is refused, so the record names none.
        applied = statement if statement is not None else (None if dropped else WRITE_ROOT)
    elif kind == DIRECTORY and statement == PARENT:
        drop, applied, basis = True, PARENT, "stated"
    else:
        drop, applied = False, WRITE_ROOT
        # What reaches this line without a statement carries the name
        # nowhere, or in a bound only: a Spec with a path that begins with
        # the name, or with a scope entry that is the name itself, was
        # refused above. No reading is recorded as confirmed by the disk.
        basis = "stated" if statement is not None else "not_needed"
    record[RECORD_KEY] = applied
    record["basis"] = basis

    def not_below(what: str, authored: str) -> DirectiveRefusal:
        message = (
            f"{OPTION} parent was stated, and {what} {authored!r} does not "
            f"begin with {name!r} followed by a path. Read against the "
            "directory above the write root it does not lie below the write "
            f"root {root}, so the run is refused instead of reading it inside "
            f"the write root. {how_to_tell} If the Spec was planned from the "
            f"directory above, no {OPTION} value makes it readable in this "
            "write root as written: correct the entry (a bound is spelled "
            f"with the leading {name!r}, like the paths), or run from a root "
            "that holds every path of the Spec."
        )
        # What a run without the statement does with the same Spec in a
        # checkout without a directory of the name: it is refused as well
        # where the Spec holds a scope entry that is the name itself and the
        # root holds no entry of that name; otherwise it reads the entry
        # inside the write root, unless a doubled name or the Spec's own
        # bound refuses it. The sentence for that case is said without a
        # look at those two refusals.
        if bare_paths and no_entry:
            message += (
                f" This checkout holds no entry named {name!r}, and "
                f"{len(bare_paths)} scope entries of the Spec are {name!r} "
                "itself: a run without a statement is refused here as well, "
                "and the way on is to correct the entry."
            )
        elif kind == REDUNDANT:
            message += (
                f" This checkout holds no directory {name!r}: a run without a "
                "statement is not refused here and reads the entry inside the "
                "write root, which is not the file a Spec planned from the "
                "directory above means."
            )
        return refuse(KIND_NOT_BELOW, message)

    def literal(authored: str, parts: tuple[str, ...], what: str) -> str:
        if statement == PARENT and not led(parts):
            raise not_below(what, authored)
        if led(parts):
            if kind != DIRECTORY or not drop:
                if parts[1] == name:
                    raise refuse(
                        KIND_REPEATS,
                        f"scope path {authored!r} repeats the name {name!r}; "
                        f"write it relative to {root}",
                    )
            kept = parts[1:] if drop else parts
            if len(kept) > 1 and kept[0] == name and kept[1] == name:
                raise refuse(
                    KIND_REPEATS,
                    f"scope path {authored!r} still begins with {name!r} twice "
                    "after one leading name is dropped; no reader can carry it",
                )
            return _text(kept, authored)
        return _text(parts, authored)

    changed = 0
    by_slice: dict[tuple[str, str], dict[str, list[str]]] = {}
    for tier_id, slice_id, field_name, authored, parts in entries:
        result = literal(authored, parts, f"the path of slice {slice_id!r}")
        changed += result != authored
        by_slice.setdefault((tier_id, slice_id), {}).setdefault(
            field_name, [],
        ).append(result)
    record["paths_changed"] = changed

    bounds: tuple[str, ...] | None = None
    if authored_bounds is not None:
        resolved_bounds: list[str] = []
        for bound, parts in bound_parts:
            if not parts:
                resolved_bounds.append("**")
            elif drop and len(parts) == 1 and parts[0] == name:
                resolved_bounds.append("**")       # the write root itself
            else:
                resolved_bounds.append(literal(
                    bound, parts, "the bound of the Spec's no_edits_outside",
                ))
        bounds = tuple(resolved_bounds)

    slices: list[ResolvedSlice] = []
    for tier in spec.tier_dag.tiers:
        for slice_ in tier.slices:
            got = by_slice.get((tier.tier_id, slice_.id), {})
            directive = slice_.directive
            regions = tuple(
                dataclasses.replace(region, file=path)
                for region, path in zip(
                    directive.mutation_regions or (),
                    got.get("mutation_regions", ()),
                    strict=False,
                )
            )
            copy = dataclasses.replace(
                directive,
                files_in_scope=tuple(got.get("files_in_scope", ())),
                creates_paths=tuple(got.get("creates_paths", ())),
                mutation_regions=regions,
            )
            if bounds:
                outside = [
                    path for path in (*copy.files_in_scope, *copy.creates_paths)
                    if not _within(path, bounds)
                ]
                if outside:
                    read_as = ", ".join(
                        f"{authored!r} as {resolved!r}"
                        for authored, resolved in zip(
                            authored_bounds or (), bounds, strict=False,
                        )
                    )
                    raise refuse(
                        KIND_BOUND,
                        f"slice {slice_.id!r}: {len(outside)} scope paths lie "
                        f"outside the Spec's no_edits_outside as this run reads "
                        f"both ({_named(outside)}). The bounds were read: "
                        f"{read_as} ('**' bounds nothing; a bound that is the "
                        f"bare name {name!r} is the directory of that name when "
                        "the paths are read as written, and the write root "
                        "itself when a leading name is dropped).",
                    )
            slices.append(ResolvedSlice(
                tier_id=tier.tier_id,
                slice_id=slice_.id,
                directive=copy if copy != directive else directive,
                materializes_paths=tuple(got.get("materializes_paths", ())),
            ))

    return DirectiveTable(
        reading=reading,
        name=name,
        read_against=applied,
        basis=basis,
        statement=statement,
        slices=tuple(slices),
        bounds=bounds,
        carries_name=carries_name,
        record_json=json.dumps(record, sort_keys=True),
    )


# --------------------------------------------------------------------------
# A refusal of a plan that the same run made
# --------------------------------------------------------------------------

def plan_made_in_this_run(refusal: DirectiveRefusal, root: Path | str) -> str:
    """The text for a directive refusal of a plan that the same run made.

    It says that the plan is not kept, that a run that plans takes no
    ``parent``, and on which run a statement about a plan is given.
    """

    return (
        f"{refusal} The commands that take {OPTION} write-root or {OPTION} "
        "parent are implement, tier, breach, handoff, reconcile and "
        "architect; swebench and deepswe take neither. Two facts about this "
        f"refusal: the plan was made in this run, from {str(root)!r}, and it "
        f"is not kept; and architect refuses {OPTION} parent on a run that "
        "plans. A statement about a plan that a run makes is given on the "
        "run that makes it (with --replan where the Spec holds an earlier "
        "plan); given on a run that reuses a retained plan, it is a "
        "statement about that retained plan."
    )
