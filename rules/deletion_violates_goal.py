# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``deletion_violates_goal`` — explicit "delete X" intent must stay in scope.

The rule implements a deliberately small lexical grammar rather than trying
to understand arbitrary English. Within one sentence, line, or semicolon
clause, a deletion predicate owns following path objects until another known
action predicate begins. It also recognizes a path followed by a conventional
passive deletion predicate. Direct negation and tightly marked historical
evidence are not mutation authority; other ambiguity remains fail-closed.

If a governed path falls outside a declared
``BlastRadiusConstraint.no_edits_outside`` prefix, the rule emits a Critical
finding. Deleting a file outside that radius is a goal violation, not ordinary
scope drift.

This is intra-spec only: it does NOT verify against any external
``goal_excluded_symbol`` list (that's a P2 enhancement). If no
BlastRadiusConstraint is declared, the rule no-ops.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass

from assay import BlastRadiusConstraint, Evidence, Finding, Spec

_PATH_EXTENSION_PATTERN = (
    r"(?:cmake\.in|cmake|proto|yaml|json|cpp|cxx|hpp|tsx|usda|toml|lock|"
    r"bzl|css|xml|csv|yml|txt|ps1|usd|py|rs|md|sh|ts|go|cuh|cu|cc|c|h)"
)
_PATH_COMPONENT_PATTERN = r"[A-Za-z0-9_.+\-]+"
_PATH_TOKEN_PATTERN = (
    rf"(?:[A-Za-z]:[\\/]?|[\\/]+)?"
    rf"(?:{_PATH_COMPONENT_PATTERN}[\\/])*"
    rf"{_PATH_COMPONENT_PATTERN}\.{_PATH_EXTENSION_PATTERN}"
)
_PATH_RE = re.compile(
    rf"(?<![A-Za-z0-9_.+:/\\-])(?:"
    rf"`(?P<backtick>{_PATH_TOKEN_PATTERN})`"
    rf'|"(?P<double>{_PATH_TOKEN_PATTERN})"'
    rf"|'(?P<single>{_PATH_TOKEN_PATTERN})'"
    rf"|(?P<bare>{_PATH_TOKEN_PATTERN})"
    rf")(?=$|[\s,;:!?\]}})]|\.(?=\s|$))",
    re.IGNORECASE,
)
_QUOTED_PATHISH_RE = re.compile(
    r"`(?=[^`\s\r\n]*[./\\])[^`\s\r\n]+`"
    r'|"(?=[^"\s\r\n]*[./\\])[^"\s\r\n]+"'
    r"|(?<![A-Za-z0-9])'(?=[^'\s\r\n]*[./\\])[^'\s\r\n]+'"
    r"(?![A-Za-z0-9])"
)

_DELETION_PREDICATE_PATTERN = (
    r"(?:"
    r"delet(?:e|es|ed|ing)"
    r"|remov(?:e|es|ed|ing)"
    r"|drop(?:s|ped|ping)?"
    r"|destroy(?:s|ed|ing)?"
    r"|purg(?:e|es|ed|ing)"
    r"|rip(?:s|ped|ping)?\s+out"
    r"|tear(?:s|ing)?\s+out"
    r"|tore\s+out"
    r"|torn\s+out"
    r")"
)
_OTHER_ACTION_PREDICATE_PATTERN = (
    r"(?:"
    r"preserv(?:e|es|ed|ing)"
    r"|retain(?:s|ed|ing)?"
    r"|keep(?:s|ing)?|kept"
    r"|inspect(?:s|ed|ing)?"
    r"|read(?:s|ing)?"
    r"|check(?:s|ed|ing)?"
    r"|verif(?:y|ies|ied|ying)"
    r"|document(?:s|ed|ing)?"
    r"|describ(?:e|es|ed|ing)"
    r"|edit(?:s|ed|ing)?"
    r"|modif(?:y|ies|ied|ying)"
    r"|add(?:s|ed|ing)?"
    r"|creat(?:e|es|ed|ing)"
    r"|renam(?:e|es|ed|ing)"
    r"|mov(?:e|es|ed|ing)"
    r"|cop(?:y|ies|ied|ying)"
    r"|updat(?:e|es|ed|ing)"
    r"|writ(?:e|es|ing|ten)|wrote"
    r"|treat(?:s|ed|ing)?"
    r")"
)
_ACTION_PREDICATE_RE = re.compile(
    rf"\b(?:(?P<deletion>{_DELETION_PREDICATE_PATTERN})"
    rf"|(?P<other>{_OTHER_ACTION_PREDICATE_PATTERN}))\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?:[;!?](?=\s|$)|\.(?=\s|$)|\n+)",
)
_NEGATION_MODIFIER_PATTERN = (
    r"(?:again|ever|(?!(?:only|merely)\b)[A-Za-z-]+ly)"
)
_NEGATION_AUXILIARY_PATTERN = (
    r"(?:do|does|did|must|shall|should|will|would|can|could|may|"
    r"is|are|was|were)"
)
_NEGATION_CONTRACTION_PATTERN = (
    r"(?:don't|doesn't|didn't|mustn't|shalln't|shouldn't|won't|wouldn't|"
    r"can't|cannot|couldn't|isn't|aren't|wasn't|weren't)"
)
_ADDITIVE_MODIFIER_PATTERN = r"(?:only|merely|just)"
_BE_OR_BEING_PATTERN = r"(?:be|being)"
_PASSIVE_AUXILIARY_PATTERN = r"(?:(?:to\s+)?be|being)"
_DIRECT_NEGATION_CUE_PATTERN = (
    rf"(?:"
    rf"\b{_NEGATION_AUXILIARY_PATTERN}\s+not"
    rf"|\b{_NEGATION_CONTRACTION_PATTERN}"
    rf"|\bnever|\bnot|\bwithout"
    rf")"
)
_OUTER_NEGATION_CUE_RE = re.compile(
    rf"(?:"
    rf"\b{_NEGATION_AUXILIARY_PATTERN}\s+not"
    rf"|\b{_NEGATION_CONTRACTION_PATTERN}"
    rf"|\bnever"
    rf")",
    re.IGNORECASE,
)
_ADJACENT_BARE_NOT_RE = re.compile(
    rf"\bnot(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s*$",
    re.IGNORECASE,
)
_DIRECT_NEGATION_RE = re.compile(
    rf"{_DIRECT_NEGATION_CUE_PATTERN}"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+{_PASSIVE_AUXILIARY_PATTERN})?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s*$",
    re.IGNORECASE,
)
_ADDITIVE_NEGATION_CUE_RE = re.compile(
    rf"(?:\bnot|\b{_NEGATION_CONTRACTION_PATTERN})\s+"
    rf"{_ADDITIVE_MODIFIER_PATTERN}\b",
    re.IGNORECASE,
)
_ADDITIVE_NOT_RE = re.compile(
    rf"(?:\bnot|\b{_NEGATION_CONTRACTION_PATTERN})\s+"
    rf"{_ADDITIVE_MODIFIER_PATTERN}"
    rf"(?:\s+{_PASSIVE_AUXILIARY_PATTERN})?\s*$",
    re.IGNORECASE,
)
_AVOIDANCE_RE = re.compile(
    rf"\bavoid(?:s|ed|ing)?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+being)?\s*$",
    re.IGNORECASE,
)
_NEGATED_AVOIDANCE_RE = re.compile(
    r"(?:"
    r"\b(?:do|does|did|must|shall|should|will|would|can|could|may)\s+"
    r"(?:not|never)"
    r"|\b(?:don't|doesn't|didn't|mustn't|shouldn't|won't|wouldn't|"
    r"can't|cannot|couldn't)"
    r"|\bnever"
    r"|\bwithout"
    rf")(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s+"
    rf"avoid(?:s|ed|ing)?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+being)?\s*$",
    re.IGNORECASE,
)
_PASSIVE_AVOIDANCE_LINK_RE = re.compile(
    rf"\s*(?:"
    rf"(?:do|does|did|must|shall|should|will|would|can|could|may)\s+"
    rf"(?:(?:not|never)\s+)?"
    rf"|{_NEGATION_CONTRACTION_PATTERN}\s+"
    rf"|never\s+"
    rf")?"
    rf"(?:{_NEGATION_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"avoid(?:s|ed|ing)?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s+being"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s*",
    re.IGNORECASE,
)
_AVOIDED_NEGATION_RE = re.compile(
    rf"\bavoid(?:s|ed|ing)?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s+"
    rf"(?:not|never)"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+{_PASSIVE_AUXILIARY_PATTERN})?\s*$",
    re.IGNORECASE,
)
_CONDITIONAL_EXCEPTION_RE = re.compile(
    r"\b(?:unless|only\s+if|except\s+when)\b",
    re.IGNORECASE,
)
_PASSIVE_MODIFIER_PATTERN = (
    rf"(?:again|already|ever|{_ADDITIVE_MODIFIER_PATTERN}|[A-Za-z-]+ly)"
)
_PASSIVE_LINK_RE = re.compile(
    rf"\s*(?:"
    rf"(?:(?:must|shall|should|will|would|can|could|may)\s+)"
    rf"(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:(?:not|never)\s+)?(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:be|being)"
    rf"|(?:(?:is|are|was|were)\s+)"
    rf"(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:(?:not|never)\s+)?(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:{_PASSIVE_AUXILIARY_PATTERN}|"
    rf"(?:scheduled|required|expected)\s+to\s+be)?"
    rf"|(?:(?:needs?|has|have)\s+)"
    rf"(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:(?:not|never)\s+)?to\s+be"
    rf"|{_NEGATION_CONTRACTION_PATTERN}\s+"
    rf"(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:{_PASSIVE_AUXILIARY_PATTERN})?"
    rf"|(?:(?:not|never)\s+)?(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"{_BE_OR_BEING_PATTERN}"
    rf")?(?:\s+[A-Za-z-]+ly){{0,2}}\s*",
    re.IGNORECASE,
)
_PASSIVE_DOUBLE_NEGATION_RE = re.compile(
    rf"\s*(?:"
    rf"{_NEGATION_AUXILIARY_PATTERN}\s+(?:not|never)"
    rf"|{_NEGATION_CONTRACTION_PATTERN}"
    rf"|(?:not|never)"
    rf")"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s+"
    rf"(?:not|never)"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+{_PASSIVE_AUXILIARY_PATTERN})?"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s*",
    re.IGNORECASE,
)
_OBJECT_COORDINATOR_RE = re.compile(
    r"\s*(?:,\s*)?(?:and|or)?\s*",
    re.IGNORECASE,
)
_PREDICATE_COORDINATOR_RE = re.compile(
    r"\s*(?:"
    r",\s*(?:(?:and|or|but)\s+)?"
    r"|(?:and|or|but)\s+"
    r"|rather\s+than\s+"
    r"|without\s+"
    r")",
    re.IGNORECASE,
)
_PREDICATE_LINK_MODIFIER_PATTERN = (
    rf"(?:then|also|{_ADDITIVE_MODIFIER_PATTERN}|"
    rf"{_NEGATION_MODIFIER_PATTERN})"
)
_PREDICATE_LINK_MODIFIERS_RE = re.compile(
    rf"{_PREDICATE_LINK_MODIFIER_PATTERN}"
    rf"(?:\s+{_PREDICATE_LINK_MODIFIER_PATTERN})?",
    re.IGNORECASE,
)
_SCOPE_COORDINATOR_RE = re.compile(
    r"\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)(?:also\s+)?",
    re.IGNORECASE,
)
_WITHOUT_SCOPE_RE = re.compile(
    rf"\s*without"
    rf"(?:\s+{_NEGATION_MODIFIER_PATTERN}){{0,2}}\s*",
    re.IGNORECASE,
)
_PAST_DELETION_FORMS = frozenset(
    {
        "deleted",
        "removed",
        "dropped",
        "destroyed",
        "purged",
        "ripped out",
        "tore out",
        "torn out",
    }
)
_PAST_PROGRESSIVE_AUXILIARY_RE = re.compile(
    rf"\b(?:"
    rf"(?:was|were)(?:\s+{_PASSIVE_MODIFIER_PATTERN}){{0,2}}"
    rf"(?:\s+(?:not|never))?"
    rf"|wasn't|weren't"
    rf")(?:\s+{_PASSIVE_MODIFIER_PATTERN}){{0,2}}\s*$",
    re.IGNORECASE,
)
_LEADING_TEMPORAL_HISTORY_RE = re.compile(
    r"^\s*(?:already|formerly|historically|previously),?\s+",
    re.IGNORECASE,
)
_ANY_TEMPORAL_HISTORY_RE = re.compile(
    r"\b(?:already|formerly|historically|previously)\b",
    re.IGNORECASE,
)
_TRAILING_TEMPORAL_HISTORY_RE = re.compile(
    r"\s+(?:already|formerly|historically|previously)\b",
    re.IGNORECASE,
)
_CONTRAST_BOUNDARY_RE = re.compile(
    r",\s*(?:but|while)\s+",
    re.IGNORECASE,
)
_PRESENT_PASSIVE_DIRECTIVE_PATTERN = (
    rf"(?:"
    rf"(?:is|are)\s+(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}(?:"
    rf"(?:not|never)\b"
    rf"|(?:(?:not|never)\s+)?"
    rf"(?:{_PASSIVE_MODIFIER_PATTERN}\s+){{0,2}}"
    rf"(?:{_PASSIVE_AUXILIARY_PATTERN}|"
    rf"(?:scheduled|required|expected)\s+to\s+be)"
    rf")"
    rf"|(?:isn't|aren't)\b"
    rf")"
)
_DIRECTIVE_NEGATION_CONTRACTION_PATTERN = (
    r"(?:don't|doesn't|mustn't|shalln't|shouldn't|won't|wouldn't|"
    r"can't|cannot|couldn't|isn't|aren't)"
)
_DIRECTIVE_MODAL_RE = re.compile(
    rf"\b(?:{_PRESENT_PASSIVE_DIRECTIVE_PATTERN}|"
    rf"{_DIRECTIVE_NEGATION_CONTRACTION_PATTERN}|"
    rf"must|shall|should|will|would|can|could|may|needs?|has|have|"
    rf"scheduled|required|expected)\b",
    re.IGNORECASE,
)
_HISTORICAL_EVIDENCE_OBJECT_RE = re.compile(
    r"\s+(?:historical|legacy|prior|previous)\s+"
    r"(?:evidence|record|violation)\s+(?:at|from|in)\s*",
    re.IGNORECASE,
)
_HISTORICAL_EVIDENCE_SUFFIX_RE = re.compile(
    r"[^;!?\n]{0,64}\b(?:absent|evidence|read[- ]only)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _HistoryIndex:
    temporal_matches: tuple[re.Match[str], ...]
    temporal_starts: tuple[int, ...]
    directive_modal_starts: tuple[int, ...]
    contrast_starts: tuple[int, ...]
    leading_history_end: int | None


def _path_value(match: re.Match[str]) -> str:
    value = (
        match.group("backtick")
        or match.group("double")
        or match.group("single")
        or match.group("bare")
    )
    assert value is not None
    return value


def _normalize_repo_path(path: str) -> str | None:
    """Normalize benign relative segments; reject absolute or escaping paths."""

    value = path.replace("\\", "/")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return None
    parts: list[str] = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            return None
        parts.append(part)
    if not parts:
        return None
    return "/".join(parts)


def _strip_project_prefix(path: str, project_prefix: str) -> str:
    normalized_project = _normalize_repo_path(project_prefix)
    if normalized_project is None:
        return path
    if path == normalized_project:
        return ""
    if path.startswith(normalized_project + "/"):
        return path[len(normalized_project) + 1 :]
    return path


def _within_any_prefix(
    path: str,
    prefixes: tuple[str, ...],
    project_prefix: str,
) -> bool:
    normalized_path = _normalize_repo_path(path)
    if normalized_path is None:
        return False
    normalized_path = _strip_project_prefix(normalized_path, project_prefix)
    for prefix in prefixes:
        if prefix in ("**", ""):
            return True
        normalized_prefix = _normalize_repo_path(prefix)
        if normalized_prefix is None:
            continue
        normalized_prefix = _strip_project_prefix(normalized_prefix, project_prefix)
        if normalized_prefix == "":
            return True
        if normalized_path == normalized_prefix or normalized_path.startswith(
            normalized_prefix.rstrip("/") + "/"
        ):
            return True
    return False


def _clause_spans(intent: str) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(intent):
        spans.append((start, boundary.end()))
        start = boundary.end()
    if start < len(intent):
        spans.append((start, len(intent)))
    return tuple(spans)


def _build_history_index(clause: str) -> _HistoryIndex:
    temporal_matches = tuple(_ANY_TEMPORAL_HISTORY_RE.finditer(clause))
    leading_history = _LEADING_TEMPORAL_HISTORY_RE.match(clause)
    return _HistoryIndex(
        temporal_matches=temporal_matches,
        temporal_starts=tuple(match.start() for match in temporal_matches),
        directive_modal_starts=tuple(
            match.start() for match in _DIRECTIVE_MODAL_RE.finditer(clause)
        ),
        contrast_starts=tuple(
            match.start() for match in _CONTRAST_BOUNDARY_RE.finditer(clause)
        ),
        leading_history_end=(
            None if leading_history is None else leading_history.end()
        ),
    )


def _has_position_between(positions: tuple[int, ...], start: int, end: int) -> bool:
    index = bisect_left(positions, start)
    return index < len(positions) and positions[index] < end


def _is_progressive_action(normalized_action: str) -> bool:
    return normalized_action.endswith("ing") or normalized_action.endswith(
        "ing out"
    )


def _leading_history_applies(
    history: _HistoryIndex,
    verb_start: int,
) -> bool:
    leading_end = history.leading_history_end
    return bool(
        leading_end is not None
        and not _has_position_between(history.contrast_starts, leading_end, verb_start)
    )


def _has_non_additive_negation(text: str) -> bool:
    if _ADJACENT_BARE_NOT_RE.search(text) is not None:
        return True
    additive_cues = tuple(_ADDITIVE_NEGATION_CUE_RE.finditer(text))
    additive_index = 0
    for cue in _OUTER_NEGATION_CUE_RE.finditer(text):
        while (
            additive_index < len(additive_cues)
            and additive_cues[additive_index].end() <= cue.start()
        ):
            additive_index += 1
        if (
            additive_index < len(additive_cues)
            and additive_cues[additive_index].start()
            <= cue.start()
            < additive_cues[additive_index].end()
        ):
            continue
        return True
    return False


def _verb_is_directly_negated(
    clause: str,
    verb: re.Match[str],
    previous_action: re.Match[str] | None,
) -> bool:
    start = 0 if previous_action is None else previous_action.end()
    prefix = clause[start : verb.start()]
    if (
        _NEGATED_AVOIDANCE_RE.search(prefix) is not None
        or _AVOIDED_NEGATION_RE.search(prefix) is not None
    ):
        return False
    if _ADDITIVE_NOT_RE.search(prefix) is not None:
        return False
    direct_negation = _DIRECT_NEGATION_RE.search(prefix)
    avoidance = _AVOIDANCE_RE.search(prefix)
    trailing_negation = direct_negation or avoidance
    if trailing_negation is None:
        return False
    if _has_non_additive_negation(prefix[: trailing_negation.start()]):
        return False
    return True


def _verb_is_historical_for_path(
    clause: str,
    verb: re.Match[str],
    path: re.Match[str],
    history: _HistoryIndex,
    past_progressive: bool,
) -> bool:
    normalized_verb = " ".join(verb.group(0).lower().split())
    historical_form = normalized_verb in _PAST_DELETION_FORMS or (
        _is_progressive_action(normalized_verb) and past_progressive
    )
    if not historical_form:
        return False

    if verb.end() <= path.start():
        temporal_index = bisect_left(history.temporal_starts, verb.start()) - 1
        if temporal_index >= 0:
            temporal = history.temporal_matches[temporal_index]
            temporal_gap = verb.start() - temporal.end()
            if (
                temporal_gap <= 32
                and temporal_gap >= 1
                and clause[temporal.end() : verb.start()].isspace()
            ):
                return True
        if _leading_history_applies(history, verb.start()):
            return True
        return bool(
            _HISTORICAL_EVIDENCE_OBJECT_RE.fullmatch(
                clause, verb.end(), path.start()
            )
            and _HISTORICAL_EVIDENCE_SUFFIX_RE.match(
                clause,
                path.end(),
                min(len(clause), path.end() + 65),
            )
        )

    if _has_position_between(
        history.directive_modal_starts, path.end(), verb.start()
    ):
        return False
    if _leading_history_applies(history, verb.start()):
        return True
    return bool(
        _has_position_between(history.temporal_starts, path.end(), verb.start())
        or _TRAILING_TEMPORAL_HISTORY_RE.match(
            clause,
            verb.end(),
            min(len(clause), verb.end() + 32),
        )
    )


def _action_predicates_outside_paths(
    clause: str,
    paths: tuple[re.Match[str], ...],
) -> tuple[re.Match[str], ...]:
    """Return action predicates, excluding verb-shaped path components."""

    actions: list[re.Match[str]] = []
    protected_spans = tuple(
        sorted(
            (*paths, *_QUOTED_PATHISH_RE.finditer(clause)),
            key=lambda item: item.start(),
        )
    )
    path_index = 0
    for action in _ACTION_PREDICATE_RE.finditer(clause):
        while (
            path_index < len(protected_spans)
            and protected_spans[path_index].end() <= action.start()
        ):
            path_index += 1
        if (
            path_index < len(protected_spans)
            and protected_spans[path_index].start()
            <= action.start()
            < protected_spans[path_index].end()
        ):
            continue
        actions.append(action)
    return tuple(actions)


def _predicate_bridge_shares_object(bridge: str) -> bool:
    """Recognize a coordinator plus one already-supported bounded linker."""

    coordinator = _PREDICATE_COORDINATOR_RE.match(bridge)
    if coordinator is None:
        return False
    linker = bridge[coordinator.end() :]
    stripped_linker = linker.strip()
    left_trimmed_linker = linker.lstrip()
    return bool(
        not stripped_linker
        or _PREDICATE_LINK_MODIFIERS_RE.fullmatch(stripped_linker)
        or _PASSIVE_LINK_RE.fullmatch(linker)
        or _PASSIVE_DOUBLE_NEGATION_RE.fullmatch(linker)
        or _PASSIVE_AVOIDANCE_LINK_RE.fullmatch(linker)
        or _DIRECT_NEGATION_RE.fullmatch(left_trimmed_linker)
        or _ADDITIVE_NOT_RE.fullmatch(left_trimmed_linker)
        or _AVOIDANCE_RE.fullmatch(left_trimmed_linker)
        or _NEGATED_AVOIDANCE_RE.fullmatch(left_trimmed_linker)
        or _AVOIDED_NEGATION_RE.fullmatch(left_trimmed_linker)
    )


def _coordinated_action_metadata(
    clause: str,
    actions: tuple[re.Match[str], ...],
) -> tuple[tuple[bool, ...], tuple[bool, ...], tuple[bool, ...]]:
    """Precompute object-sharing and finite scope links once per action pair."""

    if not actions:
        return (), (), ()
    shares_object_with_previous = [False] * len(actions)
    inherits_scope_from_previous = [False] * len(actions)
    negates_scope_from_previous = [False] * len(actions)
    for index in range(1, len(actions)):
        bridge = clause[actions[index - 1].end() : actions[index].start()]
        shares_object_with_previous[index] = _predicate_bridge_shares_object(
            bridge
        )
        inherits_scope_from_previous[index] = (
            _SCOPE_COORDINATOR_RE.fullmatch(bridge) is not None
        )
        negates_scope_from_previous[index] = (
            _WITHOUT_SCOPE_RE.fullmatch(bridge) is not None
        )

    return (
        tuple(shares_object_with_previous),
        tuple(inherits_scope_from_previous),
        tuple(negates_scope_from_previous),
    )


def _deletion_negation_states(
    clause: str,
    actions: tuple[re.Match[str], ...],
    inherits_scope_from_previous: tuple[bool, ...],
    negates_scope_from_previous: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Classify deletion polarity once, including pure deletion lists."""

    states = [False] * len(actions)
    direct_states = tuple(
        _verb_is_directly_negated(
            clause,
            action,
            actions[index - 1] if index else None,
        )
        for index, action in enumerate(actions)
    )
    for index, verb in enumerate(actions):
        if verb.lastgroup != "deletion":
            continue
        previous_action = actions[index - 1] if index else None
        negated = direct_states[index]
        if (
            previous_action is not None
            and previous_action.lastgroup == "deletion"
            and inherits_scope_from_previous[index]
        ):
            negated = negated or states[index - 1]
        if (
            previous_action is not None
            and negates_scope_from_previous[index]
            and direct_states[index - 1]
        ):
            negated = False
        if (
            direct_states[index]
            and index + 1 < len(actions)
            and negates_scope_from_previous[index + 1]
        ):
            negated = False
        states[index] = negated
    return tuple(states)


def _past_progressive_states(
    clause: str,
    actions: tuple[re.Match[str], ...],
    inherits_scope_from_previous: tuple[bool, ...],
) -> tuple[bool, ...]:
    """Classify past-progressive tense once, including coordinated sharing."""

    states = [False] * len(actions)
    for index, action in enumerate(actions):
        normalized_action = " ".join(action.group(0).lower().split())
        if not _is_progressive_action(normalized_action):
            continue
        previous_action = actions[index - 1] if index else None
        search_start = 0 if previous_action is None else previous_action.end()
        explicit_past = bool(
            _PAST_PROGRESSIVE_AUXILIARY_RE.search(
                clause,
                search_start,
                action.start(),
            )
        )
        states[index] = explicit_past or bool(
            previous_action is not None
            and inherits_scope_from_previous[index]
            and states[index - 1]
        )
    return tuple(states)


def _run_has_deletion_authority(
    clause: str,
    actions: tuple[re.Match[str], ...],
    negated_deletions: tuple[bool, ...],
    past_progressive_actions: tuple[bool, ...],
    history: _HistoryIndex,
    has_exception: bool,
    run_start: int,
    run_end: int,
    representative_path: re.Match[str],
) -> bool:
    """Decide one shared-object run without multiplying actions by paths."""

    for index in range(run_start, run_end + 1):
        verb = actions[index]
        if verb.lastgroup != "deletion":
            continue
        if negated_deletions[index] and not has_exception:
            continue
        if not _verb_is_historical_for_path(
            clause,
            verb,
            representative_path,
            history,
            past_progressive_actions[index],
        ):
            return True
    return False


def _passive_path_bounds(
    clause: str,
    paths: tuple[re.Match[str], ...],
    path_starts: tuple[int, ...],
    previous_action_end: int,
    action_start: int,
) -> tuple[int, int]:
    """Return the coordinated passive subject immediately before a run."""

    passive_start = bisect_left(path_starts, previous_action_end)
    passive_end = bisect_left(path_starts, action_start)
    if passive_start >= passive_end:
        return passive_end, passive_end
    nearest = paths[passive_end - 1]
    passive_link = clause[nearest.end() : action_start]
    if nearest.end() > action_start or not (
        _PASSIVE_LINK_RE.fullmatch(passive_link)
        or _PASSIVE_DOUBLE_NEGATION_RE.fullmatch(passive_link)
        or _PASSIVE_AVOIDANCE_LINK_RE.fullmatch(passive_link)
    ):
        return passive_end, passive_end
    cursor = passive_end - 1
    while cursor > passive_start:
        preceding = paths[cursor - 1]
        following = paths[cursor]
        if _OBJECT_COORDINATOR_RE.fullmatch(
            clause[preceding.end() : following.start()]
        ) is None:
            break
        cursor -= 1
    return cursor, passive_end


def _append_deletion(
    out: list[tuple[str, str]],
    seen: set[str],
    *,
    intent: str,
    clause_start: int,
    clause_end: int,
    path: re.Match[str],
) -> None:
    value = _path_value(path)
    if value in seen:
        return
    seen.add(value)
    excerpt_start = max(clause_start, clause_start + path.start() - 48)
    excerpt_end = min(clause_end, clause_start + path.end() + 48)
    out.append((value, intent[excerpt_start:excerpt_end]))


def _intent_deletes(intent: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for clause_start, clause_end in _clause_spans(intent):
        clause = intent[clause_start:clause_end]
        paths = tuple(_PATH_RE.finditer(clause))
        actions = _action_predicates_outside_paths(clause, paths)
        path_starts = tuple(path.start() for path in paths)
        history = _build_history_index(clause)
        has_exception = _CONDITIONAL_EXCEPTION_RE.search(clause) is not None
        (
            shares_object_with_previous,
            inherits_scope_from_previous,
            negates_scope_from_previous,
        ) = _coordinated_action_metadata(clause, actions)
        negated_deletions = _deletion_negation_states(
            clause,
            actions,
            inherits_scope_from_previous,
            negates_scope_from_previous,
        )
        past_progressive_actions = _past_progressive_states(
            clause,
            actions,
            inherits_scope_from_previous,
        )
        run_start = 0
        while run_start < len(actions):
            run_end = run_start
            while (
                run_end + 1 < len(actions)
                and shares_object_with_previous[run_end + 1]
            ):
                run_end += 1
            next_action_start = (
                actions[run_end + 1].start()
                if run_end + 1 < len(actions)
                else len(clause)
            )
            active_start = bisect_left(path_starts, actions[run_end].end())
            active_end = bisect_left(path_starts, next_action_start)
            previous_action_end = (
                0 if run_start == 0 else actions[run_start - 1].end()
            )
            passive_start, passive_end = _passive_path_bounds(
                clause,
                paths,
                path_starts,
                previous_action_end,
                actions[run_start].start(),
            )

            governed_ranges = ((active_start, active_end), (passive_start, passive_end))
            for range_start, range_end in governed_ranges:
                if range_start >= range_end or not _run_has_deletion_authority(
                    clause,
                    actions,
                    negated_deletions,
                    past_progressive_actions,
                    history,
                    has_exception,
                    run_start,
                    run_end,
                    paths[range_end - 1],
                ):
                    continue
                for path in paths[range_start:range_end]:
                    _append_deletion(
                        out,
                        seen,
                        intent=intent,
                        clause_start=clause_start,
                        clause_end=clause_end,
                        path=path,
                    )
            run_start = run_end + 1
    return out


def deletion_violates_goal(spec: Spec) -> tuple[Finding, ...]:
    if spec.tier_dag is None:
        return ()
    constraint: BlastRadiusConstraint | None = None
    for c in spec.constraints:
        if isinstance(c, BlastRadiusConstraint):
            constraint = c
            break
    if constraint is None or not constraint.no_edits_outside:
        return ()

    project_context = getattr(spec, "project_context", None)
    project_prefix = (
        (
            getattr(project_context, "submodule_path", None)
            or getattr(project_context, "project", None)
            or ""
        )
        if project_context is not None
        else ""
    )

    findings: list[Finding] = []
    for tier in spec.tier_dag.tiers:
        for s in tier.slices:
            for path, excerpt in _intent_deletes(s.directive.intent or ""):
                if _within_any_prefix(
                    path,
                    constraint.no_edits_outside,
                    project_prefix,
                ):
                    continue
                findings.append(
                    Finding(
                        id=f"delete_outside_radius_{s.id}_{path}",
                        severity="Critical",
                        category="deletion",
                        evidence=(
                            Evidence(
                                kind="intent_excerpt",
                                location=(
                                    f"tier_dag[{tier.tier_id}]"
                                    f".slices[{s.id}].directive.intent"
                                ),
                                excerpt=excerpt,
                            ),
                            Evidence(
                                kind="constraint",
                                location="spec.constraints.blast_radius",
                                excerpt=str(constraint.no_edits_outside),
                            ),
                        ),
                        suggested_action=(
                            f"{path!r} is outside the declared blast "
                            f"radius — extend no_edits_outside or remove "
                            f"the deletion from intent"
                        ),
                        source="deterministic",
                    )
                )
    return tuple(findings)
