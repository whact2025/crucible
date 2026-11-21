# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Envelopes: the reply shapes a vendor dialect may send, with their provenance.

A hand-written renderer would be a provider simulator. So a dialect renders a
script step by filling *slots* (text, tool name and arguments, token counts,
ids, a timestamp, the echoed model id) in an envelope: one file per provider
and behaviour class under ``envelopes/<dialect>/``. Everything else in a reply
is the envelope's, and the envelope says where its shape comes from.

A slot is a node ``{slot: <name>}`` that stands for a whole value. Where the
wire keys a mapping by a run-time value (usage per served model), the key is
written ``${slot:<name>}`` and is filled with a string. Nothing else in an
envelope can change.

Three grades, strongest first:

- ``recorded``: captured from the real provider through the adapter's own
  route and sanitised (date, model recorded, hash of the sanitised bytes);
- ``documented``: vendor documentation, or a field shape noted in a live run
  without a stored body (each source with its date and what it states);
- ``constructed``: built to satisfy the pinned SDK's types (type path and SDK
  version) or for a stated reason. It proves that the reply parses, not that
  the provider sends it.

The weakest envelope a cell uses is the cell's grade. There is no separate
load-time parse: a cell that uses an envelope parses it through the real SDK.
"""

from __future__ import annotations

import copy
import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml  # type: ignore[import-untyped]

from ..drivers.base import MatrixSetupError

ENVELOPE_SCHEMA = "crucible.backend-matrix.envelope.v1"
ROOT = Path(__file__).resolve().parent
GRADES: tuple[str, ...] = ("recorded", "documented", "constructed")
# The only values a renderer may put into an envelope.
SLOTS: frozenset[str] = frozenset(
    {
        "text",
        "tool_name",
        "tool_arguments",
        "call_id",
        "response_id",
        "created",
        "served_model",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        # Frames of a session-oriented wire (a vendor command-line tool that
        # owns the tool loop): the session and frame ids, a tool call's
        # decoded arguments, the tool result it echoes, and the session
        # totals of its terminal frame.
        "session_id",
        "frame_id",
        "tool_input",
        "tool_result",
        "tool_is_error",
        "turns",
        "cost_usd",
        "structured_output",
        # A stream of typed output items (a Responses route): the id of an
        # output item, and the reasoning item that precedes a tool call with
        # its id, its text and its share of the output tokens.
        "item_id",
        "reasoning_item_id",
        "reasoning_text",
        "reasoning_tokens",
        # Notifications of a thread-oriented wire (an app-server that owns the
        # tool loop): the thread and turn every notification names, the server
        # a tool call went to, and the counters of the thread so far beside
        # those of one response.
        "thread_id",
        "turn_id",
        "tool_server",
        "cumulative_input_tokens",
        "cumulative_output_tokens",
        "cumulative_total_tokens",
    }
)
# A mapping key that is a slot: the whole key, filled with a string.
_KEY_SLOT = re.compile(r"\$\{slot:([a-z][a-z0-9_]*)\}")
_FIELDS = {
    "schema_version",
    "dialect",
    "provider",
    "behaviour",
    "provenance",
    "status",
    "body",
}
_SOURCE_FIELDS = {"source", "retrieved", "states"}
# What must never be stored in an envelope, recorded or not.
_FORBIDDEN: tuple[tuple[str, str], ...] = (
    ("an authorization header", r"(?i)\bauthorization\b\s*[:=]"),
    ("a bearer credential", r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    ("an API key", r"\b(sk|nvapi|ak|pk)-[A-Za-z0-9_-]{12,}"),
    ("a key assignment", r"(?i)\b(api[_-]?key|secret|token|password)\b\s*[:=]\s*\S"),
    # Letters and digits mixed, so a long identifier or path segment is not
    # one; a digest written as "sha256:<hex>" is evidence, not a secret.
    (
        "a long opaque token",
        r"(?<![A-Za-z0-9:+/=_-])(?=[A-Za-z+/=_-]*\d)(?=[\d+/=_-]*[A-Za-z])"
        r"[A-Za-z0-9+/=_-]{40,}",
    ),
    ("an account id", r"(?i)\b(org|acct|account|user|proj|project|cus)[-_][A-Za-z0-9]{8,}"),
    ("an e-mail address", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+"),
)


def hygiene_problems(text: str) -> list[str]:
    """What ``text`` holds that an envelope must never store."""

    return [
        f"{label} ({match.group(0)[:24]!r})"
        for label, pattern in _FORBIDDEN
        for match in [re.search(pattern, text)]
        if match is not None
    ]


@dataclass(frozen=True)
class Envelope:
    """One reply shape of one provider, with where it comes from."""

    dialect: str
    provider: str
    behaviour: str
    grade: str
    provenance: Mapping[str, Any]
    status: int
    body: Any

    @property
    def name(self) -> str:
        return f"{self.dialect}/{self.provider}.{self.behaviour}"

    def slots(self) -> frozenset[str]:
        return frozenset(_slots(self.body))

    def fill(self, values: Mapping[str, Any]) -> Any:
        """The body with exactly its slots filled; nothing else can change."""

        wanted = self.slots()
        if set(values) != wanted:
            raise MatrixSetupError(
                f"envelope {self.name} has the slots {sorted(wanted)}; the "
                f"renderer supplied {sorted(values)}"
            )
        return _filled(copy.deepcopy(self.body), values)


def _is_slot(node: Any) -> bool:
    return isinstance(node, dict) and set(node) == {"slot"}


def _key_slot(key: Any) -> str | None:
    """The slot a mapping key stands for, when the whole key is one."""

    match = _KEY_SLOT.fullmatch(key) if isinstance(key, str) else None
    return None if match is None else match.group(1)


def _slots(node: Any) -> list[str]:
    if _is_slot(node):
        return [str(node["slot"])]
    if isinstance(node, dict):
        keys = [name for name in map(_key_slot, node) if name is not None]
        return keys + [name for value in node.values() for name in _slots(value)]
    if isinstance(node, list):
        return [name for value in node for name in _slots(value)]
    return []


def _filled_key(key: Any, values: Mapping[str, Any]) -> Any:
    name = _key_slot(key)
    if name is None:
        return key
    if not isinstance(values[name], str):
        raise MatrixSetupError(
            f"slot {name!r} fills a mapping key and must be a string, not "
            f"{type(values[name]).__name__}"
        )
    return values[name]


def _filled(node: Any, values: Mapping[str, Any]) -> Any:
    if _is_slot(node):
        return values[node["slot"]]
    if isinstance(node, dict):
        return {
            _filled_key(key, values): _filled(value, values)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_filled(value, values) for value in node]
    return node


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _provenance_problems(provenance: Any) -> list[str]:
    if not isinstance(provenance, dict):
        return ["the provenance block is not a mapping"]
    grade = provenance.get("grade")
    if grade == "recorded":
        missing = [
            name
            for name in ("recorded_on", "recorded_model", "sha256")
            if not _text(provenance.get(name))
        ]
        return [f"a recorded envelope must state {missing}"] if missing else []
    if grade == "documented":
        sources = provenance.get("sources")
        if not isinstance(sources, list) or not sources:
            return ["a documented envelope must list its sources"]
        return [
            f"source {index} must state exactly {sorted(_SOURCE_FIELDS)}"
            for index, source in enumerate(sources)
            if not isinstance(source, dict)
            or set(source) != _SOURCE_FIELDS
            or not all(_text(value) for value in source.values())
        ]
    if grade == "constructed":
        typed = _text(provenance.get("sdk_type")) and _text(provenance.get("sdk_version"))
        if typed or _text(provenance.get("reason")):
            return []
        return [
            "a constructed envelope must state the SDK type and version it was "
            "built for, or the reason"
        ]
    return [f"grade {grade!r} is not one of {list(GRADES)}"]


def parse(document: Any, *, where: str, text: str = "") -> Envelope:
    """Build an envelope from its YAML document; any unknown shape is an error."""

    problems: list[str] = []
    if not isinstance(document, dict) or set(document) != _FIELDS:
        raise MatrixSetupError(f"envelope {where} must hold exactly {sorted(_FIELDS)}")
    if document["schema_version"] != ENVELOPE_SCHEMA:
        problems.append(f"schema_version must be {ENVELOPE_SCHEMA!r}")
    for name in ("dialect", "provider", "behaviour"):
        if not _text(document[name]) or not re.fullmatch(r"[a-z][a-z0-9_]*", document[name]):
            problems.append(f"{name} must be a lower-case name")
    if type(document["status"]) is not int or not 100 <= document["status"] <= 599:
        problems.append("status must be an HTTP status")
    problems.extend(_provenance_problems(document["provenance"]))
    unknown = sorted(set(_slots(document["body"])) - SLOTS)
    if unknown:
        problems.append(f"slots {unknown} are not among {sorted(SLOTS)}")
    problems.extend(f"it stores {found}" for found in hygiene_problems(text))
    if problems:
        raise MatrixSetupError(f"envelope {where}: " + "; ".join(problems))
    return Envelope(
        dialect=document["dialect"],
        provider=document["provider"],
        behaviour=document["behaviour"],
        grade=document["provenance"]["grade"],
        provenance=document["provenance"],
        status=document["status"],
        body=document["body"],
    )


def _read(path: Path) -> Envelope:
    text = path.read_text(encoding="utf-8")
    envelope = parse(yaml.safe_load(text), where=path.relative_to(ROOT).as_posix(), text=text)
    if path.relative_to(ROOT).as_posix() != f"{envelope.name}.yaml":
        raise MatrixSetupError(
            f"envelope {envelope.name} is stored as {path.relative_to(ROOT).as_posix()}"
        )
    return envelope


@functools.cache
def all_envelopes() -> Mapping[str, Envelope]:
    """Every envelope file of the package, by name."""

    found = [_read(path) for path in sorted(ROOT.rglob("*.yaml"))]
    return {envelope.name: envelope for envelope in found}


def load(name: str) -> Envelope:
    try:
        return all_envelopes()[name]
    except KeyError as exc:
        raise MatrixSetupError(
            f"no envelope {name!r}; known: {sorted(all_envelopes())}"
        ) from exc


def weakest(names: Iterable[str]) -> str | None:
    """The grade of a cell: that of the weakest envelope it used, if any."""

    grades = {load(name).grade for name in names}
    for grade in reversed(GRADES):
        if grade in grades:
            return grade
    return None
