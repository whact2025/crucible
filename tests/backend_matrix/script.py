# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The neutral provider script: what the model side does, written once.

A scenario describes provider behaviour with these steps and nothing else. No
model id, endpoint or wire field appears here; a dialect renders each step
into its provider's reply (:mod:`.provider`). Tools are named by fixture
*slot*, never by tool name: the fixture (:mod:`.toolsets`) binds a slot to a
real tool and its arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Union

READ_A = "read_a"
READ_B = "read_b"
WRITE = "write"
# A tool that no ToolSet of the matrix attaches: what a model calls when it
# names a tool it was never offered.
UNATTACHED = "unattached"
SLOTS = (READ_A, READ_B, WRITE, UNATTACHED)


@dataclass(frozen=True)
class TokenUsage:
    """Tokens the provider bills for one response."""

    input_tokens: int
    output_tokens: int


DEFAULT_USAGE = TokenUsage(input_tokens=120, output_tokens=30)


@dataclass(frozen=True)
class Say:
    """A final answer with a clean stop."""

    text: str
    usage: TokenUsage = DEFAULT_USAGE


@dataclass(frozen=True)
class Call:
    """One tool call for a fixture slot; ``args`` overrides the slot's own.

    ``malformed`` is the text of the arguments as the model wrote it, where
    that text is no JSON object (no JSON at all, or a JSON document of
    another type): the call then names no arguments at all. A wire that
    carries the arguments of a call in a string carries such a text; a wire
    on which a tool call travels as a decoded object cannot.
    """

    slot: str
    args: Mapping[str, Any] | None = None
    usage: TokenUsage = DEFAULT_USAGE
    malformed: str | None = None


@dataclass(frozen=True)
class Reason:
    """A reply that carries reasoning content only."""

    usage: TokenUsage = DEFAULT_USAGE


@dataclass(frozen=True)
class Refuse:
    """The provider refuses, in one of the three shapes providers use."""

    form: Literal["content_filter", "refusal_part", "http_policy"]
    usage: TokenUsage = DEFAULT_USAGE


@dataclass(frozen=True)
class Cut:
    """A reply cut by the route's output-token cap.

    A complete reply of its wire: it carries ``text``, what was produced
    before the cap, the provider's own mark of a cut in the place where the
    wire states how a turn ended, and the usage of the response, which the
    provider bills.
    """

    text: str
    usage: TokenUsage = DEFAULT_USAGE


# What a reply that produced nothing is billed for: the input it read.
NO_OUTPUT_USAGE = TokenUsage(input_tokens=DEFAULT_USAGE.input_tokens, output_tokens=0)


@dataclass(frozen=True)
class Empty:
    """A clean stop that produced nothing.

    A complete reply of its wire with the mark of an ordinary finish: no
    text, no reasoning and no output token. ``Say`` with an empty text is
    not this step: a dialect renders that from the reply it has for a plain
    answer, and a reply without output has a rendering, and a grade, of its
    own.
    """

    usage: TokenUsage = NO_OUTPUT_USAGE

    def __post_init__(self) -> None:
        if self.usage.output_tokens:
            raise ValueError("a reply that produced nothing bills no output token")


@dataclass(frozen=True)
class HttpError:
    """An HTTP error status with the vendor's error body."""

    status: int


@dataclass(frozen=True)
class Drop:
    """The connection or the peer goes away after part of the reply.

    The reply is the one the dialect has for a final answer of ``text``.
    Only a leading part of it travels: it ends inside the turn, before
    anything that ends the turn, and then the other side is gone. On a
    wire of one document per reply the document is cut inside the text, so
    no document arrives. On a stream of frames the frame that carries the
    text travels whole and the frame after it is cut, so the client holds
    the text. ``usage`` is what the provider bills where a frame that
    travels whole states the usage of the turn; a dialect whose usage
    travels after the text states none and bills none.
    """

    text: str
    after: Literal["headers", "partial_body"] = "partial_body"
    usage: TokenUsage = DEFAULT_USAGE


@dataclass(frozen=True)
class Stall:
    """The request is accepted and never answered; ``marker`` names the moment."""

    marker: str = "stall"


@dataclass(frozen=True)
class Malformed:
    """A reply that is not a valid terminal of the dialect.

    Either a complete reply that holds no turn at all, or the frames of a
    final answer of ``text`` without the frame that ends the turn, after
    which the stream ends in order. Which of the two a wire can carry is
    its dialect's to say. ``usage`` is what the provider bills where the
    frames that do travel state the usage of the turn; a dialect whose
    usage travels with the terminal states none and bills none.
    """

    text: str
    usage: TokenUsage = DEFAULT_USAGE


@dataclass(frozen=True)
class ServedAs:
    """``step``, answered under a model id derived from the requested one."""

    step: Step
    suffix: str = "-served"


@dataclass(frozen=True)
class Loop:
    """``step`` again for every request, at most ``limit`` times.

    A request beyond ``limit`` is unscripted. Fewer than ``at_least`` requests
    leave the step unconsumed; with ``at_least=0`` the step is never
    required: the model has it ready, and a call that ended before asking
    for it leaves nothing unconsumed.
    """

    step: Step
    limit: int
    at_least: int = 1


Step = Union[
    Say, Call, Reason, Refuse, Cut, Empty, HttpError, Drop, Stall, Malformed, ServedAs, Loop
]


@dataclass(frozen=True)
class ProviderScript:
    """Replies in strict order and count.

    ``tail`` is the one reply a request that advertises no tools may get after
    the steps are used up: the forced final request some adapters send. It is
    never required.
    """

    steps: tuple[Step, ...]
    tail: Say | None = None
