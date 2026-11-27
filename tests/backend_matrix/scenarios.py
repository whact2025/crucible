# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The scenario catalogue: the starter set and pack A.

A scenario states, once and for every backend: the policy it passes (always a
complete ``BackendPolicy``, never an adapter default), what the provider does,
which ToolSet is attached, the lenses it runs in and the result every backend
must return. Where backends may legitimately differ, a difference table keyed
by the governing declared field selects ``BASE``, a named ``FALLBACK`` or
``REFUSED``; the table is total over the field's vocabulary. One table is
keyed by something that is no declared field: the state of a closed route
extra as routing states it for the route of the cell's role, total over its
three states (:mod:`.declarations`). No scenario asks which backend it runs
on.

A difference that no feature field expresses is a *declared difference* of
the ledger: the scenario defines its alternative contract under the id of the
difference, and the ledger says on which cells it is in force. An alternative
may add the optional tail of the script and change the assertions the
scenario itself makes, nothing else: it never stands in for a contract clause.
Where a declared difference restates a line of a clause (one does: what the
guarded lens asks of a route that opts into a cost upper bound), the clause
asserts the alternative itself, under a selector read from the routing file,
and the scenario's alternative leaves the plan as it is.

Every scenario lands with a lying counterpart on the toy driver
(:mod:`.drivers.toy`); a scenario without a liar cannot be enabled anywhere.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from assay import BackendExecutionTrace, BackendPolicy, ConversationTransportRequest

from .contract import (
    ACCOUNTING_FLAGS,
    EXPECTED_FALSE_FLAGS,
    READ_STOP_REASONS,
    Check,
    Deviation,
    Expectation,
    Observation,
    Reference,
)
from .declarations import (
    ADMITTED_NOT_CARRIED,
    BASE,
    CARRIED,
    FALLBACK,
    KEYING_EXTRAS,
    NOT_ADMITTED,
    REFUSED,
    SERVED_MODEL_EXTRA,
    Differences,
    Variant,
)
from .drivers.base import LENSES, MatrixSetupError
from .script import (
    READ_A,
    READ_B,
    UNATTACHED,
    WRITE,
    Call,
    Cut,
    Drop,
    Empty,
    Loop,
    Malformed,
    ProviderScript,
    Reason,
    Refuse,
    Say,
    ServedAs,
    Stall,
    TokenUsage,
)
from .toolsets import (
    READ,
    READ_BOUNDED,
    WRITE_FILE,
    WRITER,
    ToolsetSpec,
    capability_of,
)

GUARDED = "guarded"
STARTER = "starter"
PACK_A = "pack_a"
PACK_B = "pack_b"
SETS = (STARTER, PACK_A, PACK_B)

SYSTEM = "You are the matrix system prompt [canary-system-4f1c]."
USER = "Answer the matrix user prompt [canary-user-9b2e]."
TEXT = "matrix final answer"
# The answer of a turn whose terminal never arrives, and the answer that is
# on its way when the connection or the peer goes away.
UNENDED_TEXT = "matrix answer of a turn without its terminal"
DROPPED_TEXT = "matrix answer that was cut off on its way"
# What the provider had produced of an answer when it ended the reply at the
# output-token cap of the route.
CUT_TEXT = "matrix answer that the output cap cut"
SCHEMA_KEY = "matrix_answer"
# The title occurs once in the schema, so it counts renderings of the schema.
SCHEMA_TITLE = "matrix-schema-7c41"
SCHEMA: Mapping[str, Any] = {
    "title": SCHEMA_TITLE,
    "type": "object",
    "properties": {SCHEMA_KEY: {"type": "string"}},
    "required": [SCHEMA_KEY],
    "additionalProperties": False,
}
SCHEMA_ANSWER = json.dumps({SCHEMA_KEY: "forty-two"})
# The answer to a schema finalisation: valid like the draft of the last tool
# round, and another text, so a cell can tell which of the two was returned.
SCHEMA_FINAL_ANSWER = json.dumps({SCHEMA_KEY: "forty-two, as finalised"})
TURN_LIMIT = 2
CONTEXT_BUDGET = 64
BUDGET_USD = 0.001
# A prefix shorter than this inside a replayed result is coincidence, not payload.
_PAYLOAD_FLOOR = 8
# The turn count of the scenarios that write. One adapter changes what it
# offers the model after a third of the turns: a scenario of a few rounds
# stays below that.
WRITER_TURNS = 24
# How often the scenario about the offered tools reads and writes in turn.
CATALOGUE_ROUNDS = 6
# The turn count of that scenario: as many turns as its script holds
# requests, the reads, the writes and the answer. So the scenario plays
# every turn of its budget, the last one included, and an adapter that
# changes what it offers or says at any point of the budget is seen.
CATALOGUE_TURNS = 2 * CATALOGUE_ROUNDS + 1
# The deadline of that scenario's invocation. Its thirteen requests take a
# backend that starts a worker for every request most of the base policy's
# minute, and three times as long where the test shares one hardware thread
# with another busy process. The scenario is not about a deadline: the value
# is several times the longer figure, so that it ends a call that hangs and
# no other. The scenario of the malformed tool arguments passes the same
# deadline: its four requests take such a backend a third of that minute,
# and more than the minute where two other busy processes share the thread.
CATALOGUE_TIMEOUT_SECONDS = 600
# How much later than the deadline of the invocation the deadline of a write
# host may be: the two clocks are read a moment apart. Every backend that
# hosts writes is within a few milliseconds.
DEADLINE_SLACK_SECONDS = 0.5

# What the model answers where it is asked once more, without tools, at the
# turn limit: the partial answer of a call that ended there.
PARTIAL_TEXT = "partial answer at the turn limit"
# The scenario about a budget inside a tool loop: the model asks for a tool
# in each of this many rounds and then answers, and every tool round bills
# the same usage. At the matrix rate one round costs less than the budget
# and two cost more, so the budget is crossed by the second response.
LOOP_ROUNDS = 4
OVERRUN_ROUND = 2
LOOP_ROUND_USAGE = TokenUsage(150, 60)
# What a model wrote in the place of the arguments of a tool call, three
# times: none of the texts is a JSON object, so none of the calls names any
# arguments. The first is the JSON of the first read tool's arguments, cut
# before its closing brace: no JSON document. The second is a JSON document
# that says "nothing", which a reader may take for a call without
# arguments. The third is a JSON document that is neither an object nor
# empty: a list of the file name.
MALFORMED_ARGUMENT_TEXTS = ('{"file_path": "notes.txt"', "null", '["notes.txt"]')

# The conversation a workflow asks to continue in the scenario about
# continuity: a handle that no session of the cell issued, in the form both
# vendor tools name a session by, bound to two digests of the matrix. Whoever
# keeps the conversations of a wire holds none under it.
STALE_HANDLE = "00000000-0000-4000-8000-000000000050"


def _matrix_digest(what: str) -> str:
    return "sha256:" + hashlib.sha256(f"backend matrix {what}".encode()).hexdigest()


CONVERSATION = ConversationTransportRequest(
    mode="workflow_bound",
    action="reuse",
    opaque_id=STALE_HANDLE,
    identity_fingerprint=_matrix_digest("conversation identity"),
    operation_digest=_matrix_digest("conversation operation"),
)
# The one type a route that declares no continuity for a workflow refuses a
# conversation request with, before transport.
CONTINUITY_REFUSAL = NotImplementedError.__name__

# The deadline of the scenarios that play one (a stall of the provider, a
# tool handler that blocks, a cancellation that a route can honour only at
# its next step), in seconds from the call. The runner adds what the driver
# of the column states as the time its seam needs before a first request can
# reach the provider (``Driver.startup_seconds``): the deadline counts from
# the call, and a kind that starts a tool or an interpreter for a provider
# request has spent seconds of it before its request is on the wire. So the
# request is on the wire, and the handler entered, well before the deadline
# on every kind, on a busy machine too. Where it is not (the provider took
# the first request, or the held handler was entered, only after the
# deadline the cell was called with) the scenario was not played: the cell
# fails with its times and is not judged. The deadline is what ends such a
# call and so what bounds the time its cell takes: nothing of the provider
# or of the fixture ends a stall or lets a handler go by a clock of its own.
DEADLINE_SECONDS = 2
# How much later than the moment that bounds it a call may return. Wide: a
# backend that ends a session or a worker at that moment closes streams and
# reaps processes first, and the machine may be busy. A call that returns
# later fails its cell; that is no deviation and never an expected failure.
# The bound is there for a call that hangs. It is no measure of a deadline:
# a call that ends this much later than the deadline it was given is in
# time, and how late it was is kept in the cell report and asserted by no
# cell.
GRACE_SECONDS = 15.0
# How much earlier than the deadline it was called with a call that nothing
# but its deadline ends may return a result. A result that comes earlier is
# a deviation of the scenario (``return:early``): the adapter ended the
# call at a time of its own, not at the one the policy states.
EARLY_SECONDS = 0.5
# On a route that declares that it can cut a call: how long after the
# ``close()`` of the cell the call may take to return or to raise. A later
# return is a deviation of the scenario (``return:not_cut``): the call was
# not cut, it ran on, whatever it then returned. The cancellation scenarios
# pass a deadline that lies this far beyond the deadline of the others, so
# that after a ``close()`` that came within the allowance of the driver the
# call has this time to return and its deadline lies
# :data:`DEADLINE_SECONDS` beyond that: a call that ran on to its deadline
# is never taken for one that was cut.
CUT_SECONDS = 3.0
# How long before the deadline of the call that time must end for a cell to
# tell the two apart. A ``close()`` that came later was not played: the
# cell fails with its times.
CUT_CLEARANCE_SECONDS = 1.0
# Where a route can stop a call only between two steps, how long after the
# deadline (after ``close()``, where the cell closes the backend while a
# handler blocks) the held handler is let go, while the call still runs.
# After a deadline the runner adds the allowance of the driver once more: an
# adapter may count its deadline from a moment of its own inside that
# allowance (the start of its worker), and a handler that is let go before
# the deadline as the adapter counts it did not block past it.
RELEASE_AFTER_SECONDS = 1.0
# How long a cell waits, once the held handler has left, before it closes
# the backend: a provider request in that time is a request after the call.
SETTLE_SECONDS = 0.5
# How long a handler that was let go may take to leave.
LEAVE_SECONDS = 10.0
# What the provider notes when a stalled request has been taken, and what the
# runner notes when the held handler has been entered.
STALL_MARKER = "stall"
HOLD_MARKER = "hold"
# The scenarios whose plan holds a conduct. Said here, so that what a cell
# is called with can be read without building a plan; the runner refuses a
# conduct on any other scenario.
CONDUCTED: frozenset[str] = frozenset({"S15", "S16a", "S30a", "S30b", "S30c"})

# The base policy. No scenario relies on an adapter default: they differ.
BASE_POLICY = BackendPolicy(
    retry_budget=1,
    timeout_seconds=60,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)

CLEAN = Expectation(final_text=TEXT)

Bound = Literal["deadline", "release"]


@dataclass(frozen=True)
class Conduct:
    """What a cell does beside the call, and what bounds the return of the call.

    ``hold`` names the slot whose handler the cell holds at its entry.
    ``close_at`` says when the cell closes the backend from another thread:
    when the provider has taken the stalled request, or when the held
    handler has been entered. ``release`` says when the held handler is let
    go: once the call has returned (a call that can be cut returns while the
    handler still blocks), or :data:`RELEASE_AFTER_SECONDS` after the
    deadline (and the allowance of the driver) or after ``close()``, while
    the call still runs (a route that stops only between two steps returns
    once the handler has left).
    ``bounds`` are the moments the return is held to, each plus
    :data:`GRACE_SECONDS`: the deadline of the call, the release of the held
    handler. A call that misses them hangs: its cell fails, and that has no
    code. The clock of a bound never lets a stall or a handler go; the cell
    does.

    Two things more are asked of the moment the call returns, and each is a
    deviation of the scenario with a code. ``cut_seconds``: the route
    declares that it can cut a call, so the call returns or raises within
    that time of the ``close()`` of the cell (``return:not_cut`` otherwise).
    ``not_before_deadline``: nothing but its deadline ends the call, so a
    result does not come more than :data:`EARLY_SECONDS` before the deadline
    the call was made with (``return:early`` otherwise).
    """

    hold: str | None = None
    close_at: Literal["stall", "hold"] | None = None
    release: Literal["after_return", "after_deadline", "after_close"] = "after_return"
    bounds: tuple[Bound, ...] = ("deadline",)
    cut_seconds: float | None = None
    not_before_deadline: bool = False


@dataclass(frozen=True)
class Plan:
    """What one cell plays and expects, after the variant is selected."""

    script: ProviderScript
    expect: Expectation = CLEAN
    max_turns: int = 8
    budget_usd: float | None = None
    response_schema: Mapping[str, Any] | None = None
    context_budget: int | None = None
    checks: tuple[Check, ...] = ()
    # The conversation the invocation asks to continue; ``None`` for a call
    # that asks for none.
    conversation: ConversationTransportRequest | None = None
    # What the cell does beside the call; ``None`` where it only calls.
    conduct: Conduct | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    set: str
    title: str
    toolset: ToolsetSpec | None
    lenses: tuple[str, ...]
    plan: Callable[[], Plan]
    differences: Differences = field(default_factory=dict)
    fallbacks: Mapping[str, Callable[[Plan], Plan]] = field(default_factory=dict)
    policy: BackendPolicy = BASE_POLICY
    canary: bool = False
    # Alternative contracts of declared differences no feature field selects,
    # by difference id. The ledger says on which cells one is in force.
    alternatives: Mapping[str, Callable[[Plan], Plan]] = field(default_factory=dict)
    # An admission scenario builds a backend for every capability a workflow
    # can ask routing for and calls none. What routing does with each is
    # pinned per cell in the inventory, never derived from a declared value.
    admission: bool = False
    # The cell is played on a route without its credential: the backend is
    # built by routing from the shipped route, and the place the route of the
    # cell's role takes its credential from holds none at the last moment at
    # which the route reads it. A state of the route and no step of the
    # script: the driver of the column builds it and says what the place is.
    credential_unset: bool = False

    @property
    def capability(self) -> str:
        return capability_of(self.toolset)

    @property
    def governing(self) -> tuple[str, ...]:
        return tuple(self.differences)


def plan_for(
    scenario: Scenario,
    variants: Mapping[str, Variant],
    differences: Sequence[str] = (),
) -> Plan:
    """Apply the selected variants, then the declared differences in force.

    ``differences`` are the ids the ledger lists for the cell. Each must have
    an alternative here, and an alternative may only add the optional tail of
    the script and change the assertions of the scenario: the expected result,
    the scripted steps and every limit stay, so no clause is replaced.
    """

    plan = scenario.plan()
    for variant in variants.values():
        if variant.kind == "FALLBACK":
            plan = scenario.fallbacks[variant.name](plan)
        elif variant.kind == "REFUSED":
            plan = replace(
                plan,
                script=ProviderScript(()),
                expect=Expectation(outcome="refused", refused_type=variant.name),
                checks=(),
            )
    for name in differences:
        alternative = scenario.alternatives.get(name)
        if alternative is None:
            raise MatrixSetupError(
                f"scenario {scenario.id} defines no alternative for the declared "
                f"difference {name}"
            )
        changed = alternative(plan)
        if (
            replace(changed, script=plan.script, checks=plan.checks) != plan
            or changed.script.steps != plan.script.steps
        ):
            raise MatrixSetupError(
                f"the alternative of {name} on {scenario.id} changes more than "
                "the tail of the script and the assertions of the scenario; a "
                "declared difference never stands in for a contract clause"
            )
        plan = changed
    return plan


# -- scenario assertions (the part of a contract no clause states) ---------


def _matched(obs: Observation) -> list[Any]:
    return [request for request in obs.provider.requests if request.matched]


def prompt_reaches_provider(obs: Observation, _ref: Reference) -> list[Deviation]:
    requests = _matched(obs)
    if not requests or (SYSTEM in requests[0].system and USER in requests[0].user):
        return []
    return [
        Deviation(
            "request.content:prompt",
            "the prompt canaries did not reach the provider unchanged",
        )
    ]


def one_request_event(obs: Observation, _ref: Reference) -> list[Deviation]:
    assert obs.result is not None
    trace = obs.result.execution_trace
    count = 0 if trace is None else sum(event.kind == "request" for event in trace.events)
    if count <= 1:
        return []  # none at all is the expectation's event.missing:request/-
    return [
        Deviation(
            "event.extra:request/-",
            f"the trace has {count} request events for one provider request",
        )
    ]


def _replayed(obs: Observation) -> list[str]:
    return [text for request in obs.provider.requests for _id, text in request.tool_results]


def results_replayed(obs: Observation, ref: Reference) -> list[Deviation]:
    replayed = _replayed(obs)
    scripted = {binding.name for binding in ref.bindings.values()}
    for item in obs.toolset_log:
        if item.name not in scripted:
            continue  # a dispatch outside the script is C10's business
        if item.result and not any(item.result in text for text in replayed):
            return [
                Deviation(
                    "request.content:tool_results",
                    f"the result of {item.name} was not replayed to the provider",
                )
            ]
    return []


def history_replayed(obs: Observation, _ref: Reference) -> list[Deviation]:
    """On a wire that keeps no conversation, each request replays it in full.

    What a request must carry of the earlier rounds is the wire's own rule:
    the dialect of the column states it and the provider ledger keeps what it
    found in the scripted exchange. A wire that holds the conversation itself
    has no such rule and nothing to report. Asserted wherever a ToolSet is
    attached: without a tool round nothing is replayed.
    """

    problems = obs.provider.replay_problems
    if not problems:
        return []
    return [Deviation("request.content:replay", "; ".join(problems))]


def _tool_event_counts(obs: Observation, name: str) -> tuple[int, int]:
    assert obs.result is not None
    trace = obs.result.execution_trace
    events = () if trace is None else trace.events
    calls = sum(e.kind == "tool_call" and str(e.name).endswith(name) for e in events)
    results = sum(e.kind == "tool_result" and str(e.name).endswith(name) for e in events)
    return calls, results


def tool_events_required(obs: Observation, ref: Reference) -> list[Deviation]:
    out: list[Deviation] = []
    for slot, binding in sorted(ref.bindings.items()):
        name = binding.name
        dispatched = sum(item.name == name for item in obs.toolset_log)
        calls, results = _tool_event_counts(obs, name)
        if calls < dispatched:
            out.append(
                Deviation(
                    f"event.missing:tool_call/{slot}",
                    f"{dispatched} dispatch(es) of {name}, {calls} tool_call event(s)",
                )
            )
        if results < dispatched:
            out.append(
                Deviation(
                    f"event.missing:tool_result/{slot}",
                    f"{dispatched} dispatch(es) of {name}, {results} tool_result event(s)",
                )
            )
    return out


def tool_events_paired(obs: Observation, ref: Reference) -> list[Deviation]:
    out: list[Deviation] = []
    for slot, binding in sorted(ref.bindings.items()):
        calls, results = _tool_event_counts(obs, binding.name)
        if calls != results:
            kind = "tool_result" if calls > results else "tool_call"
            out.append(
                Deviation(
                    f"event.missing:{kind}/{slot}",
                    f"{calls} tool_call and {results} tool_result event(s) for "
                    f"{binding.name}: optional events must still be paired",
                )
            )
    return out


def schema_answer_returned(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The final message is the schema answer the provider gave.

    Valid is not enough: an object of the schema that the provider never sent
    passes any check of the shape. The answer is compared as the JSON value
    it is, not as text, because an adapter may serialise the object again.
    Where the optional tail of a declared difference was asked for and
    answered, the answer is the tail's, and the alternative's own check
    (:func:`finalisation_is_the_answer`) compares it.
    """

    assert obs.result is not None
    if obs.result.error is not None:
        return []  # an error result promises no answer
    try:
        answer = json.loads(obs.result.final_message)
    except json.JSONDecodeError:
        answer = None
    if not (
        isinstance(answer, dict)
        and set(answer) == {SCHEMA_KEY}
        and isinstance(answer[SCHEMA_KEY], str)
    ):
        return [
            Deviation("text:mismatch", "final_message does not validate against the schema")
        ]
    if obs.provider.tail_served or answer == json.loads(SCHEMA_ANSWER):
        return []
    return [
        Deviation(
            "text:mismatch",
            "final_message validates against the schema and is not the answer "
            "the provider gave",
        )
    ]


def schema_in_field(obs: Observation, _ref: Reference) -> list[Deviation]:
    if any(request.schema_field for request in _matched(obs)):
        return []
    return [
        Deviation(
            "request.content:schema",
            "no request carries the schema in the provider's schema field",
        )
    ]


def schema_text_in_prompt(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The prompt half of ``prompt_only``: the schema text, once, in the system prompt.

    It has a code of its own. Where the schema *field* travels is another
    check's business and another code, so a known gap on the one half never
    covers a deviation on the other.
    """

    requests = _matched(obs)
    if not requests:
        return []  # nothing reached the provider: C0 reports it
    in_prompt = requests[0].system.count(SCHEMA_TITLE)
    if in_prompt == 1:
        return []
    return [
        Deviation(
            "request.content:schema_prompt",
            f"the schema text appears {in_prompt} time(s) in the system prompt and "
            "must appear once: the route declares that the schema travels in the "
            "prompt",
        )
    ]


def schema_in_prompt(obs: Observation, ref: Reference) -> list[Deviation]:
    """``prompt_only``: the schema text once in the prompt, and no provider field."""

    out = list(schema_text_in_prompt(obs, ref))
    carrying = sum(request.schema_field for request in _matched(obs))
    if carrying:
        out.append(
            Deviation(
                "request.content:schema",
                f"{carrying} request(s) carry the schema in the schema field of the "
                "provider; the route declares that the schema travels in the "
                "prompt only",
            )
        )
    return out


def schema_in_final_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The schema travels in exactly one request, the last, which has no tools."""

    requests = _matched(obs)
    if not requests:
        return []  # nothing reached the provider: C0 reports it
    carrying = [request for request in requests if request.schema_field]
    if len(carrying) == 1 and carrying[0] is requests[-1] and not carrying[0].tools:
        return []
    return [
        Deviation(
            "request.content:schema",
            f"{len(carrying)} of {len(requests)} request(s) carry the schema in "
            "the schema field of the provider; exactly one must, the last, and "
            "it must advertise no tools",
        )
    ]


def finalisation_is_the_answer(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The answer to the schema finalisation is what the call returns.

    The script gives the finalisation another valid answer than the draft of
    the last tool round, so an adapter that pays for the finalisation and
    returns the draft is told apart. Compared as the JSON values they are.
    """

    assert obs.result is not None
    if obs.result.error is not None or not obs.provider.tail_served:
        # An error result promises no answer; a finalisation that was never
        # sent has none, and ``schema_in_final_request`` reports that.
        return []
    try:
        answer = json.loads(obs.result.final_message)
    except json.JSONDecodeError:
        answer = None
    if answer == json.loads(SCHEMA_FINAL_ANSWER):
        return []
    return [
        Deviation(
            "text:mismatch",
            "final_message is not the answer the provider gave to the schema "
            "finalisation request",
        )
    ]


def no_turn_control(obs: Observation, _ref: Reference) -> list[Deviation]:
    if all(request.turn_control is None for request in obs.provider.requests):
        return []
    return [
        Deviation(
            "request.content:turn_control",
            "a turn control was sent on a route that declares none",
        )
    ]


def turn_control_as_asked(obs: Observation, _ref: Reference) -> list[Deviation]:
    """A turn control on the wire is the suite's turn count, as the call asked.

    One completion needs one turn, so an adapter that cuts the count to a
    ceiling of its own returns the same result as one that hands it on: the
    turn control of the request is the one place that shows the difference.
    Asserted where a request carries one. Whether a request owes one follows
    from what the route declares and from its wire (``turn_count_in_request``);
    how an adapter whose wire has no field for a turn count reports a ceiling
    of its own is not asserted here.
    """

    asked = suite_default_max_turns()
    sent = sorted(
        {
            request.turn_control
            for request in _matched(obs)
            if request.turn_control is not None and request.turn_control != asked
        }
    )
    if not sent:
        return []
    return [
        Deviation(
            "request.content:turn_control",
            f"the call asked for {asked} turns and a request carries a turn "
            f"control of {sent[0]}",
        )
    ]


def turn_count_in_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    """Every request carries the turn count, where its wire has a field for one.

    Asked of a route that declares that its turn limit is enforced where the
    provider is. On a wire that has a field for the count, a request without
    it is an adapter that kept the count to itself: the session has no limit
    but what the other side applies unasked, and the result of one
    completion does not show it. A wire without the field carries no count
    from any adapter, so there the check says nothing.
    """

    requests = _matched(obs)
    untold = [request for request in requests if request.turn_control is None]
    if not obs.provider.turn_control_field or not untold:
        return []  # with no request at all C0 reports the script
    return [
        Deviation(
            "request.content:turn_control",
            f"{len(untold)} of {len(requests)} request(s) carry no turn control "
            "on a wire that has a field for one and a route that declares an "
            "enforced turn limit",
        )
    ]


def budget_in_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    if any(request.budget_control == BUDGET_USD for request in _matched(obs)):
        return []
    return [Deviation("request.content:budget", "the USD ceiling is not in the request")]


def no_overshoot(obs: Observation, _ref: Reference) -> list[Deviation]:
    assert obs.result is not None
    if obs.result.usage.cost_usd <= BUDGET_USD:
        return []
    return [Deviation("usage.cost_usd:gt", "a hard USD ceiling was overshot")]


def response_cap_in_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    """Every request carries a response cap, where its wire has a field for one.

    Asked of a route that declares that its cap on the tokens of a response
    is enforced where the provider is. On a wire that has a field for the
    cap, a request without it is an adapter that kept the cap of its route
    to itself: the provider then applies whatever it applies unasked. The
    value is the route's own and is not asserted. A wire without the field
    carries no cap from any adapter, so there the check says nothing.
    """

    requests = _matched(obs)
    untold = [
        request
        for request in requests
        if request.response_cap is None or request.response_cap < 1
    ]
    if not obs.provider.response_cap_field or not untold:
        return []  # with no request at all C0 reports the script
    return [
        Deviation(
            "request.content:response_cap",
            f"{len(untold)} of {len(requests)} request(s) carry no cap on the "
            "tokens of the response on a wire that has a field for one and a "
            "route that declares an enforced cap",
        )
    ]


def no_response_cap(obs: Observation, _ref: Reference) -> list[Deviation]:
    if all(request.response_cap is None for request in obs.provider.requests):
        return []
    return [
        Deviation(
            "request.content:response_cap",
            "a cap on the tokens of the response was sent on a route that "
            "declares none",
        )
    ]


def stale_outcome_of_the_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The result of a handle that was refused says so, for the request it answers.

    A workflow that asked to continue a conversation reads from the outcome
    what became of it. Where the handle was refused the result carries an
    outcome whose state is stale, which names no handle, and which is bound
    to the request: its identity fingerprint and its operation digest are
    those of the request. An outcome of another operation would be booked
    against the wrong record.
    """

    assert obs.result is not None
    outcome = obs.result.conversation
    if outcome is None:
        return [
            Deviation(
                "conversation:missing",
                "the invocation asked to continue a conversation and the result "
                "carries no conversation outcome",
            )
        ]
    out: list[Deviation] = []
    if outcome.state != "stale" or outcome.opaque_id is not None:
        names = "" if outcome.opaque_id is None else " and names a handle"
        out.append(
            Deviation(
                "conversation:state",
                f"the conversation outcome is {outcome.state!r}{names}; nobody "
                "holds the handle the invocation named, so the outcome is stale "
                "and names none",
            )
        )
    if (outcome.identity_fingerprint, outcome.operation_digest) != (
        CONVERSATION.identity_fingerprint,
        CONVERSATION.operation_digest,
    ):
        out.append(
            Deviation(
                "conversation:unbound",
                "the identity fingerprint or the operation digest of the "
                "conversation outcome is not that of the request",
            )
        )
    return out


def handle_asked_of_its_keeper(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The tool that keeps the conversations of the wire was asked for the handle.

    Asked on a wire where a tool beside the adapter keeps the conversations
    (the provider ledger says so). There an adapter cannot know by itself
    that nobody holds a handle: a result for a handle nobody holds rests on
    that tool having been asked for the handle of the request, and for no
    other one. An adapter that calls every handle stale without asking is
    told apart by this alone. On a wire where the adapter keeps its
    conversations itself there is nobody else to ask, and the check says
    nothing.
    """

    refused = obs.provider.refused_handles
    if not obs.provider.conversation_keeper or set(refused) == {CONVERSATION.opaque_id}:
        return []
    asked = f"for {len(set(refused))} other handle(s)" if refused else "for none"
    return [
        Deviation(
            "conversation:unasked",
            "a tool beside the adapter keeps the conversations of this wire and "
            f"was asked {asked}; the result of a handle nobody holds rests on "
            "that tool having been asked for the handle of the request, and "
            "for no other",
        )
    ]


def _visible_payload(result: str, replayed: str) -> int:
    """Bytes of ``result``'s leading payload that ``replayed`` shows the provider."""

    payload = result.encode("utf-8")
    low, high = 0, len(payload)
    while low < high:
        middle = (low + high + 1) // 2
        if payload[:middle].decode("utf-8", errors="ignore") in replayed:
            low = middle
        else:
            high = middle - 1
    return low if low >= _PAYLOAD_FLOOR else 0


def context_bounded(obs: Observation, _ref: Reference) -> list[Deviation]:
    replayed = "\n".join(_replayed(obs))
    visible = sum(_visible_payload(item.result, replayed) for item in obs.toolset_log)
    if visible <= CONTEXT_BUDGET:
        return []
    return [
        Deviation(
            "context:over_budget",
            f"{visible} bytes of tool results reached the provider under a "
            f"context budget of {CONTEXT_BUDGET}",
        )
    ]


def first_result_truncated_not_dropped(
    obs: Observation, _ref: Reference
) -> list[Deviation]:
    """The lower bound of the byte budget: the first result still arrives, cut."""

    replayed = _replayed(obs)
    if not obs.toolset_log or not replayed:
        # Nothing was dispatched (C10 reports the missing call), or no request
        # answered a call (C0 reports the steps that were never requested).
        return []
    if _visible_payload(obs.toolset_log[0].result, "\n".join(replayed)):
        return []
    return [
        Deviation(
            "request.content:tool_results",
            f"no prefix of the first tool result (at least {_PAYLOAD_FLOOR} bytes) "
            "reached the provider: a result over the context budget is "
            "truncated, not dropped",
        )
    ]


def write_is_progress(obs: Observation, _ref: Reference) -> list[Deviation]:
    snapshot = obs.wrapper_snapshot
    writes = getattr(snapshot, "write_calls", None)
    denied = getattr(snapshot, "denied_read_calls", None)
    if writes == 1 and denied == 0:
        return []
    return [
        Deviation(
            "progress:write_not_credited",
            f"the progress wrapper counts {writes} write(s) and {denied} denied "
            "read(s); it must count the one write and deny nothing",
        )
    ]


def cut_off_text_withheld(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The text of a reply that was cut off is no final message.

    Whatever the kind of the result: where the connection or the peer went
    away inside the answer, what had arrived of it is not what the call
    returns. A leading part of the scripted answer in the final message is
    that text.
    """

    assert obs.result is not None
    if DROPPED_TEXT[:_PAYLOAD_FLOOR] not in obs.result.final_message:
        return []
    return [
        Deviation(
            "text:mismatch",
            "final_message carries text of the reply that was cut off before its "
            "terminal",
        )
    ]


def write_ran_in_a_host_that_is_gone(obs: Observation, ref: Reference) -> list[Deviation]:
    """Where the write of the cell ran, and what became of that process.

    A write that the fixture's recorder saw ran in the process of the
    workflow: the recorder shadows the dispatch of the innermost ToolSet
    object in this process and nowhere else. A write that it did not see ran
    in a host, and the cell holds every host that was built while it played:
    none may still be there once ``invoke`` has returned, none may have left
    a process behind (one that was below it when it was stopped, or one of
    its process group or session), and the one that recorded the write must
    have been built with the deadline of the invocation: the moment of the
    call plus the policy's timeout, give or take
    :data:`DEADLINE_SLACK_SECONDS`. A host that is given a timeout of its
    own, counted from the moment it is built, lets a blocked write outlive
    the call by what the call had spent before. A write that is on disk, was
    not seen here and is recorded by no host cannot be judged, which is an
    error of the matrix and no deviation. A write that is not on disk is the
    missing dispatch of the clause about tools; nothing is said of its host.
    """

    if not obs.write_applied:
        return []
    name = ref.bindings[ref.write_slot].name
    if any(item.name == name for item in obs.toolset_log):
        return [
            Deviation(
                "mutation:in_process",
                f"{name} was dispatched in the process of the workflow: no process "
                "of its own stands between a blocked write and the deadline of "
                "the call",
            )
        ]
    mutation = obs.mutation
    hosts = () if mutation is None else mutation.hosts
    ran = [host for host in hosts if name in host.started_writes]
    if mutation is None or not ran:
        raise MatrixSetupError(
            f"{name} is on disk, did not run in the process of the workflow and "
            "no host the matrix watches recorded it as started: the cell cannot "
            "say where it ran"
        )
    out: list[Deviation] = []
    left = sorted(str(host.pid) for host in hosts if host.alive_at_return)
    if left:
        out.append(
            Deviation(
                "mutation:host_alive",
                f"the process(es) {', '.join(left)} that hosted writes of the call "
                "were still there when invoke had returned",
            )
        )
    behind = sorted({str(pid) for host in hosts for pid in host.left_behind})
    if behind:
        out.append(
            Deviation(
                "mutation:left_behind",
                f"the process(es) {', '.join(behind)} that a host of the call had "
                "started, or that are of its process group or session, were "
                "still there when invoke had returned",
            )
        )
    own = mutation.invoked_at + mutation.timeout_seconds
    late = [
        host.deadline - own
        for host in ran
        if host.deadline > own + DEADLINE_SLACK_SECONDS
    ]
    if late:
        out.append(
            Deviation(
                "mutation:deadline_unbound",
                f"the host of the write was built with a deadline {max(late):.1f} s "
                f"later than the invocation's own ({mutation.timeout_seconds:g} s "
                "from the call)",
            )
        )
    return out


def tools_are_the_attached_ones(obs: Observation, ref: Reference) -> list[Deviation]:
    """Every scripted request advertises exactly the tools of the attached ToolSet."""

    attached = set(ref.attached_tools)
    other = [
        (index + 1, sorted(set(request.tools) ^ attached))
        for index, request in enumerate(obs.provider.requests)
        if request.matched and set(request.tools) != attached
    ]
    if not other:
        return []
    first, differing = other[0]
    return [
        Deviation(
            "request.content:tools",
            f"{len(other)} of {len(_matched(obs))} request(s) advertise other tools "
            f"than the attached ToolSet; the first is request {first}, which "
            f"differs in {differing}",
        )
    ]


def messages_are_the_workflows(obs: Observation, ref: Reference) -> list[Deviation]:
    """Every message is the workflow's prompt, a turn of the model or a tool result.

    Three things are read. What kind of message each one is, is the wire's
    to say: the dialect of a wire that carries the conversation states it
    and the provider ledger keeps what it found; a wire that keeps the
    conversation itself has no such rule. The text of the prompt is read on
    every wire: the system and the user text of every scripted request are
    those of the first, so that an adapter says nothing of its own there in
    the middle of a call. And the text of a tool result is read on every
    wire: what a request carries as the result of a call is a text the
    attached ToolSet handed back for that tool, with whatever a wrapper of
    the workflow added and nothing else.

    A write that ran in a host was answered by no ToolSet object of the test
    process; of the result of a tool that the attached ToolSet never
    answered here, nothing is said.
    """

    problems = list(obs.provider.foreign_messages)
    requests = [
        (index + 1, request)
        for index, request in enumerate(obs.provider.requests)
        if request.matched
    ]
    for part in ("system", "user"):
        changed = [
            number
            for number, request in requests
            if getattr(request, part) != getattr(requests[0][1], part)
        ]
        if changed:
            problems.append(
                f"the {part} text of {len(changed)} request(s) is not that of the "
                f"first request of the call; the first is request {changed[0]}"
            )
    tool_of = {
        call.call_id: ref.bindings[call.slot].name
        for call in obs.provider.sent_calls
        if call.slot in ref.bindings
    }
    handed_back: dict[str, set[str]] = {}
    for item in obs.toolset_returned:
        handed_back.setdefault(item.name, set()).add(item.result)
    for number, request in requests:
        for call_id, text in request.tool_results:
            name = tool_of.get(call_id)
            texts = None if name is None else handed_back.get(name)
            if texts is not None and text not in texts:
                problems.append(
                    f"request {number} carries a result of {name} for {call_id} "
                    "that is no text the attached ToolSet handed back"
                )
    if not problems:
        return []
    return [Deviation("request.content:messages", "; ".join(problems))]


def partial_text_kept(obs: Observation, _ref: Reference) -> list[Deviation]:
    """What the model had written when the turn limit ended the call is the final message.

    A call that ends at its turn limit without an error returns the partial
    answer, and nothing in its place. Where one further request without
    tools was asked and answered, that answer is the partial answer; where
    the model was never asked again it had written no text, and the final
    message is empty: a warning of the adapter's own, or any other text, is
    not what the model said.

    So the check can fail for text that was dropped only where that further
    request was answered. Where the model is never asked again, the script
    has no text to keep: there it holds that the result carries no text of
    the adapter's own, and says nothing of an adapter that would drop what
    a model wrote beside its tool calls.
    """

    assert obs.result is not None
    if obs.result.error is not None:
        return []  # an error result promises no answer; clause C2 reports the kind
    kept = PARTIAL_TEXT if obs.provider.tail_served else ""
    if obs.result.final_message == kept:
        return []
    return [
        Deviation(
            "text:mismatch",
            "final_message is not the partial answer of the call that ended at its "
            "turn limit"
            + (
                ": the answer to the one further request without tools"
                if obs.provider.tail_served
                else ": the model had written no text, so it is empty"
            ),
        )
    ]


def _reads_and_writes_in_turn() -> ProviderScript:
    """Read, write, and again: every write replaces the file with a longer text.

    The write tool replaces an existing file only for the hash of what it
    replaces, and never by the same text. So each write but the first names
    the hash of the text before it, and every one of them succeeds.
    """

    steps: list[Call | Say] = []
    text, before = "", None
    for number in range(1, CATALOGUE_ROUNDS + 1):
        steps.append(Call(READ_A))
        text += f"line {number} of the matrix writer scenario\n"
        arguments = {"file_path": WRITE_FILE, "content": text}
        if before is not None:
            arguments["expected_sha256"] = hashlib.sha256(before.encode("utf-8")).hexdigest()
        steps.append(Call(WRITE, arguments))
        before = text
    return ProviderScript((*steps, Say(TEXT)))


# -- variant transformations ------------------------------------------------


# The name the protocol gives the field of a trace that holds the model the
# provider served: taken from the protocol's own type, from no adapter.
SERVED_MODEL_FIELD = "served_model"


def mismatch_names_the_model_field(obs: Observation, _ref: Reference) -> list[Deviation]:
    """An error of incomplete evidence names the served-model field as what is missing.

    Asked of a result whose error is of that kind: an entry of its
    ``missing`` contains the name of the trace's field for the served model.
    No spelling is asked around the name. A result of another kind is held
    by the clause about the kind, and nothing is said of it here.
    """

    assert obs.result is not None
    error = obs.result.error
    if error is None or str(error.kind) != "incomplete_evidence":
        return []
    missing = tuple(str(entry) for entry in getattr(error, "missing", ()))
    if any(SERVED_MODEL_FIELD in entry for entry in missing):
        return []
    return [
        Deviation(
            f"error.missing:{SERVED_MODEL_FIELD}",
            f"the incomplete evidence names {list(missing)} as missing and not the "
            "field of the trace for the model that was served",
        )
    ]


def _observed_and_dispatched(plan: Plan) -> Plan:
    """A route that admits a pin on the served model and leaves it out observes only.

    The reply under another id is taken: its call is dispatched, and a
    second reply under the same id ends the turn. A success, with the id the
    provider named in the trace.
    """

    return replace(
        plan,
        script=ProviderScript((ServedAs(Call(READ_A)), ServedAs(Say(TEXT)))),
        expect=replace(CLEAN, dispatch=(READ_A,)),
        checks=(results_replayed, history_replayed),
    )


def _swap_check(plan: Plan, old: Check, new: Check | None) -> Plan:
    checks = tuple(check for check in plan.checks if check is not old)
    return replace(plan, checks=checks if new is None else (*checks, new))


def _tool_rounds_then(text: str) -> ProviderScript:
    calls = tuple(Call(READ_A) for _ in range(TURN_LIMIT + 1))
    return ProviderScript((*calls, Say(text)))


def _final_request_without_tools(plan: Plan) -> Plan:
    tail = Say(PARTIAL_TEXT)
    return replace(plan, script=replace(plan.script, tail=tail))


def _typed_after_overrun(plan: Plan) -> Plan:
    return replace(
        plan,
        script=_tool_rounds_then("answer after the limit was passed"),
        expect=replace(
            plan.expect, dispatch=(READ_A,) * (TURN_LIMIT + 1), dispatch_optional=()
        ),
    )


def _no_turn_control(plan: Plan) -> Plan:
    return replace(
        plan,
        script=_tool_rounds_then(TEXT),
        expect=replace(CLEAN, dispatch=(READ_A,) * (TURN_LIMIT + 1)),
        checks=(*plan.checks, no_turn_control),
    )


def _cost_reported_only(plan: Plan) -> Plan:
    return replace(plan, expect=CLEAN)


def _noted_after_overrun(plan: Plan) -> Plan:
    """The limit is noted after more rounds than it allows; the answer is kept."""

    overrun = _typed_after_overrun(_swap_check(plan, partial_text_kept, None))
    answer = overrun.script.steps[-1]
    assert isinstance(answer, Say)
    return replace(overrun, expect=replace(overrun.expect, final_text=answer.text))


def _no_turn_control_and_an_answer(plan: Plan) -> Plan:
    """No turn limit is claimed: the finite script ends with its answer."""

    return _no_turn_control(_swap_check(plan, partial_text_kept, None))


def _loop_ends_with_its_owner(plan: Plan) -> Plan:
    """The tool loop is a vendor's: it ends where the vendor's tool ends it.

    At the earliest after the round in which the budget was crossed, at the
    latest with the answer after the last tool round. The answer is asked
    for only while the loop still runs and is never required. Every call
    after the first may or may not be dispatched.
    """

    rounds = Loop(
        Call(READ_A, usage=LOOP_ROUND_USAGE), limit=LOOP_ROUNDS, at_least=OVERRUN_ROUND
    )
    return replace(
        plan,
        script=ProviderScript((rounds, Loop(Say(TEXT), limit=1, at_least=0))),
        expect=replace(plan.expect, dispatch_optional=(READ_A,) * (LOOP_ROUNDS - 1)),
    )


def _unstated(scenario: str, field_name: str, value: str) -> Callable[[Plan], Plan]:
    """The plan of a declared value for which nothing is stated yet: none.

    No column declares the value and the scenario does not say what it owes
    under it. A column that comes to declare it cannot run the cell before
    that is stated: the cell is an error of the matrix, never a pass.
    """

    def refuse(_plan: Plan) -> Plan:
        raise MatrixSetupError(
            f"scenario {scenario} states nothing for {field_name}={value!r}: what "
            "a route that declares it owes is stated before the cell is played"
        )

    return refuse


def _complete_reply(plan: Plan) -> Plan:
    """No response cap is claimed: the reply is complete, the finish a clean one.

    The plan of the plain text scenario in full, and one assertion more: no
    request carries a cap on the tokens of the response.
    """

    return replace(
        plan,
        script=ProviderScript((Say(TEXT),)),
        expect=replace(CLEAN, events=("request/-",)),
        checks=(*_CLEAN_TEXT_CHECKS, no_response_cap),
    )


def _statement_owed(value: str, words: str) -> Callable[[Plan], Plan]:
    """The plan of a declared deadline whose contract is not stated yet.

    The row of the stall gives two values of ``deadline`` a variant that
    names the limit guard, which the raw lens does not bind, and no column
    declares either. The table is total, so each has its entry; a column
    that came to declare one could not run the cell before the statement of
    what the variant asserts exists.
    """

    def owed(_plan: Plan) -> Plan:
        raise MatrixSetupError(
            f"a route that declares deadline={value!r} is owed the variant "
            f"{words!r} of the deadline scenarios, which names the limit guard "
            "and is not stated for the raw lens yet: state what it asserts, and "
            "in which lens, before a column that declares the value runs the cell"
        )

    return owed


def _at_a_boundary_after_release(plan: Plan) -> Plan:
    """The route stops only between two steps: it returns once the handler has left.

    The cell lets the held handler go a stated time after the deadline,
    while the call still runs, and holds the return to that release.
    """

    assert plan.conduct is not None
    return replace(
        plan, conduct=replace(plan.conduct, release="after_deadline", bounds=("release",))
    )


def _at_the_next_boundary(plan: Plan) -> Plan:
    """The stalled request is not cut: the cancellation is returned by the deadline."""

    assert plan.conduct is not None
    return replace(plan, conduct=replace(plan.conduct, cut_seconds=None))


def _at_a_boundary_after_close(plan: Plan) -> Plan:
    """The held handler is let go a stated time after ``close()``; then the call returns."""

    assert plan.conduct is not None
    return replace(
        plan,
        conduct=replace(
            plan.conduct,
            release="after_close",
            bounds=("release", "deadline"),
            cut_seconds=None,
        ),
    )


def _ends_at_its_deadline_as_the_stall(plan: Plan) -> Plan:
    """No cancellation: ``close()`` changes nothing and the call ends as the stall does."""

    assert plan.conduct is not None
    return replace(
        plan,
        expect=_TIMED_OUT,
        conduct=replace(plan.conduct, cut_seconds=None, not_before_deadline=True),
    )


def _ends_at_its_deadline_as_the_blocked_read(plan: Plan) -> Plan:
    """No cancellation: the call ends as the deadline during a read does on such a route."""

    assert plan.conduct is not None
    return replace(
        plan,
        expect=replace(_TIMED_OUT, dispatch=(READ_A,)),
        conduct=replace(
            plan.conduct,
            release="after_deadline",
            bounds=("release",),
            cut_seconds=None,
            not_before_deadline=True,
        ),
    )


# Declared differences (the ledger holds their citation, owner and scope).
D_STRUCT_UNQUALIFIED = "D-STRUCT-UNQUALIFIED"
D_OAI_SCHEMA_FINAL = "D-OAI-SCHEMA-FINAL"
D_CDX_SERVED = "D-CDX-SERVED"
D_COST_BOUND_UNRANKED = "D-COST-BOUND-UNRANKED"


def _schema_in_one_final_request(plan: Plan) -> Plan:
    """One further no-tools request carries the schema and answers with it.

    The alternative restates where the schema *field* travels and whose answer
    is returned. What the declared value says about the prompt stays: a route
    that declares the schema in the prompt still owes the schema text there.
    """

    kept = tuple(
        check for check in plan.checks if check not in (schema_in_field, schema_in_prompt)
    )
    prompt_half = (schema_text_in_prompt,) if schema_in_prompt in plan.checks else ()
    return replace(
        plan,
        script=replace(plan.script, tail=Say(SCHEMA_FINAL_ANSWER)),
        checks=(*kept, *prompt_half, schema_in_final_request, finalisation_is_the_answer),
    )


def served_model_unnamed(obs: Observation, _ref: Reference) -> list[Deviation]:
    """The trace records no served model where the wire names none."""

    assert obs.result is not None
    trace = obs.result.execution_trace
    if trace is None or trace.served_model is None:
        return []  # a missing trace is clause C7's digest.missing:trace
    return [
        Deviation(
            "digest.mismatch:served_model",
            "the wire of the route names no served model and the trace records "
            f"{trace.served_model!r} as the one the provider served",
        )
    ]


def _trace_names_no_served_model(plan: Plan) -> Plan:
    """The wire has no served-model field: the trace must not present one.

    The alternative adds an assertion and takes none away. Clause C7 compares
    the served model only where the provider named one, so on such a wire it
    is silent about the field; this states what the result owes instead.
    """

    return replace(plan, checks=(*plan.checks, served_model_unnamed))


def _held_to_the_diagnostic_marks(plan: Plan) -> Plan:
    """A guarded result of a cost-bounded route is diagnostic, not rankable.

    The alternative is asserted by clause C5 itself, wherever the routing
    file of the cell's route opts into the cost upper bound: the plan of the
    scenario does not change. It is defined here so that the ledger can list
    the cells it is in force on, and static cell H03 holds that list to
    exactly the guarded cells of such routes.
    """

    return plan


# -- the catalogue -----------------------------------------------------------

_SUITE_PARSERS = ("workflows/deepswe.py", "workflows/swebench.py")


def suite_default_max_turns() -> int:
    """The larger ``--max-turns`` default of the two suite parsers.

    Read from their source at run time; the matrix imports nothing above the
    backend boundary.
    """

    root = Path(__file__).resolve().parents[2]
    defaults: list[int] = []
    for relative in _SUITE_PARSERS:
        found: list[int] = []
        for node in ast.walk(ast.parse((root / relative).read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "--max-turns"
            ):
                found.extend(
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "default"
                    and isinstance(keyword.value, ast.Constant)
                    and type(keyword.value.value) is int
                )
        if not found:
            raise MatrixSetupError(f"{relative} declares no integer --max-turns default")
        defaults.extend(found)
    return max(defaults)


# A FALLBACK cites the product by file and line. The line numbers of the
# citations in this module are those of the initial release of each
# repository; the words in parentheses say what is cited there.
_OPTIONAL_TOOL_EVENTS = FALLBACK(
    "tool_events_optional",
    "assay/src/assay/backend/features.py:26 (usage and terminal_only promise no "
    "tool events)",
)
# S01, and S11, which is S01 at the suite's turn count. That the wire carries
# the route's model and that served equals requested is clause C7, on every
# cell that expects a result.
_CLEAN_TEXT_CHECKS: tuple[Check, ...] = (prompt_reaches_provider, one_request_event)
_REFUSAL = Expectation(
    kind="content_rejected", stop=(), stop_not=("end_turn",), retriable=False, flags=()
)
# A refusal the provider bills for keeps its usage ("usage kept if reported").
_BILLED_REFUSAL = replace(_REFUSAL, flags=("token_usage_complete", "cost_complete"))


# A turn that never reached its terminal ends as a transport error whose stop
# reason is none of those a workflow reads as an outcome of its own, and the
# result says that the terminal is missing: the terminal flag is expected
# false. No event is asked beyond the one terminal event of a complete trace,
# and no flag of the error is asserted.
_BROKEN_TURN = Expectation(
    kind="transport",
    stop=(),
    stop_not=READ_STOP_REASONS,
    flags=("trace_complete",),
    flags_false=("terminal_complete",),
)


# A call on a route without its credential makes no provider request and
# returns a transport error that is not retried. Its stop reason is none of
# those a workflow reads as an outcome of its own; no spelling is asked. The
# trace is the adapter's own record of what it did, so its flag is owed and
# the trace ends in its one terminal event; no other event is asked, and a
# request event that stands for the attempt is neither asked nor refused. No
# turn was started and no usage record can have arrived: the terminal flag
# and the two accounting flags are not asked, and none is expected false.
# Clause C6 still holds what the result reports to what the provider billed,
# which is nothing.
_WITHOUT_A_REQUEST = Expectation(
    kind="transport",
    stop=(),
    stop_not=READ_STOP_REASONS,
    retriable=False,
    flags=("trace_complete",),
)


# A reply answered under another model than the route asked for is no result
# of the route. The error is incomplete evidence with the stop reason of a
# model mismatch, and nothing of the reply is dispatched. The cost flag is
# expected false: the price of the model that answered is not the route's.
# The row excepts no other accounting flag: the one reply is billed and
# reported, so the token usage flag is owed. The terminal flag is not
# asked: the row names none, and no clause says whether a result that is
# refused for the identity of the model that answered owns the terminal the
# flag speaks of. The trace flag is owed. The retriable flag is not
# asserted, and no event is asked beyond the terminal one.
_SERVED_UNDER_ANOTHER_ID = Expectation(
    kind="incomplete_evidence",
    stop=("model_mismatch",),
    flags=("token_usage_complete", "trace_complete"),
    flags_false=("cost_complete",),
)


# A call that ends at its deadline: a transport error that is not retried,
# with the stop reason of a deadline and its limit event. Nothing ended the
# turn, so the token, the cost and the terminal flag are each expected false.
_TIMED_OUT = Expectation(
    kind="transport",
    stop=("timeout",),
    retriable=False,
    flags=("trace_complete",),
    flags_false=EXPECTED_FALSE_FLAGS,
    events=("limit/deadline",),
)
# A call that ``close()`` ended: the cancellation, with an event of that kind
# under any name. Whether it may be retried is not asserted: the consumers
# read the kind. The three flags are expected false as at a deadline.
_CANCELLED = Expectation(
    kind="cancellation",
    stop=("cancelled",),
    flags=("trace_complete",),
    flags_false=EXPECTED_FALSE_FLAGS,
    events=("cancellation/-",),
)
_LIMITS_MODULE = (
    "crucible/backends/limits.py:1366-1382 (the limit module refuses a "
    "diagnostic invocation on this deadline before transport: it binds "
    "hard_invocation and request_only only)"
)
_CANCELLATION_VOCABULARY = (
    "assay/src/assay/backend/features.py:35-37 (the values of cancellation: a "
    "route of this value stops a call between two of its steps, if at all)"
)
# The table of ``deadline``, for the stall and for the deadline during a
# tool call.
_BY_DEADLINE: Mapping[str, Any] = {
    "hard_invocation": BASE,
    "request_only": BASE,
    "post_hoc": FALLBACK("guard_converts_late_result", _LIMITS_MODULE),
    "unsupported": FALLBACK("guard_refuses_route", _LIMITS_MODULE),
}
_DEADLINE_FALLBACKS: Mapping[str, Callable[[Plan], Plan]] = {
    "guard_converts_late_result": _statement_owed(
        "post_hoc", "the guard converts a late result"
    ),
    "guard_refuses_route": _statement_owed("unsupported", "the guard refuses the route"),
}
_DEADLINE_POLICY = replace(BASE_POLICY, timeout_seconds=DEADLINE_SECONDS)
# The cancellation scenarios: the deadline lies the time of a cut beyond the
# deadline of the others (see :data:`CUT_SECONDS`).
_CANCELLATION_POLICY = replace(
    BASE_POLICY, timeout_seconds=DEADLINE_SECONDS + CUT_SECONDS
)


def _blocked_handler(letter: str, title: str, toolset: ToolsetSpec, plan: Plan) -> Scenario:
    """A form of the deadline during a tool call: one request, one call, a held handler.

    The token, cost and terminal flags are expected false here as on the
    stall: the turn did not reach its terminal. It was cut at the deadline
    or by close(), or, on a route that can stop only at a boundary, ended by
    the adapter at its first boundary after the handler was released. On a
    route whose vendor tool owns the loop nothing shows that usage relayed
    before the terminal is final, and the budget ledgers decide on the two
    accounting flags alone. Read so by a decision panel under a standing
    instruction of the release owner, who may overrule. Whether a route
    whose adapter owns the loop may flag the reply it received complete, as
    it does where it ends a turn at its dispatch budget (S12), is owed a
    ruling.

    What the cell holds is a thread: the one on which the adapter, its
    bridge or its tool server dispatched the call in the test process. No
    adapter can end it. The cell asserts of that thread only that the call
    does not wait for it where the route declares that it can cut a call.
    It does not assert that no handler is still running when the call has
    returned ("host and bridge stopped" is read as: no child process, no
    listener), and it does not say what an adapter owes that finds a
    dispatch thread it cannot settle. So an adapter that leaves the thread
    behind without a word passes, and one that reports it by a raise out of
    the call deviates (clause C1). Which of the two the contract wants is
    not stated: it is owed a ruling, with the same question on the
    cancellation while a read handler is held.

    Ruled on 2026-11-27 by a decision panel under a standing instruction of
    the release owner, who may overrule: the form stands as stated. The call
    owes the typed result of the deadline, and a raise out of the call is a
    deviation of the adapter, on a reference column as on any other; the
    fixture is not changed for it. What an adapter owes beside that result
    for a dispatch thread it could not settle is left a question.
    """

    return Scenario(
        id=f"S16{letter}",
        set=PACK_B,
        title=title,
        toolset=toolset,
        lenses=("raw",),
        policy=_DEADLINE_POLICY,
        plan=lambda: plan,
        differences={
            "deadline": _BY_DEADLINE,
            "cancellation": {
                "process_kill": BASE,
                "interruptible": BASE,
                "boundary_only": FALLBACK(
                    "at_a_boundary_after_release", _CANCELLATION_VOCABULARY
                ),
                "unsupported": FALLBACK(
                    "at_a_boundary_after_release", _CANCELLATION_VOCABULARY
                ),
            },
        },
        fallbacks={
            **_DEADLINE_FALLBACKS,
            "at_a_boundary_after_release": _at_a_boundary_after_release,
        },
    )


def _cancelled_stall(letter: str, on_cancellation: str, expect: Expectation) -> Scenario:
    """A form of the cancellation of a stalled request, by the policy's value."""

    return Scenario(
        id=f"S30{letter}",
        set=PACK_B,
        title=f"Cancellation by close(), on_cancellation={on_cancellation}",
        toolset=None,
        lenses=("raw",),
        policy=replace(_CANCELLATION_POLICY, on_cancellation=on_cancellation),  # type: ignore[arg-type]
        # The provider takes the one request and never answers it, in every
        # variant; the cell closes the backend from another thread when the
        # request has been taken. A route that declares that it can cut a
        # call returns within the time of a cut; the deadline of the call
        # lies well beyond it, so a call that ran on to its deadline shows.
        plan=lambda: Plan(
            ProviderScript((Stall(STALL_MARKER),)),
            expect=expect,
            conduct=Conduct(close_at="stall", cut_seconds=CUT_SECONDS),
        ),
        differences={
            "cancellation": {
                "process_kill": BASE,
                "interruptible": BASE,
                "boundary_only": FALLBACK("at_the_next_boundary", _CANCELLATION_VOCABULARY),
                "unsupported": FALLBACK(
                    "ends_at_its_deadline_as_the_stall", _CANCELLATION_VOCABULARY
                ),
            }
        },
        fallbacks={
            "at_the_next_boundary": _at_the_next_boundary,
            "ends_at_its_deadline_as_the_stall": _ends_at_its_deadline_as_the_stall,
        },
    )


def _refusal(form: str, letter: str, *, billed: bool) -> Scenario:
    expect = _BILLED_REFUSAL if billed else _REFUSAL
    return Scenario(
        id=f"S23{letter}",
        set=PACK_A,
        title=f"Refusal ({form})",
        toolset=None,
        lenses=("raw",),
        plan=lambda: Plan(ProviderScript((Refuse(form),)), expect=expect),  # type: ignore[arg-type]
    )


_ALL: tuple[Scenario, ...] = (
    Scenario(
        id="S00",
        set=STARTER,
        title="Seam canary",
        toolset=None,
        lenses=("raw",),
        plan=lambda: Plan(ProviderScript((Say("matrix canary answer"),))),
        canary=True,
    ),
    Scenario(
        id="S01",
        set=STARTER,
        title="Text, clean finish",
        toolset=None,
        lenses=LENSES,
        plan=lambda: Plan(
            ProviderScript((Say(TEXT),)),
            expect=replace(CLEAN, events=("request/-",)),
            checks=_CLEAN_TEXT_CHECKS,
        ),
    ),
    Scenario(
        id="S03",
        set=STARTER,
        title="Tool loop, two rounds",
        toolset=READ,
        lenses=LENSES,
        plan=lambda: Plan(
            ProviderScript((Call(READ_A), Call(READ_B), Say(TEXT))),
            expect=replace(CLEAN, dispatch=(READ_A, READ_B)),
            checks=(results_replayed, history_replayed, tool_events_required),
        ),
        differences={
            "event_detail": {
                "tool_calls": BASE,
                "agent_graph": BASE,
                "usage": _OPTIONAL_TOOL_EVENTS,
                "terminal_only": _OPTIONAL_TOOL_EVENTS,
            }
        },
        fallbacks={
            "tool_events_optional": lambda plan: _swap_check(
                plan, tool_events_required, tool_events_paired
            )
        },
    ),
    Scenario(
        id="S04",
        set=STARTER,
        title="Tool loop, then schema answer (planner shape)",
        toolset=READ,
        lenses=LENSES,
        plan=lambda: Plan(
            ProviderScript((Call(READ_A), Say(SCHEMA_ANSWER))),
            # No verbatim text: an adapter may serialise the object again.
            # The answer is compared as a JSON value by the first check.
            expect=replace(CLEAN, final_text=None, dispatch=(READ_A,)),
            response_schema=SCHEMA,
            checks=(schema_answer_returned, history_replayed, schema_in_field),
        ),
        differences={
            "structured_output": {
                "native": BASE,
                "prompt_only": FALLBACK(
                    "schema_in_prompt",
                    "crucible/backends/features.py:143-145 (one canonical schema "
                    "through the Assay prompt ABI)",
                ),
                "unsupported": FALLBACK(
                    "schema_unclaimed",
                    "crucible/backends/features.py:79-81 (no native-schema "
                    "conformance claim)",
                    difference=D_STRUCT_UNQUALIFIED,
                ),
            }
        },
        fallbacks={
            "schema_in_prompt": lambda plan: _swap_check(
                plan, schema_in_field, schema_in_prompt
            ),
            "schema_unclaimed": lambda plan: _swap_check(plan, schema_in_field, None),
        },
        alternatives={D_OAI_SCHEMA_FINAL: _schema_in_one_final_request},
    ),
    Scenario(
        id="S10a",
        set=PACK_A,
        title="Turn limit, on_max_turns=raise",
        toolset=READ,
        lenses=LENSES,
        plan=lambda: Plan(
            # Every turn of the count is asked of the provider: a limit that
            # is typed after fewer leaves the loop unconsumed.
            ProviderScript((Loop(Call(READ_A), limit=TURN_LIMIT, at_least=TURN_LIMIT),)),
            expect=Expectation(
                kind="max_turns",
                stop=("max_turns",),
                retriable=False,
                events=("limit/max_turns",),
                dispatch=(READ_A,),
                dispatch_optional=(READ_A,),
            ),
            max_turns=TURN_LIMIT,
            checks=(history_replayed,),
        ),
        differences={
            "turn_limit": {
                "hard": BASE,
                "provider_stop": BASE,
                "round_boundary": FALLBACK(
                    "final_request_without_tools",
                    "crucible/backends/openai_compat.py:1619-1624 (forced "
                    "finalisation after the last tool round)",
                ),
                "post_hoc": FALLBACK(
                    "typed_after_overrun",
                    "crucible/backends/limits.py:364-373 (post_hoc is terminal "
                    "accounting)",
                ),
                "unsupported": FALLBACK(
                    "no_turn_control",
                    "crucible/backends/features.py:39-40 (no reviewed max-turn "
                    "request control)",
                ),
            }
        },
        fallbacks={
            "final_request_without_tools": _final_request_without_tools,
            "typed_after_overrun": _typed_after_overrun,
            "no_turn_control": _no_turn_control,
        },
    ),
    Scenario(
        id="S11",
        set=PACK_A,
        title="Suite default turn count",
        toolset=None,
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Say(TEXT),)),
            expect=replace(CLEAN, events=("request/-",)),
            max_turns=suite_default_max_turns(),
            checks=(*_CLEAN_TEXT_CHECKS, turn_control_as_asked),
        ),
        # An enforced turn limit is declared as hard or as provider_stop, and
        # the product places both where the provider is. Such a route owes
        # the count in every request on a wire that has a field for one. The
        # other values enforce nothing there: the adapter counts between
        # rounds or after the fact, or the route claims no turn limit.
        differences={
            "turn_limit": {
                "hard": FALLBACK(
                    "turn_count_in_request",
                    "crucible/backends/limits.py:364-373 (hard is enforced at "
                    "the provider)",
                ),
                "provider_stop": FALLBACK(
                    "turn_count_in_request",
                    "crucible/backends/limits.py:364-373 (provider_stop is "
                    "enforced by the provider)",
                ),
                "round_boundary": BASE,
                "post_hoc": BASE,
                "unsupported": BASE,
            }
        },
        fallbacks={
            "turn_count_in_request": lambda plan: replace(
                plan, checks=(*plan.checks, turn_count_in_request)
            ),
        },
    ),
    Scenario(
        id="S12",
        set=PACK_A,
        title="Workflow dispatch-budget denial",
        toolset=READ_BOUNDED(1),
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Call(READ_A), Call(READ_B))),
            expect=Expectation(
                kind="max_turns",
                # Both spellings are accepted, by a decision of the release
                # owner taken at its recommended default: the adapters do
                # not agree on one, and the matrix leaves them as they are.
                stop=("tool_budget_exhausted", "tool_dispatch_budget_exhausted"),
                retriable=False,
                events=("limit/tool_dispatch_budget",),
                dispatch=(READ_A,),
            ),
            checks=(history_replayed,),
        ),
    ),
    Scenario(
        id="S13",
        set=PACK_A,
        title="USD overrun, single response",
        toolset=None,
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Say(TEXT, TokenUsage(1000, 1000)),)),
            expect=Expectation(
                kind="budget_exhausted",
                stop=("budget_exhausted",),
                retriable=False,
                events=("limit/budget_usd",),
            ),
            budget_usd=BUDGET_USD,
        ),
        differences={
            "usd_limit": {
                "post_hoc": BASE,
                "round_boundary": BASE,
                "provider_stop": FALLBACK(
                    "ceiling_in_request",
                    "crucible/backends/limits.py:364-373 (provider_stop is "
                    "enforced by the provider)",
                ),
                "hard": FALLBACK(
                    "ceiling_without_overshoot",
                    "crucible/backends/limits.py:1339-1345 (hard is the only "
                    "admitted USD ceiling)",
                ),
                "unsupported": FALLBACK(
                    "cost_reported_only",
                    "assay/src/assay/backend/features.py:27-29 (unsupported "
                    "claims no enforcement)",
                ),
            }
        },
        fallbacks={
            "ceiling_in_request": lambda plan: replace(
                plan, checks=(*plan.checks, budget_in_request)
            ),
            "ceiling_without_overshoot": lambda plan: replace(
                plan, checks=(*plan.checks, budget_in_request, no_overshoot)
            ),
            "cost_reported_only": _cost_reported_only,
        },
    ),
    Scenario(
        id="S18",
        set=PACK_A,
        title="Tool-result byte budget",
        toolset=READ,
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Call(READ_A), Call(READ_B), Say(TEXT))),
            expect=replace(CLEAN, dispatch=(READ_A, READ_B)),
            context_budget=CONTEXT_BUDGET,
            checks=(context_bounded, first_result_truncated_not_dropped, history_replayed),
        ),
    ),
    _refusal("content_filter", "a", billed=True),
    _refusal("refusal_part", "b", billed=True),
    # An HTTP policy error carries no usage, so its accounting flags are waived.
    _refusal("http_policy", "c", billed=False),
    Scenario(
        id="S40",
        set=PACK_A,
        title="A write is progress",
        toolset=WRITER(2),
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript(
                (
                    Call(READ_A),
                    Call(READ_B),
                    Call(WRITE),
                    Call(READ_A),
                    Call(READ_B),
                    Say(TEXT),
                )
            ),
            expect=replace(CLEAN, dispatch=(READ_A, READ_B, WRITE, READ_A, READ_B)),
            # Six rounds stay below a third of max_turns, where one adapter
            # changes the advertised tools.
            max_turns=WRITER_TURNS,
            checks=(write_is_progress, history_replayed),
        ),
    ),
    # -- limits inside a tool loop ------------------------------------------
    Scenario(
        id="S10b",
        set=PACK_B,
        title="Turn limit, on_max_turns=continue_with_warning",
        toolset=READ,
        lenses=("raw",),
        policy=replace(BASE_POLICY, on_max_turns="continue_with_warning"),
        # The turn limit of the scenario above under the other policy: the
        # limit is reached and recorded as there, and the call returns no
        # error. What the model had written is kept as the final message.
        plan=lambda: Plan(
            ProviderScript((Loop(Call(READ_A), limit=TURN_LIMIT, at_least=TURN_LIMIT),)),
            expect=Expectation(
                stop=("max_turns",),
                events=("limit/max_turns",),
                dispatch=(READ_A,),
                dispatch_optional=(READ_A,),
            ),
            max_turns=TURN_LIMIT,
            checks=(partial_text_kept, history_replayed),
        ),
        differences={
            "turn_limit": {
                "hard": BASE,
                "provider_stop": BASE,
                "round_boundary": FALLBACK(
                    "final_request_without_tools",
                    "crucible/backends/openai_compat.py:1619-1624 (forced "
                    "finalisation after the last tool round)",
                ),
                "post_hoc": FALLBACK(
                    "noted_after_overrun",
                    "crucible/backends/limits.py:364-373 (post_hoc is terminal "
                    "accounting)",
                ),
                "unsupported": FALLBACK(
                    "no_turn_control",
                    "crucible/backends/features.py:39-40 (no reviewed max-turn "
                    "request control)",
                ),
            }
        },
        fallbacks={
            "final_request_without_tools": _final_request_without_tools,
            "noted_after_overrun": _noted_after_overrun,
            "no_turn_control": _no_turn_control_and_an_answer,
        },
    ),
    Scenario(
        id="S14",
        set=PACK_B,
        title="USD overrun inside a tool loop",
        toolset=READ,
        lenses=("raw",),
        # The second response takes the cost of the call above its budget.
        # Whose the tool loop is decides where the call can end: an adapter
        # that runs the loop itself asks for no third round; a loop that a
        # vendor's tool runs ends where that tool ends it, and the result
        # accounts for every round the provider billed until then.
        plan=lambda: Plan(
            ProviderScript(
                (
                    Loop(
                        Call(READ_A, usage=LOOP_ROUND_USAGE),
                        limit=OVERRUN_ROUND,
                        at_least=OVERRUN_ROUND,
                    ),
                )
            ),
            expect=Expectation(
                kind="budget_exhausted",
                stop=("budget_exhausted",),
                retriable=False,
                events=("limit/budget_usd",),
                dispatch=(READ_A,),
                dispatch_optional=(READ_A,),
            ),
            budget_usd=BUDGET_USD,
            checks=(history_replayed,),
        ),
        differences={
            "tool_loop": {
                "adapter_owned": BASE,
                "sdk_owned": FALLBACK(
                    "loop_ends_with_its_owner",
                    "crucible/backends/features.py:36 (the tool loop of the route "
                    "is run by the vendor's SDK or tool)",
                ),
                "none": FALLBACK(
                    "tool_loop_none",
                    "assay/src/assay/backend/features.py:24 (a route may declare "
                    "that it runs no tool loop)",
                ),
            }
        },
        fallbacks={
            "loop_ends_with_its_owner": _loop_ends_with_its_owner,
            "tool_loop_none": _unstated("S14", "tool_loop", "none"),
        },
    ),
    # -- tool calls that must not be dispatched -----------------------------
    Scenario(
        id="S05",
        set=PACK_B,
        title="Malformed tool arguments",
        toolset=READ,
        lenses=("raw",),
        # The model names an attached tool and writes no JSON object for its
        # arguments, three times: a text that is no JSON, a JSON text that
        # says "nothing", a JSON list. Nothing is dispatched for any of the
        # calls: not the tool with an empty object, not the value a text
        # decodes to, not the text. The model is answered each time, and
        # the turn ends as the provider ends it. Four requests and nothing
        # about a deadline: the invocation gets the long one.
        policy=replace(BASE_POLICY, timeout_seconds=CATALOGUE_TIMEOUT_SECONDS),
        plan=lambda: Plan(
            ProviderScript(
                (
                    *(Call(READ_A, malformed=text) for text in MALFORMED_ARGUMENT_TEXTS),
                    Say(TEXT),
                )
            ),
            checks=(history_replayed,),
        ),
    ),
    Scenario(
        id="S06",
        set=PACK_B,
        title="Tool outside the attached set",
        toolset=READ,
        lenses=("raw",),
        # The model names a tool it was never offered. Nothing reaches the
        # ToolSet for the call, neither that name nor an attached tool in
        # its place; the model is answered, and the turn ends as the
        # provider ends it.
        plan=lambda: Plan(
            ProviderScript((Call(UNATTACHED), Say(TEXT))),
            checks=(history_replayed,),
        ),
    ),
    # -- schema without tools; output-token cut; empty completion -----------
    Scenario(
        id="S02",
        set=PACK_B,
        title="Text with response schema",
        toolset=None,
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Say(SCHEMA_ANSWER),)),
            # No verbatim text: an adapter may serialise the object again.
            # The answer is compared as a JSON value by the first check.
            expect=replace(CLEAN, final_text=None),
            response_schema=SCHEMA,
            checks=(schema_answer_returned, schema_in_field),
        ),
        # The table of the scenario of a tool loop that ends in a schema
        # answer, value for value: where the schema travels follows the
        # declaration, with or without tools. A self-test holds the two
        # tables to each other. The declared difference of a schema
        # finalisation after a tool loop has no alternative here: without a
        # ToolSet there is no loop to finalise, and a second request is an
        # extra one.
        differences={
            "structured_output": {
                "native": BASE,
                "prompt_only": FALLBACK(
                    "schema_in_prompt",
                    "crucible/backends/features.py:143-145 (one canonical schema "
                    "through the Assay prompt ABI)",
                ),
                "unsupported": FALLBACK(
                    "schema_unclaimed",
                    "crucible/backends/features.py:79-81 (no native-schema "
                    "conformance claim)",
                    difference=D_STRUCT_UNQUALIFIED,
                ),
            }
        },
        fallbacks={
            "schema_in_prompt": lambda plan: _swap_check(
                plan, schema_in_field, schema_in_prompt
            ),
            "schema_unclaimed": lambda plan: _swap_check(plan, schema_in_field, None),
        },
    ),
    Scenario(
        id="S17",
        set=PACK_B,
        title="Output-token cut",
        toolset=None,
        lenses=("raw",),
        # The provider ends the reply at the cap of the route. The reply is
        # a complete reply of its wire that states its usage, so every flag
        # is owed and the usage is the bill. The result is an error of the
        # general kind, not retriable, told apart by its stop reason and by
        # a limit event of its own: never a turn limit, never an answer.
        # Not asserted: that no other limit event stands beside that one,
        # what the final message of the error result holds, and the value
        # of the cap a request carries.
        plan=lambda: Plan(
            ProviderScript((Cut(CUT_TEXT),)),
            expect=Expectation(
                kind="transport",
                stop=("max_tokens",),
                retriable=False,
                events=("limit/response_tokens",),
            ),
        ),
        # An enforced cap is declared as hard or as provider_stop, and the
        # product places both where the provider is: such a route owes the
        # cap in every request on a wire that has a field for one. A route
        # that claims no cap sends none, and its reply is complete.
        differences={
            "response_token_limit": {
                "hard": FALLBACK(
                    "cap_in_request",
                    "crucible/backends/limits.py:364-373 (hard is enforced at "
                    "the provider)",
                ),
                "provider_stop": FALLBACK(
                    "cap_in_request",
                    "crucible/backends/limits.py:364-373 (provider_stop is "
                    "enforced by the provider)",
                ),
                "round_boundary": BASE,
                "post_hoc": BASE,
                "unsupported": FALLBACK(
                    "complete_reply",
                    "assay/src/assay/backend/features.py:27-29 (unsupported "
                    "claims no enforcement)",
                ),
            }
        },
        fallbacks={
            "cap_in_request": lambda plan: replace(
                plan, checks=(*plan.checks, response_cap_in_request)
            ),
            "complete_reply": _complete_reply,
        },
    ),
    Scenario(
        id="S25",
        set=PACK_B,
        title="Empty completion, no tools",
        toolset=None,
        lenses=("raw",),
        # A clean stop that produced nothing is no answer: a transport error
        # that a caller may retry, at the one attempt the policy allows, and
        # never named after the stop reason of a success. The reply is
        # complete and states its usage, so every flag is owed.
        plan=lambda: Plan(
            ProviderScript((Empty(),)),
            expect=Expectation(
                kind="transport",
                stop=(),
                stop_not=READ_STOP_REASONS,
                retriable=True,
            ),
        ),
    ),
    # -- a reply under another model -----------------------------------------
    Scenario(
        id="S24",
        set=PACK_B,
        title="Served model differs",
        # The read fixture, and a tool call as the one reply: "nothing is
        # dispatched" is then an assertion. No model id stands here: the
        # dialect derives the other id from the one the request names.
        toolset=READ,
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((ServedAs(Call(READ_A)),)),
            expect=_SERVED_UNDER_ANOTHER_ID,
            checks=(mismatch_names_the_model_field, history_replayed),
        ),
        # Keyed by the state of the route extra that pins the served model,
        # as routing states it for the route of the cell's role. A route
        # that carries the pin, and a route on which routing does not admit
        # one, hold the base contract: the second can state no expected id
        # of its own and cannot leave a pin out, so nothing in it asks less
        # than a plain call does, and the product documents no observe-only
        # behaviour for it. The fallback is left to a route on which routing
        # admits the pin and which leaves it out.
        differences={
            SERVED_MODEL_EXTRA: {
                CARRIED: BASE,
                NOT_ADMITTED: BASE,
                ADMITTED_NOT_CARRIED: FALLBACK(
                    "observe_only",
                    "crucible/backends/openai_compat.py:731-741 (the served-model "
                    "check of the gateway adapter: generic operator routes retain "
                    "their historical observe-only behavior)",
                ),
            }
        },
        fallbacks={"observe_only": _observed_and_dispatched},
    ),
    # -- deadlines ----------------------------------------------------------
    Scenario(
        id="S15",
        set=PACK_B,
        title="Deadline while the provider stalls",
        toolset=None,
        lenses=("raw",),
        policy=_DEADLINE_POLICY,
        # The provider takes the one request and never answers it. Nothing
        # was reported and no terminal was received, so a result that flags
        # its tokens, its cost or its terminal complete states what it
        # cannot know.
        plan=lambda: Plan(
            ProviderScript((Stall(STALL_MARKER),)),
            expect=_TIMED_OUT,
            conduct=Conduct(not_before_deadline=True),
        ),
        differences={"deadline": _BY_DEADLINE},
        fallbacks=_DEADLINE_FALLBACKS,
    ),
    _blocked_handler(
        "a",
        "Deadline during a tool call (a read handler blocks)",
        READ,
        Plan(
            ProviderScript((Call(READ_A),)),
            expect=replace(_TIMED_OUT, dispatch=(READ_A,)),
            checks=(history_replayed,),
            conduct=Conduct(hold=READ_A, not_before_deadline=True),
        ),
    ),
    # -- broken terminals ---------------------------------------------------
    Scenario(
        id="S26",
        set=PACK_B,
        title="Malformed terminal",
        toolset=None,
        lenses=("raw",),
        # The reply is no terminal of its wire: a complete reply without a
        # turn, or a stream that ends in order before the frame that ends
        # the turn. The terminal flag is expected false on every wire. The
        # two accounting flags go by what the provider billed. On a wire
        # whose frames state the usage before the terminal it billed the
        # turn: the flags are not asked and a result may report that usage
        # and claim it. On a wire whose usage travels with the terminal it
        # billed nothing: a result that flags either complete deviates.
        plan=lambda: Plan(
            ProviderScript((Malformed(UNENDED_TEXT),)),
            expect=replace(_BROKEN_TURN, flags_false_if_unbilled=ACCOUNTING_FLAGS),
        ),
    ),
    Scenario(
        id="S27",
        set=PACK_B,
        title="Stream dropped mid-turn",
        toolset=None,
        lenses=("raw",),
        # The connection or the peer goes away inside the answer. Nothing
        # that arrived ends the turn, so the token, the cost and the
        # terminal flag are each expected false, and the text that had
        # arrived is no final message. On a stream of frames the frame that
        # carries the text arrives whole and the client holds the text; on
        # a wire of one document per reply no document arrives.
        plan=lambda: Plan(
            ProviderScript((Drop(DROPPED_TEXT),)),
            expect=replace(_BROKEN_TURN, flags_false=EXPECTED_FALSE_FLAGS),
            checks=(cut_off_text_withheld,),
        ),
    ),
    # -- a route without its credential ----------------------------------------
    Scenario(
        id="S28",
        set=PACK_B,
        title="Missing credential",
        toolset=None,
        lenses=("raw",),
        # Nothing is scripted: any request the provider sees is one too many.
        # The base policy, a retry budget of one. The scenario carries no
        # check of its own and no governing field.
        plan=lambda: Plan(ProviderScript(()), expect=_WITHOUT_A_REQUEST),
        credential_unset=True,
    ),
    # -- cancellation -------------------------------------------------------
    # The request was never answered, so the three flags are expected false
    # as on the stall. Where the policy asks for the raise, the call raises
    # its cancellation after the one request and no result clause applies.
    _cancelled_stall("a", "convert_to_backend_error", _CANCELLED),
    _cancelled_stall("b", "raise", Expectation(outcome="cancelled")),
    Scenario(
        id="S30c",
        set=PACK_B,
        title="Cancellation by close() while a read handler blocks",
        toolset=READ,
        lenses=("raw",),
        policy=_CANCELLATION_POLICY,
        # One request, answered with the read; the cell closes the backend
        # from another thread when the handler has been entered. The flags
        # are those of the deadline during a tool call, for the same ground
        # (see there): the reply was answered and the turn did not reach
        # its terminal. The backend is closed while the cell holds the
        # thread on which the read was dispatched, which no adapter can
        # end: what a close() owes that finds such a thread is not stated
        # (see the deadline during a tool call), and a raise of that close()
        # is a raise of close() like any other (clause C8). Ruled on
        # 2026-11-27 by a decision panel under a standing instruction of the
        # release owner, who may overrule: that raise is a deviation of the
        # cell, as stated here; what close() owes while a handler it cannot
        # end still runs is left a question.
        plan=lambda: Plan(
            ProviderScript((Call(READ_A),)),
            expect=replace(_CANCELLED, dispatch=(READ_A,)),
            checks=(history_replayed,),
            conduct=Conduct(hold=READ_A, close_at="hold", cut_seconds=CUT_SECONDS),
        ),
        differences={
            "cancellation": {
                "process_kill": BASE,
                "interruptible": BASE,
                "boundary_only": FALLBACK(
                    "at_a_boundary_after_close", _CANCELLATION_VOCABULARY
                ),
                "unsupported": FALLBACK(
                    "ends_at_its_deadline_as_the_blocked_read", _CANCELLATION_VOCABULARY
                ),
            }
        },
        fallbacks={
            "at_a_boundary_after_close": _at_a_boundary_after_close,
            "ends_at_its_deadline_as_the_blocked_read": (
                _ends_at_its_deadline_as_the_blocked_read
            ),
        },
    ),
    Scenario(
        id="S41",
        set=PACK_B,
        title="Writes run in a killable host",
        # The fixture of S40; the count plays no part here.
        toolset=WRITER(2),
        lenses=("raw",),
        plan=lambda: Plan(
            ProviderScript((Call(WRITE), Say(TEXT))),
            expect=replace(CLEAN, dispatch=(WRITE,)),
            max_turns=WRITER_TURNS,
            checks=(write_ran_in_a_host_that_is_gone, history_replayed),
        ),
    ),
    Scenario(
        id="S44",
        set=PACK_B,
        title="Empty final turn after a write",
        toolset=WRITER(2),
        lenses=("raw",),
        plan=lambda: Plan(
            # The last reply reasons and shows no text. The provider billed
            # output tokens for it, so it is a finished turn: a completion is
            # taken for stalled only where nothing was produced at all.
            ProviderScript((Call(WRITE), Reason())),
            expect=replace(CLEAN, final_text="", dispatch=(WRITE,)),
            max_turns=WRITER_TURNS,
            checks=(history_replayed,),
        ),
    ),
    Scenario(
        id="S45",
        set=PACK_B,
        title="Writer catalogue is stable",
        # As many reads in a row as the script holds reads are admitted. That
        # a write is credited as progress is another scenario's assertion;
        # this one is about what the model is offered and told.
        toolset=WRITER(CATALOGUE_ROUNDS),
        lenses=("raw",),
        policy=replace(BASE_POLICY, timeout_seconds=CATALOGUE_TIMEOUT_SECONDS),
        plan=lambda: Plan(
            _reads_and_writes_in_turn(),
            expect=replace(CLEAN, dispatch=(READ_A, WRITE) * CATALOGUE_ROUNDS),
            # Thirteen requests in thirteen turns: every turn of the budget.
            max_turns=CATALOGUE_TURNS,
            checks=(tools_are_the_attached_ones, messages_are_the_workflows, history_replayed),
        ),
    ),
    # -- a conversation request ---------------------------------------------
    Scenario(
        id="S50",
        set=PACK_B,
        title="Conversation request",
        toolset=None,
        lenses=("raw",),
        # The invocation asks to continue a conversation under a handle that
        # nobody holds. Whoever keeps the conversations of the wire refuses
        # it before a model turn is asked, so the script is empty, no usage
        # was reported and the three flags a scenario may waive are not
        # asked. The result is the typed stale conversation, with the
        # outcome of the request: no error a caller may try again, since
        # the same handle is refused again. Where a tool beside the adapter
        # keeps the conversations of the wire, that tool was asked for the
        # handle.
        plan=lambda: Plan(
            ProviderScript(()),
            expect=Expectation(
                kind="stale_conversation",
                stop=(),
                stop_not=READ_STOP_REASONS,
                retriable=False,
                flags=("trace_complete",),
            ),
            checks=(stale_outcome_of_the_request, handle_asked_of_its_keeper),
            conversation=CONVERSATION,
        ),
        # A route that declares continuity for a workflow answers the
        # request. One that declares none, or continuity for its own retries
        # only, refuses it before transport, with the one agreed type.
        differences={
            "session_continuity": {
                "workflow_bound": BASE,
                "retry_only": REFUSED(CONTINUITY_REFUSAL),
                "none": REFUSED(CONTINUITY_REFUSAL),
            }
        },
    ),
    Scenario(
        id="S51",
        set=PACK_B,
        title="Admission",
        toolset=None,
        lenses=("raw",),
        # No provider behaviour: nothing is called, so nothing may be asked.
        plan=lambda: Plan(ProviderScript(())),
        admission=True,
    ),
)

# A wire without a served-model field is a property of a kind's dialect, not of
# one scenario: every scenario that asserts a result defines the alternative
# of that declared difference, and the ledger says on which cells it is in
# force. The canary asserts none, and neither does an admission scenario,
# which calls nothing.
_ALL = tuple(
    scenario
    if scenario.canary or scenario.admission
    else replace(
        scenario,
        alternatives={**scenario.alternatives, D_CDX_SERVED: _trace_names_no_served_model},
    )
    for scenario in _ALL
)

# What the guarded lens asks of a route that opts into a cost upper bound is
# a property of the route, not of one scenario either: every scenario that
# runs guarded and asserts a result defines the alternative of that declared
# difference, and the ledger lists the cells of such routes.
_ALL = tuple(
    replace(
        scenario,
        alternatives={
            **scenario.alternatives,
            D_COST_BOUND_UNRANKED: _held_to_the_diagnostic_marks,
        },
    )
    if GUARDED in scenario.lenses and not scenario.canary and not scenario.admission
    else scenario
    for scenario in _ALL
)

SCENARIOS: Mapping[str, Scenario] = {scenario.id: scenario for scenario in _ALL}


def governing_fields(scenarios: Sequence[Scenario] = _ALL) -> tuple[str, ...]:
    """Every declared field that governs a defined scenario, in first-use order.

    A route extra that keys a table is no declared field and is not among
    them (:func:`keying_extras`).
    """

    fields: list[str] = []
    for scenario in scenarios:
        fields.extend(
            name
            for name in scenario.governing
            if name not in fields and name not in KEYING_EXTRAS
        )
    return tuple(fields)


def keying_extras(scenarios: Sequence[Scenario] = _ALL) -> tuple[str, ...]:
    """Every route extra that keys the table of a defined scenario, in first-use order."""

    extras: list[str] = []
    for scenario in scenarios:
        extras.extend(
            name for name in scenario.governing if name in KEYING_EXTRAS and name not in extras
        )
    return tuple(extras)


def difference_variants(scenarios: Sequence[Scenario] = _ALL) -> frozenset[str]:
    """Declared differences that a variant of a difference table asserts."""

    return frozenset(
        variant.difference
        for scenario in scenarios
        for table in scenario.differences.values()
        for variant in table.values()
        if variant.difference
    )


def difference_alternatives(
    scenarios: Sequence[Scenario] = _ALL,
) -> dict[str, frozenset[str]]:
    """Declared differences in force per cell: id -> the scenarios defining one."""

    found: dict[str, set[str]] = {}
    for scenario in scenarios:
        for name in scenario.alternatives:
            found.setdefault(name, set()).add(scenario.id)
    return {name: frozenset(ids) for name, ids in sorted(found.items())}


def _validate() -> None:
    if SERVED_MODEL_FIELD not in BackendExecutionTrace.__dataclass_fields__:
        raise MatrixSetupError(
            f"the trace of the protocol has no field {SERVED_MODEL_FIELD!r}: the "
            "scenario of a served model that differs names it"
        )
    for scenario in _ALL:
        if scenario.set not in SETS or not set(scenario.lenses) <= set(LENSES):
            raise MatrixSetupError(f"scenario {scenario.id} names an unknown set or lens")
        named = {
            variant.name
            for table in scenario.differences.values()
            for variant in table.values()
            if variant.kind == "FALLBACK"
        }
        if named != set(scenario.fallbacks):
            raise MatrixSetupError(
                f"scenario {scenario.id}: fallbacks {sorted(scenario.fallbacks)} do "
                f"not match the difference table {sorted(named)}"
            )


_validate()
