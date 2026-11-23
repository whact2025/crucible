# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The scenario catalogue: the starter set and pack A.

A scenario states, once and for every backend: the policy it passes (always a
complete ``BackendPolicy``, never an adapter default), what the provider does,
which ToolSet is attached, the lenses it runs in and the result every backend
must return. Where backends may legitimately differ, a difference table keyed
by the governing declared field selects ``BASE``, a named ``FALLBACK`` or
``REFUSED``; the table is total over the field's vocabulary. No scenario asks
which backend it runs on.

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
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from assay import BackendPolicy

from .contract import Check, Deviation, Expectation, Observation, Reference
from .declarations import BASE, FALLBACK, Differences, Variant
from .drivers.base import LENSES, MatrixSetupError
from .script import READ_A, READ_B, WRITE, Call, Loop, ProviderScript, Refuse, Say, TokenUsage
from .toolsets import READ, READ_BOUNDED, WRITER, ToolsetSpec, capability_of

GUARDED = "guarded"
STARTER = "starter"
PACK_A = "pack_a"
PACK_B = "pack_b"
SETS = (STARTER, PACK_A, PACK_B)

SYSTEM = "You are the matrix system prompt [canary-system-4f1c]."
USER = "Answer the matrix user prompt [canary-user-9b2e]."
TEXT = "matrix final answer"
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

# The base policy. No scenario relies on an adapter default: they differ.
BASE_POLICY = BackendPolicy(
    retry_budget=1,
    timeout_seconds=60,
    on_cancellation="convert_to_backend_error",
    on_max_turns="raise",
)

CLEAN = Expectation(final_text=TEXT)


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


def budget_in_request(obs: Observation, _ref: Reference) -> list[Deviation]:
    if any(request.budget_control == BUDGET_USD for request in _matched(obs)):
        return []
    return [Deviation("request.content:budget", "the USD ceiling is not in the request")]


def no_overshoot(obs: Observation, _ref: Reference) -> list[Deviation]:
    assert obs.result is not None
    if obs.result.usage.cost_usd <= BUDGET_USD:
        return []
    return [Deviation("usage.cost_usd:gt", "a hard USD ceiling was overshot")]


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


# -- variant transformations ------------------------------------------------


def _swap_check(plan: Plan, old: Check, new: Check | None) -> Plan:
    checks = tuple(check for check in plan.checks if check is not old)
    return replace(plan, checks=checks if new is None else (*checks, new))


def _tool_rounds_then(text: str) -> ProviderScript:
    calls = tuple(Call(READ_A) for _ in range(TURN_LIMIT + 1))
    return ProviderScript((*calls, Say(text)))


def _final_request_without_tools(plan: Plan) -> Plan:
    tail = Say("partial answer at the turn limit")
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
            ProviderScript((Loop(Call(READ_A), limit=TURN_LIMIT),)),
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
            checks=_CLEAN_TEXT_CHECKS,
        ),
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
            max_turns=24,
            checks=(write_is_progress, history_replayed),
        ),
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
    """Every declared field that governs a defined scenario, in first-use order."""

    fields: list[str] = []
    for scenario in scenarios:
        fields.extend(name for name in scenario.governing if name not in fields)
    return tuple(fields)


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
