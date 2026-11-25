# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The result contract: what a workflow can see of one backend call.

A cell is judged on an :class:`Observation` only: the result or the raise, the
provider ledger, the tool dispatch log, the limit terminal and what is still
running afterwards. The clauses C0-C10 are consumer-backed statements about
that observation; each returns deviation codes from one closed vocabulary, so
a ledger signature stays stable when a reply is re-recorded. Codes carry
types, phases, module names and relations, never messages or numbers.

Which clauses apply to a cell follows from the cell's definition (lens,
ToolSet, expected outcome), never from what the backend did. A clause that
applies returns ``ok``, deviation codes, ``not_evaluated`` (which must be
pinned in the inventory) or ``stood_for``: a raise stands for the result
clauses, and a guard rewrite stands for the kind and stop reason it replaced.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Mapping, NamedTuple, Sequence

import psutil

from assay import BackendFeatureSet, InvocationResult
from crucible.backends.features import feature_profile_digest
from crucible.backends.limits import InvocationLimitTerminal
from crucible.backends.pricing import ModelPricing

from .provider import ProviderLedger
from .toolsets import Dispatch, ToolBinding

FLAGS = (
    "token_usage_complete",
    "cost_complete",
    "terminal_complete",
    "trace_complete",
)
CLAUSES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10")
RESULT_CLAUSES = ("C2", "C3", "C4", "C6", "C7")
SCENARIO = "S"
# Stop reasons the limit guard writes when it replaces a result
# (crucible/backends/limits.py: _attempt_evidence_result,
# _incomplete_limit_evidence_result).
GUARD_STOP_REASONS = frozenset({"limit_contract_violated", "limit_evidence_incomplete"})
MAX_ERROR_TEXT = 2000
# What C6 asks only of a result that flags its cost complete.
PRICING_CODES = frozenset({"pricing:none", "pricing:unreconciled"})
# What C5 reports of a result the limit guard let stand and does not call
# rankable.
NOT_RANKABLE = "guard:not_rankable"
# What the limit guard names the terminal of a result it let stand under a
# cost policy (crucible/backends/limits.py).
COST_BOUND_REASON = "diagnostic_cost_upper_bound"
# What C5 reports instead where the route opts into a cost upper bound: the
# guard never calls a result of such a route rankable, so the result is held
# to the marks of a diagnostic one, and each mark that is missing has a code.
COST_BOUND_TERMINAL = "guard:cost_bound_terminal"
COST_BOUND_ATTEMPTS = "guard:cost_bound_attempts"
COST_BOUND_AUTHORITY = "guard:cost_bound_authority"
COST_BOUND_RANKABLE = "guard:cost_bound_rankable"
COST_BOUND_MARKS = frozenset(
    {COST_BOUND_TERMINAL, COST_BOUND_ATTEMPTS, COST_BOUND_AUTHORITY, COST_BOUND_RANKABLE}
)

# The closed deviation vocabulary: every code a cell may emit, with the clause
# it is booked under ("S" is the scenario contract, "D" the declaration lock).
_VOCABULARY: tuple[tuple[str, str], ...] = (
    (r"request\.(extra|unconsumed)", "C0"),
    (r"raise:\w+@after_request/[\w.]+", "C1"),
    (r"raise:\w+@before_request/[\w.]+", SCENARIO),
    (r"raise:\w+@close/[\w.]+", "C8"),
    (r"kind:\w+!=\w+", "C2"),
    (r"retriable:(True|False)", "C2"),
    (r"http_status:\w+!=\w+", "C2"),
    (r"stop:\w+!=\w+", "C3"),
    (r"event\.terminal:[\w.:-]+!=\w+", "C4"),
    (r"flag:(terminal_complete|trace_complete)=(True|False)", "C4"),
    (r"guard:\w+", "C5"),
    (r"flag:(token_usage_complete|cost_complete)=(True|False)", "C6"),
    (r"usage\.(input_tokens|output_tokens|cost_usd):(lt|gt)", "C6"),
    (r"usage\.cost_bounds:(none|uncovered|segments|counts|amounts)", "C6"),
    (r"pricing:(none|unreconciled)", "C6"),
    (r"digest\.(missing|mismatch):\w+", "C7"),
    (r"request\.content:model", "C7"),
    (r"served_model:not_requested", "C7"),
    (r"child:alive", "C8"),
    (r"secret:leak", "C9"),
    (r"text:unbounded", "C9"),
    (r"dispatch\.(extra|missing):\w+", "C10"),
    (r"event\.(missing|extra):\w+/[\w-]+", SCENARIO),
    (r"text:mismatch", SCENARIO),
    (r"request\.content:\w+", SCENARIO),
    (r"context:over_budget", SCENARIO),
    (r"progress:write_not_credited", SCENARIO),
    (r"admission:\w+:\w+!=\w+", SCENARIO),
    (r"decl\.(changed|over|under|mismatch):\w+", "D"),
    (r"static:\w+", "static"),
)


def clause_of(code: str) -> str:
    """Return the clause a code is booked under; an unknown code is an error."""

    for pattern, clause in _VOCABULARY:
        if re.fullmatch(pattern, code):
            return clause
    raise ValueError(f"deviation code {code!r} is not in the closed vocabulary")


class Deviation(NamedTuple):
    """One deviation: the code that is compared, and what was observed."""

    code: str
    detail: str


@dataclass(frozen=True)
class Raised:
    """An exception that left ``invoke`` (or build, attach or ``close``)."""

    type: str
    phase: Literal["before_request", "after_request", "close"]
    module: str
    # What a workflow would log. Never part of the code; C9 reads it.
    message: str = ""

    @property
    def code(self) -> str:
        return f"raise:{self.type}@{self.phase}/{self.module}"

    @classmethod
    def from_exception(cls, exc: BaseException, *, phase: str) -> Raised:
        """Name the module of the innermost frame that raised."""

        module = "unknown"
        traceback = exc.__traceback__
        while traceback is not None:
            module = str(traceback.tb_frame.f_globals.get("__name__", "unknown"))
            traceback = traceback.tb_next
        return cls(
            type=type(exc).__name__,
            phase=phase,  # type: ignore[arg-type]
            module=module,
            message=str(exc),
        )


@dataclass(frozen=True)
class SecondCall:
    """The guarded lens's second plain call on the backend the cell used."""

    refused: bool = False  # the limit guard refused it
    raised: str | None = None  # the type of any other exception it raised
    error_kind: str | None = None  # the error kind of the result it returned
    requests: int = 0  # provider requests it made


@dataclass(frozen=True)
class Settlement:
    """What the cell left behind after ``invoke`` and ``close()``."""

    children_alive: tuple[int, ...] = ()
    listeners: tuple[int, ...] = ()
    close_raised: Raised | None = None
    late_requests: int = 0


@dataclass(frozen=True)
class Observation:
    """Everything a cell may assert on."""

    result: InvocationResult | None
    raised: Raised | None
    provider: ProviderLedger
    toolset_log: tuple[Dispatch, ...]
    wrapper_snapshot: Any | None
    write_applied: bool | None
    declarations: BackendFeatureSet | None
    backend_name: str
    backend_model: str
    limit_terminal: InvocationLimitTerminal | None
    second_call: SecondCall | None
    settlement: Settlement
    wall_seconds: float
    debug_dump_text: str


@dataclass(frozen=True)
class Reference:
    """What the observation is compared with, none of it from the adapter."""

    routed: BackendFeatureSet
    # The model the route of the cell's role names: what the wire must carry.
    model: str
    # The endpoint that route names, as a run record may carry it (no
    # credential, no query); ``None`` where the route names none.
    endpoint: str | None
    pricing: ModelPricing
    secrets: tuple[str, ...]
    tool_authority_digest: str | None
    bindings: Mapping[str, ToolBinding]
    write_slot: str
    # Whether that route opts into a cost upper bound in its routing file: a
    # result of such a route is a cost interval, and C6 holds the interval.
    # The one selector of everything the contract asks of such a route.
    cost_bounded: bool = False
    # The cost policy that routing file names for a caller of the route: what
    # C5 holds the terminal of a guarded result to. ``None`` where the route
    # does not opt in.
    cost_policy: str | None = None
    # The shipped price row of the route's model, which is the rate table of
    # that policy: what C6 holds the two bounds of every segment of the
    # interval to. Never the matrix rate card, which stays the reference of
    # the reported cost on every route. ``None`` where no row is shipped.
    bound_pricing: ModelPricing | None = None


@dataclass(frozen=True)
class Expectation:
    """The result a scenario's selected variant expects."""

    outcome: Literal["result", "refused"] = "result"
    refused_type: str = ""
    kind: str = "none"
    stop: tuple[str, ...] = ("end_turn",)
    stop_not: tuple[str, ...] = ()
    retriable: bool | None = None
    flags: tuple[str, ...] = FLAGS
    events: tuple[str, ...] = ()
    final_text: str | None = None
    dispatch: tuple[str, ...] = ()
    dispatch_optional: tuple[str, ...] = ()

    @property
    def stop_label(self) -> str:
        return self.stop[0] if self.stop else "other"


Check = Callable[[Observation, Reference], Sequence[Deviation]]


@dataclass(frozen=True)
class ClauseOutcome:
    status: Literal["ok", "deviation", "not_evaluated", "stood_for"]
    deviations: tuple[Deviation, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class Report:
    """The outcome of every clause that applies to one cell."""

    outcomes: Mapping[str, ClauseOutcome]
    seam_problems: tuple[str, ...] = ()
    # Codes whose check the result's own flags switched off inside a clause
    # that was evaluated. A column-wide known gap is not owed there.
    unasked: frozenset[str] = frozenset()

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(
            item.code for outcome in self.outcomes.values() for item in outcome.deviations
        )

    def tripped(self) -> frozenset[str]:
        return frozenset(
            clause
            for clause, outcome in self.outcomes.items()
            if outcome.status == "deviation"
        )

    def evaluated(self) -> frozenset[str]:
        return frozenset(
            clause
            for clause, outcome in self.outcomes.items()
            if outcome.status in ("ok", "deviation")
        )

    def not_evaluated(self) -> frozenset[str]:
        return frozenset(
            clause
            for clause, outcome in self.outcomes.items()
            if outcome.status == "not_evaluated"
        )

    def describe(self) -> str:
        """Which clause, which code, what was observed: one line each."""

        lines: list[str] = []
        for clause, outcome in self.outcomes.items():
            if outcome.status == "deviation":
                lines.extend(
                    f"  {clause} {item.code}: {item.detail}"
                    for item in outcome.deviations
                )
            elif outcome.status != "ok":
                lines.append(f"  {clause} {outcome.status}: {outcome.reason}")
        return "\n".join(lines) or "  no deviation"


def applicable(
    *, lens: str, toolset: bool, expects_result: bool, call: bool = True
) -> tuple[str, ...]:
    """The clauses of a cell, from its definition alone.

    ``call`` is false for a cell that builds backends and calls none (an
    admission cell): the clauses about what ``invoke`` returns are then no
    part of it.
    """

    if not call:
        return ("C0", "C8", "C9")
    clauses = {"C0", "C1", "C8", "C9"}
    if expects_result:
        clauses.update(RESULT_CLAUSES)
    if lens == "guarded":
        clauses.add("C5")
    if toolset:
        clauses.add("C10")
    return tuple(clause for clause in CLAUSES if clause in clauses)


# -- settlement probe (C8) -------------------------------------------------


def _listening_ports(process: psutil.Process) -> frozenset[int]:
    connections = getattr(process, "net_connections", None) or process.connections
    return frozenset(
        int(item.laddr.port)
        for item in connections(kind="tcp")
        if item.status == psutil.CONN_LISTEN and item.laddr
    )


def _live_children(process: psutil.Process) -> frozenset[int]:
    live: set[int] = set()
    for child in process.children(recursive=True):
        try:
            if child.status() != psutil.STATUS_ZOMBIE:
                live.add(child.pid)
        except psutil.NoSuchProcess:
            continue
    return frozenset(live)


def settlement_baseline() -> tuple[frozenset[int], frozenset[int]]:
    """The test process's children and listening ports when a cell starts."""

    process = psutil.Process()
    return _live_children(process), _listening_ports(process)


def settle(
    baseline: tuple[frozenset[int], frozenset[int]],
    *,
    driver_children: Sequence[int],
    own_ports: Sequence[int],
    close_raised: Raised | None,
    grace_seconds: float = 0.5,
) -> Settlement:
    """Differential settlement: what exists now that did not when the cell began.

    A child is left when it is a new child of this process or one the driver
    names, whichever sees it: a detached process is no child any more.
    """

    process = psutil.Process()
    children_before, ports_before = baseline
    deadline = time.monotonic() + grace_seconds
    while True:
        left = (_live_children(process) - children_before) | frozenset(
            pid for pid in driver_children if _alive(pid)
        )
        if not left or time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    listeners = _listening_ports(process) - ports_before - frozenset(own_ports)
    return Settlement(
        children_alive=tuple(sorted(left)),
        listeners=tuple(sorted(listeners)),
        close_raised=close_raised,
    )


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# -- clauses ---------------------------------------------------------------


def _c0(obs: Observation) -> list[Deviation]:
    out: list[Deviation] = []
    if obs.provider.extra:
        out.append(
            Deviation(
                "request.extra",
                f"{obs.provider.extra} request(s) matched no scripted step "
                f"({obs.provider.request_count} seen in all)",
            )
        )
    if obs.provider.unconsumed:
        out.append(
            Deviation(
                "request.unconsumed",
                f"{obs.provider.unconsumed} scripted step(s) were never requested "
                f"({obs.provider.request_count} request(s) seen)",
            )
        )
    return out


def _guard_rewrote(obs: Observation) -> bool:
    result = obs.result
    if result is None:
        return False
    # Every call the guard lets through leaves a terminal; the runner refuses
    # a guarded cell whose driver did not bind the guard.
    assert obs.limit_terminal is not None
    return result.stop_reason in GUARD_STOP_REASONS or bool(obs.limit_terminal.overshoots)


def _c2(result: InvocationResult, exp: Expectation) -> list[Deviation]:
    out: list[Deviation] = []
    error = result.error
    got = "none" if error is None else str(error.kind)
    if got != exp.kind:
        out.append(Deviation(f"kind:{got}!={exp.kind}", f"error is {error!r}"))
    if (
        error is not None
        and exp.retriable is not None
        and bool(error.retriable) != exp.retriable
    ):
        out.append(
            Deviation(
                f"retriable:{bool(error.retriable)}",
                f"the scenario expects retriable={exp.retriable}",
            )
        )
    return out


def _c3(result: InvocationResult, exp: Expectation) -> list[Deviation]:
    stop = result.stop_reason
    if (exp.stop and stop not in exp.stop) or stop in exp.stop_not:
        accepted = f"one of {list(exp.stop)}" if exp.stop else f"not {list(exp.stop_not)}"
        return [
            Deviation(
                f"stop:{stop}!={exp.stop_label}",
                f"stop_reason is {stop!r}; the scenario accepts {accepted}",
            )
        ]
    return []


def _c4(result: InvocationResult, exp: Expectation) -> list[Deviation]:
    out: list[Deviation] = []
    flags = result.completeness
    if "terminal_complete" in exp.flags and not flags.terminal_complete:
        out.append(
            Deviation("flag:terminal_complete=False", "the result disowns its terminal")
        )
    if not flags.trace_complete:
        out.append(
            Deviation(
                "flag:trace_complete=False",
                "the trace is flagged incomplete, so its terminal cannot be checked",
            )
        )
        return out
    # With a complete trace assay guarantees exactly one terminal event, last.
    assert result.execution_trace is not None
    terminal = result.execution_trace.events[-1]
    if terminal.name != result.stop_reason:
        out.append(
            Deviation(
                f"event.terminal:{terminal.name or 'none'}!={result.stop_reason}",
                "the terminal event is not named after stop_reason",
            )
        )
    return out


def _diagnostic_marks(terminal: InvocationLimitTerminal, ref: Reference) -> list[Deviation]:
    """What C5 holds a guarded result to where the route opts into a cost bound.

    The limit guard never calls a result of such a route rankable, and the
    route's results are diagnostic by the product's own rule, so C5 does not
    ask "rankable" there. It asks for the marks of a diagnostic result
    instead, each read from the terminal the guard left: the terminal of the
    cost policy, under the policy the routing file names; exactly one
    physical attempt; the cost authority kept; and a result that is not
    called rankable. That the guard let the result stand and that a second
    call is answered are asked on every route alike.
    """

    policy = terminal.resolution.diagnostic_cost_policy
    seen = (
        f"terminal reason {terminal.terminal_reason!r}, cost policy {policy!r}, "
        f"{terminal.physical_attempts} physical attempt(s) recorded, cost authority "
        f"{terminal.diagnostic_cost_authority}, rankable {terminal.rankable}"
    )
    out: list[Deviation] = []
    if terminal.terminal_reason != COST_BOUND_REASON or policy != ref.cost_policy:
        out.append(
            Deviation(
                COST_BOUND_TERMINAL,
                "the route opts into a cost upper bound under policy "
                f"{ref.cost_policy!r} and the limit guard did not leave the "
                f"terminal of that policy ({seen})",
            )
        )
    if terminal.physical_attempts != 1:
        out.append(
            Deviation(
                COST_BOUND_ATTEMPTS,
                f"a diagnostic result rests on exactly one physical attempt ({seen})",
            )
        )
    if not terminal.diagnostic_cost_authority:
        out.append(
            Deviation(
                COST_BOUND_AUTHORITY,
                f"the terminal does not keep the cost authority of a settled bound ({seen})",
            )
        )
    if terminal.rankable:
        out.append(
            Deviation(
                COST_BOUND_RANKABLE,
                "the limit guard calls a result rankable on a route whose results "
                f"are cost intervals ({seen})",
            )
        )
    return out


def _c5(obs: Observation, ref: Reference) -> list[Deviation]:
    out: list[Deviation] = []
    terminal = obs.limit_terminal
    if _guard_rewrote(obs):
        assert obs.result is not None and terminal is not None
        out.append(
            Deviation(
                f"guard:{obs.result.stop_reason}",
                "the limit guard replaced the result (terminal reason "
                f"{terminal.terminal_reason!r}, "
                f"{terminal.physical_attempts} physical attempt(s) recorded)",
            )
        )
    elif obs.result is not None and obs.result.error is None:
        # Read, never inferred from the absence of a rewrite: under a cost
        # policy the guard lets a result stand that it never calls rankable.
        # An error result is not rankable by its kind, which C2 judges. Where
        # the route opts into a cost upper bound the result is held to the
        # marks of a diagnostic result instead (a declared difference of the
        # ledger); the selector is the routing file's, never the terminal's.
        assert terminal is not None
        if ref.cost_bounded:
            out.extend(_diagnostic_marks(terminal, ref))
        elif not terminal.rankable:
            out.append(
                Deviation(
                    NOT_RANKABLE,
                    "the limit guard let the result stand and does not call it "
                    f"rankable (terminal reason {terminal.terminal_reason!r}, cost "
                    f"policy {terminal.resolution.diagnostic_cost_policy!r}, "
                    f"{terminal.physical_attempts} physical attempt(s) recorded)",
                )
            )
    second = obs.second_call
    if second is not None and second.refused:
        out.append(
            Deviation(
                "guard:revoked",
                "a second plain call on the same backend was refused by the guard",
            )
        )
    elif (
        second is None
        or second.raised is not None
        or second.error_kind is not None
        or second.requests != 1
    ):
        # Admitted is not enough: the backend must still answer, with one request.
        seen = (
            "was not made"
            if second is None
            else f"raised {second.raised}"
            if second.raised is not None
            else f"returned error kind {second.error_kind!r} after "
            f"{second.requests} provider request(s)"
        )
        out.append(
            Deviation(
                "guard:second_call_failed",
                f"a second plain call on the same backend {seen}; it must return "
                "a result without error from exactly one provider request",
            )
        )
    return out


def _c6(obs: Observation, exp: Expectation, ref: Reference) -> list[Deviation]:
    assert obs.result is not None
    out: list[Deviation] = []
    usage, flags = obs.result.usage, obs.result.completeness
    billed = obs.provider.billed_total()
    for name in ("token_usage_complete", "cost_complete"):
        if name in exp.flags and not getattr(flags, name):
            out.append(
                Deviation(
                    f"flag:{name}=False",
                    "the provider reported usage for every response",
                )
            )
    # A route whose results are cost intervals flags its accounting incomplete
    # even where every response is accounted for: the flags then excuse
    # nothing, and where the scenario owes them the numbers are owed exactly.
    exact_tokens = ref.cost_bounded and "token_usage_complete" in exp.flags
    for field_name in ("input_tokens", "output_tokens"):
        got, want = getattr(usage, field_name), getattr(billed, field_name)
        lower_bound = not flags.token_usage_complete and got < want and not exact_tokens
        if got == want or lower_bound:
            continue  # exact, or an honest lower bound
        out.append(
            Deviation(
                f"usage.{field_name}:{'lt' if got < want else 'gt'}",
                f"result reports {got}, the provider billed {want} "
                f"(token_usage_complete={flags.token_usage_complete})",
            )
        )
    priced = math.fsum(
        ref.pricing.compute_cost(
            input_tokens=item.input_tokens, output_tokens=item.output_tokens
        )
        for item in obs.provider.billed
    )
    cost = float(usage.cost_usd)
    exact = math.isclose(cost, priced, rel_tol=1e-9, abs_tol=1e-12)
    if not exact and (flags.cost_complete or cost > priced):
        out.append(
            Deviation(
                f"usage.cost_usd:{'lt' if cost < priced else 'gt'}",
                f"result reports {cost:.9f} USD, the rate card prices the billed "
                f"usage at {priced:.9f} USD (cost_complete={flags.cost_complete})",
            )
        )
    if flags.cost_complete:
        evidence = obs.result.pricing_evidence
        if evidence is None:
            out.append(
                Deviation("pricing:none", "cost is flagged complete without evidence")
            )
        elif not evidence.complete or not evidence.reconciles(usage):
            out.append(
                Deviation(
                    "pricing:unreconciled",
                    "the pricing evidence does not explain the reported usage",
                )
            )
    if ref.cost_bounded and "cost_complete" in exp.flags:
        out.extend(_cost_interval(obs, ref))
    return out


def bound_amounts(pricing: ModelPricing, segment: Any) -> tuple[float, float]:
    """The lower and the upper bound of one segment of a cost interval.

    The one statement of the bound formula in the matrix, from the usage the
    segment records and one price row. With the count of cache reads unknown
    the lower bound prices every input token that was not written to the
    cache as read from it, and the upper bound prices none as read; with the
    count known both are the exact price. Whether the counts a segment
    records are the ones the provider's reply carried is not this
    function's to say: :func:`_cost_interval` holds them to the provider
    ledger.
    """

    read = segment.cache_read_tokens

    def priced(cache_read_tokens: int) -> float:
        return pricing.compute_cost(
            input_tokens=segment.input_tokens,
            output_tokens=segment.output_tokens,
            cache_creation_tokens=segment.cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
        )

    if read is None:
        return priced(segment.input_tokens - segment.cache_creation_tokens), priced(0)
    return priced(read), priced(read)


def _count_text(count: int | None) -> str:
    return "unknown" if count is None else str(count)


def _same_amount(got: float, want: float) -> bool:
    # The tolerance of the product's own replay of the formula
    # (crucible/backends/cost_bounds.py).
    return math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12)


def _cost_interval(obs: Observation, ref: Reference) -> list[Deviation]:
    """What C6 holds of the cost interval a cost-bounded route returns.

    Such a route never reads a missing count of cache reads as zero, so its
    results flag token usage and cost incomplete for the provider's ordinary
    reply, and the flags alone would leave every number unasserted. The
    interval is what is owed instead: it is there, it covers the reported
    usage (full coverage, the token counts are the sums of its segments, the
    reported cost is its lower bound), it has one segment for every response
    the provider billed, with the tokens of that response and with the
    counts of cache reads and of cache writes that the reply of that
    response carried, and the two bounds of every segment are what the
    shipped price row of the route's model gives for the usage the segment
    records. That row is the rate table of the route's cost policy; the
    matrix rate card is not, and stays the reference of the reported cost.

    The counts are held because the bounds follow from them: a segment that
    states a count of cache reads is priced exactly, and at the lowest price
    when it says that every input token was read. A count nobody sent would
    so lower what a budget is charged, with bounds that are right for the
    usage the segment records. What the reply said is the provider ledger's
    to know: its dialect reads the counts back from the bytes it sent, and a
    count the reply did not carry is missing there, never zero.
    """

    assert obs.result is not None
    usage = obs.result.usage
    bounds = usage.cost_bounds
    if bounds is None:
        return [
            Deviation(
                "usage.cost_bounds:none",
                "the route's results are cost intervals and the result carries none",
            )
        ]
    out: list[Deviation] = []
    if not bounds.covers(usage):
        out.append(
            Deviation(
                "usage.cost_bounds:uncovered",
                "the interval does not cover the reported usage "
                f"(coverage_complete={bounds.coverage_complete}, lower bound "
                f"{bounds.lower_bound_usd:.9f} USD, reported cost "
                f"{float(usage.cost_usd):.9f} USD)",
            )
        )
    billed = [(item.input_tokens, item.output_tokens) for item in obs.provider.billed]
    held = [(item.input_tokens, item.output_tokens) for item in bounds.segments]
    if held != billed:
        out.append(
            Deviation(
                "usage.cost_bounds:segments",
                f"the interval has {len(held)} segment(s) with (input, output) "
                f"tokens {held}; the provider billed {len(billed)} response(s) "
                f"with {billed}",
            )
        )
    # A dialect that does not read the cache counts back from its replies
    # leaves the counts of an interval to the result's own word: the cell
    # cannot be judged, which is an error of the matrix and no deviation.
    stated = obs.provider.cache_counts
    assert stated is not None, (
        "the provider side of a cost-bounded route recorded no cache counts: its "
        "dialect must read them back from the replies it sends"
    )
    assert len(stated) == len(billed), (
        f"the provider ledger states cache counts for {len(stated)} of "
        f"{len(billed)} billed response(s)"
    )
    # Compared response by response, so only where the interval has one
    # segment for each: an interval of other segments is named above.
    miscounted = [
        f"segment {index + 1} records cache reads {_count_text(item.cache_read_tokens)} "
        f"and cache writes {_count_text(item.cache_creation_tokens)}, the reply stated "
        f"cache reads {_count_text(sent.read_tokens)} and cache writes "
        f"{_count_text(sent.write_tokens)}"
        for index, (item, sent) in enumerate(zip(bounds.segments, stated, strict=False))
        if (item.cache_read_tokens, item.cache_creation_tokens)
        != (sent.read_tokens, sent.write_tokens)
    ]
    if len(bounds.segments) == len(stated) and miscounted:
        out.append(
            Deviation(
                "usage.cost_bounds:counts",
                "the cache counts of the interval are not the ones the provider's "
                "replies carried (a count a reply did not carry is unknown, never "
                "zero): " + "; ".join(miscounted),
            )
        )
    # The runner refuses a cost-bounded route whose model has no shipped row.
    assert ref.bound_pricing is not None, "a cost-bounded reference carries no price row"
    mispriced = [
        f"segment {index + 1} states {item.lower_bound_usd:.9f} to "
        f"{item.upper_bound_usd:.9f} USD, the price row gives {lower:.9f} to {upper:.9f}"
        for index, item in enumerate(bounds.segments)
        for lower, upper in (bound_amounts(ref.bound_pricing, item),)
        if not (
            _same_amount(item.lower_bound_usd, lower)
            and _same_amount(item.upper_bound_usd, upper)
        )
    ]
    if mispriced:
        out.append(
            Deviation(
                "usage.cost_bounds:amounts",
                "the bounds of the interval are not what the shipped price row of "
                "the route's model gives for the usage each segment records: "
                + "; ".join(mispriced),
            )
        )
    return out


# The flags a scenario may waive: the two accounting flags (C6) and the
# terminal flag (C4). The trace flag is waived by no scenario.
_WAIVABLE_FLAGS = ("token_usage_complete", "cost_complete", "terminal_complete")


def _unasked(obs: Observation, exp: Expectation) -> frozenset[str]:
    """Checks that are switched off inside a clause that is evaluated.

    Two ways. The scenario waives a flag (a refusal the provider never
    billed carries no usage): the check that would report the flag false is
    not asked, whatever the result says. And C6 requires pricing evidence of
    a result that flags its cost complete; a result that flags it incomplete
    owes a lower bound and no evidence. A column-wide known gap is not owed
    where its check was not asked.
    """

    result = obs.result
    if result is None:
        return frozenset()  # a raise: no result clause ran at all
    waived = {f"flag:{name}=False" for name in _WAIVABLE_FLAGS if name not in exp.flags}
    if not result.completeness.cost_complete:
        return frozenset(waived) | PRICING_CODES
    return frozenset(waived)


_CALL_RAISE = re.compile(r"raise:\w+@(before|after)_request/[\w.]+")
_COST_FLAG_CLEARED = "flag:cost_complete=False"
# The C5 codes that are not a rewrite: the two about the second call, and
# those about a result the guard let stand.
_NOT_A_REWRITE = (
    frozenset({"guard:revoked", "guard:second_call_failed", NOT_RANKABLE}) | COST_BOUND_MARKS
)


def not_owed_given(code: str, cell_codes: Iterable[str]) -> bool:
    """Whether a cell's own expected codes show that the check of ``code`` did not run.

    Three of the rules by which :func:`evaluate` leaves a check out, read
    from the codes a cell is ledgered with instead of from an observation, so
    that the verifier can tell without playing the cell where a column-wide
    code is owed. The fourth, a flag the scenario waives, shows in no code of
    the cell: the inventory pins those cells, each with its reason.

    - a raise out of the call stands for the result clauses (a raise out of
      ``close()`` does not: the result was there);
    - a guard rewrite stands for the kind and stop reason it replaced;
    - C6 asks for pricing evidence only of a result that flags its cost
      complete.
    """

    clause = clause_of(code)
    for own in cell_codes:
        if clause in RESULT_CLAUSES and _CALL_RAISE.fullmatch(own):
            return True
        if (
            clause in ("C2", "C3")
            and own.startswith("guard:")
            and own not in _NOT_A_REWRITE
        ):
            return True
        if code in PRICING_CODES and own == _COST_FLAG_CLEARED:
            return True
    return False


def _c7(obs: Observation, ref: Reference, *, toolset: bool) -> list[Deviation]:
    assert obs.result is not None
    out: list[Deviation] = []
    # No substitution: every scripted request names the model of the route the
    # cell's role resolves to, whatever the adapter says about itself.
    wire = {request.model for request in obs.provider.requests if request.matched}
    substituted = sorted(wire - {ref.model})
    if substituted:
        out.append(
            Deviation(
                "request.content:model",
                f"a request carried model {substituted[0]!r}; the route's model "
                f"is {ref.model!r}",
            )
        )
    trace = obs.result.execution_trace
    if trace is None:
        out.append(Deviation("digest.missing:trace", "the result carries no trace"))
        return out
    if trace.backend != obs.backend_name:
        out.append(
            Deviation(
                "digest.mismatch:backend",
                f"trace names {trace.backend!r}, the adapter is {obs.backend_name!r}",
            )
        )
    if trace.requested_model != obs.backend_model:
        out.append(
            Deviation(
                "digest.mismatch:requested_model",
                "trace.requested_model is not the routed model",
            )
        )
    # Where one adapter serves two routes under one name, the endpoint is what
    # tells their run records apart. A route that names none is not compared.
    if ref.endpoint is not None:
        if trace.endpoint is None:
            out.append(
                Deviation(
                    "digest.missing:endpoint",
                    f"the route names the endpoint {ref.endpoint!r} and the trace "
                    "records none",
                )
            )
        elif trace.endpoint != ref.endpoint:
            out.append(
                Deviation(
                    "digest.mismatch:endpoint",
                    f"the trace records the endpoint {trace.endpoint!r}; the route "
                    f"names {ref.endpoint!r}",
                )
            )
    served = obs.provider.served_models[-1] if obs.provider.served_models else None
    if served is not None:
        if trace.served_model is None:
            out.append(
                Deviation("digest.missing:served_model", "the provider named a model")
            )
        elif trace.served_model != served:
            out.append(
                Deviation(
                    "digest.mismatch:served_model",
                    "trace.served_model is not the model the provider named",
                )
            )
        # Served equals requested. Asserted when the provider answered under a
        # model it was asked for; a provider that names another model is a
        # scenario of its own, with its own expected result.
        if served in wire and trace.requested_model != served:
            out.append(
                Deviation(
                    "served_model:not_requested",
                    "the provider answered under the model it was asked for, and "
                    "the trace records another model as the requested one",
                )
            )
    if trace.feature_profile_digest is None:
        out.append(
            Deviation(
                "digest.missing:feature_profile",
                "the trace does not bind the route's feature profile",
            )
        )
    elif trace.feature_profile_digest != feature_profile_digest(ref.routed):
        out.append(
            Deviation(
                "digest.mismatch:feature_profile",
                "the trace digest is not that of the routed feature profile",
            )
        )
    if toolset:
        if trace.tool_authority_digest is None:
            out.append(
                Deviation(
                    "digest.missing:tool_authority",
                    "a ToolSet was attached and the trace does not bind it",
                )
            )
        elif trace.tool_authority_digest != ref.tool_authority_digest:
            out.append(
                Deviation(
                    "digest.mismatch:tool_authority",
                    "the trace digest is not the shared tool-authority digest",
                )
            )
    return out


def _c8(obs: Observation) -> list[Deviation]:
    out: list[Deviation] = []
    settlement = obs.settlement
    if settlement.close_raised is not None:
        out.append(Deviation(settlement.close_raised.code, "close() raised"))
    if settlement.children_alive or settlement.listeners or settlement.late_requests:
        out.append(
            Deviation(
                "child:alive",
                f"after close(): child processes {list(settlement.children_alive)}, "
                f"listening ports {list(settlement.listeners)}, "
                f"{settlement.late_requests} late provider request(s)",
            )
        )
    return out


def _c9(obs: Observation, ref: Reference) -> list[Deviation]:
    texts = [obs.debug_dump_text]
    # A workflow logs the text of whatever a call or close() raised.
    texts.extend(
        raised.message
        for raised in (obs.raised, obs.settlement.close_raised)
        if raised is not None
    )
    message = ""
    if obs.result is not None:
        result = obs.result
        message = "" if result.error is None else str(result.error.message)
        texts.extend(
            [result.final_message, message, repr(result.execution_trace)]
            + [item.content for item in result.transcript]
        )
    out: list[Deviation] = []
    if any(secret and secret in text for secret in ref.secrets for text in texts):
        out.append(
            Deviation(
                "secret:leak",
                "a scripted credential or token appears in a result, trace, raise, "
                "log or debug dump",
            )
        )
    if len(message) > MAX_ERROR_TEXT:
        out.append(
            Deviation(
                "text:unbounded",
                f"the error message is {len(message)} characters long",
            )
        )
    return out


def _aligned(sent: Sequence[Any], logged: Sequence[Any]) -> tuple[set[int], set[int]]:
    """Indices of a longest common subsequence of two short call sequences."""

    lengths = [[0] * (len(logged) + 1) for _ in range(len(sent) + 1)]
    for i in range(len(sent) - 1, -1, -1):
        for j in range(len(logged) - 1, -1, -1):
            lengths[i][j] = (
                lengths[i + 1][j + 1] + 1
                if sent[i] == logged[j]
                else max(lengths[i + 1][j], lengths[i][j + 1])
            )
    matched_sent: set[int] = set()
    matched_logged: set[int] = set()
    i = j = 0
    while i < len(sent) and j < len(logged):
        if sent[i] == logged[j]:
            matched_sent.add(i)
            matched_logged.add(j)
            i, j = i + 1, j + 1
        elif lengths[i + 1][j] >= lengths[i][j + 1]:
            i += 1
        else:
            j += 1
    return matched_sent, matched_logged


def _c10(obs: Observation, exp: Expectation, ref: Reference) -> list[Deviation]:
    out: list[Deviation] = []
    slot_by_name = {binding.name: slot for slot, binding in ref.bindings.items()}
    reads: list[Dispatch] = []
    for item in obs.toolset_log:
        slot = slot_by_name.get(item.name)
        if slot is None:
            out.append(
                Deviation(
                    f"dispatch.extra:{item.name}",
                    "a tool outside the scripted calls reached the ToolSet",
                )
            )
        elif slot != ref.write_slot:
            reads.append(item)
    # A write may run in another process, so it is judged by its effect; the
    # reads are aligned, in order, with the calls the provider actually sent.
    sent = list(obs.provider.sent_calls)
    sent_reads = [index for index, call in enumerate(sent) if call.slot != ref.write_slot]
    matched_sent, matched_logged = _aligned(
        [(ref.bindings[sent[i].slot].name, dict(sent[i].arguments)) for i in sent_reads],
        [(item.name, dict(item.arguments)) for item in reads],
    )
    dispatched_reads = {sent_reads[position] for position in matched_sent}
    for index, call in enumerate(sent):
        name = ref.bindings[call.slot].name
        if call.slot == ref.write_slot:
            dispatched = bool(obs.write_applied)
        else:
            dispatched = index in dispatched_reads
        required = index < len(exp.dispatch)
        allowed = index < len(exp.dispatch) + len(exp.dispatch_optional)
        if required and not dispatched:
            out.append(
                Deviation(
                    f"dispatch.missing:{call.slot}",
                    f"scripted call {index + 1} ({name}) did not reach the ToolSet "
                    "with the arguments the model sent",
                )
            )
        elif dispatched and not allowed:
            out.append(
                Deviation(
                    f"dispatch.extra:{call.slot}",
                    f"scripted call {index + 1} ({name}) must not be dispatched "
                    "after the terminal condition",
                )
            )
    for position, item in enumerate(reads):
        if position not in matched_logged:
            out.append(
                Deviation(
                    f"dispatch.extra:{slot_by_name[item.name]}",
                    f"{item.name} reached the ToolSet without a matching scripted call",
                )
            )
    return out


def _scenario(
    obs: Observation, exp: Expectation, ref: Reference, checks: Sequence[Check]
) -> list[Deviation]:
    assert obs.result is not None
    out: list[Deviation] = []
    trace = obs.result.execution_trace
    for wanted in exp.events:
        kind, _, name = wanted.partition("/")
        present = trace is not None and any(
            event.kind == kind and (name == "-" or event.name == name)
            for event in trace.events
        )
        if not present:
            out.append(
                Deviation(f"event.missing:{wanted}", "the trace has no such event")
            )
    # A final message is asserted on a result without error only: an error
    # result promises no answer, and its kind already says so.
    answered = obs.result.error is None
    if answered and exp.final_text is not None and obs.result.final_message != exp.final_text:
        out.append(
            Deviation(
                "text:mismatch",
                "final_message is not the text the provider sent",
            )
        )
    for check in checks:
        out.extend(check(obs, ref))
    return out


def _outcome(deviations: Sequence[Deviation]) -> ClauseOutcome:
    for item in deviations:
        clause_of(item.code)  # the vocabulary is closed
    if deviations:
        return ClauseOutcome("deviation", tuple(deviations))
    return ClauseOutcome("ok")


ADMITTED = "admitted"
# Routing built the adapter and logged that the profile of the route does not
# meet what the role asks: what diagnostic admission does with a role that
# enforcing admission refuses.
RETAINED = "retained"


@dataclass(frozen=True)
class Admitted:
    """What an admission cell saw of one subject, and where.

    A subject is a capability a workflow can ask routing for, the same
    capability on the routing file as shipped, or the word the profile
    catalog has for the column. ``got`` is :data:`ADMITTED`,
    :data:`RETAINED`, the type of the exception routing refused with, or the
    catalog's word.
    """

    subject: str
    got: str
    where: str = ""


def evaluate_admission(
    obs: Observation,
    seen: Sequence[Admitted],
    expected: Mapping[str, str],
    ref: Reference,
    *,
    lens: str,
) -> Report:
    """The verdict of an admission cell: routing did what the pin says it does.

    The cell builds a backend for every capability and calls none, so its
    clauses are C0 (no provider request), C8 and C9, and the scenario
    contract: every subject was seen, and each as pinned.
    """

    clauses = applicable(lens=lens, toolset=False, expects_result=False, call=False)
    evaluators: Mapping[str, Callable[[], list[Deviation]]] = {
        "C0": lambda: _c0(obs),
        "C8": lambda: _c8(obs),
        "C9": lambda: _c9(obs, ref),
    }
    outcomes = {clause: _outcome(evaluators[clause]()) for clause in clauses}
    deviations = [
        Deviation(
            f"admission:{item.subject}:{item.got}!={expected.get(item.subject, 'unpinned')}",
            f"{item.where}: {item.subject} is {item.got}; the inventory pins "
            f"{expected.get(item.subject, 'nothing')}",
        )
        for item in seen
        if item.got != expected.get(item.subject)
    ]
    deviations.extend(
        Deviation(
            f"admission:{subject}:unseen!={want}",
            f"the cell saw nothing of {subject}, which the inventory pins as {want}",
        )
        for subject, want in expected.items()
        if subject not in {item.subject for item in seen}
    )
    outcomes[SCENARIO] = _outcome(deviations)
    return Report(outcomes, tuple(obs.provider.seam_problems))


def evaluate(
    obs: Observation,
    exp: Expectation,
    ref: Reference,
    *,
    lens: str,
    toolset: bool,
    canary: bool = False,
    checks: Sequence[Check] = (),
) -> Report:
    """Evaluate every clause that applies to the cell."""

    expects_result = exp.outcome == "result"
    clauses = applicable(lens=lens, toolset=toolset, expects_result=expects_result)
    outcomes: dict[str, ClauseOutcome] = {"C0": _outcome(_c0(obs))}
    if canary:
        # The canary proves the seam and C0 only; the inventory pins the rest.
        for clause in clauses[1:]:
            outcomes[clause] = ClauseOutcome(
                "not_evaluated", reason="the canary checks the seam and C0 only"
            )
        return Report(outcomes, tuple(obs.provider.seam_problems))

    raised = obs.raised
    scenario: list[Deviation] = []
    c1: list[Deviation] = []
    if raised is not None:
        detail = f"{raised.type} left the call {raised.phase.replace('_', ' ')}"
        if raised.phase == "after_request":
            c1.append(Deviation(raised.code, detail))
        elif expects_result or raised.type != exp.refused_type:
            wanted = "a result" if expects_result else f"a {exp.refused_type} refusal"
            scenario.append(Deviation(raised.code, f"{detail}; expected {wanted}"))
    elif not expects_result:
        assert obs.result is not None
        error = obs.result.error
        got = "none" if error is None else str(error.kind)
        scenario.append(
            Deviation(
                f"kind:{got}!=refused",
                f"invoke returned a result; expected a {exp.refused_type} refusal",
            )
        )
    outcomes["C1"] = _outcome(c1)

    result = obs.result
    rewrote = "C5" in clauses and _guard_rewrote(obs)
    for clause in clauses:
        if clause in ("C0", "C1"):
            continue
        if clause in RESULT_CLAUSES and result is None:
            assert raised is not None
            outcomes[clause] = ClauseOutcome("stood_for", reason=raised.code)
        elif clause in ("C2", "C3") and rewrote:
            assert result is not None
            outcomes[clause] = ClauseOutcome(
                "stood_for", reason=f"guard:{result.stop_reason}"
            )
        elif clause == "C2":
            outcomes[clause] = _outcome(_c2(result, exp))  # type: ignore[arg-type]
        elif clause == "C3":
            outcomes[clause] = _outcome(_c3(result, exp))  # type: ignore[arg-type]
        elif clause == "C4":
            outcomes[clause] = _outcome(_c4(result, exp))  # type: ignore[arg-type]
        elif clause == "C5":
            outcomes[clause] = _outcome(_c5(obs, ref))
        elif clause == "C6":
            outcomes[clause] = _outcome(_c6(obs, exp, ref))
        elif clause == "C7":
            outcomes[clause] = _outcome(_c7(obs, ref, toolset=toolset))
        elif clause == "C8":
            outcomes[clause] = _outcome(_c8(obs))
        elif clause == "C9":
            outcomes[clause] = _outcome(_c9(obs, ref))
        elif clause == "C10":
            outcomes[clause] = _outcome(_c10(obs, exp, ref))

    if result is not None and expects_result:
        scenario.extend(_scenario(obs, exp, ref, checks))
        outcomes[SCENARIO] = _outcome(scenario)
    elif scenario:
        outcomes[SCENARIO] = _outcome(scenario)
    elif expects_result:
        assert raised is not None
        outcomes[SCENARIO] = ClauseOutcome("stood_for", reason=raised.code)
    else:
        outcomes[SCENARIO] = ClauseOutcome("ok")
    return Report(outcomes, tuple(obs.provider.seam_problems), _unasked(obs, exp))
