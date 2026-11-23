# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The toy driver: an adapter written for the matrix, honest or with one fault.

``HonestBackend`` implements the assay ``Backend`` protocol with ``features``,
``attach_toolset`` and ``close``. It talks to the ``LoopbackProvider`` over a
real loopback socket in a dialect of its own, runs its own tool loop against
the attached ToolSet, prices usage through pricing discovery and builds its
results from public assay types. It behaves as its declared profile says, so
every variant a real column selects has an honest pass before a real driver
exists.

A *liar* is ``HonestBackend`` with one named fault (two faults are the
driver's: it binds the limit guard to another profile, or builds the adapter
for another model than the route's). :data:`LIARS` fixes, beside each
fault, the scenario and lens it runs in and the exact deviation codes its cell
must yield. That is the proof that the oracle can fail: a scenario cannot be
enabled anywhere without a liar here.

The toy touches no private product name.
"""

from __future__ import annotations

import contextlib
import dataclasses
import http.client
import json
import logging
import math
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping
from urllib.parse import urlsplit

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    BackendFeatureSet,
    BudgetExhausted,
    ContentRejected,
    ExecutionCompleteness,
    Invocation,
    InvocationResult,
    MaxTurnsExceeded,
    Message,
    TransportError,
    Usage,
    UsagePricingEvidence,
    UsagePricingSegment,
)
from crucible.backends.capabilities import validate_toolset_capability
from crucible.backends.features import FeatureAdmissionError, feature_profile_digest
from crucible.backends.pricing import pricing_for_model
from crucible.backends.tool_authority import (
    APPROVAL_POLICY_PER_CALL_EXACT,
    TOOL_AUTHORITY_BRIDGE_PROTOCOL,
    TOOL_AUTHORITY_BRIDGE_VERSION,
    tool_authority_digest,
)
from crucible.tools import ToolDispatchBudgetExceeded

from .. import columns as matrix_columns
from ..columns import CANARY_CREDENTIAL, TOY_ADAPTER, TOY_KIND, Column
from ..provider import POLICY_ERROR_CODE, LoopbackProvider, ToyDialect
from ..scenarios import D_OAI_SCHEMA_FINAL, SCHEMA_KEY
from ..script import READ_A, READ_B, ProviderScript
from .base import Lens, MatrixSetupError, bind_as_routing, register

# Declared values the toy has a behaviour for. A snapshot value outside this
# table cannot have an honest pass and is a collection error.
SUPPORTED: Mapping[str, frozenset[str]] = {
    "structured_output": frozenset({"native", "prompt_only", "unsupported"}),
    "event_detail": frozenset({"terminal_only", "usage", "tool_calls", "agent_graph"}),
    "turn_limit": frozenset({"hard", "provider_stop", "round_boundary", "unsupported"}),
    "usd_limit": frozenset({"post_hoc", "round_boundary"}),
}
_OWN_TURN_CEILING = 64
_UNATTACHED_TOOL = "matrix_unattached"
# What a trace names under the fault that records another endpoint than the
# one the adapter was built for and called.
_OTHER_ENDPOINT = "http://127.0.0.1:9/elsewhere"
# A second secret the toy holds beside the canary credential, as a real
# backend holds a bridge or session token. The driver declares it.
TOY_SESSION_TOKEN = "matrix-toy-session-token-do-not-log-a94f27"
_LOG = logging.getLogger(__name__)

# -- faults -------------------------------------------------------------------
CANNED_REPLY = "canned_reply"
DROP_OUTPUT_TOKENS = "drop_output_tokens"
ZERO_COST_NO_EVIDENCE = "zero_cost_no_evidence"
RAISE_AFTER_REQUEST = "raise_after_request"
TERMINAL_MISNAMED = "terminal_misnamed"
NO_PROFILE_DIGEST = "no_profile_digest"
LEAVE_CHILD = "leave_child"
LEAK_CREDENTIAL = "leak_credential"
REPORTS_OTHER_PROFILE = "reports_other_profile"
ALTER_TEXT = "alter_text"
REWRITE_PROMPT = "rewrite_prompt"
NO_REQUEST_EVENT = "no_request_event"
DISPATCH_UNATTACHED = "dispatch_unattached"
REQUEST_EVENT_PER_ROUND = "request_event_per_round"
EMPTY_TOOL_RESULTS = "empty_tool_results"
FORGET_EARLIER_ROUNDS = "forget_earlier_rounds"
NO_TOOL_EVENTS = "no_tool_events"
CLEAR_ACCOUNTING_FLAGS = "clear_accounting_flags"
SCHEMA_MISPLACED = "schema_misplaced"
SUCCESS_AT_TURN_LIMIT = "success_at_turn_limit"
LIMIT_AFTER_TERMINAL = "limit_after_terminal"
EXTRA_TOOL_REQUEST = "extra_tool_request"
SEND_TURN_CONTROL = "send_turn_control"
OWN_TURN_CEILING = "own_turn_ceiling"
BUDGET_STOP_AS_TRANSPORT = "budget_stop_as_transport"
IGNORE_USD_BUDGET = "ignore_usd_budget"
CONTEXT_LIMIT_STOP = "context_limit_stop"
IGNORE_CONTEXT_BUDGET = "ignore_context_budget"
REFUSAL_AS_TRANSPORT = "refusal_as_transport"
WRITE_AROUND_WRAPPER = "write_around_wrapper"
SUBSTITUTE_WIRE_MODEL = "substitute_wire_model"
SUBSTITUTE_SECOND_REQUEST_MODEL = "substitute_second_request_model"
TWO_REQUEST_EVENTS = "two_request_events"
GET_BEFORE_TURN = "get_before_turn"
FOREIGN_USER_AGENT = "foreign_user_agent"
RETRIABLE_ERROR = "retriable_error"
REFUSAL_ENDS_END_TURN = "refusal_ends_end_turn"
DROP_REFUSAL_USAGE = "drop_refusal_usage"
SECRET_IN_ERROR_MESSAGE = "secret_in_error_message"
RAISE_WITH_CREDENTIAL = "raise_with_credential"
LOG_SESSION_TOKEN = "log_session_token"
DEBUG_LOG_SESSION_TOKEN = "debug_log_session_token"
LEAVE_LISTENER = "leave_listener"
LATE_REQUEST = "late_request"
SECOND_CLOSE_RAISES = "second_close_raises"
DEAD_AFTER_FIRST_CALL = "dead_after_first_call"
SECOND_CALL_ERRORS = "second_call_errors"
SECOND_CALL_RAISES = "second_call_raises"
SCHEMA_ANSWER_INVALID = "schema_answer_invalid"
NO_LIMIT_EVENT = "no_limit_event"
DISPATCH_STOP_MISNAMED = "dispatch_stop_misnamed"
SCHEMA_NOT_IN_PROMPT = "schema_not_in_prompt"
RETURN_SCHEMA_DRAFT = "return_schema_draft"
SCHEMA_ANSWER_REPLACED = "schema_answer_replaced"
TRACE_OTHER_ENDPOINT = "trace_other_endpoint"
# Not a fault but a trait: after a tool loop with a response schema the
# adapter sends one further request without tools that carries the schema in
# the provider's field, and returns the answer to it, as the declared
# difference D-OAI-SCHEMA-FINAL describes. What the declared value says about
# the prompt is unchanged. Honest where that difference is in force, an extra
# request anywhere else.
FINAL_SCHEMA_REQUEST = "final_schema_request"
# Faults of the driver, not of the adapter: the limit guard is bound to
# another profile than the one routing computes for the column, or the adapter
# is built for another model than the column's route names.
GUARD_ON_OTHER_PROFILE = "guard_on_other_profile"
BUILT_FOR_OTHER_MODEL = "built_for_other_model"
# Faults of the build, for the admission scenario: the writer role is refused
# although nothing gates the column, or a backend asks its provider something
# while it is built.
REFUSE_WRITER = "refuse_writer"
REQUEST_AT_BUILD = "request_at_build"


@dataclass
class _Run:
    """State of one invocation."""

    invocation: Invocation
    events: list[tuple[str, str | None]] = field(default_factory=list)
    segments: list[UsagePricingSegment] = field(default_factory=list)
    served_model: str | None = None
    usage_known: bool = True
    context_used: int = 0
    rounds: int = 0
    # The calls whose results earlier requests of this invocation carried:
    # the toy wire keeps no conversation, so every request names them again.
    answered: list[str] = field(default_factory=list)


class HonestBackend:
    """The toy adapter. With ``faults`` empty it keeps the whole contract."""

    name = TOY_ADAPTER

    def __init__(
        self,
        *,
        profile: BackendFeatureSet,
        model: str,
        endpoint: str,
        credential: str,
        session_token: str,
        capability: str,
        log_dir: Path,
        faults: frozenset[str] = frozenset(),
    ) -> None:
        self.model = model
        self.capability = capability
        self._profile = profile
        self._endpoint = endpoint
        self._credential = credential
        self._session_token = session_token
        self._log_dir = log_dir
        self._faults = faults
        self._toolset: Any = None
        self._context_budget: int | None = None
        self._children: list[subprocess.Popen[bytes]] = []
        self._listeners: list[socket.socket] = []
        self._calls = 0
        self._closes = 0
        self._closed = False

    # -- protocol surface ---------------------------------------------------

    @property
    def features(self) -> BackendFeatureSet:
        if REPORTS_OTHER_PROFILE in self._faults:
            other = "interruptible" if self._profile.cancellation != "interruptible" else "unsupported"
            return dataclasses.replace(self._profile, cancellation=other)  # type: ignore[arg-type]
        return self._profile

    def attach_toolset(self, toolset: Any, *, context_budget: int | None = None) -> None:
        if self._closed:
            raise RuntimeError("the toy backend is closed")
        validate_toolset_capability(self.capability, toolset)  # type: ignore[arg-type]
        self._toolset = toolset
        self._context_budget = context_budget if toolset is not None else None

    def children(self) -> tuple[int, ...]:
        return tuple(child.pid for child in self._children)

    def close(self) -> None:
        self._closes += 1
        if SECOND_CLOSE_RAISES in self._faults and self._closes > 1:
            raise RuntimeError("the toy backend is already closed")
        if LATE_REQUEST in self._faults and self._closes == 1:
            self._send("POST", ToyDialect.path, {"model": self.model}, 5.0)
        self._closed = True
        if LEAVE_CHILD not in self._faults and LEAVE_LISTENER not in self._faults:
            self.reap()

    def reap(self) -> None:
        """Stop and close whatever this backend started (the driver's last resort)."""

        for child in self._children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)
        self._children.clear()
        for listener in self._listeners:
            listener.close()
        self._listeners.clear()

    # -- wire -----------------------------------------------------------------

    def _tool_names(self) -> list[str]:
        if self._toolset is None:
            return []
        return [str(item["name"]) for item in self._toolset.to_anthropic_tools()]

    def _body(
        self,
        run: _Run,
        turn: int,
        results: list[dict[str, str]],
        tools: list[str],
        *,
        finalize: bool = False,
    ) -> dict[str, Any]:
        invocation = run.invocation
        system = invocation.system
        model = self.model
        if SUBSTITUTE_WIRE_MODEL in self._faults:
            model += "-fallback"
        if SUBSTITUTE_SECOND_REQUEST_MODEL in self._faults and run.rounds == 1:
            # ``rounds`` counts the requests already sent: this is the second.
            model += "-fallback"
        body: dict[str, Any] = {"model": model, "user": invocation.user, "turn": turn}
        declared = self._profile.structured_output
        in_field, in_prompt = declared == "native", declared == "prompt_only"
        if FINAL_SCHEMA_REQUEST in self._faults:
            # The field travels in the one final request only; the prompt half
            # of the declared value stays.
            in_field = finalize
        if SCHEMA_MISPLACED in self._faults:
            in_field, in_prompt = not in_field, not in_prompt
        if SCHEMA_NOT_IN_PROMPT in self._faults:
            in_prompt = False
        if invocation.response_schema is not None:
            schema = json.loads(json.dumps(invocation.response_schema))
            if in_field:
                body["response_schema"] = schema
            if in_prompt:
                system += "\n\nAnswer with JSON for this schema: " + json.dumps(
                    schema, sort_keys=True
                )
        if REWRITE_PROMPT in self._faults:
            system = "You are a helpful assistant."
        if self._profile.turn_limit != "unsupported" or SEND_TURN_CONTROL in self._faults:
            body["max_turns"] = invocation.max_turns
        if EMPTY_TOOL_RESULTS in self._faults:
            results = [{"id": item["id"], "text": ""} for item in results]
        answered = [] if FORGET_EARLIER_ROUNDS in self._faults else list(run.answered)
        body.update(system=system, tools=tools, tool_results=results, answered=answered)
        return body

    def _post(self, body: Mapping[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
        return self._send("POST", ToyDialect.path, body, timeout)

    def _send(
        self, method: str, path: str, body: Mapping[str, Any] | None, timeout: float
    ) -> tuple[int, dict[str, Any]]:
        target = urlsplit(self._endpoint)
        connection = http.client.HTTPConnection(
            target.hostname or "", target.port, timeout=timeout
        )
        agent = ToyDialect.user_agent
        if FOREIGN_USER_AGENT in self._faults:
            agent = "python-requests/2"
        try:
            connection.request(
                method,
                path,
                body=None if body is None else json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self._credential}",
                    "Content-Type": "application/json",
                    "User-Agent": agent,
                },
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
        finally:
            connection.close()

    def _dump(self, record: Mapping[str, Any]) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        with (self._log_dir / "toy-debug.jsonl").open("a", encoding="utf-8") as dump:
            dump.write(json.dumps(record, sort_keys=True) + "\n")

    # -- the call -------------------------------------------------------------

    def invoke(self, invocation: Invocation) -> InvocationResult:
        if self._closed:
            raise RuntimeError("the toy backend is closed")
        policy = invocation.policy
        if policy is None or policy.on_max_turns != "raise" or policy.retry_budget != 1:
            raise MatrixSetupError(
                "the toy plays on_max_turns='raise' with retry_budget=1 only; the "
                "scenario that needs another policy adds the behaviour with its liar"
            )
        if OWN_TURN_CEILING in self._faults and invocation.max_turns > _OWN_TURN_CEILING:
            raise ValueError(f"max_turns must not exceed {_OWN_TURN_CEILING}")
        run = _Run(invocation)
        self._calls += 1
        if NO_REQUEST_EVENT not in self._faults and REQUEST_EVENT_PER_ROUND not in self._faults:
            run.events.append(("request", None))
        if TWO_REQUEST_EVENTS in self._faults:
            run.events.append(("request", None))
        if DEAD_AFTER_FIRST_CALL in self._faults and self._calls > 1:
            run.usage_known = False
            return self._result(
                run,
                final="",
                stop="error",
                error=TransportError(message="the toy backend is gone", retriable=False),
            )
        if LEAVE_CHILD in self._faults:
            self._children.append(
                subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
            )
        if LEAVE_LISTENER in self._faults:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            self._listeners.append(listener)
        if LEAK_CREDENTIAL in self._faults:
            self._dump({"authorization": f"Bearer {self._credential}"})
        if LOG_SESSION_TOKEN in self._faults:
            # INFO is the level a workflow run keeps in its stderr log.
            _LOG.info("toy session %s", self._session_token)
        if DEBUG_LOG_SESSION_TOKEN in self._faults:
            _LOG.debug("toy session %s", self._session_token)
        if CANNED_REPLY in self._faults:
            run.served_model = self.model
            return self._result(run, final="canned answer", stop="end_turn", error=None)

        timeout = float(policy.timeout_seconds or 60.0)
        results: list[dict[str, str]] = []
        turn = 0
        while True:
            turn += 1
            reply = self._exchange(run, turn, results, self._tool_names(), timeout)
            if isinstance(reply, InvocationResult):
                return reply
            if reply["type"] != "tool_calls":
                text = str(reply.get("text", ""))
                if (
                    FINAL_SCHEMA_REQUEST in self._faults
                    and self._toolset is not None
                    and invocation.response_schema is not None
                ):
                    final = self._exchange(run, turn + 1, [], [], timeout, finalize=True)
                    if isinstance(final, InvocationResult):
                        return final
                    if RETURN_SCHEMA_DRAFT not in self._faults:
                        text = str(final.get("text", ""))
                return self._answer(run, text)
            limited = self._profile.turn_limit != "unsupported"
            at_limit = limited and turn >= invocation.max_turns
            if at_limit and self._profile.turn_limit != "round_boundary":
                return self._turn_limit(run, turn, results, timeout)
            results = []
            for call in reply["calls"]:
                outcome = self._dispatch(run, turn, call)
                if isinstance(outcome, InvocationResult):
                    return outcome
                results.append({"id": str(call["id"]), "text": outcome})
            if at_limit:
                # round_boundary: one last request without tools, usage counted.
                reply = self._exchange(run, turn + 1, results, [], timeout)
                if isinstance(reply, InvocationResult):
                    return reply
                return self._turn_limit(
                    run, turn, [], timeout, partial=str(reply.get("text", ""))
                )

    def _exchange(
        self,
        run: _Run,
        turn: int,
        results: list[dict[str, str]],
        tools: list[str],
        timeout: float,
        *,
        finalize: bool = False,
    ) -> dict[str, Any] | InvocationResult:
        """One provider round: a reply to continue with, or the typed terminal."""

        body = self._body(run, turn, results, tools, finalize=finalize)
        self._dump({"turn": turn, "request": body})
        run.rounds += 1
        run.answered.extend(item["id"] for item in results)
        if REQUEST_EVENT_PER_ROUND in self._faults:
            run.events.append(("request", None))
        try:
            if GET_BEFORE_TURN in self._faults:
                self._send("GET", "/toy/v1/models", None, timeout)
            status, reply = self._post(body, timeout)
        except OSError as exc:
            run.usage_known = False
            return self._result(
                run,
                final="",
                stop="error",
                error=TransportError(message=f"provider unreachable: {type(exc).__name__}"),
            )
        if self._calls > 1 and SECOND_CALL_RAISES in self._faults:
            raise RuntimeError("the toy adapter lost the reply to its second call")
        if self._calls > 1 and SECOND_CALL_ERRORS in self._faults:
            run.usage_known = False
            return self._result(
                run,
                final="",
                stop="error",
                error=TransportError(
                    message="the toy adapter lost the reply to its second call",
                    retriable=False,
                ),
            )
        if RAISE_AFTER_REQUEST in self._faults:
            raise RuntimeError("the toy adapter lost its reply")
        if RAISE_WITH_CREDENTIAL in self._faults:
            raise RuntimeError(
                f"provider rejected 'Authorization: Bearer {self._credential}'"
            )
        if status != 200:
            run.usage_known = False
            code = str(reply.get("error", {}).get("code", ""))
            if code == POLICY_ERROR_CODE:
                return self._refusal(run, "content_policy")
            return self._result(
                run,
                final="",
                stop="error",
                error=TransportError(
                    http_status=status,
                    message=f"provider answered HTTP {status} ({code})",
                    retriable=status == 429 or status >= 500,
                ),
            )
        run.served_model = str(reply.get("model", "")) or None
        usage = reply.get("usage", {})
        tokens_in, tokens_out = int(usage["input_tokens"]), int(usage["output_tokens"])
        cost = pricing_for_model(self.model).compute_cost(
            input_tokens=tokens_in, output_tokens=tokens_out
        )
        if DROP_OUTPUT_TOKENS in self._faults:
            tokens_out = 0
        run.segments.append(UsagePricingSegment(tokens_in, tokens_out, 0, 0, cost))
        if reply["type"] == "refusal":
            return self._refusal(run, "refusal")
        if reply["type"] == "message" and reply.get("finish") == "content_filter":
            return self._refusal(run, "content_filter")
        return reply

    def _refusal(self, run: _Run, stop: str) -> InvocationResult:
        if REFUSAL_AS_TRANSPORT in self._faults:
            error: Any = TransportError(message="provider error", retriable=False)
            return self._result(run, final="", stop="error", error=error)
        if DROP_REFUSAL_USAGE in self._faults:
            run.segments.clear()
            run.usage_known = False
        if REFUSAL_ENDS_END_TURN in self._faults:
            stop = "end_turn"
        message = "the provider refused the request"
        if SECRET_IN_ERROR_MESSAGE in self._faults:
            message += f" sent with 'Authorization: Bearer {self._credential}'"
        return self._result(
            run, final="", stop=stop, error=ContentRejected(message=message)
        )

    def _answer(self, run: _Run, text: str) -> InvocationResult:
        invocation = run.invocation
        if DISPATCH_UNATTACHED in self._faults and self._toolset is not None:
            self._toolset.dispatch(_UNATTACHED_TOOL, {})
        if ALTER_TEXT in self._faults:
            text = text.upper()
        if SCHEMA_ANSWER_INVALID in self._faults and invocation.response_schema is not None:
            text = "Here is the answer: " + text
        if SCHEMA_ANSWER_REPLACED in self._faults and invocation.response_schema is not None:
            # Another object of the schema: it validates, and the provider
            # never sent it.
            text = json.dumps({SCHEMA_KEY: "an answer the provider never gave"})
        spent = math.fsum(segment.cost_usd for segment in run.segments)
        over = invocation.budget_usd is not None and spent > invocation.budget_usd
        if over and IGNORE_USD_BUDGET not in self._faults:
            run.events.append(("limit", "budget_usd"))
            return self._result(
                run,
                final=text,
                stop="budget_exhausted",
                error=BudgetExhausted(
                    spent_usd=spent,
                    limit_usd=float(invocation.budget_usd or 0.0),
                    message="the call cost more than its USD budget",
                ),
            )
        return self._result(run, final=text, stop="end_turn", error=None)

    def _turn_limit(
        self,
        run: _Run,
        turn: int,
        results: list[dict[str, str]],
        timeout: float,
        *,
        partial: str = "",
    ) -> InvocationResult:
        if EXTRA_TOOL_REQUEST in self._faults:
            self._post(self._body(run, turn + 1, results, self._tool_names()), timeout)
        if NO_LIMIT_EVENT not in self._faults:
            run.events.append(("limit", "max_turns"))
        if SUCCESS_AT_TURN_LIMIT in self._faults:
            return self._result(run, final=partial, stop="end_turn", error=None)
        return self._result(
            run,
            final=partial,
            stop="max_turns",
            error=MaxTurnsExceeded(
                turns_used=turn,
                turns_limit=run.invocation.max_turns,
                message="the model asked for another tool round at the turn limit",
            ),
        )

    def _dispatch(
        self, run: _Run, turn: int, call: Mapping[str, Any]
    ) -> str | InvocationResult:
        name, arguments = str(call["name"]), dict(call["arguments"])
        events = (
            self._profile.event_detail in ("tool_calls", "agent_graph")
            and NO_TOOL_EVENTS not in self._faults
        )
        if events:
            run.events.append(("tool_call", name))
        try:
            if WRITE_AROUND_WRAPPER in self._faults and name in self._toolset.write_names:
                text = self._toolset.isolated_mutation_inner_toolset().dispatch(
                    name, arguments
                )
            else:
                text = self._toolset.dispatch(name, arguments)
        except ToolDispatchBudgetExceeded as exc:
            return self._dispatch_budget_stop(run, turn, exc)
        if events:
            run.events.append(("tool_result", name))
        return self._bounded(run, str(text))

    def _dispatch_budget_stop(
        self, run: _Run, turn: int, exc: Exception
    ) -> InvocationResult:
        if BUDGET_STOP_AS_TRANSPORT in self._faults:
            return self._result(
                run,
                final="",
                stop="tool_budget_exhausted",
                error=TransportError(message="tool loop failed", retriable=False),
            )
        run.events.append(("limit", "tool_dispatch_budget"))
        misnamed = DISPATCH_STOP_MISNAMED in self._faults
        return self._result(
            run,
            final="",
            stop="max_turns" if misnamed else "tool_budget_exhausted",
            error=MaxTurnsExceeded(
                turns_used=turn,
                turns_limit=run.invocation.max_turns,
                message=f"the tool loop stopped at a dispatch budget: {exc}",
            ),
        )

    def _bounded(self, run: _Run, text: str) -> str | InvocationResult:
        """Apply the tool-result byte budget: truncate, then refuse, never stop."""

        budget = self._context_budget
        if budget is None or IGNORE_CONTEXT_BUDGET in self._faults:
            return text
        payload = text.encode("utf-8")
        if CONTEXT_LIMIT_STOP in self._faults and run.context_used + len(payload) > budget:
            run.events.append(("limit", "context_bytes"))
            return self._result(
                run,
                final="",
                stop="context_limit",
                error=TransportError(
                    message="tool-result context budget exceeded", retriable=False
                ),
            )
        available = budget - run.context_used
        if available <= 0:
            return "[BUDGET EXHAUSTED] No more tool results will be returned."
        if len(payload) <= available:
            run.context_used += len(payload)
            return text
        run.context_used = budget
        prefix = payload[:available].decode("utf-8", errors="ignore")
        return prefix + "\n\n[TRUNCATED] Result cut to the remaining context budget."

    # -- result ---------------------------------------------------------------

    def _result(
        self, run: _Run, *, final: str, stop: str, error: Any
    ) -> InvocationResult:
        faults = self._faults
        events = list(run.events)
        terminal = ("terminal", "done" if TERMINAL_MISNAMED in faults else stop)
        if LIMIT_AFTER_TERMINAL in faults and ("limit", "max_turns") in events:
            events.remove(("limit", "max_turns"))
            events += [terminal, ("limit", "max_turns")]
        else:
            events.append(terminal)
        usage = Usage(
            input_tokens=sum(segment.input_tokens for segment in run.segments),
            output_tokens=sum(segment.output_tokens for segment in run.segments),
            cost_usd=math.fsum(segment.cost_usd for segment in run.segments),
        )
        accounted = run.usage_known
        evidence: UsagePricingEvidence | None = UsagePricingEvidence(
            tuple(run.segments), complete=accounted
        )
        if ZERO_COST_NO_EVIDENCE in faults:
            usage, evidence = dataclasses.replace(usage, cost_usd=0.0), None
        if CLEAR_ACCOUNTING_FLAGS in faults and run.invocation.response_schema is not None:
            accounted, evidence = False, None
        if RETRIABLE_ERROR in faults and error is not None:
            error = dataclasses.replace(error, retriable=True)
        toolset = self._toolset
        trace = BackendExecutionTrace(
            backend=self.name,
            requested_model=self.model,
            served_model=run.served_model,
            endpoint=_OTHER_ENDPOINT if TRACE_OTHER_ENDPOINT in faults else self._endpoint,
            feature_profile_digest=(
                None
                if NO_PROFILE_DIGEST in faults
                else feature_profile_digest(self._profile)
            ),
            tool_authority_digest=(
                None
                if toolset is None
                else tool_authority_digest(
                    toolset,
                    capability=self.capability,
                    approval_policy=APPROVAL_POLICY_PER_CALL_EXACT,
                    bridge_protocol=TOOL_AUTHORITY_BRIDGE_PROTOCOL,
                    bridge_version=TOOL_AUTHORITY_BRIDGE_VERSION,
                    context_budget=self._context_budget,
                )
            ),
            events=tuple(
                BackendExecutionEvent(sequence=index, kind=kind, name=name)  # type: ignore[arg-type]
                for index, (kind, name) in enumerate(events)
            ),
        )
        return InvocationResult(
            final_message=final,
            transcript=(Message("assistant", final),) if final else (),
            usage=usage,
            stop_reason=stop,
            error=error,
            completeness=ExecutionCompleteness(
                token_usage_complete=accounted,
                cost_complete=accounted,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=trace,
            pricing_evidence=evidence,
        )


# -- the driver ---------------------------------------------------------------


@dataclass
class _Bound:
    backend: HonestBackend
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    secrets: tuple[str, ...] = (TOY_SESSION_TOKEN,)
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "function_call"

    def cancel_when(self, marker: str) -> None:
        raise MatrixSetupError(
            f"the toy dialect has no marker {marker!r}: cancellation scenarios add "
            "it with their liars"
        )

    def children(self) -> tuple[int, ...]:
        return self.backend.children()


class ToyDriver:
    """Drive the toy adapter. ``faults`` turn it into a liar."""

    kind: str = TOY_KIND
    dialect: str = ToyDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    bypassed_layers: tuple[str, ...] = (
        "routing (the toy is no catalog profile: bound with routing's two binders)",
        "any vendor SDK, worker or executable",
        "DNS and TLS",
    )
    private_seams: tuple[str, ...] = ()
    inexpressible: Mapping[str, str] = {}
    envelopes: tuple[str, ...] = ()  # the toy dialect renders its own replies

    def __init__(self, faults: frozenset[str] = frozenset()) -> None:
        self.faults = faults

    def preflight(self) -> None:
        return None

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        return {}

    def guarded_profile(self, profile: BackendFeatureSet) -> BackendFeatureSet:
        """The profile the limit guard is bound to: the column's, unless faulted."""

        if GUARD_ON_OTHER_PROFILE not in self.faults:
            return profile
        other = "unsupported" if profile.turn_limit != "unsupported" else "hard"
        return dataclasses.replace(profile, turn_limit=other)  # type: ignore[arg-type]

    def built_model(self, column: Column) -> str:
        """The model the adapter is built for: the column's, unless faulted."""

        if BUILT_FOR_OTHER_MODEL not in self.faults:
            return column.model
        # Another model of the matrix rate card: it has a price, at the same rate.
        return next(
            model for model in sorted(matrix_columns.rate_card()) if model != column.model
        )

    @contextlib.contextmanager
    def open(
        self,
        column: Column,
        *,
        capability: str,
        lens: Lens,
        script: ProviderScript,
        tmp_path: Path,
    ) -> Iterator[_Bound]:
        profile = column.features
        if profile is None:
            raise MatrixSetupError(f"column {column.id} is not a toy column")
        unsupported = unsupported_values(profile)
        if unsupported:
            raise MatrixSetupError(
                f"toy column {column.id}: no toy behaviour for {unsupported}"
            )
        if REFUSE_WRITER in self.faults and capability == "workspace_write":
            raise FeatureAdmissionError(
                f"the toy route of column {column.id} refuses the writer role"
            )
        log_dir = tmp_path / "toy-debug"
        with LoopbackProvider(script, credential=CANARY_CREDENTIAL) as provider:
            backend = HonestBackend(
                profile=profile,
                model=self.built_model(column),
                endpoint=provider.url,
                credential=CANARY_CREDENTIAL,
                session_token=TOY_SESSION_TOKEN,
                capability=capability,
                log_dir=log_dir,
                faults=self.faults,
            )
            role = matrix_columns.ROLE_BY_CAPABILITY[capability]
            bind_as_routing(
                backend,
                self.guarded_profile(profile),
                lens=lens,
                route=f"{TOY_KIND}/{column.model} for role {role!r}",
                # Every open-weight route is diagnostic; the toy rehearses them.
                admission="diagnostic",
            )
            if REQUEST_AT_BUILD in self.faults:
                backend._send("GET", "/toy/v1/models", None, 5.0)
            try:
                yield _Bound(backend, provider, log_dirs=(log_dir,))
            finally:
                backend.reap()


def unsupported_values(profile: BackendFeatureSet) -> list[str]:
    """Governing values of ``profile`` the toy has no behaviour for."""

    return [
        f"{name}={getattr(profile, name)}"
        for name, values in SUPPORTED.items()
        if getattr(profile, name) not in values
    ]


register(ToyDriver())


# -- liars --------------------------------------------------------------------


@dataclass(frozen=True)
class Liar:
    """One fault, where it runs, and exactly what its cell must yield."""

    scenario: str
    faults: frozenset[str]
    codes: frozenset[str]
    trips: tuple[str, ...]
    lens: str = "raw"
    rehearses: tuple[str, ...] = ()
    needs: Mapping[str, str] = field(default_factory=dict)
    composite: bool = False
    via: str = "cell"  # "cell", or "P02" for the static profile-binding check
    # The declared differences in force on the liar's cell. The ledger lists
    # them for a real cell; a liar names its own.
    differences: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        parts = [self.scenario, "+".join(sorted(self.faults)), self.lens]
        parts.extend(self.needs[name] for name in sorted(self.needs))
        return "/".join(parts)


def _liar(scenario: str, fault: str, codes: tuple[str, ...], trips: tuple[str, ...], **more: Any) -> Liar:
    return Liar(scenario, frozenset({fault}), frozenset(codes), trips, **more)


_RAISE = f"@after_request/{__name__}"
_BEFORE = f"@before_request/{__name__}"
_UNTOLD = ("flag:token_usage_complete=False", "flag:cost_complete=False")
_SUBSTITUTED = ("request.content:model", "served_model:not_requested")
_S40_COMMON = (
    "progress:write_not_credited",
    "stop:tool_budget_exhausted!=end_turn",
    f"dispatch.missing:{READ_A}",
    "request.unconsumed",
)

LIARS: tuple[Liar, ...] = (
    _liar("S00", CANNED_REPLY, ("request.unconsumed",), ("C0",)),
    _liar("S01", DROP_OUTPUT_TOKENS, ("usage.output_tokens:lt",), ("C6",)),
    _liar(
        "S01",
        ZERO_COST_NO_EVIDENCE,
        ("usage.cost_usd:lt", "pricing:none"),
        ("C6",),
        rehearses=("G32",),
    ),
    _liar("S01", RAISE_AFTER_REQUEST, (f"raise:RuntimeError{_RAISE}",), ("C1",)),
    _liar("S01", TERMINAL_MISNAMED, ("event.terminal:done!=end_turn",), ("C4",)),
    _liar(
        "S01",
        NO_PROFILE_DIGEST,
        ("digest.missing:feature_profile",),
        ("C7",),
        rehearses=("G24",),
    ),
    _liar("S01", LEAVE_CHILD, ("child:alive",), ("C8",)),
    _liar("S01", LEAVE_LISTENER, ("child:alive",), ("C8",)),
    _liar("S01", LATE_REQUEST, ("child:alive",), ("C8",)),
    _liar("S01", SECOND_CLOSE_RAISES, (f"raise:RuntimeError@close/{__name__}",), ("C8",)),
    _liar("S01", LEAK_CREDENTIAL, ("secret:leak",), ("C9",)),
    # A workflow run keeps INFO and above; the cell reads every level.
    _liar("S01", LOG_SESSION_TOKEN, ("secret:leak",), ("C9",)),
    _liar("S01", DEBUG_LOG_SESSION_TOKEN, ("secret:leak",), ("C9",)),
    _liar(
        "S01",
        RAISE_WITH_CREDENTIAL,
        (f"raise:RuntimeError{_RAISE}", "secret:leak"),
        ("C1", "C9"),
    ),
    _liar("S01", REPORTS_OTHER_PROFILE, ("static:profile_binding",), ("P02",), via="P02"),
    _liar("S01", GUARD_ON_OTHER_PROFILE, ("static:profile_binding",), ("P02",), via="P02"),
    _liar("S01", ALTER_TEXT, ("text:mismatch",), ("S",)),
    _liar("S01", REWRITE_PROMPT, ("request.content:prompt",), ("S",)),
    _liar("S01", NO_REQUEST_EVENT, ("event.missing:request/-",), ("S",)),
    _liar("S01", TWO_REQUEST_EVENTS, ("event.extra:request/-",), ("S",)),
    _liar("S01", SUBSTITUTE_WIRE_MODEL, _SUBSTITUTED, ("C7",)),
    # Consistent about a model that is not the route's: only the wire shows it.
    _liar("S01", BUILT_FOR_OTHER_MODEL, ("request.content:model",), ("C7",)),
    # The request went where the route says; the run record names another place.
    _liar("S01", TRACE_OTHER_ENDPOINT, ("digest.mismatch:endpoint",), ("C7",)),
    _liar("S01", GET_BEFORE_TURN, ("request.extra",), ("C0",)),
    _liar(
        "S01",
        GET_BEFORE_TURN,
        ("request.extra", "guard:second_call_failed"),
        ("C0", "C5"),
        lens="guarded",
    ),
    # The second call makes no request, makes its one request and returns an
    # error, or makes it and raises: three ways not to be usable any more.
    *(
        _liar("S01", fault, ("guard:second_call_failed",), ("C5",), lens="guarded")
        for fault in (DEAD_AFTER_FIRST_CALL, SECOND_CALL_ERRORS, SECOND_CALL_RAISES)
    ),
    _liar("S03", DISPATCH_UNATTACHED, (f"dispatch.extra:{_UNATTACHED_TOOL}",), ("C10",)),
    # A tool cell runs through another routing role than a text cell.
    _liar("S03", SUBSTITUTE_WIRE_MODEL, _SUBSTITUTED, ("C7",)),
    # Another model on one later round only (a retry on a fallback, a cheaper
    # model for one request): every scripted request is read, not the first.
    # The last reply is served under the route's model, so only the wire shows it.
    _liar("S03", SUBSTITUTE_SECOND_REQUEST_MODEL, ("request.content:model",), ("C7",)),
    _liar(
        "S03",
        REQUEST_EVENT_PER_ROUND,
        ("guard:limit_contract_violated", "guard:revoked"),
        ("C5",),
        lens="guarded",
        rehearses=("G16", "G28"),
    ),
    _liar("S03", EMPTY_TOOL_RESULTS, ("request.content:tool_results",), ("S",)),
    # The results of the last round arrive; the rounds before it are forgotten
    # on a wire that keeps no conversation. Only the dialect's rule shows it.
    _liar("S03", FORGET_EARLIER_ROUNDS, ("request.content:replay",), ("S",)),
    _liar(
        "S03",
        NO_TOOL_EVENTS,
        (
            f"event.missing:tool_call/{READ_A}",
            f"event.missing:tool_result/{READ_A}",
            f"event.missing:tool_call/{READ_B}",
            f"event.missing:tool_result/{READ_B}",
        ),
        ("S",),
        needs={"event_detail": "tool_calls"},
    ),
    _liar("S04", CLEAR_ACCOUNTING_FLAGS, _UNTOLD, ("C6",), rehearses=("G29",)),
    _liar("S04", SCHEMA_ANSWER_INVALID, ("text:mismatch",), ("S",)),
    # Valid is not enough: an object of the schema that the provider never
    # sent is not the answer.
    _liar("S04", SCHEMA_ANSWER_REPLACED, ("text:mismatch",), ("S",)),
    _liar(
        "S04",
        SCHEMA_MISPLACED,
        ("request.content:schema",),
        ("S",),
        needs={"structured_output": "native"},
    ),
    # A route that declares the schema in the prompt owes two things, each with
    # its code: the schema text there, and no provider field.
    _liar(
        "S04",
        SCHEMA_MISPLACED,
        ("request.content:schema", "request.content:schema_prompt"),
        ("S",),
        needs={"structured_output": "prompt_only"},
    ),
    _liar(
        "S04",
        SCHEMA_NOT_IN_PROMPT,
        ("request.content:schema_prompt",),
        ("S",),
        needs={"structured_output": "prompt_only"},
    ),
    # With the declared difference of the schema finalisation in force the
    # trait is the honest play, and a fault beside it breaks one line of the
    # alternative: the finalisation is paid for and the draft of the last tool
    # round is returned; or a route that declares the schema in the prompt
    # sends no schema text there.
    Liar(
        "S04",
        frozenset({FINAL_SCHEMA_REQUEST, RETURN_SCHEMA_DRAFT}),
        frozenset({"text:mismatch"}),
        ("S",),
        differences=(D_OAI_SCHEMA_FINAL,),
    ),
    Liar(
        "S04",
        frozenset({FINAL_SCHEMA_REQUEST, SCHEMA_NOT_IN_PROMPT}),
        frozenset({"request.content:schema_prompt"}),
        ("S",),
        rehearses=("M04",),
        needs={"structured_output": "prompt_only"},
        differences=(D_OAI_SCHEMA_FINAL,),
    ),
    _liar(
        "S10a",
        SUCCESS_AT_TURN_LIMIT,
        ("kind:none!=max_turns", "stop:end_turn!=max_turns"),
        ("C2", "C3"),
        needs={"turn_limit": "hard"},
    ),
    _liar(
        "S10a",
        LIMIT_AFTER_TERMINAL,
        ("raise:ValueError@after_request/assay.types.invocation",),
        ("C1",),
        rehearses=("G14",),
        needs={"turn_limit": "hard"},
    ),
    _liar(
        "S10a",
        EXTRA_TOOL_REQUEST,
        ("request.extra",),
        ("C0",),
        needs={"turn_limit": "hard"},
    ),
    _liar(
        "S10a",
        SEND_TURN_CONTROL,
        ("request.content:turn_control",),
        ("S",),
        needs={"turn_limit": "unsupported"},
    ),
    _liar(
        "S10a",
        NO_LIMIT_EVENT,
        ("event.missing:limit/max_turns",),
        ("S",),
        needs={"turn_limit": "hard"},
    ),
    _liar(
        "S10a",
        RETRIABLE_ERROR,
        ("retriable:True",),
        ("C2",),
        needs={"turn_limit": "hard"},
    ),
    _liar(
        "S11",
        OWN_TURN_CEILING,
        (f"raise:ValueError{_BEFORE}", "request.unconsumed"),
        ("S", "C0"),
        rehearses=("G15",),
    ),
    _liar("S11", NO_REQUEST_EVENT, ("event.missing:request/-",), ("S",)),
    _liar("S11", TWO_REQUEST_EVENTS, ("event.extra:request/-",), ("S",)),
    _liar("S11", SUBSTITUTE_WIRE_MODEL, _SUBSTITUTED, ("C7",)),
    _liar(
        "S12",
        BUDGET_STOP_AS_TRANSPORT,
        ("kind:transport!=max_turns", "event.missing:limit/tool_dispatch_budget"),
        ("C2",),
        rehearses=("G02",),
    ),
    _liar(
        "S12",
        DISPATCH_STOP_MISNAMED,
        ("stop:max_turns!=tool_budget_exhausted",),
        ("C3",),
    ),
    _liar("S12", RETRIABLE_ERROR, ("retriable:True",), ("C2",)),
    _liar("S13", RETRIABLE_ERROR, ("retriable:True",), ("C2",)),
    _liar(
        "S13",
        IGNORE_USD_BUDGET,
        (
            "kind:none!=budget_exhausted",
            "stop:end_turn!=budget_exhausted",
            "event.missing:limit/budget_usd",
        ),
        ("C2", "C3"),
        rehearses=("G08",),
    ),
    _liar(
        "S18",
        CONTEXT_LIMIT_STOP,
        ("kind:transport!=none", "stop:context_limit!=end_turn", "request.unconsumed"),
        ("C3",),
        rehearses=("G39",),
    ),
    _liar("S18", IGNORE_CONTEXT_BUDGET, ("context:over_budget",), ("S",)),
    _liar("S18", EMPTY_TOOL_RESULTS, ("request.content:tool_results",), ("S",)),
    *(
        _liar(
            scenario,
            REFUSAL_AS_TRANSPORT,
            ("kind:transport!=content_rejected",),
            ("C2",),
            rehearses=("G18",),
        )
        for scenario in ("S23a", "S23b", "S23c")
    ),
    # The two forms the provider bills for must keep their usage.
    *(_liar(scenario, DROP_REFUSAL_USAGE, _UNTOLD, ("C6",)) for scenario in ("S23a", "S23b")),
    _liar("S23a", RETRIABLE_ERROR, ("retriable:True",), ("C2",)),
    _liar("S23a", REFUSAL_ENDS_END_TURN, ("stop:end_turn!=other",), ("C3",)),
    _liar("S23c", SECRET_IN_ERROR_MESSAGE, ("secret:leak",), ("C9",)),
    _liar(
        "S40",
        WRITE_AROUND_WRAPPER,
        (*_S40_COMMON, "kind:max_turns!=none"),
        ("S", "C2"),
        rehearses=("G17",),
    ),
    Liar(
        "S40",
        frozenset({WRITE_AROUND_WRAPPER, BUDGET_STOP_AS_TRANSPORT}),
        frozenset((*_S40_COMMON, "kind:transport!=none")),
        ("S", "C2"),
        rehearses=("G03",),
        composite=True,
    ),
    # Admission: a capability the column's pin admits is refused, and a
    # backend that is only built asks its provider something.
    _liar(
        "S51",
        REFUSE_WRITER,
        ("admission:workspace_write:FeatureAdmissionError!=admitted",),
        ("S",),
    ),
    _liar("S51", REQUEST_AT_BUILD, ("request.extra",), ("C0",)),
)
