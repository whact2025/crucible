# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""The two gateway kinds: their cells, and the proof of their driver.

The cells run the starter set on the hosted gateway columns and the local
column through the real adapter, the root ``openai`` SDK and a loopback
socket. The remaining tests show that the driver's seam can fail: a request
that did not come through the redirect, a redirect that is missing, a request
for a foreign origin, a dialect that answers under a fixed model id or does
not read the model on the wire. They show that the conversation is replayed
as the wire needs and that a cell deviates when it is not. They also pin the
envelopes and their grades and show the declared difference of the schema
finalisation asserted on the real column.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import http.server
import importlib.metadata
import json
import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import urlsplit

import httpx
import pytest
import yaml  # type: ignore[import-untyped]
from openai.resources.chat.completions import Completions
from openai.types.chat import ChatCompletion

from . import cells, contract, declarations, envelopes, ledger, netguard, runner, tap
from . import columns as matrix_columns
from .cells import Cell
from .columns import HOSTED_GATEWAY_KIND, LOCAL_GATEWAY_KIND, Column
from .drivers import MatrixSetupError
from .drivers import openai_compat as gateway
from .provider import LoopbackProvider, Reply, SeenRequest, SentCall, StatelessDialect
from .scenarios import (
    D_OAI_SCHEMA_FINAL,
    SCENARIOS,
    SCHEMA_ANSWER,
    SCHEMA_FINAL_ANSWER,
    history_replayed,
)
from .script import READ_A, READ_B, Call, ProviderScript, Say, ServedAs, Step, TokenUsage

LEDGER = ledger.load()
KINDS = (HOSTED_GATEWAY_KIND, LOCAL_GATEWAY_KIND)
CELLS = cells.cells_of(KINDS)
COLUMNS = tuple(column for column in matrix_columns.real_columns() if column.kind in KINDS)
PRODUCTION = matrix_columns.SCRIPTED_ENV[matrix_columns.ENDPOINT_ENV[HOSTED_GATEWAY_KIND]]


@pytest.mark.parametrize("cell", CELLS, ids=lambda cell: cell.id)
def test_cell(cell: Cell, tmp_path: Path, matrix_preflight: dict[str, str | None]) -> None:
    runner.run_cell(cell, tmp_path, preflight=matrix_preflight, known=LEDGER)


def _column(kind: str, role: str = "primary") -> Column:
    return next(item for item in COLUMNS if (item.kind, item.role) == (kind, role))


def _played(
    column: Column, scenario: str, tmp_path: Path, lens: str = "raw", **more: Any
) -> runner.Execution:
    cell = Cell(column, scenario, lens)
    more.setdefault("differences", LEDGER.differences_for(cell.id))
    return runner.execute(cell, tmp_path, **more)


# -- the seam canary: the real layers ran ---------------------------------------


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_canary_shows_the_sdks_own_request_to_the_host_of_the_route(
    column: Column, tmp_path: Path
) -> None:
    execution = _played(column, "S00", tmp_path)
    (request,) = execution.observation.provider.requests
    host = urlsplit(f"//{request.host}").hostname

    assert execution.report.seam_problems == () and execution.report.codes == frozenset()
    assert request.matched and request.path == "/v1/chat/completions"
    # The installed SDK built the request: its user agent, not a typed one.
    assert request.user_agent == f"OpenAI/Python {importlib.metadata.version('openai')}"
    assert request.credential_ok
    # The model on the wire is the catalog's, read at run time.
    assert request.model == column.model != ""
    if column.kind == HOSTED_GATEWAY_KIND:
        # The client addressed production; the redirect answered it on loopback.
        assert request.host == urlsplit(PRODUCTION).netloc and not netguard.is_loopback_host(host)
    else:
        assert netguard.is_loopback_host(host)
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    assert result.execution_trace.endpoint == (
        PRODUCTION if column.kind == HOSTED_GATEWAY_KIND else f"http://{request.host}/v1"
    )


def test_both_kinds_build_one_adapter_through_routing(tmp_path: Path) -> None:
    names = {}
    for column in COLUMNS:
        with gateway_driver(column).open(
            column,
            capability="text",
            lens="raw",
            script=ProviderScript((Say("unused"),)),
            tmp_path=tmp_path / column.id.replace("/", "-"),
        ) as bound:
            names[column.id] = (bound.backend.name, bound.backend.features)
            assert bound.tool_path == "function_call" and bound.children() == ()

    # The local route builds the same class: the adapter name is one constant.
    assert {name for name, _features in names.values()} == {"openai_compat"}
    assert {column.adapter for column in COLUMNS} == {"openai_compat"}
    for column in COLUMNS:
        assert names[column.id][1] == declarations.routed(column)


def gateway_driver(column: Column) -> Any:
    return (
        gateway.HostedGatewayDriver()
        if column.kind == HOSTED_GATEWAY_KIND
        else gateway.LocalGatewayDriver()
    )


class _AroundTheRedirect(gateway.HostedGatewayDriver):
    """The route's endpoint is the loopback provider: the redirect never acts."""

    def _endpoint(self, provider: LoopbackProvider) -> Mapping[str, str]:
        return {matrix_columns.ENDPOINT_ENV[self.kind]: f"{provider.url}/v1"}


class _WithoutRedirect(gateway.HostedGatewayDriver):
    """The route keeps its production endpoint and nothing answers it."""

    def _redirect(self, stack: contextlib.ExitStack, provider: LoopbackProvider) -> None:
        return None


def test_canary_fails_when_the_request_did_not_come_through_the_redirect(
    tmp_path: Path,
) -> None:
    cell = Cell(_column(HOSTED_GATEWAY_KIND), "S00", "raw")

    with pytest.raises(pytest.fail.Exception, match="did not come through the redirect"):
        runner.run_cell(
            cell,
            tmp_path,
            preflight={HOSTED_GATEWAY_KIND: None},
            known=LEDGER,
            driver=_AroundTheRedirect(),
        )


def test_a_hosted_cell_without_the_redirect_is_stopped_by_the_network_guard(
    tmp_path: Path,
) -> None:
    execution = runner.execute(
        Cell(_column(HOSTED_GATEWAY_KIND), "S00", "raw"), tmp_path, driver=_WithoutRedirect()
    )
    blocked = netguard.take_violations()
    production = urlsplit(PRODUCTION).hostname or ""

    # The SDK turns the blocked call into a connection error and the adapter
    # into a typed result; the guard's record and C0 are what show it.
    assert production and blocked and all(production in entry for entry in blocked)
    assert execution.observation.provider.request_count == 0
    assert execution.report.codes == {"request.unconsumed"}


def test_seam_names_what_is_not_the_sdks_request_for_the_route() -> None:
    hosted = gateway.HostedGatewayDriver()._dialect()
    local = gateway.LocalGatewayDriver()._dialect()

    def seen(dialect: gateway.OpenAIChatDialect, **headers: str) -> SeenRequest:
        sent = {
            "User-Agent": dialect.user_agent,
            "Authorization": f"Bearer {matrix_columns.CANARY_CREDENTIAL}",
            "Host": urlsplit(PRODUCTION).netloc,
            **headers,
        }
        return dialect.parse(
            "POST", dialect.path, sent, b"{}", credential=matrix_columns.CANARY_CREDENTIAL
        )

    assert hosted.seam_problems(seen(hosted)) == []
    for changed, named in (
        ({"User-Agent": "python-httpx/0.28"}, "user agent"),
        ({"Authorization": "Bearer another-key"}, "canary credential"),
        ({"Host": "127.0.0.1:8000"}, "did not come through the redirect"),
    ):
        (problem,) = hosted.seam_problems(seen(hosted, **changed))
        assert named in problem
    (path,) = hosted.seam_problems(dataclasses.replace(seen(hosted), path="/v1/responses"))
    assert "request path" in path
    # The local route carries no credential of its own: any bearer value is the
    # SDK's, and the client must have addressed loopback.
    assert local.seam_problems(seen(local, Host="127.0.0.1:8000", Authorization="Bearer x")) == []
    (remote,) = local.seam_problems(seen(local))
    assert "not loopback" in remote
    (bare,) = local.seam_problems(seen(local, Host="localhost:1", Authorization="x"))
    assert "no bearer credential" in bare


# -- the origin redirect ----------------------------------------------------------


@contextlib.contextmanager
def _echo_server() -> Iterator[tuple[str, list[dict[str, str]]]]:
    """A loopback HTTP server that records the Host header and path it saw."""

    seen: list[dict[str, str]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - http.server's name
            seen.append({"host": self.headers.get("Host", ""), "path": self.path})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b"ok")

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_redirect_answers_the_production_origin_on_loopback_and_refuses_others(
    tmp_path: Path,
) -> None:
    sync_handler = httpx.HTTPTransport.handle_request
    async_handler = httpx.AsyncHTTPTransport.handle_async_request
    marker = tmp_path / "tap-installed"

    async def asked(url: str) -> str:
        async with httpx.AsyncClient() as client:
            return (await client.get(url)).text

    with _echo_server() as (target, seen):
        with tap.redirect(PRODUCTION, target, marker=marker) as record:
            assert tap.installed() and "installed in process" in marker.read_text("utf-8")
            with pytest.raises(RuntimeError, match="already installed"):
                tap.install(PRODUCTION, target)
            with httpx.Client() as client:
                assert client.get(f"{PRODUCTION}/models?page=2").text == "ok"
                assert client.get(f"{target}/direct").text == "ok"
                with pytest.raises(tap.TapViolation, match="neither the route's"):
                    client.get("https://example.invalid/v1/models")
                # Another port of the production host is another origin.
                with pytest.raises(tap.TapViolation):
                    client.get(PRODUCTION.replace("https://", "http://") + "/models")
            assert asyncio.run(asked(f"{PRODUCTION}/async")) == "ok"
            with pytest.raises(tap.TapViolation):
                asyncio.run(asked("https://example.invalid/async"))
        loopback = urlsplit(target).netloc

        # Path and query are the client's, and so is the Host header: the
        # provider side can tell that production was addressed.
        assert seen == [
            {"host": urlsplit(PRODUCTION).netloc, "path": "/v1/models?page=2"},
            {"host": loopback, "path": "/direct"},
            {"host": urlsplit(PRODUCTION).netloc, "path": "/v1/async"},
        ]
        assert (record.redirected, record.passed, len(record.violations)) == (2, 1, 3)
        production = urlsplit(PRODUCTION).hostname or ""
        assert production and all(
            "example.invalid" in entry or production in entry for entry in record.violations
        )
    # Nothing was resolved or connected beyond loopback, and nothing is left patched.
    assert netguard.take_violations() == ()
    assert not tap.installed()
    assert httpx.HTTPTransport.handle_request is sync_handler
    assert httpx.AsyncHTTPTransport.handle_async_request is async_handler
    with pytest.raises(RuntimeError, match="not installed"):
        tap.remove()
    with pytest.raises(ValueError, match="must be loopback"):
        tap.install(PRODUCTION, "https://example.invalid")


def test_a_request_for_a_foreign_origin_fails_the_cell_even_when_it_is_swallowed(
    tmp_path: Path,
) -> None:
    column = _column(HOSTED_GATEWAY_KIND)

    # An SDK wraps what its transport raises into a connection error, and an
    # adapter may type that as a result. The record decides, not the raise.
    opened = gateway.HostedGatewayDriver().open(
        column,
        capability="text",
        lens="raw",
        script=ProviderScript((Say("unused"),)),
        tmp_path=tmp_path,
    )
    with pytest.raises(MatrixSetupError, match="left for another origin.*example.invalid"), opened:
        with contextlib.suppress(tap.TapViolation):
            httpx.get("https://example.invalid/telemetry")
    assert not tap.installed()


# -- the dialect ------------------------------------------------------------------


class _FixedServedId(gateway.OpenAIChatDialect):
    """Answers every request under one id, whatever model was asked for."""

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        reply = super().render(step, seen, call_id)
        return dataclasses.replace(reply, served_model="a-fixed-model-id")


class _ModelBlind(gateway.OpenAIChatDialect):
    """Does not read the model a request names."""

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), model="")


def _with_dialect(dialect: type[gateway.OpenAIChatDialect]) -> gateway.HostedGatewayDriver:
    class Driver(gateway.HostedGatewayDriver):
        def _dialect(self) -> gateway.OpenAIChatDialect:
            return dialect(
                self.provider, host=urlsplit(PRODUCTION).netloc, route_credential=True
            )

    return Driver()


def test_a_dialect_that_answers_under_a_fixed_model_is_a_seam_problem(tmp_path: Path) -> None:
    cell = Cell(_column(HOSTED_GATEWAY_KIND), "S01", "raw")

    # It would switch off "served equals requested" for the whole kind.
    with pytest.raises(pytest.fail.Exception, match="answered under model 'a-fixed-model-id'"):
        runner.run_cell(
            cell,
            tmp_path,
            preflight={HOSTED_GATEWAY_KIND: None},
            known=LEDGER,
            driver=_with_dialect(_FixedServedId),
        )
    # A scripted ServedAs step is the one way to name another model.
    dialect = gateway.HostedGatewayDriver()._dialect()
    asked = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")
    assert dialect.render(Say("x"), asked, "1").served_model == "asked"
    assert dialect.render(ServedAs(Say("x")), asked, "1").served_model == "asked-served"


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_every_scripted_request_carries_the_model_of_its_route(
    column: Column, tmp_path: Path
) -> None:
    execution = _played(column, "S03", tmp_path)
    requests = execution.observation.provider.requests

    # Three rounds through the planner route: each names the route's model,
    # so clause C7 compares every request, not the first.
    assert SCENARIOS["S03"].capability == "read_only" and len(requests) == 3
    assert {request.model for request in requests} == {execution.reference.model}
    assert execution.reference.model == column.model
    # The endpoint C7 compares the trace with is the route's own: production
    # on the hosted kind, the scripted provider the client addressed on the
    # local kind. It is read from the routing file, not from the adapter.
    assert execution.reference.endpoint == (
        PRODUCTION if column.kind == HOSTED_GATEWAY_KIND else f"http://{requests[0].host}/v1"
    )
    assert execution.report.outcomes["C7"].status == "ok"
    # Clause C9 read what the SDK and its HTTP client logged at every level
    # while the cell played, and found no credential there.
    dump = execution.observation.debug_dump_text
    assert "/chat/completions" in dump and "connect_tcp" in dump
    assert matrix_columns.CANARY_CREDENTIAL not in dump
    assert execution.report.outcomes["C9"].status == "ok"


def test_c7_is_live_on_a_real_column_through_what_the_dialect_reads(tmp_path: Path) -> None:
    blind = runner.execute(
        Cell(_column(HOSTED_GATEWAY_KIND), "S01", "raw"),
        tmp_path,
        driver=_with_dialect(_ModelBlind),
    )

    assert "request.content:model" in blind.report.codes


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_a_trace_that_names_another_endpoint_fails_c7_on_a_real_column(
    column: Column, tmp_path: Path
) -> None:
    execution = _played(column, "S01", tmp_path)
    result = execution.observation.result
    assert result is not None and result.execution_trace is not None
    trace = result.execution_trace
    shown = execution.report.codes

    # The adapter is honest about where it sent the request ...
    assert trace.endpoint == execution.reference.endpoint is not None
    assert not any(code.endswith(":endpoint") for code in shown)
    # ... and a run record that names another origin, or none, is a deviation
    # of the cell: one adapter serves both kinds, the endpoint tells them apart.
    for recorded, code in (
        ("https://example.invalid/v1", "digest.mismatch:endpoint"),
        (PRODUCTION + "/other", "digest.mismatch:endpoint"),
        (None, "digest.missing:endpoint"),
    ):
        changed = dataclasses.replace(
            result, execution_trace=dataclasses.replace(trace, endpoint=recorded)
        )
        judged = contract.evaluate(
            dataclasses.replace(execution.observation, result=changed),
            execution.plan.expect,
            execution.reference,
            lens="raw",
            toolset=False,
            checks=execution.plan.checks,
        )
        assert judged.codes - shown == {code}, recorded
        assert "C7" in judged.tripped()


def test_dialect_reads_the_replayed_history_and_the_schema_field() -> None:
    dialect = gateway.LocalGatewayDriver()._dialect()
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "ask"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "first"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c2"}, {"id": "c3"}]},
            {"role": "tool", "tool_call_id": "c2", "content": "second"},
            {"role": "tool", "tool_call_id": "c3", "content": "third"},
            {"role": "user", "content": "nudge"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_source"}}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
    }
    # The wire is JSON: a body in another encoding reads as nothing.
    assert dialect.parse("POST", dialect.path, {}, b"model: m", credential="").model == ""
    seen = dialect.parse(
        "POST", dialect.path, {"host": "127.0.0.1:1"}, json.dumps(body).encode(), credential=""
    )
    # What answers the last reply is what follows the last model turn.
    assert seen.tool_results == (("c2", "second"), ("c3", "third"))
    assert (seen.model, seen.system, seen.user) == ("m", "sys", "ask\nnudge")
    assert seen.tools == ("read_source",) and seen.schema_field
    assert seen.turn_control is None and seen.budget_control is None
    # The conversation is kept by its shape, roles and call ids, without text.
    assert gateway.conversation(seen) == [
        "system",
        "user",
        "assistant[c1]",
        "tool[c1]",
        "assistant[c2, c3]",
        "tool[c2]",
        "tool[c3]",
        "user",
    ]
    assert "first" not in json.dumps(seen.evidence)
    plain = dialect.parse(
        "POST",
        dialect.path,
        {},
        json.dumps({**body, "response_format": {"type": "json_object"}}).encode(),
        credential="",
    )
    assert not plain.schema_field


# -- the conversation is replayed as the wire needs ---------------------------------

_OPENING: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "ask"},
]
_WORD = {"role": "user", "content": "go on"}
_DRAFT = {"role": "assistant", "content": "a draft"}


def _asking(*call_ids: str) -> dict[str, Any]:
    """An assistant turn that asks for these tool calls."""

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "type": "function"} for call_id in call_ids],
    }


def _answering(call_id: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": "result"}


def _carrying(messages: Sequence[Mapping[str, Any]], *, matched: bool = True) -> SeenRequest:
    """A request with these messages, as the dialect reads it off the wire."""

    dialect = gateway.LocalGatewayDriver()._dialect()
    body = json.dumps({"model": "m", "messages": list(messages)}).encode()
    seen = dialect.parse("POST", dialect.path, {}, body, credential="")
    return dataclasses.replace(seen, matched=matched)


def test_a_conversation_that_drops_or_reorders_a_turn_is_named() -> None:
    one = [_asking("call-1"), _answering("call-1")]
    two = [_asking("call-2"), _answering("call-2")]
    calls = [SentCall(READ_A, "call-1", 0, {}), SentCall(READ_B, "call-2", 1, {})]
    first, second = _carrying(_OPENING), _carrying([*_OPENING, *one])

    def named(third: Sequence[Mapping[str, Any]]) -> list[str]:
        return gateway.replay_problems([first, second, _carrying(third)], calls)

    # The wire keeps no conversation: after the opening messages a request
    # carries every earlier round, the turn that asked and then its result.
    assert named([*_OPENING, *one, *two]) == []
    # What the client says beside the rounds is its own and the wire takes
    # it: the draft and the instruction of a schema finalisation, the
    # instruction after the last round of a turn limit, a word between rounds.
    assert named([*_OPENING, *one, *two, _DRAFT, _WORD]) == []
    assert named([*_OPENING, *one, *two, _WORD]) == []
    assert named([*_OPENING, *one, _WORD, *two]) == []
    # One turn may ask for several calls: one result for each follows it.
    together = [SentCall(READ_A, "call-1", 0, {}), SentCall(READ_B, "call-2", 0, {})]
    for results in (("call-1", "call-2"), ("call-2", "call-1")):
        both = [_asking("call-1", "call-2"), *(_answering(call_id) for call_id in results)]
        assert gateway.replay_problems([first, _carrying([*_OPENING, *both])], together) == []
    for broken in (
        [*_OPENING, *two],  # the earlier round is forgotten
        [*_OPENING, *one, two[1]],  # a result that no turn asked for
        [*_OPENING, *one, two[1], two[0]],  # the result before the turn that asked
        [*_OPENING, *one, two[0], _answering("call-9")],  # the result of another call
        [*_OPENING, *one, two[0]],  # the call has no result
        [*_OPENING, *one, two[0], _WORD, two[1]],  # a word between the two
        [*_OPENING, *one, *two, _WORD, two[1]],  # a result once more, after a word
        [*_OPENING, *two, *one],  # the rounds in another order
        [*_OPENING, *one, *one, *two],  # a round twice
        [*_OPENING, *one, *two, _asking("call-7"), _answering("call-7")],  # never asked
        [_OPENING[1], *one, *two],  # the system message is gone
        [],  # nothing is carried
    ):
        (problem,) = named(broken)
        assert problem.startswith("request 3 carries"), broken
    # The first request of a call replays nothing.
    (early,) = gateway.replay_problems([_carrying([*_OPENING, *one])], calls[:1])
    assert early.startswith("request 1 carries")
    # A request that matched no scripted step is an extra request, which
    # clause C0 reports; the rule judges the scripted ones, by their place.
    stray = _carrying([two[1]], matched=False)
    later = [SentCall(READ_A, "call-1", 1, {})]
    assert gateway.replay_problems([stray, first, second], later) == []
    # The dialect of both kinds states this rule, which is how a cell comes
    # to assert it.
    for driver in (gateway.HostedGatewayDriver(), gateway.LocalGatewayDriver()):
        dialect = driver._dialect()
        assert isinstance(dialect, StatelessDialect)
        assert dialect.replay_problems([first, second], calls[:1]) == []
        (stated,) = dialect.replay_problems([first, _carrying([*_OPENING, one[1]])], calls[:1])
        assert stated.startswith("request 2 carries")


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_a_tool_round_is_replayed_as_the_turn_that_asked_and_its_result(
    column: Column, tmp_path: Path
) -> None:
    loop = _played(column, "S03", tmp_path / "loop")
    first, second = (call.call_id for call in loop.observation.provider.sent_calls)
    opening = ["system", "user"]
    one = [f"assistant[{first}]", f"tool[{first}]"]
    two = [f"assistant[{second}]", f"tool[{second}]"]

    # The cell asserts it: the dialect states the rule, the provider applied
    # it to the exchange and the scenario reports what it names.
    assert history_replayed in loop.plan.checks
    assert [gateway.conversation(item) for item in loop.observation.provider.requests] == [
        opening,
        [*opening, *one],
        [*opening, *one, *two],
    ]
    assert loop.observation.provider.replay_problems == []
    assert "request.content:replay" not in loop.report.codes
    # The finalisation request of the planner shape carries the round too,
    # then the draft and the instruction to finalise it.
    planner = _played(column, "S04", tmp_path / "planner")
    (call,) = (item.call_id for item in planner.observation.provider.sent_calls)
    assert planner.observation.provider.tail_served
    assert gateway.conversation(planner.observation.provider.requests[-1]) == [
        *opening,
        f"assistant[{call}]",
        f"tool[{call}]",
        "assistant",
        "user",
    ]
    assert history_replayed in planner.plan.checks
    assert planner.observation.provider.replay_problems == []
    assert "request.content:replay" not in planner.report.codes


_CREATE = Completions.create


def _sending(
    change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
) -> Callable[..., Any]:
    """The SDK's own call, with the conversation of each request changed on its way.

    What an adapter that built another conversation hands to the SDK. The
    request that leaves is still the SDK's, so the seam holds.
    """

    def create(self: Any, *arguments: Any, **named: Any) -> Any:
        named["messages"] = change([dict(item) for item in named["messages"]])
        return _CREATE(self, *arguments, **named)

    return create


def _forgetting_earlier_rounds(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The opening messages, then the last turn that asked for a tool and what follows it."""

    asking = [index for index, item in enumerate(messages) if item.get("tool_calls")]
    if len(asking) < 2:
        return messages
    return [*messages[: asking[0]], *messages[asking[-1] :]]


def _without_the_asking_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The tool results go back without the model turns that asked for them."""

    return [item for item in messages if not item.get("tool_calls")]


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
@pytest.mark.parametrize(
    ("scenario", "change", "named"),
    [
        ("S03", _forgetting_earlier_rounds, "request 3 carries"),
        ("S04", _without_the_asking_turns, "request 2 carries"),
    ],
    ids=["S03-earlier-rounds-forgotten", "S04-result-without-the-turn-that-asked"],
)
def test_a_conversation_that_is_not_replayed_in_full_is_a_deviation_of_the_cell(
    column: Column,
    scenario: str,
    change: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    named: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = Cell(column, scenario, "raw")
    honest = _played(column, scenario, tmp_path / "honest")
    monkeypatch.setattr(Completions, "create", _sending(change))

    lying = _played(column, scenario, tmp_path / "lying")

    # The scripted provider answered all the same: it picks a reply on
    # position and call ids, where a real endpoint refuses a tool message
    # that no assistant turn asked for. Nothing else of the cell changed ...
    told, seen = honest.observation.provider, lying.observation.provider
    assert lying.report.seam_problems == ()
    assert (seen.request_count, seen.extra, seen.unconsumed) == (told.request_count, 0, 0)
    assert seen.tail_served == told.tail_served
    # ... so the codes are those of the honest cell and the one of the replay
    # rule, which names the request. As a test item the cell fails, with an
    # entry or without: no entry covers the code.
    assert told.replay_problems == []
    assert lying.report.codes == honest.report.codes | {"request.content:replay"}
    (problem, *_later) = seen.replay_problems
    assert problem.startswith(named)
    with pytest.raises(ledger.LedgerMismatch) as refused:
        ledger.compare(
            LEDGER,
            cell.id,
            lying.report.codes,
            current=lying.current,
            evaluated=lying.report.evaluated(),
            unasked=lying.report.unasked,
            report=lying.report.describe(),
        )
    assert "request.content:replay" in str(refused.value).splitlines()[0]
    assert "unexpected" in str(refused.value).splitlines()[0]


# -- envelopes and grades -----------------------------------------------------------


@pytest.mark.parametrize("provider", ["gateway", "local"])
def test_envelopes_parse_through_the_installed_sdk(provider: str) -> None:
    dialect = gateway.OpenAIChatDialect(provider, host=None, route_credential=False)
    asked = dialect.parse("POST", dialect.path, {}, b'{"model": "asked"}', credential="")

    said = dialect.render(Say("text", TokenUsage(7, 3)), asked, "call-1")
    called = dialect.render(Call(READ_A), asked, "call-2")
    completion = ChatCompletion.model_validate(said.body)
    tool_turn = ChatCompletion.model_validate(called.body)

    assert said.envelope == f"openai_chat/{provider}.completion"
    assert called.envelope == f"openai_chat/{provider}.tool_call"
    assert (completion.model, completion.choices[0].finish_reason) == ("asked", "stop")
    assert completion.choices[0].message.content == "text"
    assert completion.usage is not None
    assert (completion.usage.prompt_tokens, completion.usage.completion_tokens) == (7, 3)
    tool_calls = tool_turn.choices[0].message.tool_calls
    assert tool_calls is not None and [item.id for item in tool_calls] == ["call-2"]
    assert tool_turn.choices[0].finish_reason == "tool_calls"
    assert called.call is not None and called.call[1] == "call-2"


def test_the_weakest_envelope_a_cell_used_is_its_grade(tmp_path: Path) -> None:
    hosted, local = _column(HOSTED_GATEWAY_KIND), _column(LOCAL_GATEWAY_KIND)
    grades = {
        (column.kind, scenario): _played(column, scenario, tmp_path / f"{index}").grade
        for index, (column, scenario) in enumerate(
            [(hosted, "S01"), (hosted, "S03"), (local, "S01"), (local, "S03")]
        )
    }

    # A plain gateway completion was observed live; a tool call was not, and
    # nothing was observed of a local server.
    assert grades == {
        (HOSTED_GATEWAY_KIND, "S01"): "documented",
        (HOSTED_GATEWAY_KIND, "S03"): "constructed",
        (LOCAL_GATEWAY_KIND, "S01"): "constructed",
        (LOCAL_GATEWAY_KIND, "S03"): "constructed",
    }
    assert envelopes.weakest(()) is None
    assert envelopes.weakest(["openai_chat/gateway.completion"]) == "documented"


def test_envelope_rules_refuse_what_a_renderer_or_a_recording_must_not_do() -> None:
    path = envelopes.ROOT / "openai_chat" / "gateway.completion.yaml"
    text = path.read_text(encoding="utf-8")
    good = yaml.safe_load(text)
    envelope = envelopes.parse(good, where="x", text=text)

    def refused(match: str, **changes: Any) -> None:
        with pytest.raises(MatrixSetupError, match=match):
            envelopes.parse({**good, **changes}, where="x")

    # Provenance: every grade states its evidence.
    refused("is not one of", provenance={"grade": "guessed"})
    refused("must list its sources", provenance={"grade": "documented"})
    refused(
        "must state exactly",
        provenance={"grade": "documented", "sources": [{"source": "a page"}]},
    )
    refused("must state .*sha256", provenance={"grade": "recorded", "recorded_on": "d"})
    refused("built for, or the reason", provenance={"grade": "constructed"})
    assert envelopes.parse(
        {**good, "provenance": {"grade": "constructed", "reason": "no source exists"}}, where="x"
    ).grade == "constructed"
    # A renderer fills slots, the known ones, all of them, and nothing else.
    refused("slots \\['price'\\]", body={"cost": {"slot": "price"}})
    slots = {name: 1 for name in envelope.slots()}
    assert envelope.fill(slots)["usage"]["total_tokens"] == 1
    for wrong in ({**slots, "text2": 1}, {k: v for k, v in slots.items() if k != "text"}):
        with pytest.raises(MatrixSetupError, match="has the slots"):
            envelope.fill(wrong)
    assert good["body"]["object"] == envelope.fill(slots)["object"]  # not a slot: unchanged
    # A file holds exactly the closed set of keys.
    with pytest.raises(MatrixSetupError, match="must hold exactly"):
        envelopes.parse({**good, "headers": {}}, where="x")
    # Nothing derived from a credential is stored, in any envelope.
    for stored, named in (
        ("authorization: Bearer abcdef123456", "authorization header"),
        ("x-api-key=abc", "key assignment"),
        ("key sk-live0123456789abcdef", "API key"),
        ("org-1234567890ab", "account id"),
        ("mail someone@example.com", "e-mail address"),
        ("id 0123456789abcdef0123456789abcdef01234567", "long opaque token"),
    ):
        assert any(named in found for found in envelopes.hygiene_problems(stored)), stored
        with pytest.raises(MatrixSetupError, match="it stores"):
            envelopes.parse(good, where="x", text=text + "\n# " + stored)
    assert envelopes.hygiene_problems("sha256:" + "0a" * 32) == []
    for shipped in envelopes.all_envelopes().values():
        assert shipped.grade in envelopes.GRADES


# -- the declared difference of the schema finalisation -----------------------------


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_the_schema_finalisation_is_asserted_where_the_ledger_puts_it_in_force(
    column: Column, tmp_path: Path
) -> None:
    cell = Cell(column, "S04", "raw")
    in_force = LEDGER.differences_for(cell.id)
    execution = runner.execute(cell, tmp_path / "in-force", differences=in_force)
    requests = execution.observation.provider.requests

    assert in_force == (D_OAI_SCHEMA_FINAL,)
    # The shape of the exchange, and only that; which codes the cell shows is
    # between the cell and the ledger. One further request without tools
    # carries the schema, no other does, and the provider answered it from the
    # tail of the script.
    assert [(bool(item.tools), item.schema_field) for item in requests] == [
        (True, False),
        (True, False),
        (False, True),
    ]
    assert all(item.matched for item in requests)
    assert execution.observation.provider.tail_served
    assert execution.report.outcomes["C0"].status == "ok"
    # Where it is not in force the same request is an extra request.
    without = runner.execute(cell, tmp_path / "not-in-force")
    assert "request.extra" in without.report.codes
    assert not without.observation.provider.tail_served


class _ReturnsTheDraft:
    """The real adapter, except that its call returns the draft of the last tool round.

    It still sends the finalisation request and accounts for it as the adapter
    does: only the message it hands back is the earlier one.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def invoke(self, invocation: Any) -> Any:
        result = self._backend.invoke(invocation)
        return dataclasses.replace(result, final_message=SCHEMA_ANSWER)


@pytest.mark.parametrize("column", COLUMNS, ids=lambda column: column.id)
def test_an_adapter_that_returns_the_draft_fails_the_cell_whatever_its_entry(
    column: Column, tmp_path: Path
) -> None:
    class Driver(type(gateway_driver(column))):  # type: ignore[misc]
        @contextlib.contextmanager
        def open(self, *arguments: Any, **named: Any) -> Iterator[Any]:
            with super().open(*arguments, **named) as bound:
                yield dataclasses.replace(bound, backend=_ReturnsTheDraft(bound.backend))

    cell = Cell(column, "S04", "raw")
    honest = _played(column, "S04", tmp_path / "honest")
    result = honest.observation.result

    # The adapter itself returns the answer to the finalisation request, which
    # the script makes another valid text than the draft ...
    assert result is not None and honest.observation.provider.tail_served
    assert json.loads(result.final_message) == json.loads(SCHEMA_FINAL_ANSWER)
    assert json.loads(SCHEMA_FINAL_ANSWER) != json.loads(SCHEMA_ANSWER)
    assert "text:mismatch" not in honest.report.codes
    # ... and one that paid for the finalisation and returned the draft is not
    # covered by any entry of the cell: both texts validate, only one is owed.
    with pytest.raises(ledger.LedgerMismatch, match=r"unexpected .*'text:mismatch'"):
        runner.run_cell(
            cell,
            tmp_path / "draft",
            preflight={column.kind: None},
            known=LEDGER,
            driver=Driver(),
        )


class _SchemaOnEveryRequest(gateway.OpenAIChatDialect):
    """Reads the schema field into every request.

    What the provider would see of an adapter that carried the schema on its
    tool-bearing requests as well as on the final one.
    """

    def parse(self, *arguments: Any, **named: Any) -> SeenRequest:
        return dataclasses.replace(super().parse(*arguments, **named), schema_field=True)


@pytest.mark.parametrize("lens", SCENARIOS["S04"].lenses)
def test_a_finalisation_request_that_went_wrong_fails_the_cell_whatever_its_entry(
    lens: str, tmp_path: Path
) -> None:
    cell = Cell(_column(HOSTED_GATEWAY_KIND), "S04", lens)

    # The cell may be ledgered today for other reasons (the accounting of the
    # finalisation, the schema text the route declares in the prompt). None of
    # its entries covers the placement of the schema field: that code is
    # reported as an unexpected one, with or without an entry.
    with pytest.raises(ledger.LedgerMismatch, match=r"unexpected .*'request\.content:schema'"):
        runner.run_cell(
            cell,
            tmp_path,
            preflight={HOSTED_GATEWAY_KIND: None},
            known=LEDGER,
            driver=_with_dialect(_SchemaOnEveryRequest),
        )


# -- the module ends with the canary check --------------------------------------------


def test_canary_passed_for_every_column(request: pytest.FixtureRequest) -> None:
    outcomes = request.config.stash.get(runner.OUTCOMES, {})

    assert runner.canary_problems(outcomes, CELLS) == []
