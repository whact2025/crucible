# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Drivers of the two gateway kinds: the hosted gateway and a local server.

Both build the backend exactly as a workflow does, with ``resolve_for_role``
on the column's shipped routing file, and answer it from the scripted
loopback provider in the OpenAI chat-completions dialect. The adapter, the
root ``openai`` SDK with its own transport, the in-process tool loop and a
real socket all run; no product name is patched.

- **hosted**: the route's endpoint variable expands to the production gateway
  URL, so the host-keyed feature profile and the route's shims stay on. The
  origin redirect (:mod:`..tap`) answers that origin on loopback. Bypassed:
  DNS, TLS, the gateway.
- **local**: the route's endpoint variable expands to the loopback provider.
  Nothing is patched; only the model server is bypassed.

The seam canary of a column fails unless the request is the SDK's own: the
documented path, the installed SDK's user agent, a bearer credential (the
route's, where the route has one) and the host the client addressed
(production behind the redirect, loopback for the local kind).

Chat completions keeps no conversation, so the dialect states what each
request must replay of the earlier rounds (:func:`replay_problems`), for the
scenarios that attach a ToolSet.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence
from urllib.parse import urlsplit

from crucible.backends.config import load_routing, resolve_for_role

from .. import columns as matrix_columns
from .. import envelopes as envelope_files
from .. import netguard, tap
from ..columns import CANARY_CREDENTIAL, COLUMN_PRODUCT, Column
from ..provider import LoopbackProvider, Reply, SeenRequest, SentCall
from ..script import Call, ProviderScript, Say, ServedAs, Step
from ..toolsets import BINDINGS
from .base import GUARD_POLICY, Lens, MatrixSetupError, register

_SDK = "openai"
_CREATED = 1_700_000_000


def _text(content: Any) -> str:
    """The text of one chat message's content: a string, or text parts."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _shape(message: Mapping[str, Any]) -> dict[str, Any]:
    """What one chat message is to the conversation: its role and its call ids."""

    role = str(message.get("role", ""))
    calls = message.get("tool_calls")
    return {
        "role": role,
        # The calls an assistant turn asks for, in its order.
        "asks": [
            str(call.get("id", ""))
            for call in (calls if isinstance(calls, list) else ())
            if isinstance(call, dict)
        ],
        # The call a tool message answers.
        "answers": str(message.get("tool_call_id", "")) if role == "tool" else None,
    }


class OpenAIChatDialect:
    """The OpenAI chat-completions wire, one JSON request and reply per round."""

    name = "openai_chat"
    path = "/v1/chat/completions"

    def __init__(self, provider: str, *, host: str | None, route_credential: bool) -> None:
        self._provider = provider
        # The host the client must have addressed; ``None`` means loopback.
        self._host = host
        # Whether the route itself carries the credential the canary knows.
        self._route_credential = route_credential

    @property
    def user_agent(self) -> str:
        """What the installed SDK names itself: read, never typed."""

        return f"OpenAI/Python {importlib.metadata.version(_SDK)}"

    def envelope_names(self) -> tuple[str, ...]:
        return tuple(
            f"{self.name}/{self._provider}.{behaviour}"
            for behaviour in ("completion", "tool_call")
        )

    def parse(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes,
        *,
        credential: str,
    ) -> SeenRequest:
        lowered = {name.lower(): value for name, value in headers.items()}
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        messages = [
            item for item in payload.get("messages") or () if isinstance(item, dict)
        ]
        last_model_turn = max(
            (index for index, item in enumerate(messages) if item.get("role") == "assistant"),
            default=-1,
        )
        authorization = lowered.get("authorization", "")
        response_format = payload.get("response_format")
        return SeenRequest(
            method=method,
            path=path,
            user_agent=lowered.get("user-agent", ""),
            credential_ok=(
                authorization == f"Bearer {credential}"
                if self._route_credential
                else authorization.startswith("Bearer ") and len(authorization) > 7
            ),
            model=str(payload.get("model", "")),
            system="\n".join(
                _text(item.get("content")) for item in messages if item.get("role") == "system"
            ),
            user="\n".join(
                _text(item.get("content")) for item in messages if item.get("role") == "user"
            ),
            tools=tuple(
                str(item.get("function", {}).get("name", ""))
                for item in payload.get("tools") or ()
                if isinstance(item, dict)
            ),
            # The history is replayed in full on every request: what answers
            # the last reply is what follows the last model turn.
            tool_results=tuple(
                (str(item.get("tool_call_id", "")), _text(item.get("content")))
                for item in messages[last_model_turn + 1 :]
                if item.get("role") == "tool"
            ),
            # The wire has no field for a turn count or a spend ceiling.
            turn_control=None,
            budget_control=None,
            schema_field=(
                isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
            ),
            host=lowered.get("host", ""),
            # The shape of the conversation the request carries: roles and
            # call ids, no text.
            evidence={"messages": [_shape(item) for item in messages]},
        )

    def seam_problems(self, seen: SeenRequest) -> list[str]:
        """Why this request cannot be the SDK's own request for this route."""

        problems: list[str] = []
        if seen.path != self.path:
            problems.append(f"request path {seen.path!r}, not {self.path!r}")
        if seen.user_agent != self.user_agent:
            problems.append(
                f"user agent {seen.user_agent!r}, not the installed SDK's "
                f"{self.user_agent!r}"
            )
        if not seen.credential_ok:
            problems.append(
                "the canary credential is not in the Authorization header"
                if self._route_credential
                else "the Authorization header carries no bearer credential"
            )
        if self._host is None:
            if not netguard.is_loopback_host(urlsplit(f"//{seen.host}").hostname):
                problems.append(f"the client addressed {seen.host!r}, not loopback")
        elif seen.host != self._host:
            problems.append(
                f"the client addressed {seen.host!r}, not the production host "
                f"{self._host!r}: the request did not come through the redirect"
            )
        return problems

    def replay_problems(
        self, requests: Sequence[SeenRequest], calls: Sequence[SentCall]
    ) -> list[str]:
        """The stateless-replay rule of the wire, for the cells of both kinds.

        The scripted provider answers on position and call ids alone, so it
        answers a conversation that a real endpoint refuses. The provider
        applies this rule to the scripted exchange when it ends, and the
        scenarios with a ToolSet report what it names.
        """

        return replay_problems(requests, calls)

    def render(self, step: Step, seen: SeenRequest, call_id: str) -> Reply:
        served = seen.model
        if isinstance(step, ServedAs):
            served = seen.model + step.suffix
            step = step.step
        if isinstance(step, Say):
            behaviour, call = "completion", None
            slots: dict[str, Any] = {"text": step.text}
        elif isinstance(step, Call):
            binding = BINDINGS[step.slot]
            arguments = dict(binding.arguments if step.args is None else step.args)
            behaviour, call = "tool_call", (step.slot, call_id, arguments)
            slots = {
                "call_id": call_id,
                "tool_name": binding.name,
                # The wire carries the arguments as a JSON document in a string.
                "tool_arguments": json.dumps(arguments, sort_keys=True),
            }
        else:
            raise MatrixSetupError(
                f"the {self.name} dialect does not render {type(step).__name__} yet; "
                "the scenario that needs it adds the envelope with its liar"
            )
        envelope = envelope_files.load(f"{self.name}/{self._provider}.{behaviour}")
        usage = step.usage
        slots.update(
            response_id=f"chatcmpl-{call_id}",
            created=_CREATED,
            served_model=served,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
        return Reply(
            envelope.status, envelope.fill(slots), usage, served, call, envelope.name
        )


def conversation(seen: SeenRequest) -> list[str]:
    """The conversation a request carries, one word per message.

    The role of the message, with the call ids of an assistant turn that asks
    for tools and of the tool message that answers one:
    ``system, user, assistant[call-1], tool[call-1]``.
    """

    words: list[str] = []
    for message in seen.evidence.get("messages") or ():
        ids = [message["answers"]] if message["role"] == "tool" else message["asks"]
        words.append(message["role"] + (f"[{', '.join(ids)}]" if ids else ""))
    return words


def replay_problems(
    requests: Sequence[SeenRequest], calls: Sequence[SentCall]
) -> list[str]:
    """Why the requests of one call do not replay the conversation in full.

    The wire keeps no conversation. The first request of a call carries the
    opening messages, the prompts, and nothing of a tool round. Every later
    request begins with the same opening and carries each earlier reply that
    asked for a tool, in order, as the assistant turn that names the call
    directly followed by the tool message that answers it (where one turn
    names several calls, by one tool message for each). A real endpoint
    refuses a tool message that answers no call of the assistant turn before
    it, and an assistant turn whose calls are not all answered.

    What else the client says is its own and the wire takes it: a user
    message between two rounds or after the last one (the instruction of a
    forced finalisation) and an assistant turn without a tool call (the draft
    before a schema finalisation).

    ``requests`` are all requests of the exchange in order, as the provider
    ledger keeps them; a tool call names the request it answered by its index
    there. A request that matched no scripted step is not judged: it is an
    extra request, which clause C0 reports.
    """

    problems: list[str] = []
    opening: list[str] | None = None
    for index, request in enumerate(requests):
        if not request.matched:
            continue
        messages = list(request.evidence.get("messages") or ())
        roles = [message["role"] for message in messages]
        why: list[str] = []
        if opening is None:
            # The prompts: what precedes the first turn of the model.
            first_turn = next(
                (place for place, role in enumerate(roles) if role in ("assistant", "tool")),
                len(roles),
            )
            opening = roles[:first_turn]
        elif roles[: len(opening)] != opening:
            why.append(f"it does not begin with the opening messages ({', '.join(opening)})")
        rounds: list[str] = []
        at = 0
        while at < len(messages):
            message = messages[at]
            at += 1
            if message["role"] == "tool":
                why.append(
                    f"the tool message for {message['answers']} follows no assistant "
                    "turn that asked for it"
                )
            elif message["role"] == "assistant" and message["asks"]:
                results: list[str] = []
                while at < len(messages) and messages[at]["role"] == "tool":
                    results.append(messages[at]["answers"])
                    at += 1
                if sorted(results) != sorted(message["asks"]):
                    why.append(
                        f"the assistant turn that asked for {message['asks']} is "
                        f"followed by the results of {results}"
                    )
                rounds.extend(message["asks"])
        answered = [call.call_id for call in calls if call.request_index < index]
        if rounds != answered:
            why.append(
                f"it replays the tool rounds {rounds} where the provider had asked "
                f"for {answered}"
            )
        if why:
            problems.append(
                f"request {index + 1} carries {', '.join(conversation(request))}: "
                + "; ".join(why)
            )
    return problems


@dataclass
class _Bound:
    backend: Any
    provider: LoopbackProvider
    log_dirs: tuple[Path, ...]
    # The adapter holds no token beside the route's credential.
    secrets: tuple[str, ...] = ()
    tool_path: Literal["function_call", "mcp_bridge", "control_channel"] = "function_call"

    def cancel_when(self, marker: str) -> None:
        raise MatrixSetupError(
            f"the gateway drivers have no marker {marker!r}: cancellation "
            "scenarios add it with their liars"
        )

    def children(self) -> tuple[int, ...]:
        return ()  # the adapter runs in this process


class _GatewayDriver:
    """What the two gateway kinds share: routing, the dialect, the bound backend."""

    kind: str
    provider: str
    bypassed_layers: tuple[str, ...]
    dialect: str = OpenAIChatDialect.name
    fidelity: Literal["wire", "worker_protocol"] = "wire"
    # The origin redirect patches httpx's transports, not a product name.
    private_seams: tuple[str, ...] = ()

    @property
    def inexpressible(self) -> Mapping[str, str]:
        """Script steps this kind's wire cannot carry: the inventory's rows only."""

        from .. import cells  # noqa: PLC0415 - cells imports the drivers

        return {
            f"{row['scenario']}/{row['step']}": row["reason"]
            for row in cells.load_inventory().not_expressible
            if row["kind"] == self.kind
        }

    @property
    def envelopes(self) -> tuple[str, ...]:
        return self._dialect().envelope_names()

    def preflight(self) -> None:
        for module in (_SDK, "httpx"):
            if importlib.util.find_spec(module) is None:
                raise MatrixSetupError(
                    f"kind {self.kind} needs the root environment's {module!r}; "
                    "run `uv sync --frozen --group dev`"
                )

    def overlay(self, column: Column, env: Mapping[str, str]) -> Mapping[str, str]:
        return {}  # the shipped route is used as it is

    def _dialect(self) -> OpenAIChatDialect:
        raise NotImplementedError

    def _endpoint(self, provider: LoopbackProvider) -> Mapping[str, str]:
        """What the route's endpoint variable expands to for this cell."""

        return {}

    def _redirect(
        self, stack: contextlib.ExitStack, provider: LoopbackProvider
    ) -> tap.TapRecord | None:
        return None

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
        if column.kind != self.kind:
            raise MatrixSetupError(f"column {column.id} is not of kind {self.kind}")
        log_dir, cwd = tmp_path / "backend-logs", tmp_path / "backend-cwd"
        cwd.mkdir(parents=True, exist_ok=True)
        with contextlib.ExitStack() as stack:
            provider = stack.enter_context(
                LoopbackProvider(script, credential=CANARY_CREDENTIAL, dialect=self._dialect())
            )
            stack.enter_context(
                matrix_columns.scripted_environment(**self._endpoint(provider))
            )
            record = self._redirect(stack, provider)
            # The boundary of the matrix: exactly what a workflow calls.
            backend = resolve_for_role(
                matrix_columns.ROLE_BY_CAPABILITY[capability],
                routing=load_routing(config_path=column.routing_path(COLUMN_PRODUCT)),
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(cwd),
                log_dir=log_dir,
                limit_policy=GUARD_POLICY if lens == "guarded" else None,
            )
            try:
                yield _Bound(backend, provider, log_dirs=(log_dir,))
            finally:
                with contextlib.suppress(Exception):
                    backend.close()  # the runner has observed close() already
                if record is not None and record.violations:
                    # An SDK wraps what its transport raises; the record decides.
                    raise MatrixSetupError(
                        f"column {column.id}: a request left for another origin "
                        f"than the route's: {'; '.join(record.violations)}"
                    )


class HostedGatewayDriver(_GatewayDriver):
    """The hosted gateway: production URL in the route, answered on loopback."""

    kind = matrix_columns.HOSTED_GATEWAY_KIND
    provider = "gateway"
    bypassed_layers: tuple[str, ...] = (
        "DNS and TLS (the origin redirect answers the production origin on loopback)",
        "the gateway and its models (replies are scripted envelopes)",
    )

    def _production(self) -> str:
        return matrix_columns.SCRIPTED_ENV[matrix_columns.ENDPOINT_ENV[self.kind]]

    def _dialect(self) -> OpenAIChatDialect:
        return OpenAIChatDialect(
            self.provider, host=urlsplit(self._production()).netloc, route_credential=True
        )

    def _redirect(
        self, stack: contextlib.ExitStack, provider: LoopbackProvider
    ) -> tap.TapRecord | None:
        return stack.enter_context(tap.redirect(self._production(), provider.url))


class LocalGatewayDriver(_GatewayDriver):
    """A local model server: the route's endpoint is the loopback provider."""

    kind = matrix_columns.LOCAL_GATEWAY_KIND
    provider = "local"
    bypassed_layers: tuple[str, ...] = (
        "the model server (replies are scripted envelopes)",
    )

    def _dialect(self) -> OpenAIChatDialect:
        # The shipped local route carries no credential of its own.
        return OpenAIChatDialect(self.provider, host=None, route_credential=False)

    def _endpoint(self, provider: LoopbackProvider) -> Mapping[str, str]:
        return {matrix_columns.ENDPOINT_ENV[self.kind]: f"{provider.url}/v1"}


register(HostedGatewayDriver())
register(LocalGatewayDriver())
