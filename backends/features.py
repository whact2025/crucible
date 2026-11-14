# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Concrete-route backend feature profiles and admission policy decisions.

Profiles are deliberately resolved from backend *and* route identity.  The
``openai_compat`` adapter, in particular, spans heterogeneous gateways whose
observed behavior cannot safely be inferred from the Python class alone.

Feature admission is deliberately separate from adapter construction.  A
workflow declares the enforcement levels it can safely consume, then routing
evaluates those requirements before a transport object (and therefore before
any paid request) exists.  Routes default to enforcing rejection; the routing
document must explicitly select diagnostic mode to retain a rejected route as
non-promotable evidence.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping
from urllib.parse import urlsplit

from assay import (
    BACKEND_CONFORMANCE_VERSION,
    BackendFeatureSet,
    CONSERVATIVE_BACKEND_FEATURES,
)


_CODEX_FEATURES = BackendFeatureSet(
    structured_output="native",
    tool_loop="sdk_owned",
    session_continuity="workflow_bound",
    event_detail="usage",
    # app-server currently has no reviewed max-turn request control.
    turn_limit="unsupported",
    response_token_limit="unsupported",
    context_byte_limit="unsupported",
    usd_limit="post_hoc",
    # One absolute wrapper deadline spans attempts and interruptible backoff,
    # and cancellation kills the active app-server process. Keep the
    # conservative request_only label because the public provider surface
    # does not acknowledge a hard server-side deadline or prove billed work
    # stopped when the wrapper kills the process.
    deadline="request_only",
    retry_scope="invocation",
    cache_control="provider_owned",
    cancellation="process_kill",
    project_instructions="suppressible",
    native_delegation="none",
)

_CLAUDE_AGENT_FEATURES = BackendFeatureSet(
    structured_output="native",
    tool_loop="sdk_owned",
    session_continuity="workflow_bound",
    event_detail="usage",
    turn_limit="provider_stop",
    response_token_limit="unsupported",
    context_byte_limit="unsupported",
    usd_limit="post_hoc",
    # One absolute wrapper deadline spans Agent-SDK attempts and backoff, with
    # the active CLI process cancelled on expiry. This remains request_only:
    # the public stream does not acknowledge a hard provider-side deadline or
    # prove server-side/billed work stopped after local process cancellation.
    deadline="request_only",
    retry_scope="invocation",
    cache_control="provider_owned",
    cancellation="process_kill",
    project_instructions="suppressible",
    native_delegation="none",
)

_OPENAI_COMPAT_FEATURES = BackendFeatureSet(
    # This adapter sends response_format when requested, but an arbitrary
    # compatible gateway has not earned a native-schema conformance claim.
    structured_output="unsupported",
    tool_loop="adapter_owned",
    session_continuity="none",
    event_detail="usage",
    turn_limit="round_boundary",
    response_token_limit="provider_stop",
    # A hard tool-result cap exists only after a caller attaches one; the
    # static route profile cannot claim it unconditionally.
    context_byte_limit="unsupported",
    usd_limit="post_hoc",
    # One absolute wrapper deadline is forwarded as a decreasing request
    # timeout and also bounds retries, backoff, tool boundaries, and forced
    # finalization. Synchronous ToolSet handlers are not preemptible, so the
    # route cannot honestly claim hard-invocation enforcement.
    deadline="request_only",
    retry_scope="per_completion",
    cache_control="unsupported",
    cancellation="boundary_only",
    project_instructions="not_applicable",
    native_delegation="none",
)

_NVIDIA_K26_OPENAI_FEATURES = dataclasses.replace(
    _OPENAI_COMPAT_FEATURES,
    # The configured-inference route is a diagnostic control. Its current
    # schema behavior remains route-qualified rather than assumed native.
    structured_output="prompt_only",
)

_NVIDIA_GLM52_OPENAI_FEATURES = dataclasses.replace(
    _OPENAI_COMPAT_FEATURES,
    # The exact NVIDIA GLM-5.2 route has a qualified constrained-finalization
    # compatibility path, followed by Assay's deterministic schema validation.
    structured_output="prompt_only",
)

_KIMI_SDK_FEATURES = BackendFeatureSet(
    structured_output="prompt_only",
    tool_loop="sdk_owned",
    session_continuity="none",
    event_detail="tool_calls",
    turn_limit="hard",
    # The pinned native Kimi provider receives the model-specific response
    # cap through its qualified generation-kwargs shape (max_tokens for
    # K2.7, max_completion_tokens for K3).
    response_token_limit="provider_stop",
    context_byte_limit="unsupported",
    usd_limit="post_hoc",
    # One wrapper-owned absolute deadline and step ledger span physical
    # provider retries.
    deadline="hard_invocation",
    retry_scope="invocation",
    cache_control="unsupported",
    # The SDK worker is killable, but a synchronous ToolSet handler already
    # running in the parent cannot be killed until the tool host is
    # process-isolated.
    cancellation="boundary_only",
    project_instructions="suppressible",
    native_delegation="none",
)

_DEEPSEEK_RESPONSES_CONSTRUCTION_FEATURES = BackendFeatureSet(
    # The native Responses route receives one canonical schema through the
    # Assay prompt ABI; provider-native response_format remains unqualified.
    structured_output="prompt_only",
    tool_loop="adapter_owned",
    session_continuity="none",
    event_detail="tool_calls",
    turn_limit="hard",
    response_token_limit="provider_stop",
    context_byte_limit="unsupported",
    usd_limit="post_hoc",
    # The trusted subreaper proves ordinary local descendant settlement, but
    # the shared delegated-cgroup Gate has not closed supervisor SIGKILL.
    # DSK-00b also has not established provider-side cancellation or billing
    # settlement after disconnect.
    deadline="request_only",
    retry_scope="unsupported",
    cache_control="provider_owned",
    # Ordinary SDK-worker cancellation is subreaper-settled; the cgroup Gate
    # and in-process client function handler boundary remain open.
    cancellation="boundary_only",
    project_instructions="not_applicable",
    native_delegation="none",
)


def feature_profile_for_route(
    backend: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> BackendFeatureSet:
    """Return the conservative observed profile for one resolved route."""

    canonical = backend.strip().lower()
    if canonical == "codex":
        return _CODEX_FEATURES
    if canonical == "claude":
        return _CLAUDE_AGENT_FEATURES
    if canonical == "deepseek":
        if model == "deepseek-v4-flash" and base_url is None:
            return _DEEPSEEK_RESPONSES_CONSTRUCTION_FEATURES
        return CONSERVATIVE_BACKEND_FEATURES
    if canonical in {"local", "openai_compat"}:
        try:
            route_host = (
                urlsplit(base_url).hostname.lower()
                if base_url is not None
                and urlsplit(base_url).hostname is not None
                else None
            )
        except (TypeError, ValueError):
            route_host = None
        if route_host in {
            "inference-api.nvidia.com",
            "inference-api-dev.nvidia.com",
        }:
            if model == "nvidia/moonshotai/kimi-k2.6":
                return _NVIDIA_K26_OPENAI_FEATURES
            if model == "nvidia/zai-org/glm-5.2":
                return _NVIDIA_GLM52_OPENAI_FEATURES
        return _OPENAI_COMPAT_FEATURES
    if canonical == "kimi":
        try:
            parsed_route = urlsplit(base_url) if base_url is not None else None
            exact_moonshot_route = bool(
                parsed_route is not None
                and parsed_route.scheme.lower() == "https"
                and parsed_route.hostname is not None
                and parsed_route.hostname.lower() == "api.moonshot.ai"
                and parsed_route.port is None
                and parsed_route.username is None
                and parsed_route.password is None
                and parsed_route.path.rstrip("/") == "/v1"
                and not parsed_route.query
                and not parsed_route.fragment
            )
        except (TypeError, ValueError):
            exact_moonshot_route = False
        if model in {"kimi-k2.7-code-highspeed", "kimi-k3"} and exact_moonshot_route:
            return _KIMI_SDK_FEATURES
        return CONSERVATIVE_BACKEND_FEATURES
    return CONSERVATIVE_BACKEND_FEATURES


def feature_profile_digest(profile: BackendFeatureSet) -> str:
    """Return a stable version-bound digest for checkpoint provenance."""

    payload = {
        "conformance_version": BACKEND_CONFORMANCE_VERSION,
        "features": dataclasses.asdict(profile),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FeatureRequirement:
    """One execution-policy requirement evaluated before transport."""

    field: str
    accepted: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise TypeError("FeatureRequirement.field must be a non-empty string")
        if not isinstance(self.accepted, frozenset) or not all(
            isinstance(value, str) for value in self.accepted
        ):
            raise TypeError(
                "FeatureRequirement.accepted must be a frozenset of strings"
            )
        known = {item.name for item in dataclasses.fields(BackendFeatureSet)}
        known.remove("schema_version")
        if self.field not in known:
            raise ValueError(f"unknown backend feature field {self.field!r}")
        if not self.accepted:
            raise ValueError("FeatureRequirement.accepted must not be empty")
        allowed = BackendFeatureSet.allowed_values(self.field)
        unknown = self.accepted - allowed
        if unknown:
            raise ValueError(
                f"FeatureRequirement.accepted for {self.field!r} contains "
                f"unknown values {sorted(unknown)}; expected values from "
                f"{sorted(allowed)}"
            )


@dataclass(frozen=True)
class FeatureDecision:
    """Allow/reject result with stable human-readable reasons."""

    allowed: bool
    reasons: tuple[str, ...] = ()


def evaluate_feature_requirements(
    profile: BackendFeatureSet,
    requirements: Iterable[FeatureRequirement],
) -> FeatureDecision:
    """Evaluate requirements without performing or authorizing transport."""

    reasons: list[str] = []
    for requirement in requirements:
        observed = str(getattr(profile, requirement.field))
        if observed not in requirement.accepted:
            reasons.append(
                f"{requirement.field}={observed!r} is not one of "
                f"{sorted(requirement.accepted)}"
            )
    return FeatureDecision(allowed=not reasons, reasons=tuple(reasons))


FeatureAdmissionMode = Literal["enforce", "diagnostic"]
FEATURE_ADMISSION_MODES: frozenset[str] = frozenset(
    {"enforce", "diagnostic"}
)


class FeatureAdmissionError(ValueError):
    """A selected route cannot satisfy the workflow's declared contract."""


def execution_feature_requirements(
    *,
    capability: str,
    native_shell: bool | None,
    additional: Iterable[FeatureRequirement] = (),
) -> tuple[FeatureRequirement, ...]:
    """Return common requirements for one concrete workflow-role execution.

    Role capability is an authority ceiling; it must also be realizable by the
    selected route.  Tool-bearing calls require a real adapter/SDK loop, and a
    caller that explicitly disables native shell/project surfaces requires an
    adapter capable of suppressing them (or a direct transport where they are
    not applicable).  Workflows may append narrower requirements, such as
    accepting ``prompt_only`` but not ``unsupported`` structured output.
    """

    from .capabilities import normalize_capability

    normalized = normalize_capability(capability)
    requirements: list[FeatureRequirement] = []
    if normalized != "text":
        requirements.append(
            FeatureRequirement(
                "tool_loop",
                frozenset({"sdk_owned", "adapter_owned"}),
            )
        )
    if native_shell is False:
        requirements.append(
            FeatureRequirement(
                "project_instructions",
                frozenset({"suppressible", "not_applicable"}),
            )
        )
    requirements.extend(additional)
    return tuple(requirements)


def admit_feature_requirements(
    profile: BackendFeatureSet,
    requirements: Iterable[FeatureRequirement],
    *,
    mode: FeatureAdmissionMode,
    route: str,
) -> FeatureDecision:
    """Evaluate and, in enforcing mode, reject before adapter transport.

    Diagnostic mode is intentionally explicit and returns the same rejected
    decision to its caller for logging/provenance.  It does not turn a route
    into a conforming route.
    """

    if mode not in FEATURE_ADMISSION_MODES:
        raise ValueError(
            f"unknown feature admission mode {mode!r}; expected one of "
            f"{sorted(FEATURE_ADMISSION_MODES)}"
        )
    decision = evaluate_feature_requirements(profile, requirements)
    if not decision.allowed and mode == "enforce":
        raise FeatureAdmissionError(
            f"backend route {route} does not satisfy workflow feature "
            f"requirements: {'; '.join(decision.reasons)}"
        )
    return decision


def profile_identity(
    backend: str,
    *,
    model: str | None,
    base_url: str | None,
) -> Mapping[str, str]:
    """Return the secret-free profile identity embedded in route provenance."""

    profile = feature_profile_for_route(
        backend,
        model=model,
        base_url=base_url,
    )
    return {
        "conformance_version": BACKEND_CONFORMANCE_VERSION,
        "profile_digest": feature_profile_digest(profile),
    }


__all__ = [
    "FEATURE_ADMISSION_MODES",
    "FeatureAdmissionError",
    "FeatureAdmissionMode",
    "FeatureDecision",
    "FeatureRequirement",
    "admit_feature_requirements",
    "evaluate_feature_requirements",
    "execution_feature_requirements",
    "feature_profile_digest",
    "feature_profile_for_route",
    "profile_identity",
]
