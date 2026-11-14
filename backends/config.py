# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Per-role backend routing configuration.

Every LLM workflow resolves named roles through this module. The built-in
route is Codex with GPT-6 Astra and Max reasoning for every role; operator
YAML may replace that default or pin individual roles. Backend and model are
configuration concerns rather than independent workflow CLI flags.

Discovery order (first hit wins):

1. ``--backend-config`` CLI flag (passed by workflows to ``load_routing``).
2. ``CRUCIBLE_BACKEND_CONFIG`` env var — absolute path to a YAML file.
3. ``<spec_dir>/.crucible/backends.yaml`` — spec-local override.
4. ``<workspace>/.crucible/backends.yaml`` — workspace-wide default.
5. Built-in defaults (``DEFAULT_ROUTING``).

Config schema (closed and versioned)::

    schema_version: crucible.backend-routing.v1
    default:
      backend: codex             # claude | codex | deepseek | kimi |
                                 # local | openai_compat
      model: gpt-6-astra
      effort: max                # Claude/Codex reasoning effort
      base_url: ...              # openai_compat or dedicated kimi route
      api_key: ${INFER_AUTH_TOKEN} # ${VAR} expands
                                 #   from os.environ at load time so the
                                 #   secret never lands in the YAML
      feature_admission: enforce # enforce (default) | diagnostic
      extra:                     # adapter-specific options are explicit
        max_tokens: 8192
        # Codex only: legacy_app_server (default) | official_sdk
        # transport: official_sdk

    roles:
      plan_critic:      { backend: claude, model: aws/anthropic/bedrock-claude-fable-5-1 }
      goal_judge:       { backend: codex,  model: gpt-6-astra, effort: max }
      # qwen2.5-coder-32b must also have an explicit zero-rate pricing entry.
      impl_breaker:     { backend: openai_compat, model: qwen2.5-coder-32b }
      # ... unspecified roles inherit ``default``

Registered roles omitted from the config inherit ``default``. Unknown role
names, unknown root fields, and misspelled role fields are rejected. There is
no family-name collapsing; each workflow role names itself and declares its
maximum capability in :mod:`crucible.backends.roles`.

The built-in default routes every role to Codex with GPT-6 Astra and Max
reasoning. ``DEFAULT_ROUTING`` intentionally has no per-role pins, so changing
``default`` in an operator file switches the whole cascade. Mixing remains
possible through explicit role entries, which always win.

Use ``resolve_for_role(role, routing=..., log_dir=..., cwd=...)`` at
each workflow call site. The routing object is cheap to pass around;
the actual ``Backend`` instances are constructed lazily.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, cast

from crucible.common.workspace import workspace_root
from crucible.common.authority_io import AuthorityIOError, read_exact_regular

from . import resolve as _resolve_backend
from .features import (
    FEATURE_ADMISSION_MODES,
    FeatureAdmissionError,
    FeatureAdmissionMode,
    FeatureRequirement,
    admit_feature_requirements,
    execution_feature_requirements,
    feature_profile_for_route,
    profile_identity,
)
from .limits import (
    InvocationLimitIdentity,
    InvocationLimitLedger,
    InvocationLimitPolicy,
    InvocationLimitRequest,
    InvocationLimitState,
    bind_invocation_limit_contract,
    bind_invocation_limit_policy,
    preflight_invocation_limit_policy,
    resolve_invocation_limits,
)
from .evidence import bind_result_evidence_admission
from .roles import (
    role_feature_requirements,
    role_spec,
    validate_role_capability,
)

logger = logging.getLogger(__name__)

_ROUTING_SCHEMA = "crucible.backend-routing.v1"
_BACKENDS = frozenset(
    {
        "claude",
        "codex",
        "deepseek",
        "kimi",
        "local",
        "openai_compat",
    }
)
_CLAUDE_CREDENTIAL_SOURCES = frozenset({"environment", "claude_session"})
_CLAUDE_SESSION_MODELS = frozenset({
    "claude-fable-5-1",
    "claude-mythos-5-1",
    "claude-opus-5",
})
_CODEX_CREDENTIAL_SOURCES = frozenset({"environment", "chatgpt_session"})
_CODEX_CHATGPT_MODELS = frozenset({"gpt-5.6-sol", "gpt-6-astra"})
_KIMI_DIAGNOSTIC_WRITE_OPTION = "diagnostic_workspace_write"
_KIMI_DIAGNOSTIC_MODELS = frozenset({"kimi-k2.7-code-highspeed", "kimi-k3"})
_KIMI_DIAGNOSTIC_ENDPOINT = "https://api.moonshot.ai/v1"
_KIMI_DIAGNOSTIC_ENDPOINT_REF = "${KIMI_BASE_URL}"
_KIMI_WORKER_SOURCES = frozenset({"environment", "project_locked"})
_KIMI_RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_KIMI_MAX_CONTEXT_SIZES = {"kimi-k2.7-code-highspeed": 262_144, "kimi-k3": 1_048_576}
_DEEPSEEK_MODEL = "deepseek-v4-flash"
_DEEPSEEK_MAX_RESPONSE_TOKENS = 32_000
_DEEPSEEK_RUNTIME_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DEEPSEEK_DIAGNOSTIC_WRITE_OPTION = "diagnostic_workspace_write"
_ALLOWED_EXTRA_FIELDS = {
    "claude": frozenset(
        {
            "cli_path",
            "credential_source",
            "expected_org_id",
            "expected_org_name",
        }
    ),
    "codex": frozenset(
        {
            "cli_path",
            "credential_source",
            "output_schema",
            "transport",
            "trusted_project_instructions",
            "timeout_seconds",
        }
    ),
    "deepseek": frozenset(
        {
            _DEEPSEEK_DIAGNOSTIC_WRITE_OPTION,
            "expected_worker_runtime_fingerprint",
            "max_response_tokens",
            "timeout_seconds",
        }
    ),
    "kimi": frozenset(
        {
            _KIMI_DIAGNOSTIC_WRITE_OPTION,
            "diagnostic_cost_upper_bound",
            "expected_worker_runtime_fingerprint",
            "max_response_tokens",
            "timeout_seconds",
            "worker_source",
        }
    ),
    "local": frozenset(
        {"reasoning_effort", "max_tokens", "expected_served_model"}
    ),
    "openai_compat": frozenset(
        {"reasoning_effort", "max_tokens", "expected_served_model"}
    ),
}


@dataclass(frozen=True)
class RoleConfig:
    """Per-role backend selection.

    All fields are optional at the role level — unset fields inherit
    from ``default`` at resolution time. ``extra`` is the catch-all
    for adapter-specific kwargs the factory understands.
    """

    backend: str | None = None
    model: str | None = None
    effort: str | None = None
    base_url: str | None = None
    # Expanded credentials must never appear in config/routing reprs or take
    # part in structural equality. Route identity records only whether
    # authentication is configured, never the credential value.
    api_key: str | None = field(default=None, repr=False, compare=False)
    # ``None`` is meaningful on a role override: inherit the default route's
    # policy. Effective routes always resolve to enforce or diagnostic.
    feature_admission: FeatureAdmissionMode | None = "enforce"
    extra: dict[str, Any] = field(default_factory=dict)


def _overlay_role_config(base: RoleConfig, override: RoleConfig) -> RoleConfig:
    """Overlay one route without leaking settings across backend families."""

    backend = override.backend or base.backend
    same_backend = backend == base.backend

    def inherited(value: Any, fallback: Any) -> Any:
        if value is not None:
            return value
        return fallback if same_backend else None

    return RoleConfig(
        backend=backend,
        model=inherited(override.model, base.model),
        effort=inherited(override.effort, base.effort),
        base_url=inherited(override.base_url, base.base_url),
        api_key=inherited(override.api_key, base.api_key),
        # Admission is control-plane policy rather than an adapter setting,
        # so a role inherits it even when switching backend families.
        feature_admission=(
            override.feature_admission
            if override.feature_admission is not None
            else base.feature_admission
        ),
        extra=(
            {**base.extra, **override.extra} if same_backend else dict(override.extra)
        ),
    )


@dataclass(frozen=True)
class BackendRouting:
    """Resolved routing table: a fully-populated default + role overrides."""

    default: RoleConfig
    roles: dict[str, RoleConfig] = field(default_factory=dict)
    schema_version: str = _ROUTING_SCHEMA
    source_sha256: str | None = None

    def for_role(self, role: str) -> RoleConfig:
        """Return the effective config for ``role``, with default fall-through.

        Role-level fields override defaults; unset role fields inherit.
        Unknown roles are a configuration error, while registered roles that
        are not explicitly configured inherit the default.
        """
        role_spec(role)
        role_cfg = self.roles.get(role)
        if role_cfg is None:
            return self.default
        return _overlay_role_config(self.default, role_cfg)


_ROUTING_SNAPSHOT: contextvars.ContextVar[BackendRouting | None] = (
    contextvars.ContextVar("crucible_backend_routing_snapshot", default=None)
)


@contextlib.contextmanager
def bind_routing_snapshot(
    routing: BackendRouting,
) -> Iterator[BackendRouting]:
    """Freeze one resolved routing table across a nested workflow cascade.

    Benchmark workers execute Architect, Tier, and Breach synchronously in the
    same thread.  Binding the already-resolved table here prevents those child
    workflows from rediscovering a different spec-local file or observing an
    operator edit midway through an expensive task.  ``ContextVar`` keeps
    parallel benchmark workers isolated without mutating process environment.
    """

    _validate_routing(routing)
    token = _ROUTING_SNAPSHOT.set(routing)
    try:
        yield routing
    finally:
        _ROUTING_SNAPSHOT.reset(token)


def _routing_identity_role(config: RoleConfig) -> dict[str, Any]:
    """Project effective behavior without retaining credential material."""

    return {
        "backend": config.backend,
        "model": config.model,
        "effort": config.effort,
        "base_url": config.base_url,
        # Credential rotation must not invalidate an otherwise identical paid
        # result, but adding/removing authentication is still a policy change.
        "api_key_configured": config.api_key is not None,
        "feature_admission": config.feature_admission,
        "extra": config.extra,
        "backend_features": profile_identity(
            str(config.backend or ""),
            model=config.model,
            base_url=config.base_url,
        ),
    }


def routing_fingerprint(routing: BackendRouting) -> str:
    """Return a secret-free identity for one fully resolved routing table."""

    _validate_routing(routing)
    payload = {
        "schema_version": routing.schema_version,
        "default": _routing_identity_role(routing.default),
        "roles": {
            name: _routing_identity_role(routing.for_role(name))
            for name in sorted(routing.roles)
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def routing_source_fingerprint(routing: BackendRouting) -> str | None:
    """Return the exact loaded routing-document identity, when one exists.

    ``routing_fingerprint`` intentionally describes effective, secret-free
    behavior.  This separate identity preserves that semantic contract while
    allowing benchmark provenance and resume gates to bind the exact routing
    bytes selected by discovery.
    """

    _validate_routing(routing)
    source_sha256 = routing.source_sha256
    if source_sha256 is None:
        return None
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError(
            "backend routing source_sha256 must be 64 lowercase hex characters"
        )
    return f"sha256:{source_sha256}"


def routing_feature_fingerprint(routing: BackendRouting) -> str:
    """Return a secret-free digest of the route-resolved feature profiles."""

    _validate_routing(routing)

    def identity(config: RoleConfig) -> dict[str, str]:
        identity = dict(
            profile_identity(
                str(config.backend or ""),
                model=config.model,
                base_url=config.base_url,
            )
        )
        identity["feature_admission"] = config.feature_admission or "enforce"
        return identity

    payload = {
        "default": identity(routing.default),
        "roles": {
            name: identity(routing.for_role(name)) for name in sorted(routing.roles)
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


# ----------------------------------------------------------------------
# Built-in defaults
# ----------------------------------------------------------------------

_CODEX_DEFAULT_MODEL = "gpt-6-astra"
DEFAULT_ROUTING: BackendRouting = BackendRouting(
    default=RoleConfig(
        backend="codex",
        model=_CODEX_DEFAULT_MODEL,
        effort="max",
        feature_admission="enforce",
    ),
    roles={},
)


# ----------------------------------------------------------------------
# Discovery + parsing
# ----------------------------------------------------------------------


def _candidate_paths(
    *,
    config_path: Path | str | None,
    spec_dir: Path | str | None,
) -> list[Path]:
    out: list[Path] = []
    if config_path is not None:
        # An explicit workflow/CLI route is authoritative.  In particular,
        # an ambient CRUCIBLE_BACKEND_CONFIG must not be able to redirect a
        # frozen provider arm after the operator supplied --backend-config.
        out.append(Path(config_path))
    env = os.environ.get("CRUCIBLE_BACKEND_CONFIG", "").strip()
    if env:
        out.append(Path(env))
    if spec_dir is not None:
        out.append(Path(spec_dir) / ".crucible" / "backends.yaml")
    out.append(workspace_root() / ".crucible" / "backends.yaml")
    return out


_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_refs(value: Any) -> Any:
    """Expand ``${VAR}`` references in a string against ``os.environ``.

    Used for ``base_url`` / ``api_key`` so a routing YAML can name a
    secret's env var (``api_key: ${NVIDIA_API_KEY}``) WITHOUT committing
    the secret itself. Non-strings pass through. An unset, empty, or
    whitespace-only ``${VAR}`` raises — silently sending a literal or empty
    bearer token would 401 confusingly at call time.
    """
    if not isinstance(value, str):
        return value

    def _sub(m: re.Match[str]) -> str:
        name = m.group(1)
        val = os.environ.get(name)
        if val is None:
            raise ValueError(
                f"routing config references ${{{name}}} but that env var "
                f"is unset; export it before running (it carries a secret, "
                f"so it must not be written into the YAML)."
            )
        if not val.strip():
            raise ValueError(
                f"routing config references ${{{name}}} but that env var "
                "is empty or whitespace; export a non-empty value before "
                "running."
            )
        return val

    return _ENV_REF_RE.sub(_sub, value)


def _parse_role_config(
    raw: dict[str, Any],
    *,
    location: str,
    expand_env: bool = True,
    inherited_backend: str | None = None,
) -> RoleConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{location} must be a mapping; got {type(raw).__name__}")
    known = {
        "backend",
        "model",
        "effort",
        "base_url",
        "api_key",
        "feature_admission",
        "extra",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"{location} has unknown fields {sorted(unknown)}; adapter-specific "
            "options belong under the explicit `extra` mapping"
        )
    for key in ("backend", "model", "effort", "base_url", "api_key"):
        if key in raw and (not isinstance(raw[key], str) or not raw[key].strip()):
            raise ValueError(f"{location}.{key} must be a non-empty string")
    backend = raw.get("backend")
    if backend is not None and backend not in _BACKENDS:
        raise ValueError(
            f"{location}.backend is {backend!r}; expected one of {sorted(_BACKENDS)}"
        )
    effective_backend = backend or inherited_backend
    if effective_backend == "deepseek" and ("base_url" in raw or "api_key" in raw):
        # Reject before environment expansion. The dedicated route reads its
        # sole credential only inside DeepSeekBackend transport dispatch.
        raise ValueError(
            f"{location} dedicated DeepSeek route forbids base_url/api_key; "
            "the endpoint is fixed and DEEPSEEK_API_KEY is dispatch-only"
        )
    extra_raw = raw.get("extra", {})
    if not isinstance(extra_raw, dict) or not all(
        isinstance(key, str) and key for key in extra_raw
    ):
        raise ValueError(f"{location}.extra must be a string-keyed mapping")
    expand = _expand_env_refs if expand_env else (lambda value: value)
    feature_admission = raw.get("feature_admission")
    if feature_admission is not None and (
        not isinstance(feature_admission, str)
        or feature_admission not in FEATURE_ADMISSION_MODES
    ):
        raise ValueError(
            f"{location}.feature_admission is {feature_admission!r}; "
            f"expected one of {sorted(FEATURE_ADMISSION_MODES)}"
        )
    return RoleConfig(
        backend=backend,
        model=raw.get("model"),
        effort=raw.get("effort"),
        base_url=expand(raw.get("base_url")),
        api_key=expand(raw.get("api_key")),
        feature_admission=feature_admission,
        extra=dict(extra_raw),
    )


def _parse_routing_doc(
    doc: dict[str, Any],
    *,
    expand_env: bool = True,
) -> BackendRouting:
    if not isinstance(doc, dict):
        raise ValueError("routing config root must be a mapping")
    expected = {"schema_version", "default", "roles"}
    if set(doc) - expected:
        raise ValueError(
            f"routing config has unknown root fields {sorted(set(doc) - expected)}"
        )
    if doc.get("schema_version") != _ROUTING_SCHEMA:
        raise ValueError(f"routing config schema_version must be {_ROUTING_SCHEMA!r}")
    # Only omission means "inherit the built-in default". Treat an explicit
    # falsey non-mapping (for example ``default: []``) as malformed instead of
    # silently falling back to Codex and running a different route.
    default_raw = doc["default"] if "default" in doc else {}
    default_role = _parse_role_config(
        default_raw,
        location="default",
        expand_env=expand_env,
        inherited_backend=DEFAULT_ROUTING.default.backend,
    )
    # Same-backend partial overrides inherit built-in settings. Switching
    # backend families starts from a clean route so Codex model/effort values
    # cannot leak into an explicit Claude or OpenAI-compatible selection.
    default = _overlay_role_config(DEFAULT_ROUTING.default, default_role)
    # Likewise, an omitted roles table is empty, while an explicitly malformed
    # falsey value must not disappear into an implicit no-overrides fallback.
    roles_raw = doc["roles"] if "roles" in doc else {}
    if not isinstance(roles_raw, dict):
        raise ValueError("`roles` must be a mapping of role-name → role-config")
    roles: dict[str, RoleConfig] = {}
    for name, raw in roles_raw.items():
        if not isinstance(name, str) or not name:
            raise ValueError("routing role names must be non-empty strings")
        role_spec(name)
        roles[name] = _parse_role_config(
            raw,
            location=f"roles.{name}",
            expand_env=expand_env,
            inherited_backend=default.backend,
        )
    routing = BackendRouting(
        default=default,
        roles=roles,
        schema_version=_ROUTING_SCHEMA,
    )
    _validate_routing(routing)
    return routing


def _validate_routing(routing: BackendRouting) -> None:
    if routing.schema_version != _ROUTING_SCHEMA:
        raise ValueError(f"unsupported routing schema {routing.schema_version!r}")
    if routing.default.backend not in _BACKENDS:
        raise ValueError("routing default must resolve to a supported backend")
    for name in routing.roles:
        role_spec(name)
    effective = {"default": routing.default}
    effective.update({name: routing.for_role(name) for name in routing.roles})
    for name, config in effective.items():
        if config.model is None:
            raise ValueError(
                f"{name} config must declare a model when switching backend"
            )
        if (
            not isinstance(config.feature_admission, str)
            or config.feature_admission not in FEATURE_ADMISSION_MODES
        ):
            raise ValueError(
                f"{name} config must resolve feature_admission to one of "
                f"{sorted(FEATURE_ADMISSION_MODES)}"
            )
        if config.effort is not None and config.backend not in {
            "claude",
            "codex",
            "openai_compat",
        } and not (config.backend == "kimi" and config.model == "kimi-k3"):
            raise ValueError(
                f"{name} config supplies effort to backend "
                f"{config.backend!r}, which does not support that setting; "
                "remove it or use an adapter-specific supported field"
            )
        if config.backend not in {
            "claude",
            "kimi",
            "local",
            "openai_compat",
        } and (config.base_url is not None or config.api_key is not None):
            raise ValueError(
                f"{name} config supplies base_url/api_key to backend "
                f"{config.backend!r}; those fields are supported only by "
                "OpenAI-compatible, Claude, and dedicated Kimi routes"
            )
        allowed_extra = _ALLOWED_EXTRA_FIELDS.get(str(config.backend), frozenset())
        unknown_extra = set(config.extra) - allowed_extra
        if unknown_extra:
            raise ValueError(
                f"{name} config has unsupported {config.backend!r} extra fields "
                f"{sorted(unknown_extra)}; allowed fields are "
                f"{sorted(allowed_extra)}"
            )
        if "cli_path" in config.extra and (
            not isinstance(config.extra["cli_path"], str)
            or not config.extra["cli_path"].strip()
        ):
            raise ValueError(f"{name}.extra.cli_path must be a non-empty string")
        if config.backend == "claude" and "credential_source" in config.extra:
            credential_source = config.extra["credential_source"]
            if (
                type(credential_source) is not str
                or credential_source not in _CLAUDE_CREDENTIAL_SOURCES
            ):
                raise ValueError(
                    f"{name}.extra.credential_source must be one of "
                    f"{sorted(_CLAUDE_CREDENTIAL_SOURCES)}"
                )
            if credential_source == "claude_session":
                if config.model not in _CLAUDE_SESSION_MODELS:
                    raise ValueError(
                        f"{name}.extra.credential_source='claude_session' "
                        "requires exact model from reviewed set "
                        f"{sorted(_CLAUDE_SESSION_MODELS)!r}"
                    )
                if config.base_url is not None or config.api_key is not None:
                    raise ValueError(
                        f"{name}.extra.credential_source='claude_session' "
                        "cannot be combined with base_url or api_key"
                    )
                expected_org_id = config.extra.get("expected_org_id")
                if (
                    type(expected_org_id) is not str
                    or not expected_org_id
                    or expected_org_id != expected_org_id.strip()
                ):
                    raise ValueError(
                        f"{name}.extra.credential_source='claude_session' "
                        "requires an exact non-empty expected_org_id"
                    )
                expected_org_name = config.extra.get("expected_org_name")
                if expected_org_name is not None and (
                    type(expected_org_name) is not str
                    or not expected_org_name
                    or expected_org_name != expected_org_name.strip()
                ):
                    raise ValueError(
                        f"{name}.extra.expected_org_name must be an exact "
                        "non-empty string when set"
                    )
            elif (
                "expected_org_id" in config.extra or "expected_org_name" in config.extra
            ):
                raise ValueError(
                    f"{name}.extra.expected_org_id/expected_org_name require "
                    "credential_source='claude_session'"
                )
        elif config.backend == "claude" and (
            "expected_org_id" in config.extra or "expected_org_name" in config.extra
        ):
            raise ValueError(
                f"{name}.extra.expected_org_id/expected_org_name require "
                "credential_source='claude_session'"
            )
        if config.backend == "codex":
            credential_source = config.extra.get(
                "credential_source",
                "environment",
            )
            if (
                type(credential_source) is not str
                or credential_source not in _CODEX_CREDENTIAL_SOURCES
            ):
                raise ValueError(
                    f"{name}.extra.credential_source must be one of "
                    f"{sorted(_CODEX_CREDENTIAL_SOURCES)}"
                )
            if credential_source == "chatgpt_session":
                if config.model not in _CODEX_CHATGPT_MODELS:
                    raise ValueError(
                        f"{name}.extra.credential_source='chatgpt_session' "
                        "requires exact model from reviewed set "
                        f"{sorted(_CODEX_CHATGPT_MODELS)!r}"
                    )
                if config.extra.get("transport", "legacy_app_server") != (
                    "legacy_app_server"
                ):
                    raise ValueError(
                        f"{name}.extra.credential_source='chatgpt_session' "
                        "requires transport='legacy_app_server'"
                    )
            elif "cli_path" in config.extra:
                raise ValueError(
                    f"{name}.extra.cli_path requires "
                    "credential_source='chatgpt_session'"
                )
            timeout_seconds = config.extra.get("timeout_seconds")
            if timeout_seconds is not None and (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise ValueError(
                    f"{name}.extra.timeout_seconds must be a finite positive number"
                )
        if config.backend == "deepseek":
            if config.model != _DEEPSEEK_MODEL:
                raise ValueError(
                    f"{name} dedicated DeepSeek route requires exact model "
                    f"{_DEEPSEEK_MODEL!r}"
                )
            max_response_tokens = config.extra.get("max_response_tokens")
            if max_response_tokens is not None and (
                type(max_response_tokens) is not int
                or not 1 <= max_response_tokens <= _DEEPSEEK_MAX_RESPONSE_TOKENS
            ):
                raise ValueError(
                    f"{name}.extra.max_response_tokens must be an integer "
                    f"from 1 through {_DEEPSEEK_MAX_RESPONSE_TOKENS}"
                )
            timeout_seconds = config.extra.get("timeout_seconds")
            if timeout_seconds is not None and (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise ValueError(
                    f"{name}.extra.timeout_seconds must be a finite positive number"
                )
            expected_runtime = config.extra.get("expected_worker_runtime_fingerprint")
            if expected_runtime is not None and (
                type(expected_runtime) is not str
                or _DEEPSEEK_RUNTIME_DIGEST_RE.fullmatch(expected_runtime) is None
            ):
                raise ValueError(
                    f"{name}.extra.expected_worker_runtime_fingerprint must "
                    "be sha256:<64 lowercase hex>"
                )
            if _DEEPSEEK_DIAGNOSTIC_WRITE_OPTION in config.extra:
                diagnostic_write = config.extra[
                    _DEEPSEEK_DIAGNOSTIC_WRITE_OPTION
                ]
                if type(diagnostic_write) is not bool:
                    raise ValueError(
                        f"{name}.extra.{_DEEPSEEK_DIAGNOSTIC_WRITE_OPTION} "
                        "must be a bool"
                    )
                if (
                    diagnostic_write is True
                    and config.feature_admission != "diagnostic"
                ):
                    raise ValueError(
                        f"{name}.extra.{_DEEPSEEK_DIAGNOSTIC_WRITE_OPTION}=true "
                        "requires feature_admission='diagnostic'"
                    )
        if (
            config.backend == "kimi"
            and _KIMI_DIAGNOSTIC_WRITE_OPTION in config.extra
        ):
            diagnostic_write = config.extra[_KIMI_DIAGNOSTIC_WRITE_OPTION]
            if type(diagnostic_write) is not bool:
                raise ValueError(
                    f"{name}.extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION} must be a bool"
                )
            if diagnostic_write is True:
                if config.feature_admission != "diagnostic":
                    raise ValueError(
                        f"{name}.extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION}=true "
                        "requires feature_admission='diagnostic'"
                    )
                if config.model not in _KIMI_DIAGNOSTIC_MODELS:
                    raise ValueError(
                        f"{name}.extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION}=true "
                        f"requires an exact admitted model: {sorted(_KIMI_DIAGNOSTIC_MODELS)}"
                    )
                if config.base_url not in {
                    _KIMI_DIAGNOSTIC_ENDPOINT,
                    _KIMI_DIAGNOSTIC_ENDPOINT_REF,
                }:
                    raise ValueError(
                        f"{name}.extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION}=true "
                        "requires the exact Moonshot public endpoint"
                    )
                if type(config.api_key) is not str or not config.api_key.strip():
                    raise ValueError(
                        f"{name}.extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION}=true "
                        "requires a configured api_key"
                    )
        if config.backend == "kimi":
            if "diagnostic_cost_upper_bound" in config.extra:
                cost_bound = config.extra["diagnostic_cost_upper_bound"]
                if type(cost_bound) is not bool:
                    raise ValueError(f"{name}.extra.diagnostic_cost_upper_bound must be a bool")
                if cost_bound and (
                    config.model != "kimi-k3" or config.effort != "max"
                    or config.feature_admission != "diagnostic"
                    or config.base_url not in {_KIMI_DIAGNOSTIC_ENDPOINT, _KIMI_DIAGNOSTIC_ENDPOINT_REF}
                    or config.extra.get("worker_source") != "project_locked"
                ):
                    raise ValueError(
                        f"{name}.extra.diagnostic_cost_upper_bound requires exact K3/max, "
                        "diagnostic admission, public Moonshot endpoint and project_locked worker"
                    )
            if config.model not in _KIMI_MAX_CONTEXT_SIZES:
                raise ValueError(
                    f"{name} requires an exact admitted Kimi model: "
                    f"{sorted(_KIMI_MAX_CONTEXT_SIZES)}"
                )
            if config.effort is not None and (
                config.model != "kimi-k3" or config.effort != "max"
            ):
                raise ValueError(
                    f"{name}.effort is supported only as 'max' for kimi-k3"
                )
            max_context_size = _KIMI_MAX_CONTEXT_SIZES[config.model]
            max_response_tokens = config.extra.get("max_response_tokens")
            if max_response_tokens is not None and (
                type(max_response_tokens) is not int
                or not 1 <= max_response_tokens <= max_context_size
            ):
                raise ValueError(
                    f"{name}.extra.max_response_tokens must be an integer "
                    f"from 1 through {max_context_size}"
                )
            timeout_seconds = config.extra.get("timeout_seconds")
            if timeout_seconds is not None and (
                isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds)
                or timeout_seconds <= 0
            ):
                raise ValueError(
                    f"{name}.extra.timeout_seconds must be a finite positive number"
                )
            worker_source = config.extra.get("worker_source", "environment")
            if (
                type(worker_source) is not str
                or worker_source not in _KIMI_WORKER_SOURCES
            ):
                raise ValueError(
                    f"{name}.extra.worker_source must be one of "
                    f"{sorted(_KIMI_WORKER_SOURCES)}"
                )
            expected_runtime = config.extra.get("expected_worker_runtime_fingerprint")
            if expected_runtime is not None:
                if (
                    type(expected_runtime) is not str
                    or _KIMI_RUNTIME_DIGEST_RE.fullmatch(expected_runtime) is None
                ):
                    raise ValueError(
                        f"{name}.extra.expected_worker_runtime_fingerprint "
                        "must be sha256:<64 lowercase hex>"
                    )
                if worker_source != "project_locked":
                    raise ValueError(
                        f"{name}.extra.expected_worker_runtime_fingerprint "
                        "requires worker_source='project_locked'"
                    )
        if "output_schema" in config.extra and not isinstance(
            config.extra["output_schema"], dict
        ):
            raise ValueError(f"{name}.extra.output_schema must be a mapping")
        if "transport" in config.extra and (
            not isinstance(config.extra["transport"], str)
            or config.extra["transport"] not in {"legacy_app_server", "official_sdk"}
        ):
            raise ValueError(
                f"{name}.extra.transport must be 'legacy_app_server' or 'official_sdk'"
            )
        if (
            "trusted_project_instructions" in config.extra
            and type(config.extra["trusted_project_instructions"]) is not bool
        ):
            raise ValueError(
                f"{name}.extra.trusted_project_instructions must be a bool"
            )
        if "reasoning_effort" in config.extra and (
            not isinstance(config.extra["reasoning_effort"], str)
            or not config.extra["reasoning_effort"].strip()
        ):
            raise ValueError(
                f"{name}.extra.reasoning_effort must be a non-empty string"
            )
        if "max_tokens" in config.extra and (
            isinstance(config.extra["max_tokens"], bool)
            or not isinstance(config.extra["max_tokens"], int)
            or config.extra["max_tokens"] <= 0
        ):
            raise ValueError(f"{name}.extra.max_tokens must be a positive integer")
        if "expected_served_model" in config.extra and (
            type(config.extra["expected_served_model"]) is not str
            or not config.extra["expected_served_model"]
            or config.extra["expected_served_model"]
            != config.extra["expected_served_model"].strip()
        ):
            raise ValueError(
                f"{name}.extra.expected_served_model must be an exact "
                "non-empty string"
            )


_MAX_ROUTING_CONFIG_BYTES = 4 * 1024 * 1024


def _routing_payload(path: Path | str) -> tuple[bytes, Path]:
    source = Path(os.path.abspath(Path(path)))
    try:
        payload, _metadata = read_exact_regular(
            source,
            maximum=_MAX_ROUTING_CONFIG_BYTES,
        )
    except AuthorityIOError as exc:
        raise ValueError(
            f"backend routing config {source} is not one exact regular file: {exc}"
        ) from exc
    return payload, source


def lint_routing_config(path: Path | str | bytes) -> BackendRouting:
    """Validate a routing document without resolving secret environment refs.

    Frozen launch consumers pass the bytes already measured through their held
    bundle descriptor.  General callers may still provide a path, which is
    admitted through the same bounded non-blocking regular-file primitive.
    """

    import yaml  # type: ignore[import-untyped]

    if isinstance(path, bytes):
        payload = path
        source = Path("<measured-routing-bytes>")
    else:
        payload, source = _routing_payload(path)
    try:
        doc = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"backend routing config {source} is invalid: {exc}") from exc
    routing = _parse_routing_doc(doc or {}, expand_env=False)
    measured = BackendRouting(
        default=routing.default,
        roles=routing.roles,
        schema_version=routing.schema_version,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _validate_routing(measured)
    return measured


def load_routing(
    *,
    config_path: Path | str | None = None,
    spec_dir: Path | str | None = None,
) -> BackendRouting:
    """Discover + parse a routing config, with fall-through to defaults.

    Logs (at INFO) which path was used. Returns the built-in
    ``DEFAULT_ROUTING`` when no config file is found (no error). A
    malformed file raises ``ValueError`` — operators want to know,
    not have it silently ignored.
    """
    import yaml  # type: ignore[import-untyped]

    snapshot = _ROUTING_SNAPSHOT.get()
    if snapshot is not None:
        return snapshot

    if config_path is not None and not Path(config_path).is_file():
        # Missing explicit input is configuration drift, not discovery.
        # Falling through to an ambient/spec/workspace route (or the Codex
        # default) would silently change a named provider arm.
        raise ValueError(
            f"explicit backend routing config is not a file: {Path(config_path)}"
        )

    for path in _candidate_paths(config_path=config_path, spec_dir=spec_dir):
        if not path.is_file():
            continue
        try:
            payload, _source = _routing_payload(path)
            doc = yaml.safe_load(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(
                f"backend routing config {path} is not valid YAML: {exc}"
            ) from exc
        routing = _parse_routing_doc(doc or {})
        merged = BackendRouting(
            default=routing.default,
            roles=routing.roles,
            schema_version=_ROUTING_SCHEMA,
            source_sha256=hashlib.sha256(payload).hexdigest(),
        )
        _validate_routing(merged)
        logger.info("crucible: loaded backend routing from %s", path)
        return merged
    logger.debug("crucible: no backend routing config found; using defaults")
    return DEFAULT_ROUTING


# ----------------------------------------------------------------------
# Per-role backend construction
# ----------------------------------------------------------------------


def resolve_for_role(
    role: str,
    *,
    routing: BackendRouting | None = None,
    log_dir: Path | str | None = None,
    cwd: str | None = None,
    config_path: Path | str | None = None,
    spec_dir: Path | str | None = None,
    capability: str = "text",
    native_shell: bool | None = None,
    allow_trusted_project_instructions: bool = True,
    feature_requirements: Iterable[FeatureRequirement] = (),
    limit_policy: InvocationLimitPolicy | None = None,
    limit_state: InvocationLimitState | None = None,
    limit_request: InvocationLimitRequest | None = None,
    limit_ledger: InvocationLimitLedger | None = None,
) -> Any:
    """Return an ``assay.Backend`` configured for ``role``.

    If ``routing`` is None, loads via :func:`load_routing`. Pass an
    explicit ``routing`` when constructing many backends in one
    workflow run so the config is parsed once, not per-role.

    Adapter-specific settings (model effort, endpoint, and authentication)
    flow through routing config. ``native_shell`` is a workflow security
    boundary for native-tool adapters: containerized authoring sets it false
    while retaining path-confined MCP writes.

    ``limit_request`` is an exact, per-invocation UIR-03 envelope. Its strict
    admission resolves before adapter construction; callers using it must
    construct a fresh backend for that exact invocation rather than cache the
    backend across differing limits.

    ``limit_policy`` is the reusable equivalent for workflow pools. It admits
    the route before construction and resolves the actual turn/USD values at
    each vendor-neutral ``invoke`` boundary. The two forms are mutually
    exclusive.
    """
    if limit_request is not None and limit_policy is not None:
        raise ValueError("limit_request and limit_policy are mutually exclusive")
    if limit_state is not None and limit_policy is None:
        raise ValueError("limit_state requires limit_policy")
    if limit_ledger is not None and (limit_request is None and limit_policy is None):
        raise ValueError("limit_ledger requires limit_request or limit_policy")
    if routing is None:
        routing = load_routing(config_path=config_path, spec_dir=spec_dir)
    _validate_routing(routing)
    capability = validate_role_capability(role, capability)
    if type(allow_trusted_project_instructions) is not bool:
        raise TypeError("allow_trusted_project_instructions must be an exact bool")
    cfg = routing.for_role(role)
    if cfg.backend is None:
        raise ValueError(
            f"role {role!r} has no resolved backend; default fell through "
            f"as None. Check your backends.yaml or DEFAULT_ROUTING."
        )
    if limit_ledger is not None and (
        not isinstance(cfg.model, str) or not cfg.model.strip()
    ):
        raise ValueError(
            "durable invocation-limit evidence requires an explicit model "
            f"for role {role!r}"
        )
    if not allow_trusted_project_instructions and cfg.extra.get(
        "trusted_project_instructions", False
    ):
        raise FeatureAdmissionError(
            "workflow policy requires project instructions to be suppressed "
            f"for role {role!r}; trusted_project_instructions cannot be "
            "enabled on this execution boundary"
        )
    if cfg.backend == "kimi" and capability == "workspace_write":
        # The dedicated SDK route has a qualified local mutation/lifecycle
        # core, but its released dependency tuple predates K2.7, Moonshot's
        # catalogue exposes no immutable deployment revision, and exact-route
        # write/cancellation/cost qualification remains incomplete. Static
        # feature support is therefore insufficient authorization for
        # production workflow-owned writes. A deliberately diagnostic route
        # may opt into the already process-isolated mutation core, but only
        # for the exact public Moonshot frontier lane and the most confined
        # workflow boundary. Its routing fingerprint and result-evidence mode
        # remain diagnostic, so this cannot promote KIM-03 or satisfy E12.
        diagnostic_write = cfg.extra.get(
            _KIMI_DIAGNOSTIC_WRITE_OPTION,
            False,
        )
        if diagnostic_write is not True:
            raise FeatureAdmissionError(
                "dedicated Kimi workspace_write routing is promotion-blocked "
                "pending KIM-03 dependency, immutable-revision, and "
                "exact-route qualification; an explicitly diagnostic "
                "frontier smoke may use the closed "
                f"extra.{_KIMI_DIAGNOSTIC_WRITE_OPTION} boundary"
            )
        diagnostic_reasons: list[str] = []
        if cfg.feature_admission != "diagnostic":
            diagnostic_reasons.append("feature_admission must be diagnostic")
        if cfg.model not in _KIMI_DIAGNOSTIC_MODELS:
            diagnostic_reasons.append(
                f"model must be an exact admitted model: {sorted(_KIMI_DIAGNOSTIC_MODELS)}"
            )
        if cfg.base_url != _KIMI_DIAGNOSTIC_ENDPOINT:
            diagnostic_reasons.append(
                f"base_url must resolve exactly to {_KIMI_DIAGNOSTIC_ENDPOINT!r}"
            )
        if type(cfg.api_key) is not str or not cfg.api_key.strip():
            diagnostic_reasons.append("api_key must be configured at runtime")
        if native_shell is not False:
            diagnostic_reasons.append("native_shell must be disabled")
        if allow_trusted_project_instructions is not False:
            diagnostic_reasons.append("trusted project instructions must be disabled")
        if type(cwd) is not str or not cwd.strip():
            diagnostic_reasons.append("an explicit non-empty confined cwd is required")
        elif not Path(cwd).is_absolute():
            diagnostic_reasons.append("the explicit confined cwd must be absolute")
        if diagnostic_reasons:
            raise FeatureAdmissionError(
                "dedicated Kimi diagnostic workspace_write route rejected: "
                + "; ".join(diagnostic_reasons)
            )
    if cfg.backend == "deepseek":
        # DSK-00b/DSK-02 remain promotion gates, but an exact, explicitly
        # diagnostic route may exercise the project-locked worker. Writers
        # additionally require the process-isolated mutation boundary and the
        # same confined workflow controls as the Kimi diagnostic lane.
        deepseek_diagnostic_reasons: list[str] = []
        if cfg.feature_admission != "diagnostic":
            deepseek_diagnostic_reasons.append(
                "feature_admission must be diagnostic"
            )
        expected_runtime = cfg.extra.get(
            "expected_worker_runtime_fingerprint"
        )
        if (
            type(expected_runtime) is not str
            or _DEEPSEEK_RUNTIME_DIGEST_RE.fullmatch(expected_runtime) is None
        ):
            deepseek_diagnostic_reasons.append(
                "an exact expected_worker_runtime_fingerprint is required"
            )
        if capability == "workspace_write":
            if (
                cfg.extra.get(_DEEPSEEK_DIAGNOSTIC_WRITE_OPTION)
                is not True
            ):
                deepseek_diagnostic_reasons.append(
                    f"extra.{_DEEPSEEK_DIAGNOSTIC_WRITE_OPTION} must be true"
                )
            if native_shell is not False:
                deepseek_diagnostic_reasons.append(
                    "native_shell must be disabled"
                )
            if allow_trusted_project_instructions is not False:
                deepseek_diagnostic_reasons.append(
                    "trusted project instructions must be disabled"
                )
            if type(cwd) is not str or not cwd.strip():
                deepseek_diagnostic_reasons.append(
                    "an explicit non-empty confined cwd is required"
                )
            elif not Path(cwd).is_absolute():
                deepseek_diagnostic_reasons.append(
                    "the explicit confined cwd must be absolute"
                )
        if deepseek_diagnostic_reasons:
            raise FeatureAdmissionError(
                "dedicated DeepSeek diagnostic route rejected: "
                + "; ".join(deepseek_diagnostic_reasons)
            )
    trusted_project_instructions = bool(
        cfg.extra.get("trusted_project_instructions", False)
    )
    if trusted_project_instructions and (
        cfg.backend != "codex"
        or capability != "read_only"
        or native_shell is not False
        or cwd is None
        or cfg.feature_admission != "enforce"
    ):
        raise ValueError(
            "trusted_project_instructions requires an enforcing Codex route "
            "with capability='read_only', native_shell=False, and an explicit "
            "cwd; confined, text-only, diagnostic, and writer routes must "
            "leave trusted repository context disabled"
        )
    requirements = execution_feature_requirements(
        capability=capability,
        native_shell=native_shell,
        additional=(
            *role_feature_requirements(role),
            *tuple(feature_requirements),
        ),
    )
    profile = feature_profile_for_route(
        cfg.backend,
        model=cfg.model,
        base_url=cfg.base_url,
    )
    if cfg.backend == "codex" and cfg.effort == "ultra":
        # GPT-6 Astra's ``ultra`` effort adds automatic task delegation.
        # That is a different authority boundary, not merely more reasoning.
        # Reject before adapter construction (and therefore before paid
        # dispatch) while this route truthfully advertises no qualified native
        # delegation. A future native-delegation profile alone is not enough:
        # the workflow must gain a separate typed NAT-01 admission before
        # this rejection can be opened. Never silently clamp the requested or
        # provenance effort.
        raise FeatureAdmissionError(
            "Codex effort 'ultra' enables automatic task delegation, but "
            "the selected route has native_delegation='none'; use 'max' "
            "for maximum non-delegating reasoning or qualify native "
            "delegation explicitly"
        )
    route_identity = (
        f"{cfg.backend}/{cfg.model}"
        + (f" at {cfg.base_url}" if cfg.base_url is not None else "")
        + f" for role {role!r}"
    )
    limit_resolution = (
        resolve_invocation_limits(
            profile,
            limit_request,
            # Persist a secret-free route label. The feature digest binds the
            # qualified endpoint profile without copying URL credentials or
            # query material into terminal traces.
            route=f"{cfg.backend}/{cfg.model} for role {role!r}",
        )
        if limit_request is not None
        else None
    )
    secret_free_route = f"{cfg.backend}/{cfg.model} for role {role!r}"
    if limit_policy is not None:
        preflight_invocation_limit_policy(
            profile,
            limit_policy,
            route=secret_free_route,
        )
    decision = admit_feature_requirements(
        profile,
        requirements,
        mode=cfg.feature_admission or "enforce",
        route=route_identity,
    )
    if not decision.allowed:
        logger.warning(
            "routing: diagnostic feature admission retained rejected route %s: %s",
            route_identity,
            "; ".join(decision.reasons),
        )
    kw: dict[str, Any] = {}
    if cfg.model is not None:
        kw["model"] = cfg.model
    if cfg.effort is not None:
        kw["effort"] = cfg.effort
    if cfg.base_url is not None:
        kw["base_url"] = cfg.base_url
    if cfg.api_key is not None:
        kw["api_key"] = cfg.api_key
    if cfg.backend in {"claude", "codex"} and native_shell is not None:
        kw["native_shell"] = native_shell
    kw.update(cfg.extra)
    # ``effort`` routing:
    # - claude: passed through as-is (ClaudeBackend ``effort`` kwarg).
    # - openai_compat: translated to ``reasoning_effort`` — hosted
    #   reasoning models (deepseek/kimi on the NVIDIA hub) honor the
    #   top-level ``reasoning_effort`` param; both published SWE-bench
    #   numbers for these models assume "high". An explicit
    #   ``reasoning_effort:`` in ``extra`` wins over a translated
    #   ``effort:``.
    # - codex: passed through to TurnOverrides.effort after rejecting
    #   ``ultra`` while the route has no qualified native delegation.
    if cfg.backend == "openai_compat" and kw.get("effort"):
        kw.setdefault("reasoning_effort", kw["effort"])
        kw.pop("effort", None)
    if cfg.backend in {"claude", "kimi"}:
        # Claude reads INFER_* while the dedicated Kimi adapter reads
        # KIMI_BASE_URL/MOONSHOT_API_KEY at invocation time. Bind the
        # configured endpoint so either adapter can detect environment drift,
        # but never forward credential material through constructor kwargs
        # where it could be retained or repr'd. Token rotation therefore does
        # not change route identity.
        base_url = kw.pop("base_url", None)
        configured_api_key = kw.pop("api_key", None)
        # This is a control-plane authorization bit, not an adapter kwarg.
        # Its presence remains bound into the route fingerprint above.
        kw.pop(_KIMI_DIAGNOSTIC_WRITE_OPTION, None)
        if base_url is not None:
            kw["expected_base_url"] = base_url
        if cfg.backend == "claude" and (
            base_url is not None or configured_api_key is not None
        ):
            # A configured Claude gateway route is resolved from the neutral
            # INFER_* runtime variables. Do not let unrelated ambient
            # ANTHROPIC_* credentials or endpoints silently change it.
            kw["prefer_infer_credentials"] = True
    elif cfg.backend not in ("local", "openai_compat"):
        for key in ("base_url", "api_key"):
            if kw.get(key):
                logger.warning(
                    "routing: role %r → backend %r has ``%s`` in config; "
                    "dropped (%s is openai_compat-only)",
                    role,
                    cfg.backend,
                    key,
                    key,
                )
                kw.pop(key, None)
    backend = _resolve_backend(
        cfg.backend,
        log_dir=log_dir,
        cwd=cwd,
        capability=capability,
        **kw,
    )
    limit_identity = (
        InvocationLimitIdentity(
            role=role,
            backend=cfg.backend,
            model=cast(str, cfg.model),
        )
        if limit_ledger is not None
        else None
    )
    if limit_resolution is not None:
        bind_invocation_limit_contract(
            backend,
            limit_resolution,
            evidence_ledger=limit_ledger,
            identity=limit_identity,
        )
    elif limit_policy is not None:
        bind_invocation_limit_policy(
            backend,
            profile,
            limit_policy,
            route=secret_free_route,
            shared_state=limit_state,
            evidence_ledger=limit_ledger,
            identity=limit_identity,
        )
    # The same explicit diagnostic escape that retains a rejected feature
    # profile also marks its successful results non-promotable. Enforcing
    # routes fail closed when provider terminal/token/cost/trace evidence is
    # incomplete. Bind this before returning the adapter so no paid call can
    # race the policy decision.
    # Tests and embedders may replace the construction factory with an opaque
    # sentinel while validating kwarg routing. Real Backend implementations
    # always expose the protocol's callable invoke boundary.
    if callable(getattr(backend, "invoke", None)):
        bind_result_evidence_admission(
            backend,
            mode=cfg.feature_admission or "enforce",
        )
    return backend


__all__ = [
    "BackendRouting",
    "RoleConfig",
    "DEFAULT_ROUTING",
    "bind_routing_snapshot",
    "lint_routing_config",
    "load_routing",
    "routing_fingerprint",
    "routing_feature_fingerprint",
    "routing_source_fingerprint",
    "resolve_for_role",
]
