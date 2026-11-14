# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Backend adapters for Assay invocations.

Each module here translates an ``assay.Invocation`` into a vendor SDK
call and returns an ``assay.InvocationResult``. Crucible owns these adapters,
their transport integration, and workflow-level routing.

Public surface:
- ``claude.ClaudeBackend`` — wraps the claude-agent-sdk
- ``codex.CodexBackend`` — wraps the codex-app-server-sdk
- ``codex_official.OfficialCodexBackend`` — optional exact-locked official
  OpenAI SDK transport behind the Codex route rollback switch
- ``deepseek.DeepSeekBackend`` — exact project-locked native DeepSeek
  Responses construction (routing remains promotion-blocked)
- ``kimi.KimiBackend`` — wraps the qualified Kimi Agent SDK runtime
- ``openai_compat.OpenAIBackend`` — wraps any OpenAI-compatible HTTP
  endpoint: local servers (vLLM, llama.cpp, Ollama, TGI, lm-studio)
  AND hosted gateways like the NVIDIA Inference Hub. Carries a real
  multi-turn tool-use loop over an attached ToolSet.
- ``resolve(name, ...)`` — low-level name-to-constructor factory used
  by the per-role routing layer in :mod:`crucible.backends.config`.

The built-in routing contract is Codex + GPT-6 Astra + Max for every
role. ``resolve`` itself is intentionally thinner: it maps one explicit
backend name to one adapter. Workflows use ``backend_pool`` /
``resolve_for_role`` so config discovery, role capability checks, and
model attribution stay centralized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .claude import ClaudeBackend, claude_model_identities_match
from .codex import CodexBackend
from .codex_official import OfficialCodexBackend
from .deepseek import (
    DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
    DeepSeekBackend,
    DeepSeekInvocationEvidence,
    DeepSeekProjectLockedRuntimeIdentity,
    DeepSeekResponseEvidence,
    capture_project_locked_deepseek_runtime_identity,
)
from .kimi import (
    KIMI_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
    KimiBackend,
    KimiProjectLockedRuntimeIdentity,
    capture_project_locked_kimi_runtime_identity,
)
from .limits import (
    INVOCATION_LIMIT_LEDGER_LEAF,
    INVOCATION_LIMIT_LEDGER_SCHEMA,
    InvocationLimitAdmissionError,
    InvocationLimitIdentity,
    InvocationLimitLedger,
    InvocationLimitPolicy,
    InvocationLimitRequest,
    InvocationLimitState,
    invocation_limit_record_from_result,
)
from .openai_compat import OpenAIBackend

__all__ = [
    "ClaudeBackend",
    "claude_model_identities_match",
    "CodexBackend",
    "OfficialCodexBackend",
    "DeepSeekBackend",
    "DeepSeekInvocationEvidence",
    "DeepSeekProjectLockedRuntimeIdentity",
    "DeepSeekResponseEvidence",
    "DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA",
    "capture_project_locked_deepseek_runtime_identity",
    "KimiBackend",
    "KimiProjectLockedRuntimeIdentity",
    "KIMI_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA",
    "capture_project_locked_kimi_runtime_identity",
    "INVOCATION_LIMIT_LEDGER_LEAF",
    "INVOCATION_LIMIT_LEDGER_SCHEMA",
    "InvocationLimitAdmissionError",
    "InvocationLimitIdentity",
    "InvocationLimitLedger",
    "InvocationLimitPolicy",
    "InvocationLimitRequest",
    "InvocationLimitState",
    "invocation_limit_record_from_result",
    "OpenAIBackend",
    "resolve",
    "BACKEND_NAMES",
]


BACKEND_NAMES: tuple[str, ...] = (
    "claude",
    "codex",
    "deepseek",
    "kimi",
    "local",
    "openai_compat",
)


def resolve(
    name: str,
    *,
    model: str | None = None,
    log_dir: Path | str | None = None,
    cwd: str | None = None,
    **extra: Any,
) -> Any:
    """Return the configured ``assay.Backend`` for ``name``.

    Args:
      name: ``"claude"``, ``"codex"``, ``"deepseek"``, ``"kimi"``,
        ``"local"``, or ``"openai_compat"`` (case-insensitive). Any other
        value raises
        ``ValueError``.
      model: Optional model override. When None, each adapter's
        adapter-local fallback applies. This is distinct from the
        global routing default, which explicitly selects Codex,
        ``gpt-6-astra``, and Max effort.
      log_dir: Per-invocation SDK debug dump directory (typically
        ``handle.run_dir`` from ``corpus.open_run``).
      cwd: Working directory supplied to the adapter. Adapters use it for
        transport context, tool execution, or request metadata as supported.
      **extra: Passed through to the adapter constructor. Lets callers
        thread adapter-specific kwargs (``effort`` and explicit
        ``transport`` for Codex, ``output_schema`` for Codex,
        ``base_url`` / ``api_key`` for openai_compat) without bypassing the
        factory.

    The factory deliberately doesn't validate that the chosen route supports
    the workflow's verb. Every registered adapter implements the workflow
    ToolSet attachment boundary; role capability and feature admission still
    decide whether a particular route may use it for promoted or diagnostic
    work.
    """
    canonical = name.strip().lower()
    if canonical == "claude":
        kw: dict[str, Any] = {"log_dir": log_dir, "cwd": cwd}
        if model is not None:
            kw["model"] = model
        kw.update(extra)
        return ClaudeBackend(**kw)
    if canonical == "codex":
        kw = {"log_dir": log_dir, "cwd": cwd}
        if model is not None:
            kw["model"] = model
        transport = extra.pop("transport", "legacy_app_server")
        if transport not in {"legacy_app_server", "official_sdk"}:
            raise ValueError(
                "codex transport must be 'legacy_app_server' or "
                "'official_sdk'"
            )
        kw.update(extra)
        if transport == "official_sdk":
            return OfficialCodexBackend(**kw)
        return CodexBackend(**kw)
    if canonical == "deepseek":
        kw = {"log_dir": log_dir, "cwd": cwd}
        if model is not None:
            kw["model"] = model
        kw.update(extra)
        return DeepSeekBackend(**kw)
    if canonical == "kimi":
        kw = {"log_dir": log_dir, "cwd": cwd}
        if model is not None:
            kw["model"] = model
        kw.update(extra)
        return KimiBackend(**kw)
    if canonical in ("openai_compat", "local"):
        kw = {"log_dir": log_dir, "cwd": cwd}
        if model is not None:
            kw["model"] = model
        kw.update(extra)
        return OpenAIBackend(**kw)
    raise ValueError(
        f"unknown backend {name!r}; choices: {BACKEND_NAMES}"
    )
