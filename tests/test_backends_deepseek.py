# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from assay import (
    CONSERVATIVE_BACKEND_FEATURES,
    BackendPolicy,
    BudgetExhausted,
    Invocation,
    MaxTurnsExceeded,
    TransportError,
)

from crucible.backends import DeepSeekBackend
from crucible.backends import deepseek as deepseek_mod
from crucible.backends.config import (
    BackendRouting,
    RoleConfig,
    _parse_role_config,
    resolve_for_role,
    routing_fingerprint,
)
from crucible.backends.features import (
    FeatureAdmissionError,
    feature_profile_for_route,
)
from crucible.backends.pricing import (
    DEFAULT_PRICING,
    PricingConfigurationError,
    compute_cost,
    pricing_for_model,
)


_DIGEST = "sha256:" + "a" * 64


def _pin() -> deepseek_mod._PinnedDeepSeekWorkerRuntime:
    components = tuple(
        deepseek_mod.DeepSeekSBOMComponent(
            name=name,
            version=version,
            record_sha256=_DIGEST,
        )
        for name, version in deepseek_mod._LOCKED_LINUX_COMPONENT_VERSIONS
    )
    return deepseek_mod._PinnedDeepSeekWorkerRuntime(
        logical_python=Path("/project/deepseek/.venv/bin/python"),
        resolved_python=Path("/usr/bin/python3.12"),
        resolved_python_sha256=_DIGEST,
        lock_source_fingerprint=_DIGEST,
        installed_tree_fingerprint=_DIGEST,
        installed_sbom_fingerprint=_DIGEST,
        runtime_fingerprint=_DIGEST,
        sbom=components,
    )


def _usage(
    *,
    input_tokens: int = 1_000,
    cached_input_tokens: int | None = 250,
    cache_write_tokens: int | None = 0,
    output_tokens: int = 100,
    reasoning_output_tokens: int | None = 40,
) -> deepseek_mod._DeepSeekUsage:
    return deepseek_mod._DeepSeekUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        cache_write_tokens=cache_write_tokens,
        output_tokens=output_tokens,
        reasoning_output_tokens=reasoning_output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def _terminal(
    *,
    response_id: str = "resp_1",
    model: str = "deepseek-flash",
    text: str = "done",
    calls: tuple[deepseek_mod._DeepSeekToolCall, ...] = (),
    usage: deepseek_mod._DeepSeekUsage | None = None,
    terminal_type: str = "response.completed",
) -> deepseek_mod._DeepSeekWorkerTerminal:
    continuation: list[dict[str, Any]] = []
    if calls:
        if text:
            continuation.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            )
        continuation.extend(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        )
    return deepseek_mod._DeepSeekWorkerTerminal(
        terminal_type=terminal_type,
        response_id=response_id,
        model=model,
        system_fingerprint="fp_test",
        output_text=text,
        tool_calls=calls,
        continuation_json=json.dumps(continuation, sort_keys=True),
        usage=_usage() if usage is None else usage,
        event_types=("response.created", "response.output_text.delta"),
    )


def _invocation(**overrides: Any) -> Invocation:
    values: dict[str, Any] = {
        "system": "system",
        "user": "user",
        "tools": None,
        "max_turns": 1,
        "budget_usd": None,
        "policy": BackendPolicy(
            retry_budget=1,
            timeout_seconds=30,
            on_cancellation="convert_to_backend_error",
            on_max_turns="continue_with_warning",
        ),
    }
    values.update(overrides)
    return Invocation(**values)


def _backend(runner, **overrides: Any) -> DeepSeekBackend:
    return DeepSeekBackend(
        _worker_pin=_pin(),
        _worker_runner=runner,
        **overrides,
    )


def _wire_line(sequence: int, **fields: Any) -> bytes:
    if fields.get("type") in {"event", "terminal"}:
        fields.setdefault("provider_sequence", sequence)
    return (
        json.dumps(
            {
                "protocol": "crucible.deepseek-worker.v1",
                "sequence": sequence,
                **fields,
            },
            sort_keys=True,
        )
        + "\n"
    ).encode()


_WORKER_PROJECT = Path(deepseek_mod.__file__).with_name("deepseek_worker")


def _project_entries(project: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every entry of a worker project.

    The installed environment is left out: nothing here imports from it.
    """

    entries: dict[str, tuple[int, int]] = {}
    for root, directories, files in os.walk(project):
        if Path(root) == project:
            directories[:] = [name for name in directories if name != ".venv"]
        for name in (*directories, *files):
            path = Path(root) / name
            metadata = path.lstat()
            entries[path.relative_to(project).as_posix()] = (
                metadata.st_size,
                metadata.st_mtime_ns,
            )
    return entries


def _load_worker_source(name: str, source: Path, *, project: Path = _WORKER_PROJECT) -> Any:
    """Load a source file of the worker project by path and leave no bytecode there.

    The adapter's pin hashes the worker's source directory with every
    bytecode entry in it and is revalidated before every launch. A
    ``__pycache__`` written by this process would change the pin of the
    checkout under every other test that runs in it, while those tests run.
    """

    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    before = _project_entries(project)
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    assert _project_entries(project) == before, "loading worker source changed its project"
    return module


@pytest.fixture(scope="module")
def worker_module():
    return _load_worker_source(
        "crucible_deepseek_worker_test_module",
        _WORKER_PROJECT / "src" / "crucible_deepseek_worker" / "__main__.py",
    )


@pytest.fixture(scope="module")
def transport_check_module():
    return _load_worker_source(
        "crucible_deepseek_transport_check_test_module",
        _WORKER_PROJECT / "provider_free_transport_check.py",
    )


def test_worker_source_is_loaded_without_leaving_bytecode_in_the_worker_project(
    worker_module,
    transport_check_module,
    monkeypatch,
    tmp_path,
):
    # Both fixtures above went through the loader, which compares the worker
    # project before and after the load. What it guards against, on a
    # stand-in project: a plain load by path leaves bytecode beside the
    # source, which is an entry the adapter's pin would hash.
    assert {worker_module.__name__, transport_check_module.__name__} == {
        "crucible_deepseek_worker_test_module",
        "crucible_deepseek_transport_check_test_module",
    }
    project = tmp_path / "standin_worker"
    source = project / "src" / "standin.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 7\n", encoding="utf-8")
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)

    spec = importlib.util.spec_from_file_location("standin_plain_load", source)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    cache = source.parent / "__pycache__"
    assert [path.suffix for path in cache.iterdir()] == [".pyc"]
    assert "src/__pycache__" in _project_entries(project)

    # The loader of this module leaves the project as it found it, and the
    # interpreter's switch as it was.
    shutil.rmtree(cache)
    before = _project_entries(project)
    loaded = _load_worker_source("standin_guarded_load", source, project=project)
    assert loaded.VALUE == 7
    assert _project_entries(project) == before and not cache.exists()
    assert sys.dont_write_bytecode is False
    # A load that does change the project is refused, whatever it wrote.
    source.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('written-during-the-load').write_text('x')\n",
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="changed its project"):
        _load_worker_source("standin_writing_load", source, project=project)
    assert sys.dont_write_bytecode is False


def _terminal_wire(
    sequence: int,
    *,
    cached_input_tokens: int | None = 25,
) -> bytes:
    return _wire_line(
        sequence,
        type="terminal",
        terminal_type="response.completed",
        response_id="resp_1",
        model="deepseek-flash",
        system_fingerprint="fp_1",
        output_text="ok",
        tool_calls=[],
        continuation_items=[],
        usage={
            "input_tokens": 100,
            "cached_input_tokens": cached_input_tokens,
            "cache_write_tokens": 0,
            "output_tokens": 20,
            "reasoning_output_tokens": 5,
            "total_tokens": 120,
        },
    )


def test_exact_request_identity_and_dispatch_only_credential(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-used")
    monkeypatch.setenv("HTTPS_PROXY", "http://hostile.invalid:9999")
    captured: dict[str, Any] = {}

    def runner(request, **kwargs):
        captured["request"] = request
        captured["kwargs"] = kwargs
        return _terminal()

    result = _backend(runner).invoke(_invocation())

    assert result.error is None
    assert captured["request"] == {
        "protocol": "crucible.deepseek-worker.v1",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "max_output_tokens": 32_000,
        "model": "deepseek-flash",
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [],
    }
    assert captured["kwargs"]["api_key"] == "secret-deepseek-test-key"
    assert "api_key" not in captured["request"]
    assert "base_url" not in captured["request"]
    assert deepseek_mod._worker_environment() == {"PYTHONNOUSERSITE": "1"}
    trace = result.execution_trace
    assert trace is not None
    assert trace.backend == "deepseek"
    assert trace.requested_model == trace.served_model == "deepseek-flash"
    assert trace.endpoint == "https://api.deepseek.com/responses"
    assert trace.requested_effort == "high"
    assert trace.served_effort is None


def test_alias_host_writer_and_ignored_request_fields_fail_locally(monkeypatch):
    monkeypatch.delenv("CRUCIBLE_DEEPSEEK_WORKER_PYTHON", raising=False)
    with pytest.raises(ValueError, match="only exact request alias"):
        DeepSeekBackend(model="deepseek-v4-pro", _worker_pin=_pin())
    with pytest.raises(ValueError, match="workspace_write is promotion-blocked"):
        DeepSeekBackend(capability="workspace_write", _worker_pin=_pin())
    with pytest.raises(TypeError, match="exact bool"):
        DeepSeekBackend(
            diagnostic_workspace_write=1,  # type: ignore[arg-type]
            _worker_pin=_pin(),
        )
    with pytest.raises(TypeError):
        DeepSeekBackend(base_url="https://attacker.invalid", _worker_pin=_pin())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DeepSeekBackend(api_key="ambient", _worker_pin=_pin())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="1 through 32000"):
        DeepSeekBackend(max_response_tokens=32_001, _worker_pin=_pin())

    backend = _backend(lambda *_args, **_kwargs: _terminal())
    with pytest.raises(ValueError, match="only attached client function tools"):
        backend.invoke(_invocation(tools=(SimpleNamespace(name="provider_tool"),)))
    with pytest.raises(ValueError, match="retry_budget == 1"):
        backend.invoke(
            _invocation(
                policy=BackendPolicy(
                    retry_budget=2,
                    timeout_seconds=30,
                    on_cancellation="convert_to_backend_error",
                    on_max_turns="continue_with_warning",
                )
            )
        )


def test_route_identity_tables_agree_on_current_flash_alias(
    worker_module,
    transport_check_module,
):
    import crucible.backends.config as config_mod

    # The request alias is a closed constant in the parent, routing
    # validation, the worker and the provider-free conformance script. A
    # partial move leaves a route that one table admits and another refuses.
    assert deepseek_mod._MODEL == "deepseek-flash"
    assert config_mod._DEEPSEEK_MODEL == "deepseek-flash"
    assert worker_module._MODEL == "deepseek-flash"
    assert transport_check_module.MODEL == "deepseek-flash"
    assert transport_check_module.EXPECTED_VERSION == deepseek_mod._OPENAI_VERSION
    assert worker_module._OPENAI_VERSION == deepseek_mod._OPENAI_VERSION
    profile = feature_profile_for_route("deepseek", model="deepseek-flash")
    assert profile != CONSERVATIVE_BACKEND_FEATURES
    assert profile.structured_output == "prompt_only"
    assert profile.turn_limit == "hard"
    # The adapter resolves this price before every dispatch.
    assert "deepseek-flash" in DEFAULT_PRICING
    assert pricing_for_model("deepseek-flash").output_per_mtok > 0


def test_retired_flash_name_is_refused_by_every_table(worker_module):
    # The vendor still answers the retired name, so only the closed tables
    # can show that no alias for it remains.
    retired = "deepseek-v4-flash"
    with pytest.raises(ValueError, match="only exact request alias"):
        DeepSeekBackend(model=retired, _worker_pin=_pin())
    with pytest.raises(worker_module.WorkerContractError, match="Flash alias"):
        worker_module._validate_request(
            {
                "protocol": "crucible.deepseek-worker.v1",
                "api_key": "secret-deepseek-test-key",
                "input": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
                "max_output_tokens": 32_000,
                "model": retired,
                "reasoning": {"effort": "high"},
                "stream": True,
                "tools": [],
            }
        )
    with pytest.raises(ValueError, match="requires exact model"):
        routing_fingerprint(
            BackendRouting(
                default=RoleConfig(
                    backend="deepseek",
                    model=retired,
                    feature_admission="diagnostic",
                )
            )
        )
    assert (
        feature_profile_for_route("deepseek", model=retired)
        == CONSERVATIVE_BACKEND_FEATURES
    )
    assert retired not in DEFAULT_PRICING
    with pytest.raises(PricingConfigurationError):
        pricing_for_model(retired, pricing=dict(DEFAULT_PRICING))


def test_runtime_identity_binds_sdk_artifacts_sbom_and_expected_fingerprint(
    monkeypatch,
):
    monkeypatch.delenv("CRUCIBLE_DEEPSEEK_WORKER_PYTHON", raising=False)
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="requires a predeclared worker runtime fingerprint",
    ):
        DeepSeekBackend()
    identity = _pin().public_identity().to_mapping()
    assert identity["runtime_fingerprint"] == _DIGEST
    assert identity["openai_version"] == "2.53.0"
    assert identity["installed_tree_fingerprint"] == _DIGEST
    assert identity["openai_distribution_hashes"] == {
        "wheel_sha256": (
            "sha256:c694ffc747a3c4d1663ef2b07b811315a476164ee5efa3a993967349ebca7618"
        ),
        "source_sha256": (
            "sha256:baf5802ad08980e1d9d561e1b996e800c8bcd14af5847c6d0e7a5cc59e4d4116"
        ),
    }
    assert [
        (component["name"], component["version"])
        for component in identity["sbom"]
    ] == list(deepseek_mod._LOCKED_LINUX_COMPONENT_VERSIONS)
    DeepSeekBackend(
        _worker_pin=_pin(),
        expected_worker_runtime_fingerprint=_DIGEST,
    )
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="configured runtime fingerprint",
    ):
        DeepSeekBackend(
            _worker_pin=_pin(),
            expected_worker_runtime_fingerprint="sha256:" + "b" * 64,
        )
    monkeypatch.setenv(
        "CRUCIBLE_DEEPSEEK_WORKER_PYTHON",
        "/tmp/ambient-python",
    )
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="ambient interpreter overrides",
    ):
        DeepSeekBackend(_worker_pin=_pin())


def test_diagnostic_routing_requires_predeclared_worker_identity():
    routing = BackendRouting(
        default=RoleConfig(
            backend="deepseek",
            model="deepseek-flash",
            feature_admission="diagnostic",
            extra={"max_response_tokens": 32_000},
        )
    )
    with pytest.raises(
        FeatureAdmissionError,
        match="expected_worker_runtime_fingerprint",
    ):
        resolve_for_role("plan_critic", routing=routing, capability="text")


def test_exact_diagnostic_route_resolves_text_and_confined_writer(monkeypatch):
    import crucible.backends.config as config_mod

    constructed: list[tuple[str, dict[str, Any]]] = []
    sentinel = object()

    def fake_resolve(name, **kwargs):
        constructed.append((name, kwargs))
        return sentinel

    monkeypatch.setattr(config_mod, "_resolve_backend", fake_resolve)
    routing = BackendRouting(
        default=RoleConfig(
            backend="deepseek",
            model="deepseek-flash",
            feature_admission="diagnostic",
            extra={
                "diagnostic_workspace_write": True,
                "expected_worker_runtime_fingerprint": _DIGEST,
                "max_response_tokens": 32_000,
            },
        )
    )

    assert (
        resolve_for_role("plan_critic", routing=routing, capability="text")
        is sentinel
    )
    assert (
        resolve_for_role(
            "implementer",
            routing=routing,
            capability="workspace_write",
            native_shell=False,
            allow_trusted_project_instructions=False,
            cwd="/confined/workspace",
        )
        is sentinel
    )
    assert [name for name, _kwargs in constructed] == ["deepseek", "deepseek"]
    assert constructed[1][1]["diagnostic_workspace_write"] is True
    assert constructed[1][1]["expected_worker_runtime_fingerprint"] == _DIGEST


def test_deepseek_writer_route_rejects_unconfined_controls(monkeypatch):
    import crucible.backends.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "_resolve_backend",
        lambda *_args, **_kwargs: pytest.fail("backend must not be constructed"),
    )
    routing = BackendRouting(
        default=RoleConfig(
            backend="deepseek",
            model="deepseek-flash",
            feature_admission="diagnostic",
            extra={"expected_worker_runtime_fingerprint": _DIGEST},
        )
    )

    with pytest.raises(
        FeatureAdmissionError,
        match=(
            "diagnostic_workspace_write.*native_shell.*trusted project.*confined cwd"
        ),
    ):
        resolve_for_role(
            "implementer",
            routing=routing,
            capability="workspace_write",
        )


@pytest.mark.parametrize("control", (1, "true", None))
def test_deepseek_writer_route_control_requires_exact_bool(control):
    routing = BackendRouting(
        default=RoleConfig(
            backend="deepseek",
            model="deepseek-flash",
            feature_admission="diagnostic",
            extra={
                "diagnostic_workspace_write": control,
                "expected_worker_runtime_fingerprint": _DIGEST,
            },
        )
    )

    with pytest.raises(ValueError, match="diagnostic_workspace_write.*bool"):
        resolve_for_role("plan_critic", routing=routing, capability="text")


def test_prompt_only_schema_is_canonical_and_not_duplicated(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    canonical = json.dumps(
        schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    requests: list[dict[str, Any]] = []

    def runner(request, **_kwargs):
        requests.append(request)
        return _terminal(text='{"answer":42}')

    result = _backend(runner).invoke(_invocation(response_schema=schema))

    assert result.error is None
    system = requests[0]["input"][0]["content"]
    assert system.count(canonical) == 1
    assert system.endswith(canonical)
    assert "CRUCIBLE OUTPUT CONTRACT:" in system

    abi_system = (
        "<!-- assay.prompt.v2 -->\n"
        "Return the requested result.\n\n"
        "## Required output contract\n"
        f"```json\n{canonical}\n```"
    )
    requests.clear()
    result = _backend(runner).invoke(
        _invocation(system=abi_system, response_schema=schema)
    )
    assert result.error is None
    assert requests[0]["input"][0]["content"] == abi_system
    assert requests[0]["input"][0]["content"].count(canonical) == 1


def test_exact_route_advertises_only_prompt_structured_output():
    profile = _backend(lambda *_args, **_kwargs: _terminal()).features

    assert profile.structured_output == "prompt_only"
    assert profile.cancellation == "boundary_only"
    assert profile.native_delegation == "none"


def test_prompt_only_schema_rejects_malformed_abi_before_dispatch():
    schema = {"type": "object"}
    runner_called = False

    def runner(*_args, **_kwargs):
        nonlocal runner_called
        runner_called = True
        return _terminal()

    malformed = (
        "<!-- assay.prompt.v2 -->\n"
        "## Required output contract\n"
        "```json\n{\"type\":\"array\"}\n```"
    )
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="exactly one matching canonical response schema",
    ):
        _backend(runner).invoke(
            _invocation(system=malformed, response_schema=schema)
        )
    assert runner_called is False


@pytest.mark.parametrize(
    "schema",
    (
        {"enum": [object()]},
        {"minimum": float("nan")},
    ),
)
def test_prompt_only_schema_rejects_noncanonical_json(schema):
    with pytest.raises(ValueError, match="JSON serializable"):
        _backend(lambda *_args, **_kwargs: _terminal()).invoke(
            _invocation(response_schema=schema)
        )


def test_deepseek_routing_rejects_credentials_before_environment_expansion(
    monkeypatch,
):
    monkeypatch.delenv("SHOULD_NOT_BE_READ", raising=False)
    with pytest.raises(ValueError, match="dispatch-only"):
        _parse_role_config(
            {
                "backend": "deepseek",
                "model": "deepseek-flash",
                "api_key": "${SHOULD_NOT_BE_READ}",
            },
            location="default",
        )


def test_worker_stream_requires_monotonic_exactly_one_final_terminal():
    progress = _wire_line(
        0,
        type="event",
        event_type="response.created",
    )
    terminal = _terminal_wire(1)
    parsed = deepseek_mod._parse_worker_stdout(
        progress + terminal,
        returncode=0,
    )
    assert parsed.event_types == ("response.created",)
    assert parsed.response_id == "resp_1"

    with pytest.raises(deepseek_mod.DeepSeekSDKQualificationError, match="EOF"):
        deepseek_mod._parse_worker_stdout(progress, returncode=0)
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="monotonic and contiguous",
    ):
        deepseek_mod._parse_worker_stdout(
            progress + _terminal_wire(2),
            returncode=0,
        )
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="provider SSE sequence",
    ):
        deepseek_mod._parse_worker_stdout(
            _wire_line(
                0,
                type="event",
                event_type="response.created",
                provider_sequence=5,
            )
            + _terminal_wire(1),
            returncode=0,
        )
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="terminal was not its final record",
    ):
        deepseek_mod._parse_worker_stdout(
            progress + terminal + _terminal_wire(2),
            returncode=0,
        )


def test_worker_preserves_provider_field_presence_for_cache_accounting(
    worker_module,
):
    class Event:
        def model_dump(self, **kwargs):
            assert kwargs == {"mode": "json", "exclude_unset": True}
            return {"type": "response.created", "sequence_number": 0}

    assert worker_module._event_mapping(Event()) == {
        "type": "response.created",
        "sequence_number": 0,
    }
    usage = worker_module._usage(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {},
                "output_tokens_details": {"reasoning_tokens": 5},
            }
        }
    )
    assert usage["cached_input_tokens"] is None
    usage_with_write = worker_module._usage(
        {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {
                    "cached_tokens": 25,
                    "cache_write_tokens": 10,
                },
                "output_tokens_details": {"reasoning_tokens": 5},
            }
        }
    )
    assert usage_with_write["cache_write_tokens"] == 10


def test_worker_rejects_silently_ignored_request_and_tool_fields(worker_module):
    request = {
        "protocol": "crucible.deepseek-worker.v1",
        "api_key": "secret-deepseek-test-key",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "max_output_tokens": 32_000,
        "model": "deepseek-flash",
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [],
    }
    assert worker_module._validate_request(request)["model"] == ("deepseek-flash")

    with pytest.raises(worker_module.WorkerContractError, match="unknown"):
        worker_module._validate_request({**request, "temperature": 0})
    with pytest.raises(worker_module.WorkerContractError, match="unknown"):
        worker_module._validate_request(
            {**request, "previous_response_id": "silently-ignored"}
        )
    with pytest.raises(worker_module.WorkerContractError, match="Flash alias"):
        worker_module._validate_request({**request, "model": "deepseek-v4-pro"})
    with pytest.raises(worker_module.WorkerContractError, match="client function"):
        worker_module._validate_request(
            {
                **request,
                "tools": [
                    {
                        "type": "web_search",
                        "name": "search",
                        "description": "search",
                        "parameters": {},
                    }
                ],
            }
        )
    with pytest.raises(worker_module.WorkerContractError, match="provider grammar"):
        worker_module._validate_request(
            {
                **request,
                "tools": [
                    {
                        "type": "function",
                        "name": "invalid.name",
                        "description": "invalid provider name",
                        "parameters": {},
                    }
                ],
            }
        )


def test_worker_requires_full_stateless_history_and_preserves_reasoning_in_memory(
    worker_module,
):
    response_output = [
        {
            "type": "reasoning",
            "content": [
                {"type": "reasoning_text", "text": "private-reasoning-sentinel"}
            ],
            "summary": [],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"life"}',
        },
    ]
    text, calls, continuation = worker_module._terminal_output(
        {"output": response_output}
    )
    assert text == ""
    assert calls == [
        {
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"life"}',
        }
    ]
    assert continuation == [
        {
            "type": "reasoning",
            "content": [
                {"type": "reasoning_text", "text": "private-reasoning-sentinel"}
            ],
        },
        {"type": "function_call", **calls[0]},
    ]

    request = {
        "protocol": "crucible.deepseek-worker.v1",
        "api_key": "secret-deepseek-test-key",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            *continuation,
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "answer=42",
            },
        ],
        "max_output_tokens": 32_000,
        "model": "deepseek-flash",
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [],
    }
    validated = worker_module._validate_request(request)
    assert validated["input"] == request["input"]
    with pytest.raises(worker_module.WorkerContractError, match="lack outputs"):
        worker_module._validate_request(
            {**request, "input": request["input"][:-1]}
        )


def test_worker_refuses_unpriced_usage_and_server_tool_output(worker_module):
    base = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
    # Server-side tool use and safety review carry charges the Flash row does
    # not price, and neither can be requested through the closed request.
    with pytest.raises(
        worker_module.WorkerContractError,
        match="usage fields changed",
    ):
        worker_module._usage(
            {"usage": {**base, "server_tool_use": {"tool_search": 1}}}
        )
    with pytest.raises(
        worker_module.WorkerContractError,
        match="output token detail fields changed",
    ):
        worker_module._usage(
            {
                "usage": {
                    **base,
                    "output_tokens_details": {
                        "reasoning_tokens": 5,
                        "safety_tokens": 3,
                    },
                }
            }
        )
    for item_type in ("tool_search_call", "web_search_call"):
        with pytest.raises(
            worker_module.WorkerContractError,
            match="unsupported output item type",
        ):
            worker_module._terminal_output(
                {"output": [{"type": item_type, "id": "item_1"}]}
            )


def test_worker_tolerates_message_phase_and_replays_exact_fields(worker_module):
    message = {
        "type": "message",
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
    }
    text, calls, continuation = worker_module._terminal_output(
        {"output": [message]}
    )
    assert text == "ok"
    assert calls == []

    function_call = {
        "type": "function_call",
        "call_id": "call_1",
        "name": "lookup",
        "arguments": '{"q":"life"}',
    }
    text, calls, continuation = worker_module._terminal_output(
        {"output": [message, function_call]}
    )
    assert text == "ok"
    assert calls == [
        {"call_id": "call_1", "name": "lookup", "arguments": '{"q":"life"}'}
    ]
    # Only the exact fields the closed request admits are replayed.
    assert continuation == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok"}],
        },
        function_call,
    ]
    request = {
        "protocol": "crucible.deepseek-worker.v1",
        "api_key": "secret-deepseek-test-key",
        "input": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            *continuation,
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "answer=42",
            },
        ],
        "max_output_tokens": 32_000,
        "model": "deepseek-flash",
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [],
    }
    assert worker_module._validate_request(request)["input"] == request["input"]


def test_worker_sdk_call_is_exact_native_responses_without_fallback(
    monkeypatch,
    worker_module,
):
    captured: dict[str, Any] = {}

    class FakeHttpClient:
        def __init__(self, **kwargs):
            captured["http"] = kwargs

        def close(self):
            captured["http_closed"] = True

    class FakeEvent:
        def model_dump(self, **kwargs):
            assert kwargs == {"mode": "json", "exclude_unset": True}
            return {
                "type": "response.completed",
                "sequence_number": 7,
                "response": {
                    "id": "resp_exact",
                    "model": "deepseek-flash",
                    "status": "completed",
                    "system_fingerprint": "fp_exact",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "ok"}
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 10,
                        "input_tokens_details": {"cached_tokens": 2},
                        "output_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 1},
                        "total_tokens": 14,
                    },
                },
            }

    class FakeResponses:
        def create(self, **kwargs):
            captured["create"] = kwargs
            return iter([FakeEvent()])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.responses = FakeResponses()

        def close(self):
            captured["client_closed"] = True

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(worker_module.httpx, "Client", FakeHttpClient)
    monkeypatch.setattr(worker_module.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(worker_module, "_emit", emitted.append)
    request = worker_module._validate_request(
        {
            "protocol": "crucible.deepseek-worker.v1",
            "api_key": "secret-deepseek-test-key",
            "input": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            "max_output_tokens": 32_000,
            "model": "deepseek-flash",
            "reasoning": {"effort": "high"},
            "stream": True,
            "tools": [],
        }
    )

    assert worker_module._invoke(request) == 0
    assert captured["http"] == {"trust_env": False, "follow_redirects": False}
    assert captured["client"]["api_key"] == "secret-deepseek-test-key"
    assert captured["client"]["base_url"] == "https://api.deepseek.com"
    assert captured["client"]["max_retries"] == 0
    assert captured["client"]["http_client"].__class__ is FakeHttpClient
    assert captured["create"] == {
        "model": "deepseek-flash",
        "input": request["input"],
        "max_output_tokens": 32_000,
        "reasoning": {"effort": "high"},
        "stream": True,
        "tools": [],
    }
    assert emitted[-1]["terminal_type"] == "response.completed"
    assert emitted[-1]["provider_sequence"] == 7
    assert emitted[-1]["continuation_items"] == []
    assert captured["client_closed"] is True
    assert captured["http_closed"] is True


def test_missing_cache_counter_remains_unknown_and_uses_incomplete_upper_bound(
    monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminal = _terminal(usage=_usage(cached_input_tokens=None))
    result = _backend(lambda *_args, **_kwargs: terminal).invoke(_invocation())

    assert result.error is None
    assert result.usage.cache_read_tokens == 0
    assert result.usage.cost_usd == pytest.approx(
        compute_cost(
            "deepseek-flash",
            input_tokens=1_000,
            output_tokens=100,
            cache_read_tokens=0,
        )
    )
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False
    assert result.pricing_evidence is not None
    assert result.pricing_evidence.complete is False


def test_missing_reasoning_counter_keeps_usage_incomplete(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminal = _terminal(usage=_usage(reasoning_output_tokens=None))
    result = _backend(lambda *_args, **_kwargs: terminal).invoke(_invocation())

    assert result.error is None
    assert result.usage.cache_read_tokens == 250
    assert result.completeness.token_usage_complete is False
    assert result.completeness.cost_complete is False


def test_exact_cache_partition_and_flash_pricing(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    result = _backend(lambda *_args, **_kwargs: _terminal()).invoke(_invocation())
    # The row holds the published peak rates (a ceiling off-peak).
    expected = 750 * 0.30 / 1_000_000 + 250 * 0.006 / 1_000_000
    expected += 100 * 1.20 / 1_000_000

    pricing = pricing_for_model("deepseek-flash")
    assert pricing.input_per_mtok == 0.30
    assert pricing.cache_read_per_mtok == 0.006
    assert pricing.output_per_mtok == 1.20
    assert expected == pytest.approx(0.0003465)
    assert result.usage.cost_usd == pytest.approx(expected)
    assert result.usage.cache_read_tokens == 250
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.pricing_evidence is not None
    assert result.pricing_evidence.complete is True


def test_cache_write_counter_is_retained_at_cache_miss_rate(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminal = _terminal(usage=_usage(cache_write_tokens=100))
    result = _backend(lambda *_args, **_kwargs: terminal).invoke(_invocation())

    assert result.usage.cache_creation_tokens == 100
    assert result.usage.cache_read_tokens == 250
    assert result.usage.cost_usd == pytest.approx(
        compute_cost(
            "deepseek-flash",
            input_tokens=1_000,
            output_tokens=100,
            cache_creation_tokens=100,
            cache_read_tokens=250,
        )
    )
    assert result.pricing_evidence is not None
    assert result.pricing_evidence.complete is True


def test_prompt_free_response_evidence_retains_physical_accounting(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminal = _terminal(usage=_usage(cache_write_tokens=100))
    result, evidence = _backend(
        lambda *_args, **_kwargs: terminal
    ).invoke_with_response_evidence(_invocation())

    assert result.error is None
    assert evidence.complete is True
    assert evidence.worker_runtime_fingerprint == _DIGEST
    assert evidence.to_mapping() == {
        "schema_version": "crucible.deepseek-invocation-evidence.v1",
        "worker_runtime_fingerprint": _DIGEST,
        "logical_result_complete": True,
        "responses": [
            {
                "response_ordinal": 0,
                "response_id": "resp_1",
                "served_model": "deepseek-flash",
                "system_fingerprint": "fp_test",
                "terminal_type": "response.completed",
                "input_tokens": 1_000,
                "cached_input_tokens": 250,
                "cache_write_tokens": 100,
                "output_tokens": 100,
                "reasoning_output_tokens": 40,
                "cost_usd": result.usage.cost_usd,
                "usage_complete": True,
            }
        ],
        "complete": True,
    }
    serialized = json.dumps(evidence.to_mapping(), sort_keys=True)
    assert "secret-deepseek-test-key" not in serialized
    assert "system\"" not in serialized
    assert "user\"" not in serialized


@pytest.mark.parametrize(
    "served_model",
    (
        # The snapshot label of the vendor's pricing page instead of the alias.
        "DeepSeek-V4.1-Flash-1017",
        # The retired compatibility name of the same route.
        "deepseek-v4-flash",
        # The other current native id.
        "deepseek-v4-pro",
    ),
)
def test_served_alias_mismatch_is_terminal_but_not_promotable(
    monkeypatch,
    served_model,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    result, evidence = _backend(
        lambda *_args, **_kwargs: _terminal(model=served_model)
    ).invoke_with_response_evidence(_invocation())

    assert result.error is not None
    assert result.stop_reason == "error"
    assert result.usage.input_tokens == 1_000
    assert result.usage.output_tokens == 100
    assert result.usage.cost_usd > 0
    assert result.completeness.terminal_complete is True
    assert result.completeness.cost_complete is False
    assert result.completeness.trace_complete is False
    trace = result.execution_trace
    assert trace is not None
    assert trace.requested_model == "deepseek-flash"
    assert trace.served_model == served_model
    assert any(
        event.kind == "terminal" and event.name == "served_model_mismatch"
        for event in trace.events
    )
    assert evidence.responses[0].served_model == served_model
    assert evidence.complete is False


def test_post_response_budget_guard_uses_the_peak_ceiling(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    # The standard usage is recorded at the published peak rate. Off-peak the
    # vendor charges half of it. The guard compares the recorded value, so it
    # can stop an invocation early and can never let one pass late.
    peak_cost = 0.0003465
    off_peak_charge = peak_cost / 2

    inside = _backend(lambda *_args, **_kwargs: _terminal()).invoke(
        _invocation(budget_usd=0.0004)
    )
    assert inside.error is None
    assert inside.stop_reason == "end_turn"
    assert inside.usage.cost_usd == pytest.approx(peak_cost)

    assert off_peak_charge < 0.0002 < peak_cost
    between = _backend(lambda *_args, **_kwargs: _terminal()).invoke(
        _invocation(budget_usd=0.0002)
    )
    assert between.stop_reason == "budget_exhausted"
    assert type(between.error) is BudgetExhausted
    assert between.error.spent_usd == pytest.approx(peak_cost)
    assert between.error.limit_usd == 0.0002


class _ReadOnlyToolSet:
    write_names: set[str] = set()
    read_only_names = {"lookup"}

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "lookup",
                "description": "Look up a value",
                "input_schema": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "answer=42"


def _writer_toolset(tmp_path: Path):
    from crucible.tools import ToolContext
    from crucible.tools.write import build_write_tools

    context = ToolContext(
        workspace_root=tmp_path,
        repo_root=tmp_path,
        write_scope=("output.txt",),
    )
    return build_write_tools(context).filter_to({"write_source"})


def test_writer_tools_run_only_through_invocation_scoped_process_host(
    monkeypatch,
    tmp_path,
):
    from crucible.tools import _isolated_mutation_host as mutation_mod

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")

    class Host:
        instances: list[Host] = []

        def __init__(self, spec, *, invocation_deadline) -> None:
            assert invocation_deadline > time.monotonic()
            self.spec = spec
            self.start_calls = 0
            self.stop_calls = 0
            self.dispatches: list[tuple[str, dict[str, Any]]] = []
            type(self).instances.append(self)

        @property
        def tool_names(self):
            return self.spec.names

        def start(self) -> None:
            self.start_calls += 1

        def stop(self) -> None:
            self.stop_calls += 1

        def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
            self.dispatches.append((name, arguments))
            return "write completed"

    monkeypatch.setattr(mutation_mod, "IsolatedMutationHost", Host)
    terminals = iter(
        (
            _terminal(
                response_id="resp_write",
                text="",
                calls=(
                    deepseek_mod._DeepSeekToolCall(
                        call_id="call_write",
                        name="write_source",
                        arguments=(
                            '{"file_path":"output.txt",'
                            '"content":"settled\\n"}'
                        ),
                    ),
                ),
            ),
            _terminal(response_id="resp_final", text="done"),
        )
    )
    backend = _backend(
        lambda *_args, **_kwargs: next(terminals),
        capability="workspace_write",
        diagnostic_workspace_write=True,
    )
    backend.attach_toolset(_writer_toolset(tmp_path), context_budget=4096)

    result = backend.invoke(_invocation(max_turns=2))

    assert result.error is None
    assert len(Host.instances) == 1
    host = Host.instances[0]
    assert host.start_calls == 1
    assert host.stop_calls == 1
    assert host.dispatches == [
        (
            "write_source",
            {"file_path": "output.txt", "content": "settled\n"},
        )
    ]
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


def test_real_isolated_host_applies_diagnostic_writer_without_parent_dispatch(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminals = iter(
        (
            _terminal(
                response_id="resp_write",
                text="",
                calls=(
                    deepseek_mod._DeepSeekToolCall(
                        call_id="call_write",
                        name="write_source",
                        arguments=(
                            '{"file_path":"output.txt",'
                            '"content":"settled\\n"}'
                        ),
                    ),
                ),
            ),
            _terminal(response_id="resp_final", text="done"),
        )
    )
    backend = _backend(
        lambda *_args, **_kwargs: next(terminals),
        capability="workspace_write",
        diagnostic_workspace_write=True,
    )
    backend.attach_toolset(_writer_toolset(tmp_path), context_budget=4096)

    result = backend.invoke(_invocation(max_turns=2))

    assert result.error is None
    assert (tmp_path / "output.txt").read_text(encoding="utf-8") == "settled\n"
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


def test_indeterminate_writer_dispatch_is_terminal_before_provider_continuation(
    monkeypatch,
    tmp_path,
):
    from crucible.tools import ToolMutationIndeterminate
    from crucible.tools import _isolated_mutation_host as mutation_mod

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    terminal = ToolMutationIndeterminate("source settlement is unknown")

    class Host:
        instances: list[Host] = []

        def __init__(self, spec, *, invocation_deadline) -> None:
            del invocation_deadline
            self.spec = spec
            self.stop_calls = 0
            type(self).instances.append(self)

        @property
        def tool_names(self):
            return self.spec.names

        def start(self) -> None:
            return

        def stop(self) -> None:
            self.stop_calls += 1

        def dispatch(self, _name: str, _arguments: dict[str, Any]) -> str:
            raise terminal

    monkeypatch.setattr(mutation_mod, "IsolatedMutationHost", Host)
    worker_calls = 0

    def runner(*_args, **_kwargs):
        nonlocal worker_calls
        worker_calls += 1
        return _terminal(
            response_id="resp_write",
            text="",
            calls=(
                deepseek_mod._DeepSeekToolCall(
                    call_id="call_write",
                    name="write_source",
                    arguments='{"file_path":"output.txt","content":"x"}',
                ),
            ),
        )

    backend = _backend(
        runner,
        capability="workspace_write",
        diagnostic_workspace_write=True,
    )
    backend.attach_toolset(_writer_toolset(tmp_path), context_budget=4096)

    with pytest.raises(ToolMutationIndeterminate) as caught:
        backend.invoke(_invocation(max_turns=2))

    assert caught.value is terminal
    assert worker_calls == 1
    assert Host.instances[0].stop_calls == 1
    assert backend._active_mutation_hosts == set()  # noqa: SLF001
    backend.close()


def test_writer_host_must_settle_before_return_and_close_can_retry(
    monkeypatch,
    tmp_path,
):
    from crucible.tools import _isolated_mutation_host as mutation_mod

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")

    class Host:
        instances: list[Host] = []

        def __init__(self, spec, *, invocation_deadline) -> None:
            del invocation_deadline
            self.spec = spec
            self.allow_settlement = False
            self.stop_calls = 0
            type(self).instances.append(self)

        @property
        def tool_names(self):
            return self.spec.names

        def start(self) -> None:
            return

        def stop(self) -> None:
            self.stop_calls += 1
            if not self.allow_settlement:
                raise mutation_mod.MutationHostLifecycleError(
                    "injected live mutation child"
                )

        def dispatch(self, _name: str, _arguments: dict[str, Any]) -> str:
            return "unused"

    monkeypatch.setattr(mutation_mod, "IsolatedMutationHost", Host)
    backend = _backend(
        lambda *_args, **_kwargs: _terminal(),
        capability="workspace_write",
        diagnostic_workspace_write=True,
    )
    backend.attach_toolset(_writer_toolset(tmp_path), context_budget=4096)

    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="mutation-host lifecycle did not settle",
    ):
        backend.invoke(_invocation())

    host = Host.instances[0]
    assert host in backend._active_mutation_hosts  # noqa: SLF001
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="could not settle every confined resource",
    ):
        backend.close()
    assert host in backend._active_mutation_hosts  # noqa: SLF001
    host.allow_settlement = True
    backend.close()
    assert host.stop_calls == 4
    assert backend._active_mutation_hosts == set()  # noqa: SLF001


def test_client_function_tool_loop_preserves_ids_and_prices_every_response(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    toolset = _ReadOnlyToolSet()
    first_terminal = _terminal(
        response_id="resp_tool",
        text="",
        calls=(
            deepseek_mod._DeepSeekToolCall(
                call_id="call_1",
                name="lookup",
                arguments='{"q":"life"}',
            ),
        ),
    )
    first_terminal = replace(
        first_terminal,
        continuation_json=json.dumps(
            [
                {
                    "type": "reasoning",
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "private-reasoning-sentinel",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"life"}',
                },
            ],
            sort_keys=True,
        ),
    )
    terminals = iter(
        [
            first_terminal,
            _terminal(response_id="resp_final", text="42"),
        ]
    )
    requests: list[dict[str, Any]] = []

    def runner(request, **_kwargs):
        requests.append(request)
        return next(terminals)

    backend = _backend(runner, capability="read_only", log_dir=tmp_path)
    backend.attach_toolset(toolset, context_budget=1024)
    result, evidence = backend.invoke_with_response_evidence(
        _invocation(max_turns=2)
    )

    assert result.error is None
    assert result.final_message == "42"
    assert toolset.calls == [("lookup", {"q": "life"})]
    assert requests[0]["tools"] == [
        {
            "type": "function",
            "name": "lookup",
            "description": "Look up a value",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    assert "previous_response_id" not in requests[1]
    assert requests[1]["input"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {
            "type": "reasoning",
            "content": [
                {
                    "type": "reasoning_text",
                    "text": "private-reasoning-sentinel",
                }
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{"q":"life"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "answer=42",
        }
    ]
    assert result.pricing_evidence is not None
    assert len(result.pricing_evidence.segments) == 2
    assert len(evidence.responses) == 2
    assert "private-reasoning-sentinel" not in json.dumps(
        evidence.to_mapping(), sort_keys=True
    )
    retained_debug = b"".join(
        path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    )
    assert b"private-reasoning-sentinel" not in retained_debug


def test_dispatch_budget_terminal_stops_before_provider_continuation(monkeypatch):
    from crucible.tools import ToolDispatchBudgetExceeded

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")

    class ExhaustedToolSet(_ReadOnlyToolSet):
        def dispatch(self, _name: str, _arguments: dict[str, Any]) -> str:
            raise ToolDispatchBudgetExceeded("read budget exhausted")

    worker_calls = 0

    def runner(*_args, **_kwargs):
        nonlocal worker_calls
        worker_calls += 1
        return _terminal(
            response_id="resp_tool",
            text="",
            calls=(
                deepseek_mod._DeepSeekToolCall(
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q":"life"}',
                ),
            ),
        )

    backend = _backend(runner, capability="read_only")
    backend.attach_toolset(ExhaustedToolSet(), context_budget=1024)

    result = backend.invoke(_invocation(max_turns=2))

    assert worker_calls == 1
    assert result.stop_reason == "tool_budget_exhausted"
    assert isinstance(result.error, MaxTurnsExceeded)
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert any(
        event.kind == "limit" and event.name == "tool_dispatch_budget"
        for event in result.execution_trace.events
    )


@pytest.mark.parametrize(
    ("limit_name", "stop_reason", "error_type"),
    (
        ("budget_usd", "budget_exhausted", BudgetExhausted),
        ("max_turns", "max_turns", MaxTurnsExceeded),
        ("context_bytes", "context_limit", TransportError),
    ),
)
def test_limit_exits_return_typed_results_with_one_final_terminal(
    monkeypatch,
    limit_name,
    stop_reason,
    error_type,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    if limit_name == "budget_usd":
        backend = _backend(lambda *_args, **_kwargs: _terminal())
        invocation = _invocation(budget_usd=0.0002)
        response_id = "resp_1"
    else:
        pending_call = _terminal(
            response_id="resp_tool",
            text="",
            calls=(
                deepseek_mod._DeepSeekToolCall(
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q":"life"}',
                ),
            ),
        )
        backend = _backend(
            lambda *_args, **_kwargs: pending_call,
            capability="read_only",
        )
        response_id = "resp_tool"
        if limit_name == "max_turns":
            # The only admitted round ends with a pending function call.
            backend.attach_toolset(_ReadOnlyToolSet(), context_budget=1024)
            invocation = _invocation(max_turns=1)
        else:
            # The tool result is nine bytes; the attached bound admits eight.
            backend.attach_toolset(_ReadOnlyToolSet(), context_budget=8)
            invocation = _invocation(max_turns=2)

    result, evidence = backend.invoke_with_response_evidence(invocation)

    assert result.stop_reason == stop_reason
    assert type(result.error) is error_type
    # The response that reached the limit was paid for. Its usage and both
    # evidence flags survive the limit exit.
    assert result.usage.input_tokens == 1_000
    assert result.usage.output_tokens == 100
    assert result.usage.cost_usd == pytest.approx(0.0003465)
    assert result.completeness.token_usage_complete is True
    assert result.completeness.cost_complete is True
    assert result.completeness.terminal_complete is True
    assert result.completeness.trace_complete is True
    assert result.pricing_evidence is not None
    assert result.pricing_evidence.complete is True
    assert len(result.pricing_evidence.segments) == 1
    assert [item.response_id for item in evidence.responses] == [response_id]
    trace = result.execution_trace
    assert trace is not None
    assert [(event.kind, event.name) for event in trace.events[-2:]] == [
        ("limit", limit_name),
        ("terminal", stop_reason),
    ]
    assert [event.kind for event in trace.events].count("terminal") == 1


def test_cross_response_tool_call_id_reuse_rejects_before_dispatch_or_accounting(
    monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    toolset = _ReadOnlyToolSet()
    duplicate_call = deepseek_mod._DeepSeekToolCall(
        call_id="call_reused",
        name="lookup",
        arguments='{"q":"life"}',
    )
    terminals = iter(
        (
            _terminal(response_id="resp_1", text="", calls=(duplicate_call,)),
            _terminal(response_id="resp_2", text="", calls=(duplicate_call,)),
        )
    )
    backend = _backend(
        lambda *_args, **_kwargs: next(terminals),
        capability="read_only",
    )
    backend.attach_toolset(toolset, context_budget=1024)

    result, evidence = backend.invoke_with_response_evidence(
        _invocation(max_turns=3)
    )

    assert result.error is not None
    assert result.stop_reason == "error"
    assert toolset.calls == [("lookup", {"q": "life"})]
    assert result.usage.input_tokens == 1_000
    assert result.usage.output_tokens == 100
    assert result.pricing_evidence is not None
    assert len(result.pricing_evidence.segments) == 1
    assert [item.response_id for item in evidence.responses] == ["resp_1"]
    assert evidence.complete is False


def test_cross_response_physical_id_reuse_rejects_before_second_accounting(
    monkeypatch,
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    toolset = _ReadOnlyToolSet()
    first = _terminal(
        response_id="resp_reused",
        text="",
        calls=(
            deepseek_mod._DeepSeekToolCall(
                call_id="call_1",
                name="lookup",
                arguments='{"q":"life"}',
            ),
        ),
    )
    terminals = iter((first, _terminal(response_id="resp_reused", text="done")))
    backend = _backend(
        lambda *_args, **_kwargs: next(terminals),
        capability="read_only",
    )
    backend.attach_toolset(toolset, context_budget=1024)

    result, evidence = backend.invoke_with_response_evidence(
        _invocation(max_turns=2)
    )

    assert result.error is not None
    assert toolset.calls == [("lookup", {"q": "life"})]
    assert result.usage.input_tokens == 1_000
    assert result.pricing_evidence is not None
    assert len(result.pricing_evidence.segments) == 1
    assert [item.response_id for item in evidence.responses] == ["resp_reused"]
    assert evidence.complete is False


def test_invocation_evidence_rejects_duplicate_physical_response_ids():
    response = deepseek_mod.DeepSeekResponseEvidence(
        response_ordinal=0,
        response_id="resp_reused",
        served_model="deepseek-flash",
        system_fingerprint=None,
        terminal_type="response.completed",
        input_tokens=1,
        cached_input_tokens=0,
        cache_write_tokens=0,
        output_tokens=1,
        reasoning_output_tokens=0,
        cost_usd=0.0,
        usage_complete=True,
    )
    with pytest.raises(ValueError, match="response IDs are not unique"):
        deepseek_mod.DeepSeekInvocationEvidence(
            worker_runtime_fingerprint=_DIGEST,
            logical_result_complete=True,
            responses=(response, replace(response, response_ordinal=1)),
        )


def test_later_unparseable_response_retains_prior_physical_evidence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek-test-key")
    toolset = _ReadOnlyToolSet()
    responses: list[Any] = [
        _terminal(
            response_id="resp_tool",
            text="",
            calls=(
                deepseek_mod._DeepSeekToolCall(
                    call_id="call_1",
                    name="lookup",
                    arguments='{"q":"life"}',
                ),
            ),
        ),
        deepseek_mod.DeepSeekSDKQualificationError("hostile malformed terminal"),
    ]

    def runner(_request, **_kwargs):
        outcome = responses.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    backend = _backend(runner, capability="read_only")
    backend.attach_toolset(toolset, context_budget=1024)
    result, evidence = backend.invoke_with_response_evidence(
        _invocation(max_turns=2)
    )

    assert result.error is not None
    assert result.usage.input_tokens == 1_000
    assert result.usage.output_tokens == 100
    assert result.usage.cost_usd > 0
    assert result.completeness.cost_complete is False
    assert len(evidence.responses) == 1
    assert evidence.responses[0].response_id == "resp_tool"
    assert evidence.logical_result_complete is False
    assert evidence.complete is False


def test_probe_binds_no_retry_env_redirect_policy_and_exact_sbom():
    probe = {
        "protocol": "crucible.deepseek-worker.v1",
        "type": "probe",
        "ok": True,
        "openai_version": "2.53.0",
        "client_policy": {
            "base_url": "https://api.deepseek.com",
            "responses_endpoint": "https://api.deepseek.com/responses",
            "max_retries": 0,
            "trust_env": False,
            "follow_redirects": False,
        },
        "sbom": [
            {
                "name": name,
                "version": version,
                "record_sha256": _DIGEST,
            }
            for name, version in deepseek_mod._LOCKED_LINUX_COMPONENT_VERSIONS
        ],
    }
    components = deepseek_mod._validate_probe(probe)
    assert [(component.name, component.version) for component in components] == list(
        deepseek_mod._LOCKED_LINUX_COMPONENT_VERSIONS
    )
    for field, bad_value in (
        ("max_retries", 1),
        ("trust_env", True),
        ("follow_redirects", True),
        ("base_url", "https://proxy.invalid"),
    ):
        bad = json.loads(json.dumps(probe))
        bad["client_policy"][field] = bad_value
        with pytest.raises(
            deepseek_mod.DeepSeekSDKQualificationError,
            match="client contract",
        ):
            deepseek_mod._validate_probe(bad)
    for mutation in ("missing", "extra", "version"):
        bad = json.loads(json.dumps(probe))
        if mutation == "missing":
            bad["sbom"].pop()
        elif mutation == "extra":
            bad["sbom"].append(
                {
                    "name": "unexpected",
                    "version": "1",
                    "record_sha256": _DIGEST,
                }
            )
        else:
            bad["sbom"][0]["version"] = "999"
        with pytest.raises(
            deepseek_mod.DeepSeekSDKQualificationError,
            match="exact Linux lock closure",
        ):
            deepseek_mod._validate_probe(bad)


def test_installed_tree_fingerprint_rejects_unowned_links(tmp_path):
    tree = tmp_path / "site-packages"
    package = tree / "demo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    first = hashlib.sha256()
    deepseek_mod._hash_tree_into(first, tree, label="site-packages")

    (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    second = hashlib.sha256()
    deepseek_mod._hash_tree_into(second, tree, label="site-packages")
    assert first.digest() != second.digest()

    (tree / "hostile.pth").symlink_to(package / "__init__.py")
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="unsupported entry",
    ):
        deepseek_mod._hash_tree_into(
            hashlib.sha256(), tree, label="site-packages"
        )


def _test_python_identity() -> tuple[Path, str]:
    executable = Path(sys.executable).resolve(strict=True)
    return executable, deepseek_mod._sha256_file(
        executable,
        label="test Python",
    )


def _wait_pid_gone(process_id: int, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while Path(f"/proc/{process_id}").exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_timeout_uses_trusted_subreaper_and_reaps_target(tmp_path):
    executable, executable_sha = _test_python_identity()
    target_pid_path = tmp_path / "target.pid"
    target = tmp_path / "target.py"
    target.write_text(
        "from pathlib import Path\n"
        "import os, sys, time\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )

    completed = deepseek_mod._run_supervised_command(
        (str(executable), "-I", str(target), str(target_pid_path)),
        stdin_bytes=b"",
        cwd=tmp_path.resolve(strict=True),
        env={"PYTHONNOUSERSITE": "1"},
        timeout_seconds=0.15,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
        executable_path=executable,
        executable_sha256=executable_sha,
    )

    assert completed.timed_out is True
    assert completed.returncode == -124
    assert target_pid_path.is_file()
    assert _wait_pid_gone(int(target_pid_path.read_text(encoding="ascii")))


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_backend_close_waits_for_authenticated_supervisor_settlement(tmp_path):
    executable, executable_sha = _test_python_identity()
    target_pid_path = tmp_path / "target.pid"
    target = tmp_path / "close_target.py"
    target.write_text(
        "from pathlib import Path\n"
        "import os, sys, time\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    backend = _backend(lambda *_args, **_kwargs: _terminal())
    started = threading.Event()
    results: list[deepseek_mod._SupervisedCommandResult] = []
    errors: list[BaseException] = []

    def register(active):
        backend._register_process(active)  # noqa: SLF001
        started.set()

    def invoke_target() -> None:
        try:
            results.append(
                deepseek_mod._run_supervised_command(
                    (str(executable), "-I", str(target), str(target_pid_path)),
                    stdin_bytes=b"",
                    cwd=tmp_path.resolve(strict=True),
                    env={"PYTHONNOUSERSITE": "1"},
                    timeout_seconds=60.0,
                    max_stdout_bytes=1024,
                    max_stderr_bytes=1024,
                    executable_path=executable,
                    executable_sha256=executable_sha,
                    on_process_start=register,
                    on_process_stop=backend._unregister_process,  # noqa: SLF001
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=invoke_target, daemon=True)
    thread.start()
    assert started.wait(5.0)
    deadline = time.monotonic() + 5.0
    while not target_pid_path.is_file():
        if time.monotonic() >= deadline:
            pytest.fail("supervised close target did not start")
        time.sleep(0.01)

    backend.close()
    thread.join(timeout=5.0)

    assert thread.is_alive() is False
    assert errors == []
    assert len(results) == 1
    assert results[0].returncode == -signal.SIGTERM
    assert _wait_pid_gone(int(target_pid_path.read_text(encoding="ascii")))


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_setsid_double_fork_is_settled_and_never_accepted(tmp_path):
    executable, executable_sha = _test_python_identity()
    escaped_pid_path = tmp_path / "escaped.pid"
    target = tmp_path / "double_fork.py"
    target.write_text(
        "from pathlib import Path\n"
        "import os, sys, time\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    grandchild = os.fork()\n"
        "    if grandchild > 0:\n"
        "        os._exit(0)\n"
        "    Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')\n"
        "    time.sleep(60)\n"
        "os.waitpid(child, 0)\n",
        encoding="utf-8",
    )

    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="background descendants; all were settled",
    ):
        deepseek_mod._run_supervised_command(
            (str(executable), "-I", str(target), str(escaped_pid_path)),
            stdin_bytes=b"",
            cwd=tmp_path.resolve(strict=True),
            env={"PYTHONNOUSERSITE": "1"},
            timeout_seconds=5.0,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
            executable_path=executable,
            executable_sha256=executable_sha,
        )

    assert escaped_pid_path.is_file()
    assert _wait_pid_gone(int(escaped_pid_path.read_text(encoding="ascii")))


@pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG requires Linux")
def test_controller_death_fail_stops_supervisor_and_direct_target(tmp_path):
    executable, _executable_sha = _test_python_identity()
    supervisor_pid_path = tmp_path / "supervisor.pid"
    target_pid_path = tmp_path / "target.pid"
    target = tmp_path / "parent_death_target.py"
    target.write_text(
        "from pathlib import Path\n"
        "import os, sys, time\n"
        "marker = Path(sys.argv[1])\n"
        "pending = marker.with_name(marker.name + '.pending')\n"
        "pending.write_text(str(os.getpid()), encoding='ascii')\n"
        "pending.replace(marker)\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    controller_source = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from crucible.backends import deepseek",
            "def publish_pid(path, process_id):",
            "    pending = path.with_name(path.name + '.pending')",
            "    pending.write_text(str(process_id), encoding='ascii')",
            "    pending.replace(path)",
            "executable = Path(sys.executable).resolve(strict=True)",
            "digest = deepseek._sha256_file(executable, label='controller Python')",
            "deepseek._run_supervised_command(",
            "    (str(executable), '-I', sys.argv[1], sys.argv[2]),",
            "    stdin_bytes=b'', cwd=Path(sys.argv[3]),",
            "    env={'PYTHONNOUSERSITE': '1'}, timeout_seconds=60.0,",
            "    max_stdout_bytes=1024, max_stderr_bytes=1024,",
            "    executable_path=executable, executable_sha256=digest,",
            "    on_process_start=lambda active: publish_pid(",
            "        Path(sys.argv[4]), active.process.pid),",
            ")",
        )
    )
    controller = subprocess.Popen(
        (
            str(Path(sys.executable).absolute()),
            "-c",
            controller_source,
            str(target),
            str(target_pid_path),
            str(tmp_path.resolve(strict=True)),
            str(supervisor_pid_path),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=dict(os.environ),
        start_new_session=True,
    )
    tracked_pids: list[int] = []
    try:
        deadline = time.monotonic() + 10.0
        while not (supervisor_pid_path.is_file() and target_pid_path.is_file()):
            if controller.poll() is not None or time.monotonic() >= deadline:
                pytest.fail("controller did not launch supervised target")
            time.sleep(0.01)
        tracked_pids = [
            int(supervisor_pid_path.read_text(encoding="ascii")),
            int(target_pid_path.read_text(encoding="ascii")),
        ]
        os.kill(controller.pid, signal.SIGKILL)
        controller.wait(timeout=5)
        assert all(_wait_pid_gone(process_id) for process_id in tracked_pids)
    finally:
        if controller.poll() is None:
            os.kill(controller.pid, signal.SIGKILL)
            controller.wait(timeout=5)
        for process_id in tracked_pids:
            if Path(f"/proc/{process_id}").exists():
                os.kill(process_id, signal.SIGKILL)


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_supervisor_caps_stdout_during_read(tmp_path):
    executable, executable_sha = _test_python_identity()
    target = tmp_path / "stdout_flood.py"
    target.write_text(
        "import os\n"
        "os.write(1, b'x' * (1024 * 1024))\n",
        encoding="utf-8",
    )

    completed = deepseek_mod._run_supervised_command(
        (str(executable), "-I", str(target)),
        stdin_bytes=b"",
        cwd=tmp_path.resolve(strict=True),
        env={"PYTHONNOUSERSITE": "1"},
        timeout_seconds=5.0,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
        executable_path=executable,
        executable_sha256=executable_sha,
    )

    assert completed.output_complete is False
    assert len(completed.stdout) == 1024


def test_lock_contains_only_frozen_closure_and_openai_artifacts(
    monkeypatch,
    tmp_path,
):
    project = Path(deepseek_mod.__file__).with_name("deepseek_worker")
    lock_path = project / "uv.lock"
    lock = lock_path.read_text(encoding="utf-8")
    deepseek_mod._validate_frozen_openai_lock()
    assert hashlib.sha256(lock.encode("utf-8")).hexdigest() == (
        deepseek_mod._WORKER_LOCK_SHA256
    )
    assert 'name = "openai"' in lock
    assert 'version = "2.53.0"' in lock
    assert (
        "sha256:c694ffc747a3c4d1663ef2b07b811315a476164ee5efa3a993967349ebca7618"
        in lock
    )
    assert (
        "sha256:baf5802ad08980e1d9d561e1b996e800c8bcd14af5847c6d0e7a5cc59e4d4116"
        in lock
    )
    hostile_project = tmp_path / "worker"
    hostile_project.mkdir()
    (hostile_project / "uv.lock").write_text(
        lock + "\n# hidden drift\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(deepseek_mod, "_WORKER_PROJECT", hostile_project)
    with pytest.raises(
        deepseek_mod.DeepSeekSDKQualificationError,
        match="lock digest differs",
    ):
        deepseek_mod._validate_frozen_openai_lock()
