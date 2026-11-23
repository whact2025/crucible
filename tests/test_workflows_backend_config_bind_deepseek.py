# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from crucible.backends.config import (
    load_routing,
    resolve_for_role,
    routing_source_fingerprint,
)
from crucible.backends.deepseek import (
    DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
    DeepSeekSDKUnavailable,
)
from crucible.backends.roles import role_spec
from crucible.workflows import backend_config_bind_deepseek as binding
from crucible.workflows.__main__ import _discover_workflows


_ROOT = Path(__file__).resolve().parents[1]
_DIGEST = "sha256:" + "a" * 64
_SECRET = "must-not-appear-deepseek-key"
_TEMPLATES = (
    _ROOT / "products/swebench/backends_deepseek_diagnostic.yaml",
    _ROOT / "products/deepswe/backends_deepseek_diagnostic.yaml",
)
_INSTALL_DOCS = (
    _ROOT / "backends/deepseek_worker/README.md",
    _ROOT / "docs/workflows/backend_config_bind_deepseek.md",
    _ROOT / "docs/workflows/swebench.md",
    _ROOT / "docs/workflows/deepswe.md",
)


def _capture_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"count": 0}

    def capture() -> SimpleNamespace:
        calls["count"] += 1
        return SimpleNamespace(runtime_fingerprint=_DIGEST)

    monkeypatch.setattr(
        binding,
        "capture_project_locked_deepseek_runtime_identity",
        capture,
    )
    return calls


def _without_runtime_binding(document: dict) -> dict:
    for route in (document["default"], *document["roles"].values()):
        del route["extra"]["expected_worker_runtime_fingerprint"]
    return document


@pytest.mark.parametrize("template", _TEMPLATES, ids=("swebench", "deepswe"))
def test_checked_in_product_template_becomes_exact_usable_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    template: Path,
) -> None:
    calls = _capture_counter(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _SECRET)
    output = tmp_path / "bound.yaml"

    def reject_adapter_construction(*_args: object, **_kwargs: object) -> None:
        pytest.fail("provider-free binding must not construct an adapter")

    monkeypatch.setattr(
        "crucible.backends.config._resolve_backend",
        reject_adapter_construction,
    )

    receipt = binding.bind_deepseek_backend_config(template, output)

    assert calls == {"count": 1}
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not tuple(tmp_path.glob(".crucible-deepseek-bind-*.tmp"))
    source_document = yaml.safe_load(template.read_text(encoding="utf-8"))
    output_bytes = output.read_bytes()
    output_document = yaml.safe_load(output_bytes)
    for route in (output_document["default"], *output_document["roles"].values()):
        assert route["extra"]["expected_worker_runtime_fingerprint"] == _DIGEST
    assert _without_runtime_binding(output_document) == source_document
    assert _SECRET.encode() not in output_bytes

    loaded = load_routing(config_path=output)
    assert loaded.source_sha256 == hashlib.sha256(output_bytes).hexdigest()
    assert routing_source_fingerprint(loaded) == receipt.output_sha256
    assert loaded.default.extra["expected_worker_runtime_fingerprint"] == _DIGEST
    assert receipt.to_mapping() == {
        "schema_version": "crucible.deepseek-backend-config-binding-receipt.v1",
        "source_path": str(template),
        "output_path": str(output),
        "source_sha256": "sha256:" + hashlib.sha256(template.read_bytes()).hexdigest(),
        "output_sha256": "sha256:" + hashlib.sha256(output_bytes).hexdigest(),
        "worker_identity_schema": DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
        "worker_runtime_fingerprint": _DIGEST,
    }
    assert _SECRET not in json.dumps(receipt.to_mapping())

    constructed: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "crucible.backends.config._resolve_backend",
        lambda name, **kwargs: constructed.append((name, kwargs)) or object(),
    )
    for role in loaded.roles:
        capability = role_spec(role).max_capability
        assert (
            resolve_for_role(
                role,
                routing=loaded,
                capability=capability,
                native_shell=False,
                allow_trusted_project_instructions=False,
                cwd=str(tmp_path),
            )
            is not None
        )
    assert len(constructed) == len(loaded.roles)
    assert {name for name, _kwargs in constructed} == {"deepseek"}
    assert all(
        kwargs["expected_worker_runtime_fingerprint"] == _DIGEST
        for _name, kwargs in constructed
    )


def test_cli_prints_only_secret_free_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _capture_counter(monkeypatch)
    monkeypatch.setenv("DEEPSEEK_API_KEY", _SECRET)
    output = tmp_path / "bound.yaml"

    assert (
        binding.main(["--template", str(_TEMPLATES[0]), "--output", str(output)]) == 0
    )

    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert captured.err == ""
    assert receipt["output_path"] == str(output)
    assert receipt["worker_runtime_fingerprint"] == _DIGEST
    assert _SECRET not in captured.out


def test_existing_output_is_untouched_and_capture_does_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    output.write_bytes(b"operator-owned\n")

    with pytest.raises(binding.DeepSeekConfigBindingError, match="already exists"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert output.read_bytes() == b"operator-owned\n"
    assert calls == {"count": 0}


def test_dispatcher_discovers_operator_command() -> None:
    assert _discover_workflows()["backend_config_bind_deepseek"] == (
        "crucible.workflows.backend_config_bind_deepseek"
    )


@pytest.mark.parametrize("document", _INSTALL_DOCS)
def test_operator_install_commands_require_single_link_copy_mode(
    document: Path,
) -> None:
    text = document.read_text(encoding="utf-8")
    assert (
        "uv sync --project backends/deepseek_worker --frozen "
        "--reinstall --link-mode copy"
    ) in text
    assert "uv sync --project backends/deepseek_worker --frozen\n" not in text


def test_output_creation_race_never_overwrites_competing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    real_link = os.link

    def racing_link(source: str, destination: str, **kwargs: object) -> None:
        destination_fd = kwargs["dst_dir_fd"]
        assert isinstance(destination_fd, int)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=destination_fd,
        )
        try:
            os.write(descriptor, b"competing-output\n")
        finally:
            os.close(descriptor)
        real_link(source, destination, **kwargs)

    monkeypatch.setattr(binding.os, "link", racing_link)
    with pytest.raises(binding.DeepSeekConfigBindingError, match="appeared"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert calls == {"count": 1}
    assert output.read_bytes() == b"competing-output\n"
    assert not tuple(tmp_path.glob(".crucible-deepseek-bind-*.tmp"))


def test_link_success_followed_by_base_exception_cleans_ambiguous_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    real_link = os.link

    def link_then_interrupt(*args: object, **kwargs: object) -> None:
        real_link(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(binding.os, "link", link_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert calls == {"count": 1}
    assert not output.exists()
    assert not tuple(tmp_path.glob(".crucible-deepseek-bind-*.tmp"))


def test_failed_stage_cleanup_is_an_explicit_do_not_dispatch_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    real_unlink = os.unlink

    def fail_write(*_args: object, **_kwargs: object) -> int:
        raise OSError("injected staging write failure")

    def fail_stage_unlink(path: object, **kwargs: object) -> None:
        if str(path).startswith(".crucible-deepseek-bind-"):
            raise OSError("injected staging cleanup failure")
        real_unlink(path, **kwargs)

    monkeypatch.setattr(binding.os, "write", fail_write)
    monkeypatch.setattr(binding.os, "unlink", fail_stage_unlink)
    with pytest.raises(
        binding.DeepSeekConfigBindingError,
        match="uncertain staging cleanup; do not dispatch",
    ):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    stages = tuple(tmp_path.glob(".crucible-deepseek-bind-*.tmp"))
    assert calls == {"count": 1}
    assert not output.exists()
    assert len(stages) == 1
    real_unlink(stages[0])


def test_ambiguous_final_parent_close_does_not_invalidate_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    real_open_parent = binding._open_output_parent
    real_close = os.close
    pinned_descriptor: list[int] = []

    def remember_parent(path: Path) -> binding._PinnedOutputParent:
        parent = real_open_parent(path)
        pinned_descriptor.append(parent.descriptor)
        return parent

    def close_then_report_ambiguous(descriptor: int) -> None:
        real_close(descriptor)
        if pinned_descriptor and descriptor == pinned_descriptor[0]:
            raise OSError("injected ambiguous close completion")

    monkeypatch.setattr(binding, "_open_output_parent", remember_parent)
    monkeypatch.setattr(binding.os, "close", close_then_report_ambiguous)

    receipt = binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert receipt.output_path == str(output)
    assert output.is_file()


@pytest.mark.parametrize(
    "document,match",
    [
        (
            """schema_version: crucible.backend-routing.v1
default: {backend: codex, model: gpt-6-astra, effort: max}
roles: {}
""",
            "not an exact DeepSeek route",
        ),
        (
            """schema_version: crucible.backend-routing.v1
default:
  backend: deepseek
  model: deepseek-flash
  feature_admission: diagnostic
roles:
  architect_critic: {backend: codex, model: gpt-6-astra, effort: max}
""",
            "mixed templates are refused",
        ),
        (
            """schema_version: crucible.backend-routing.v1
schema_version: crucible.backend-routing.v1
default: {backend: deepseek, model: deepseek-flash, feature_admission: diagnostic}
roles: {}
""",
            "unambiguous UTF-8 YAML",
        ),
        (
            f"""schema_version: crucible.backend-routing.v1
default:
  backend: deepseek
  model: deepseek-flash
  feature_admission: diagnostic
  extra: {{expected_worker_runtime_fingerprint: {_DIGEST}}}
roles: {{}}
""",
            "already bound",
        ),
        (
            """schema_version: crucible.backend-routing.v1
default: {backend: deepseek, model: deepseek-flash}
roles: {}
""",
            "feature_admission='diagnostic'",
        ),
    ],
)
def test_non_deepseek_mixed_ambiguous_bound_or_enforcing_template_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: str,
    match: str,
) -> None:
    calls = _capture_counter(monkeypatch)
    template = tmp_path / "template.yaml"
    template.write_text(document, encoding="utf-8")
    output = tmp_path / "bound.yaml"

    with pytest.raises(binding.DeepSeekConfigBindingError, match=match):
        binding.bind_deepseek_backend_config(template, output)

    assert not output.exists()
    assert calls == {"count": 0}


def test_relative_output_and_symlinked_parent_are_refused_before_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    with pytest.raises(binding.DeepSeekConfigBindingError, match="absolute normalized"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], Path("bound.yaml"))

    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(binding.DeepSeekConfigBindingError, match="non-symlinked"):
        binding.bind_deepseek_backend_config(
            _TEMPLATES[0], linked_parent / "bound.yaml"
        )

    assert calls == {"count": 0}
    assert not (real_parent / "bound.yaml").exists()


def test_symlinked_template_and_output_are_refused_without_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_counter(monkeypatch)
    linked_template = tmp_path / "template.yaml"
    linked_template.symlink_to(_TEMPLATES[0])
    output = tmp_path / "bound.yaml"
    with pytest.raises(binding.DeepSeekConfigBindingError, match="non-symlinked"):
        binding.bind_deepseek_backend_config(linked_template, output)

    target = tmp_path / "operator.yaml"
    target.write_bytes(b"operator-owned\n")
    output.symlink_to(target)
    with pytest.raises(binding.DeepSeekConfigBindingError, match="already exists"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert calls == {"count": 0}
    assert output.is_symlink()
    assert target.read_bytes() == b"operator-owned\n"


def test_failed_post_publication_validation_removes_only_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    real_verify = binding._verify_bound_bytes
    calls = {"count": 0}

    def fail_second_verify(*args: object, **kwargs: object) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise binding.DeepSeekConfigBindingError("injected post-write failure")
        real_verify(*args, **kwargs)

    monkeypatch.setattr(binding, "_verify_bound_bytes", fail_second_verify)
    with pytest.raises(binding.DeepSeekConfigBindingError, match="post-write"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".crucible-deepseek-bind-*.tmp"))


def test_parent_swap_after_publication_fails_and_cleans_through_held_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    moved_parent = tmp_path.with_name(tmp_path.name + "-moved")
    real_publish = binding._write_atomic_exclusive

    def publish_then_swap(*args: object, **kwargs: object):
        publication = real_publish(*args, **kwargs)
        tmp_path.rename(moved_parent)
        tmp_path.mkdir()
        return publication

    monkeypatch.setattr(binding, "_write_atomic_exclusive", publish_then_swap)
    try:
        with pytest.raises(binding.DeepSeekConfigBindingError, match="routing path"):
            binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

        assert not output.exists()
        assert not (moved_parent / "bound.yaml").exists()
    finally:
        tmp_path.rmdir()
        moved_parent.rename(tmp_path)


def test_parent_swap_during_capture_is_rejected_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bound.yaml"
    moved_parent = tmp_path.with_name(tmp_path.name + "-moved")
    calls = {"count": 0}

    def capture_then_swap() -> SimpleNamespace:
        calls["count"] += 1
        tmp_path.rename(moved_parent)
        tmp_path.mkdir()
        return SimpleNamespace(runtime_fingerprint=_DIGEST)

    monkeypatch.setattr(
        binding,
        "capture_project_locked_deepseek_runtime_identity",
        capture_then_swap,
    )
    try:
        with pytest.raises(binding.DeepSeekConfigBindingError, match="parent path"):
            binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

        assert calls == {"count": 1}
        assert not output.exists()
        assert not (moved_parent / "bound.yaml").exists()
        assert not tuple(moved_parent.glob(".crucible-deepseek-bind-*.tmp"))
    finally:
        tmp_path.rmdir()
        moved_parent.rename(tmp_path)


def test_parent_swap_after_second_verification_fails_at_success_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture_counter(monkeypatch)
    output = tmp_path / "bound.yaml"
    moved_parent = tmp_path.with_name(tmp_path.name + "-moved")
    real_verify = binding._verify_bound_bytes
    calls = {"count": 0}

    def verify_then_swap(*args: object, **kwargs: object) -> None:
        real_verify(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 2:
            tmp_path.rename(moved_parent)
            tmp_path.mkdir()

    monkeypatch.setattr(binding, "_verify_bound_bytes", verify_then_swap)
    try:
        with pytest.raises(binding.DeepSeekConfigBindingError, match="routing path"):
            binding.bind_deepseek_backend_config(_TEMPLATES[0], output)

        assert calls == {"count": 2}
        assert not output.exists()
        assert not (moved_parent / "bound.yaml").exists()
        assert not tuple(moved_parent.glob(".crucible-deepseek-bind-*.tmp"))
    finally:
        tmp_path.rmdir()
        moved_parent.rename(tmp_path)


def test_capture_failure_or_malformed_identity_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "bound.yaml"

    def fail_capture() -> None:
        raise DeepSeekSDKUnavailable("worker is not installed")

    monkeypatch.setattr(
        binding,
        "capture_project_locked_deepseek_runtime_identity",
        fail_capture,
    )
    with pytest.raises(binding.DeepSeekConfigBindingError, match="capture failed"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)
    assert not output.exists()

    monkeypatch.setattr(
        binding,
        "capture_project_locked_deepseek_runtime_identity",
        lambda: SimpleNamespace(runtime_fingerprint="sha256:BAD"),
    )
    with pytest.raises(binding.DeepSeekConfigBindingError, match="malformed"):
        binding.bind_deepseek_backend_config(_TEMPLATES[0], output)
    assert not output.exists()
