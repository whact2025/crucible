# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._build_verify``.

Stubs the subprocess layer by providing tiny runner modules under
``tests/fixtures/build_verify_runners/`` and parameterizing the Spec's
``project_context.runners`` to point at them. No live build is invoked.

Coverage:
- ``build_and_verify`` returns the right classification for each path:
  - success (build 0 + verify 0)
  - build_failed (build non-zero)
  - verify_failed (build 0, verify non-zero)
  - partial (verify 0, tests_failed > 0)
  - runner_missing (no build runner in runners dict)
- ``_validate_runner_path`` rejects shell-meta / non-identifier paths.
- The helper writes ``<log_dir>/build.log`` and ``<log_dir>/verify.log``.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import textwrap
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

from assay import (
    ImplementationDirective,
    ProjectContext,
    Slice,
    Spec,
    Tier,
    TierDAG,
)
from crucible.common.runners.capabilities import (
    DECLARED_VERIFICATION_CHECKPOINT_V1,
    EXACT_PYTEST_OBSERVATION_V1,
    trusted_runner_has_capability,
)
from crucible.workflows import _build_verify as build_verify_module
from crucible.workflows._build_verify import (
    _run_runner,
    _validate_runner_path,
    build_and_verify,
    runner_parameters,
    trusted_runner_overrides,
)


_DEEPSWE_DEV_VERIFY_RUNNER = (
    "crucible.products.deepswe.runners.deepswe_dev_verify"
)
_SWEBENCH_DEV_VERIFY_RUNNER = (
    "crucible.products.swebench.runners.swebench_pytest"
)


class _MemoryWriterEpochStorage:
    """Small exact-CAS epoch adapter for runner-boundary tests."""

    def __init__(
        self,
        transaction_id: str = "runner-journal",
        *,
        parent_transaction_id: str | None = None,
    ) -> None:
        from crucible.workflows._writer_epoch import (
            initial_writer_epoch_record,
        )

        self.transaction_id = transaction_id
        self.parent_transaction_id = parent_transaction_id
        self.active = True
        self.record = initial_writer_epoch_record()

    def read_writer_epoch_record(self) -> object:
        return copy.deepcopy(self.record)

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        if self.record != expected:
            raise RuntimeError("writer epoch compare-and-write conflict")
        self.record = copy.deepcopy(replacement)
        return copy.deepcopy(self.record)


class _FileWriterEpochStorage:
    """Fork-visible epoch storage for real controller-death coverage."""

    def __init__(self, path: Path) -> None:
        from crucible.workflows._writer_epoch import initial_writer_epoch_record

        self.transaction_id = "forked-runner-journal"
        self.parent_transaction_id = None
        self.active = True
        self.path = path
        self.path.write_text(
            json.dumps(initial_writer_epoch_record(), sort_keys=True),
            encoding="ascii",
        )

    def read_writer_epoch_record(self) -> object:
        return json.loads(self.path.read_text(encoding="ascii"))

    def compare_and_write_writer_epoch_record(
        self,
        expected: object,
        replacement: dict[str, object],
    ) -> object:
        current = self.read_writer_epoch_record()
        if current != expected:
            raise RuntimeError("writer epoch compare-and-write conflict")
        replacement_path = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.replacement"
        )
        replacement_path.write_text(
            json.dumps(replacement, sort_keys=True),
            encoding="ascii",
        )
        os.replace(replacement_path, self.path)
        return self.read_writer_epoch_record()

# ----------------------------------------------------------------------
# Stub runner modules — written to a tmp dir + injected onto sys.path
# ----------------------------------------------------------------------


_RUNNER_OK = '''\
import json, sys
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
sys.exit(0)
'''

_RUNNER_OK_TRAILING_NEWLINE = '''\
import json, sys
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}) + "\\n")
sys.exit(0)
'''

_RUNNER_FAIL = '''\
import json, sys
sys.stderr.write("build broke\\n")
sys.exit(2)
'''

_RUNNER_PARTIAL = '''\
import json, sys
sys.stdout.write(json.dumps({"values": {"tests_passed": 3.0, "tests_failed": 1.0}}))
sys.exit(0)
'''

_RUNNER_ASSERT_HOOKS_DISABLED = '''\
import json, os, sys
assert os.environ["GIT_CONFIG_COUNT"] == "1"
assert os.environ["GIT_CONFIG_KEY_0"] == "core.hooksPath"
assert os.environ["GIT_CONFIG_VALUE_0"] == "/dev/null"
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
'''

_RUNNER_ACKNOWLEDGE_BENCH = '''\
import json, sys
cfg = json.load(sys.stdin)
params = cfg["parameters"]
assert params["role"] == "verify"
assert params["runner_mode"] == "bench"
sys.stdout.write(json.dumps({
    "values": {"returncode": 0.0},
    "runner_mode": "bench",
}))
'''

_RUNNER_ASSERT_SUPERVISED_TOPOLOGY = '''\
import json, os, sys
cfg = json.load(sys.stdin)
params = cfg["parameters"]
assert params["role"] == "build"
assert os.getpid() == os.getpgrp()
assert os.getpid() == os.getsid(0)
assert os.getppid() > 1
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
'''

_RUNNER_DOUBLE_FORK_LATE_WRITE = '''\
import json, os, sys, time
from pathlib import Path

cfg = json.load(sys.stdin)
params = cfg["parameters"]
first = os.fork()
if first == 0:
    os.setsid()
    grandchild = os.fork()
    if grandchild > 0:
        os._exit(0)
    Path(params["background_pid_path"]).write_text(
        str(os.getpid()), encoding="utf-8"
    )
    time.sleep(0.75)
    Path(params["late_write_path"]).write_text("late\\n", encoding="utf-8")
    os._exit(0)
os.waitpid(first, 0)
deadline = time.monotonic() + 2
while not Path(params["background_pid_path"]).exists():
    if time.monotonic() >= deadline:
        raise RuntimeError("detached grandchild did not publish its pid")
    time.sleep(0.005)
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
'''

_RUNNER_PREEXITED_ZOMBIE = '''\
import json, os, sys, time
from pathlib import Path

cfg = json.load(sys.stdin)
params = cfg["parameters"]
child = os.fork()
if child == 0:
    os._exit(0)
deadline = time.monotonic() + 2
while time.monotonic() < deadline:
    raw = Path(f"/proc/{child}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    fields = raw[close + 2:].split() if close >= 0 else []
    if fields and fields[0] == "Z":
        break
    time.sleep(0.001)
else:
    raise RuntimeError("child did not become a zombie")
Path(params["zombie_pid_path"]).write_text(str(child), encoding="ascii")
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
'''

_RUNNER_STUBBORN_CHILD = '''\
import os, signal, sys, time
pid_path, term_path = sys.argv[1:3]
staged_term_path = f"{term_path}.staged.{os.getpid()}"
with open(staged_term_path, "x", encoding="utf-8") as stream:
    stream.write("term received\\n")
term_requested = False

def on_term(_signum, _frame):
    global term_requested
    if term_requested:
        return
    term_requested = True
    os.replace(staged_term_path, term_path)

signal.signal(signal.SIGTERM, on_term)
with open(pid_path, "w", encoding="utf-8") as stream:
    stream.write(str(os.getpid()))
while True:
    time.sleep(1)
'''

_RUNNER_STUBBORN_TREE = '''\
import json, os, signal, subprocess, sys, time
cfg = json.load(sys.stdin)
params = cfg["parameters"]
staged_term_path = (
    f'{params["runner_term_path"]}.staged.{os.getpid()}'
)
with open(staged_term_path, "x", encoding="utf-8") as stream:
    stream.write("term received\\n")
term_requested = False

def on_term(_signum, _frame):
    global term_requested
    if term_requested:
        return
    term_requested = True
    os.replace(staged_term_path, params["runner_term_path"])

signal.signal(signal.SIGTERM, on_term)
with open(params["runner_pid_path"], "w", encoding="utf-8") as stream:
    stream.write(str(os.getpid()))
child = subprocess.Popen([
    sys.executable,
    "-m",
    "stub_runners.stubborn_child",
    params["child_pid_path"],
    params["child_term_path"],
], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
   stderr=subprocess.DEVNULL)
deadline = time.monotonic() + 5
while not os.path.exists(params["child_pid_path"]):
    if time.monotonic() >= deadline:
        raise RuntimeError("stubborn child did not start")
    time.sleep(0.01)
sys.stdout.write("runner-started\\n")
sys.stdout.flush()
sys.stderr.write("runner-stderr\\n")
sys.stderr.flush()
while True:
    if term_requested and child.poll() is not None:
        raise SystemExit(0)
    time.sleep(0.01)
'''

_RUNNER_OVERSIZED_STREAMS = '''\
import json, sys
json.load(sys.stdin)
sys.stdout.buffer.write(
    b"stdout-head\\n" + b"x" * (9 * 1024 * 1024) + b"\\nstdout-tail\\n"
)
sys.stdout.buffer.flush()
sys.stderr.buffer.write(
    b"stderr-head\\n" + b"y" * (2 * 1024 * 1024) + b"\\nstderr-tail\\n"
)
sys.stderr.buffer.flush()
'''

_RUNNER_RECURSIVE_JSON = '''\
import json, sys
json.load(sys.stdin)
sys.stdout.write("[" * 50000 + "0" + "]" * 50000)
'''

_RUNNER_INVALID_UTF8 = '''\
import json, sys
json.load(sys.stdin)
sys.stdout.buffer.write(b"\\xff\\xfe")
'''

_RUNNER_LOG_THEN_JSON = '''\
import json, sys
json.load(sys.stdin)
sys.stdout.write("runner diagnostic\\n")
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}) + "\\n")
'''

_RUNNER_OVERSIZED_STDERR = '''\
import json, sys
json.load(sys.stdin)
sys.stdout.write(json.dumps({"values": {"returncode": 0.0}}))
sys.stdout.flush()
sys.stderr.buffer.write(
    b"stderr-head\\n" + b"y" * (2 * 1024 * 1024) + b"\\nstderr-tail\\n"
)
sys.stderr.buffer.flush()
'''


@pytest.fixture
def runner_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write runner modules and bind their parent as an explicit import root.

    The production boundary invokes configured non-Crucible runners in a fresh
    interpreter.  Supplying the fixture root through ``PYTHONPATH`` makes that
    child authority explicit and keeps the tests valid when the parent uses
    ``PYTHONSAFEPATH=1`` instead of implicitly trusting the child working
    directory.
    """
    pkg = tmp_path / "stub_runners"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "ok.py").write_text(textwrap.dedent(_RUNNER_OK))
    (pkg / "ok_trailing_newline.py").write_text(
        textwrap.dedent(_RUNNER_OK_TRAILING_NEWLINE)
    )
    (pkg / "fail.py").write_text(textwrap.dedent(_RUNNER_FAIL))
    (pkg / "partial.py").write_text(textwrap.dedent(_RUNNER_PARTIAL))
    (pkg / "hooks_disabled.py").write_text(
        textwrap.dedent(_RUNNER_ASSERT_HOOKS_DISABLED)
    )
    (pkg / "bench.py").write_text(
        textwrap.dedent(_RUNNER_ACKNOWLEDGE_BENCH)
    )
    (pkg / "session.py").write_text(
        textwrap.dedent(_RUNNER_ASSERT_SUPERVISED_TOPOLOGY)
    )
    (pkg / "double_fork_late_write.py").write_text(
        textwrap.dedent(_RUNNER_DOUBLE_FORK_LATE_WRITE)
    )
    (pkg / "preexited_zombie.py").write_text(
        textwrap.dedent(_RUNNER_PREEXITED_ZOMBIE)
    )
    (pkg / "stubborn_child.py").write_text(
        textwrap.dedent(_RUNNER_STUBBORN_CHILD)
    )
    (pkg / "stubborn_tree.py").write_text(
        textwrap.dedent(_RUNNER_STUBBORN_TREE)
    )
    (pkg / "oversized.py").write_text(
        textwrap.dedent(_RUNNER_OVERSIZED_STREAMS)
    )
    (pkg / "recursive_json.py").write_text(
        textwrap.dedent(_RUNNER_RECURSIVE_JSON)
    )
    (pkg / "invalid_utf8.py").write_text(
        textwrap.dedent(_RUNNER_INVALID_UTF8)
    )
    (pkg / "log_then_json.py").write_text(
        textwrap.dedent(_RUNNER_LOG_THEN_JSON)
    )
    (pkg / "oversized_stderr.py").write_text(
        textwrap.dedent(_RUNNER_OVERSIZED_STDERR)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            value
            for value in (str(tmp_path), inherited_pythonpath)
            if value
        ),
    )
    return "stub_runners"


def _spec(
    runners: dict[str, str],
    *,
    parameters: dict[str, object] | None = None,
) -> Spec:
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="bv_test",
            repo_url="https://example.invalid/bv.git",
            base_commit="0" * 40,
            branch=None,
            submodule_path=None,
            runners=runners,
            runner_parameters=parameters or {},
        ),
        execution=None,
        intent="t",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )


def test_spec_lineage_uses_persisted_digest_for_legacy_raw_manifest(
    tmp_path: Path,
) -> None:
    """Hydration defaults cannot replace the predecessor's persisted hash."""
    import hashlib

    from assay import ContentHash, PlanPolicy
    from assay.canonical import parse, serialize
    from assay.corpus.store import Corpus

    corpus = Corpus(tmp_path / "legacy-lineage")
    root = replace(_spec({}), plan_policy=PlanPolicy())
    legacy_payload = parse(serialize(root))
    del legacy_payload["plan_policy"][
        "max_verification_only_empty_scope_slices"
    ]
    legacy_blob = serialize(legacy_payload)
    legacy_short = hashlib.sha256(legacy_blob).hexdigest()[:12]
    (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{legacy_short}.json"
    ).write_bytes(legacy_blob)

    legacy = corpus.latest_spec()
    assert legacy is not None
    staged_legacy = legacy.with_appended_refs()
    assert build_verify_module._validate_spec_lineage(
        corpus,
        staged_legacy,
        allow_staged_tip=True,
    ) == (True, "")
    assert build_verify_module._validate_spec_lineage(
        corpus,
        staged_legacy,
    )[0] is False
    staged_with_implementation = replace(
        staged_legacy,
        implementation_refs=(ContentHash("sha256:" + "a" * 64),),
    )
    accepted, reason = build_verify_module._validate_spec_lineage(
        corpus,
        staged_with_implementation,
        allow_staged_tip=True,
    )
    assert accepted is False
    assert "workflow-owned implementation_refs" in reason
    corpus.put_spec(legacy.with_appended_refs())
    latest = corpus.latest_spec()
    assert latest is not None

    assert build_verify_module._validate_spec_lineage(corpus, latest) == (
        True,
        "",
    )


def test_spec_bound_runner_parameters_are_generic_and_deep_merged():
    spec = _spec(
        {},
        parameters={
            "per_language": {
                "python": {"test_args": ["-q"]},
                "rust": {
                    "crate": "fabric-render-ovrtx",
                    "cargo_features": ["ovrtx"],
                },
            },
        },
    )

    merged = runner_parameters(
        spec,
        {"swebench_metadata": {"instance_id": "x"}},
        languages=("rust",),
    )
    assert merged == {
        "languages": ["rust"],
        "per_language": {
            "python": {"test_args": ["-q"]},
            "rust": {
                "crate": "fabric-render-ovrtx",
                "cargo_features": ["ovrtx"],
            },
        },
        "swebench_metadata": {"instance_id": "x"},
    }


def test_spec_bound_runner_parameters_apply_slice_before_language_projection():
    spec = _spec(
        {},
        parameters={
            "per_language": {
                "python": {"test_args": ["-q"]},
                "rust": {
                    "project_subdir": "native/spikes/rust",
                    "cargo_features": ["trace"],
                },
            },
            "per_slice": {
                "runtime": {
                    "per_language": {
                        "rust": {"project_subdir": "native/runtime-rust"},
                    },
                },
            },
        },
    )

    assert runner_parameters(
        spec,
        languages=("rust",),
        slice_id="runtime",
    ) == {
        "languages": ["rust"],
        "per_language": {
            "python": {"test_args": ["-q"]},
            "rust": {
                "project_subdir": "native/runtime-rust",
                "cargo_features": ["trace"],
            },
        },
    }


def test_exact_pytest_observation_capability_is_closed_to_bundled_runners():
    capable = {
        "crucible.common.runners.lang.python",
        "crucible.common.runners.polyglot",
    }
    assert all(
        trusted_runner_has_capability(
            runner,
            EXACT_PYTEST_OBSERVATION_V1,
        )
        for runner in capable
    )
    assert not trusted_runner_has_capability(
        "third_party.lookalike_python_runner",
        EXACT_PYTEST_OBSERVATION_V1,
    )
    assert not trusted_runner_has_capability(
        "crucible.products.deepswe.runners.deepswe_dev_verify",
        EXACT_PYTEST_OBSERVATION_V1,
    )
    assert not trusted_runner_has_capability(
        "crucible.products.swebench.runners.swebench_pytest",
        EXACT_PYTEST_OBSERVATION_V1,
    )


@pytest.mark.parametrize(
    "parameters,match",
    [
        ({"per_slice": []}, "must be a mapping"),
        ({"per_slice": {"": {}}}, "non-empty string keys"),
        ({"per_slice": {"s0": []}}, "must be a mapping"),
        (
            {"per_slice": {"s0": {"languages": ["rust"]}}},
            "workflow-owned controls",
        ),
        (
            {"per_slice": {"s0": {"per_slice": {"s1": {}}}}},
            "may not nest",
        ),
        (
            {
                "per_slice": {
                    "s0": {
                        "per_language": {
                            "rust": {"per_slice": {"s1": {}}},
                        },
                    },
                },
            },
            "may not nest",
        ),
    ],
)
def test_spec_bound_runner_parameters_reject_malformed_slice_overlays(
    parameters,
    match,
):
    with pytest.raises(ValueError, match=match):
        runner_parameters(_spec({}, parameters=parameters), slice_id="s0")


def test_spec_bound_runner_parameters_reject_unknown_planned_slice_overlay():
    directive = ImplementationDirective(
        slice_id="known",
        intent="test",
        files_in_scope=(),
        creates_paths=("src/lib.rs",),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    spec = replace(
        _spec({}, parameters={"per_slice": {"unknown": {}}}),
        tier_dag=TierDAG(tiers=(Tier(
            tier_id="t0",
            slices=(Slice(id="known", directive=directive),),
            depends_on=(),
        ),)),
    )

    with pytest.raises(ValueError, match="unknown canonical slice ids"):
        runner_parameters(spec, slice_id="known")


@pytest.mark.parametrize("gate", ["build", "verify", "dev_verify"])
def test_slice_overlay_reaches_each_runner_dispatch(
    monkeypatch,
    tmp_path: Path,
    gate: str,
):
    captured = {}

    def fake_runner(role, runner, payload, **_kwargs):
        captured[role] = payload["parameters"]
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    directive = ImplementationDirective(
        slice_id="runtime",
        intent="test per-slice routing",
        files_in_scope=(),
        creates_paths=("native/runtime-rust/src/lib.rs",),
        verify_mode="tests",
        blast_radius={"max_files": 1, "languages": ["rust"]},
    )
    spec = _spec(
        {gate: "tests.fixtures.runner"},
        parameters={
            "per_language": {
                "rust": {"project_subdir": "native/spikes/rust"},
            },
            "per_slice": {
                "runtime": {
                    "per_language": {
                        "rust": {"project_subdir": "native/runtime-rust"},
                    },
                },
            },
        },
    )

    function = getattr(build_verify_module, gate)
    result = function(
        spec,
        slice_id="runtime",
        cwd=str(tmp_path),
        slice_directive=directive,
    )

    assert result is not None
    assert captured[gate]["languages"] == ["rust"]
    assert captured[gate]["per_language"]["rust"]["project_subdir"] == (
        "native/runtime-rust"
    )
    assert "per_slice" not in captured[gate]


def test_empty_spec_runner_parameters_preserve_runtime_metadata():
    spec = _spec({})
    assert runner_parameters(spec, {"swebench_metadata": {"id": "x"}}) == {
        "swebench_metadata": {"id": "x"},
    }


def test_runtime_metadata_cannot_replace_resolved_languages():
    spec = _spec({})
    with pytest.raises(ValueError, match="workflow-owned"):
        runner_parameters(
            spec,
            trusted_runner_overrides({"languages": ["python"]}),
            languages=("rust",),
        )


def test_runtime_metadata_cannot_override_spec_bound_selection():
    spec = _spec(
        {},
        parameters={
            "per_language": {"rust": {"crate": "fabric-render-ovrtx"}},
        },
    )

    with pytest.raises(ValueError, match="may not override canonical Spec"):
        runner_parameters(
            spec,
            trusted_runner_overrides({
                "per_language": {"rust": {"crate": "different-crate"}},
            }),
        )


@pytest.mark.parametrize(
    "runtime",
    [
        {"per_language": {"rust": {"crate": "different-crate"}}},
        {"test_args": ["--test", "only-this-target"]},
        {"working_dir": "/unlocked/checkout"},
    ],
)
def test_public_runtime_envelope_accepts_metadata_namespaces_only(
    runtime,
):
    spec = _spec({})
    with pytest.raises(ValueError, match="metadata namespaces only"):
        runner_parameters(spec, runtime)


def test_spec_runner_parameters_reject_workflow_owned_controls():
    spec = _spec(
        {},
        parameters={
            "per_language": {
                "rust": {
                    "role": "build",
                    "runner_mode": "bench",
                    "breach_observations": False,
                },
            },
        },
    )
    with pytest.raises(ValueError, match="workflow-owned controls"):
        runner_parameters(spec)


def test_spec_runner_parameters_reject_workflow_owned_languages():
    spec = _spec({}, parameters={"languages": ["python"]})

    with pytest.raises(ValueError, match="workflow-owned"):
        runner_parameters(spec)


def test_spec_runner_parameters_reject_declared_surface_spoofing():
    spec = _spec(
        {},
        parameters={
            "declared_verification_surface": {
                "schema_version": "crucible.declared-verification-surface.v1",
                "slices": [],
            },
        },
    )
    with pytest.raises(ValueError, match="workflow-owned controls"):
        runner_parameters(spec)


def test_spec_runner_parameters_reject_accepted_reference_spoofing():
    spec = _spec(
        {},
        parameters={
            "accepted_declared_verification_reference": {
                "schema_version": "attacker-controlled",
            },
        },
    )
    with pytest.raises(ValueError, match="workflow-owned controls"):
        runner_parameters(spec)


def test_spec_runner_parameters_reject_recursive_language_envelope():
    spec = _spec(
        {},
        parameters={
            "per_language": {
                "rust": {
                    "crate": "expected",
                    "per_language": {
                        "rust": {"crate": "unexpected"},
                    },
                },
            },
        },
    )

    with pytest.raises(ValueError, match="may not nest"):
        runner_parameters(spec)


def test_language_projection_accepts_frozen_mapping_and_tuple_values():
    from types import MappingProxyType

    from crucible.common.runners.parameters import for_language, validate_static

    configured = MappingProxyType({
        "per_language": MappingProxyType({
            "rust": MappingProxyType({
                "cargo_features": ("ovrtx",),
            }),
        }),
    })

    validate_static(configured)
    assert for_language(configured, "rust") == {
        "cargo_features": ("ovrtx",),
    }


def test_project_subdir_narrows_runner_beneath_workflow_root(tmp_path: Path):
    from crucible.common.runners.parameters import resolve_project_working_dir

    project = tmp_path / "native" / "spike"
    project.mkdir(parents=True)

    assert resolve_project_working_dir(
        {"project_subdir": "native/spike"},
        process_cwd=tmp_path,
    ) == project.resolve()


@pytest.mark.parametrize("subdir", ["../escape", "/absolute", ""])
def test_project_subdir_rejects_unconfined_values(
    tmp_path: Path,
    subdir: str,
):
    from crucible.common.runners.parameters import resolve_project_working_dir

    with pytest.raises(ValueError, match="project_subdir"):
        resolve_project_working_dir(
            {"project_subdir": subdir},
            process_cwd=tmp_path,
        )


def test_project_subdir_rejects_symlink_escape(tmp_path: Path):
    from crucible.common.runners.parameters import resolve_project_working_dir

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        resolve_project_working_dir(
            {"project_subdir": "linked"},
            process_cwd=tmp_path,
        )


# ----------------------------------------------------------------------
# Classification matrix
# ----------------------------------------------------------------------


def test_success_path(runner_pkg: str, tmp_path: Path):
    spec = _spec({
        "build": f"{runner_pkg}.ok",
        "verify": f"{runner_pkg}.ok",
    })
    outcome = build_and_verify(
        spec, slice_id="s0", cwd=str(tmp_path),
        log_dir=tmp_path / "logs",
    )
    assert outcome.classification == "success"
    assert outcome.build is not None and outcome.build.ok
    assert outcome.verify is not None and outcome.verify.ok
    # Logs were written.
    assert (tmp_path / "logs" / "build.log").is_file()
    assert (tmp_path / "logs" / "verify.log").is_file()
    assert len(tuple((tmp_path / "logs").glob("build_*.log"))) == 1
    assert len(tuple((tmp_path / "logs").glob("verify_*.log"))) == 1


def _directive(verify_mode: str) -> ImplementationDirective:
    return ImplementationDirective(
        slice_id="s0",
        intent="exercise verify routing",
        files_in_scope=(),
        creates_paths=(),
        verify_mode=verify_mode,
        blast_radius={},
    )


def test_bench_mode_is_workflow_owned_and_acknowledged(
    runner_pkg: str,
    tmp_path: Path,
):
    outcome = build_and_verify(
        _spec({
            "build": f"{runner_pkg}.ok",
            "verify": f"{runner_pkg}.bench",
        }),
        slice_id="s0",
        cwd=str(tmp_path),
        slice_directive=_directive("bench"),
    )
    assert outcome.classification == "success"
    assert outcome.verify is not None and outcome.verify.ok


def test_bench_mode_fails_closed_when_adapter_silently_runs_default(
    runner_pkg: str,
    tmp_path: Path,
):
    outcome = build_and_verify(
        _spec({
            "build": f"{runner_pkg}.ok",
            "verify": f"{runner_pkg}.ok",
        }),
        slice_id="s0",
        cwd=str(tmp_path),
        slice_directive=_directive("bench"),
    )
    assert outcome.classification == "verify_failed"
    assert outcome.verify is not None
    assert outcome.verify.returncode == 2
    assert b"did not acknowledge" in outcome.verify.stderr


def test_bench_mode_fails_closed_without_verify_adapter(
    runner_pkg: str,
    tmp_path: Path,
):
    outcome = build_and_verify(
        _spec({"build": f"{runner_pkg}.ok"}),
        slice_id="s0",
        cwd=str(tmp_path),
        slice_directive=_directive("bench"),
    )
    assert outcome.classification == "verify_failed"
    assert outcome.verify is not None
    assert outcome.verify.runner == "<unconfigured>"
    assert outcome.verify.process_settlement_digest is None


@pytest.mark.parametrize("verify_mode", ["skip", "build_only"])
def test_non_verifying_modes_still_skip_the_verify_adapter(
    runner_pkg: str,
    tmp_path: Path,
    verify_mode: str,
):
    outcome = build_and_verify(
        _spec({
            "build": f"{runner_pkg}.ok",
            "verify": f"{runner_pkg}.fail",
        }),
        slice_id="s0",
        cwd=str(tmp_path),
        slice_directive=_directive(verify_mode),
    )
    assert outcome.classification == "success"
    assert outcome.verify is None


def test_runner_subprocess_cannot_execute_repository_git_hooks(
    runner_pkg: str,
    tmp_path: Path,
):
    outcome = build_and_verify(
        _spec({"build": f"{runner_pkg}.hooks_disabled"}),
        slice_id="s0",
        cwd=str(tmp_path),
    )
    assert outcome.classification == "success"


def test_runner_subprocess_uses_isolated_supervised_group_and_session(
    runner_pkg: str,
    tmp_path: Path,
):
    result = _run_runner(
        "build",
        f"{runner_pkg}.session",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 0
    assert result.values == {"returncode": 0.0}
    assert result.stderr == b""
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        result.process_settlement_digest or "",
    )


@pytest.mark.parametrize("publication_count", [0, 2])
def test_runner_refuses_missing_or_ambiguous_process_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_count: int,
) -> None:
    def _unsettled_subprocess(*_args, **kwargs):
        callback = kwargs["settlement_callback"]
        for _index in range(publication_count):
            callback("sha256:" + "a" * 64)
        return subprocess.CompletedProcess(
            args=["unsettled"],
            returncode=0,
            stdout=b'{"values":{"returncode":0}}',
            stderr=b"",
        )

    monkeypatch.setattr(
        build_verify_module,
        "_run_runner_subprocess",
        _unsettled_subprocess,
    )

    with pytest.raises(
        build_verify_module._RunnerSettlementError,
        match="did not publish exactly one process settlement digest",
    ):
        _run_runner(
            "build",
            "trusted.runner",
            {"parameters": {}},
            cwd=str(tmp_path),
            timeout=5,
            log_dir=None,
        )


def test_runner_result_rejects_malformed_process_settlement_digest() -> None:
    with pytest.raises(ValueError, match="must be exact sha256"):
        build_verify_module.RunnerResult(
            role="build",
            runner="trusted.runner",
            returncode=0,
            values={},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
            process_settlement_digest="sha256:not-a-digest",
        )


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_normal_runner_double_fork_is_settled_and_fails_gate_before_late_write(
    runner_pkg: str,
    tmp_path: Path,
) -> None:
    background_pid_path = tmp_path / "background.pid"
    late_write_path = tmp_path / "late.txt"

    result = _run_runner(
        "verify",
        f"{runner_pkg}.double_fork_late_write",
        {
            "parameters": {
                "background_pid_path": str(background_pid_path),
                "late_write_path": str(late_write_path),
            },
        },
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 2
    assert b"background descendants outlived normal exit" in result.stderr
    assert re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        result.process_settlement_digest or "",
    )
    background_pid = int(background_pid_path.read_text(encoding="utf-8"))
    time.sleep(0.85)
    assert not late_write_path.exists()
    assert not psutil.pid_exists(background_pid)


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_preexited_zombie_is_reaped_without_false_background_failure(
    runner_pkg: str,
    tmp_path: Path,
) -> None:
    zombie_pid_path = tmp_path / "zombie.pid"

    result = _run_runner(
        "build",
        f"{runner_pkg}.preexited_zombie",
        {"parameters": {"zombie_pid_path": str(zombie_pid_path)}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 0
    assert result.values == {"returncode": 0.0}
    assert result.stderr == b""
    zombie_pid = int(zombie_pid_path.read_text(encoding="ascii"))
    assert not psutil.pid_exists(zombie_pid)


def test_runner_without_supervisor_control_proof_fails_closed(
    runner_pkg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    del runner_pkg
    from crucible.workflows._process_supervisor_contract import (
        capture_trusted_supervisor_source,
    )

    fake_supervisor = tmp_path / "fake_supervisor.py"
    fake_supervisor.write_text("pass\n", encoding="utf-8")
    fake_source = capture_trusted_supervisor_source(fake_supervisor)
    writer_epoch = _MemoryWriterEpochStorage()
    monkeypatch.setattr(
        build_verify_module,
        "acquire_trusted_supervisor_source",
        fake_source.acquire,
    )

    settlement_calls: list[str] = []
    with pytest.raises(
        build_verify_module._RunnerSettlementError,
        match="control record is unavailable",
    ):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "raise SystemExit(0)"],
            config_json=b"{}",
            cwd=str(tmp_path),
            timeout=1,
            env=os.environ.copy(),
            settlement_callback=settlement_calls.append,
            writer_epoch_members=(writer_epoch,),
        )
    assert settlement_calls == []
    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    assert inspect_writer_epoch_family((writer_epoch,)).state == "never_armed"


def test_trusted_supervisor_capture_rejects_compile_time_semantic_error(
    tmp_path: Path,
) -> None:
    from crucible.workflows._process_supervisor_contract import (
        ProcessSupervisorContractError,
        capture_trusted_supervisor_source,
    )

    invalid_source = tmp_path / "invalid_supervisor.py"
    invalid_source.write_text("return 1\n", encoding="utf-8")

    with pytest.raises(ProcessSupervisorContractError):
        capture_trusted_supervisor_source(invalid_source)


def test_queued_command_observer_has_runtime_handshake_guard() -> None:
    """The frozen supervisor must not optimize a trust check into assert."""

    from crucible.workflows import _process_supervisor as supervisor

    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    observers = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "observe_queued_commands"
    ]
    assert len(observers) == 1
    observer_source = ast.get_source_segment(source, observers[0])
    assert observer_source is not None
    assert "if handshake_launch_id is None:" in observer_source
    assert "command_protocol_failed = True\n            return" in observer_source
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(observers[0]))


@pytest.mark.skipif(sys.platform != "linux", reason="prctl requires Linux")
def test_runner_rejects_receipt_when_nondumpable_boundary_cannot_be_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No target starts and no epoch settles without control-fd isolation."""

    from crucible.workflows import _process_supervisor as supervisor
    from crucible.workflows._process_supervisor_contract import (
        capture_trusted_supervisor_source,
    )

    source = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert source.count("_PR_SET_DUMPABLE = 4") == 1
    rejected_source = tmp_path / "rejected_process_supervisor.py"
    rejected_source.write_text(
        source.replace(
            "_PR_SET_DUMPABLE = 4",
            "_PR_SET_DUMPABLE = 0x7fffffff",
            1,
        ),
        encoding="utf-8",
    )
    captured = capture_trusted_supervisor_source(rejected_source)
    marker = tmp_path / "target-started"
    writer_epoch = _MemoryWriterEpochStorage()
    settlement_calls: list[str] = []
    monkeypatch.setattr(
        build_verify_module,
        "acquire_trusted_supervisor_source",
        captured.acquire,
    )
    try:
        with pytest.raises(
            build_verify_module._RunnerSettlementError,
            match="did not prove launch readiness",
        ):
            build_verify_module._run_runner_subprocess(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib,sys; "
                        "pathlib.Path(sys.argv[1]).write_text('started')"
                    ),
                    str(marker),
                ],
                config_json=b"",
                cwd=str(tmp_path),
                timeout=2,
                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
                settlement_callback=settlement_calls.append,
                writer_epoch_members=(writer_epoch,),
            )
    finally:
        os.close(captured.descriptor)

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    assert not marker.exists()
    assert settlement_calls == []
    assert inspect_writer_epoch_family((writer_epoch,)).state == "never_armed"


@pytest.mark.skipif(sys.platform != "linux", reason="sealed memfd requires Linux")
def test_runner_launch_failure_before_ready_does_not_arm_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer_epoch = _MemoryWriterEpochStorage()
    settlement_calls: list[str] = []

    def fail_launch(*_args, **_kwargs):
        raise OSError("supervisor launch refused")

    monkeypatch.setattr(build_verify_module.subprocess, "Popen", fail_launch)

    with pytest.raises(OSError, match="supervisor launch refused"):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "raise SystemExit(0)"],
            config_json=b"{}",
            cwd=str(tmp_path),
            timeout=1,
            env=os.environ.copy(),
            settlement_callback=settlement_calls.append,
            writer_epoch_members=(writer_epoch,),
        )

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    assert settlement_calls == []
    assert inspect_writer_epoch_family((writer_epoch,)).state == "never_armed"


@pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG requires Linux")
def test_controller_sigkill_stops_direct_target_and_retains_armed_epoch(
    tmp_path: Path,
) -> None:
    """Controller death permits neither a late write nor false settlement."""

    epoch_storage = _FileWriterEpochStorage(tmp_path / "writer-epoch.json")
    identities_path = tmp_path / "runner-parent-death-identities"
    early_write_path = tmp_path / "runner-parent-death-early"
    late_write_path = tmp_path / "runner-parent-death-late"
    settlement_path = tmp_path / "runner-parent-death-settlement"
    target_script = """
import os
import sys
import time

with open(sys.argv[1], "w", encoding="ascii") as stream:
    stream.write("mutated-before-controller-death")
with open(sys.argv[2], "w", encoding="ascii") as stream:
    stream.write(f"{os.getpid()} {os.getppid()}")
time.sleep(1.5)
with open(sys.argv[3], "w", encoding="ascii") as stream:
    stream.write("escaped-after-controller-death")
time.sleep(30)
"""

    def run_controller() -> None:
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                target_script,
                str(early_write_path),
                str(identities_path),
                str(late_write_path),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=30,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")},
            settlement_callback=lambda digest: settlement_path.write_text(
                digest,
                encoding="ascii",
            ),
            writer_epoch_members=(epoch_storage,),
        )

    process_context = multiprocessing.get_context("fork")
    controller = process_context.Process(target=run_controller)
    controller.start()
    target_id: int | None = None
    supervisor_id: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if identities_path.exists():
                raw = identities_path.read_text(encoding="ascii").split()
                if len(raw) == 2:
                    target_id, supervisor_id = (int(item) for item in raw)
                    break
            if not controller.is_alive():
                pytest.fail("runner controller exited before target startup")
            time.sleep(0.01)

        assert target_id is not None
        assert supervisor_id is not None
        assert controller.pid is not None
        assert supervisor_id != controller.pid
        from crucible.workflows._writer_epoch import (
            inspect_writer_epoch_family,
        )

        assert inspect_writer_epoch_family((epoch_storage,)).state == "armed"
        os.kill(controller.pid, signal.SIGKILL)
        controller.join(timeout=3)
        assert controller.exitcode == -signal.SIGKILL

        cleanup_deadline = time.monotonic() + 5
        while time.monotonic() < cleanup_deadline:
            if (
                not Path(f"/proc/{target_id}").exists()
                and not Path(f"/proc/{supervisor_id}").exists()
            ):
                break
            time.sleep(0.01)

        assert not Path(f"/proc/{target_id}").exists()
        assert not Path(f"/proc/{supervisor_id}").exists()
        time.sleep(1.6)
        assert early_write_path.read_text(encoding="ascii") == (
            "mutated-before-controller-death"
        )
        assert not late_write_path.exists()
        assert not settlement_path.exists()
        assert inspect_writer_epoch_family((epoch_storage,)).state == "armed"
    finally:
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=3)
        for process_id in (target_id, supervisor_id):
            if process_id is None:
                continue
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_supervisor_signals_only_pidfd_pinned_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _process_supervisor as supervisor

    events: list[object] = []
    monkeypatch.setattr(
        supervisor,
        "_open_pidfd",
        lambda process_id: events.append(("open", process_id)) or 91,
    )
    monkeypatch.setattr(
        supervisor,
        "_process_start_time",
        lambda process_id: events.append(("stat", process_id)) or 700,
    )
    monkeypatch.setattr(
        supervisor,
        "_pidfd_send_signal",
        lambda descriptor, signal_number: events.append(
            ("send", descriptor, signal_number)
        ) or True,
    )
    monkeypatch.setattr(
        supervisor.os,
        "close",
        lambda descriptor: events.append(("close", descriptor)),
    )
    monkeypatch.setattr(
        supervisor.os,
        "kill",
        lambda *_args: pytest.fail("supervisor used a bare PID signal"),
    )

    supervisor._signal_identity((44, 700), signal.SIGTERM)

    assert events == [
        ("open", 44),
        ("stat", 44),
        ("send", 91, signal.SIGTERM),
        ("close", 91),
    ]


def test_supervisor_pidfd_recheck_rejects_recycled_diagnostic_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows import _process_supervisor as supervisor

    events: list[object] = []
    monkeypatch.setattr(supervisor, "_open_pidfd", lambda _process_id: 92)
    monkeypatch.setattr(supervisor, "_process_start_time", lambda _pid: 701)
    monkeypatch.setattr(
        supervisor,
        "_pidfd_send_signal",
        lambda *_args: pytest.fail("recycled identity was signaled"),
    )
    monkeypatch.setattr(
        supervisor.os,
        "close",
        lambda descriptor: events.append(("close", descriptor)),
    )

    supervisor._signal_identity((44, 700), signal.SIGKILL)

    assert events == [("close", 92)]


@pytest.mark.skipif(sys.platform != "linux", reason="directory fd requires Linux")
def test_supervisor_target_cwd_survives_path_swap_after_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows._process_supervisor_contract import (
        open_target_working_directory,
    )

    requested = tmp_path / "requested"
    pinned = tmp_path / "pinned-original"
    replacement = tmp_path / "replacement"
    requested.mkdir()
    replacement.mkdir()
    (requested / "identity.txt").write_text("original", encoding="ascii")
    (replacement / "identity.txt").write_text("replacement", encoding="ascii")
    captured_descriptor = open_target_working_directory(requested)
    swapped = False

    def lease_then_swap(_cwd) -> int:
        nonlocal swapped
        descriptor = os.dup(captured_descriptor)
        requested.rename(pinned)
        requested.symlink_to(replacement, target_is_directory=True)
        swapped = True
        return descriptor

    monkeypatch.setattr(
        build_verify_module,
        "open_target_working_directory",
        lease_then_swap,
    )
    try:
        completed = build_verify_module.run_supervised_process(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "print(Path('identity.txt').read_text(), end=''); "
                    "Path('target-write.txt').write_text('original-inode')"
                ),
            ],
            cwd=requested,
            capture_output=True,
            timeout=2,
        )
    finally:
        os.close(captured_descriptor)

    assert swapped
    assert completed.returncode == 0
    assert completed.stdout == b"original"
    assert (pinned / "target-write.txt").read_text(encoding="ascii") == (
        "original-inode"
    )
    assert not (replacement / "target-write.txt").exists()


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_settlement_digest_binds_exact_validated_control_record(
    tmp_path: Path,
) -> None:
    settlement_calls: list[str] = []

    completed = build_verify_module.run_supervised_process(
        [sys.executable, "-c", "raise SystemExit(0)"],
        cwd=tmp_path,
        capture_output=True,
        timeout=2,
        settlement_callback=settlement_calls.append,
    )

    assert completed.returncode == 0
    assert len(settlement_calls) == 1
    digest = settlement_calls[0]
    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_terminal_digest_binds_every_authenticated_field() -> None:
    launch_id = "runner-digest-binding"
    canonical_control = json.dumps(
        {
            "error": None,
            "epoch": None,
            "launch_id": launch_id,
            "outcome": "completed",
            "phase": "terminal",
            "returncode": 0,
            "schema_version": build_verify_module._RUNNER_SUPERVISOR_SCHEMA,
            "settled": True,
            "target_started": True,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    controller, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    try:
        peer.send(canonical_control)
        peer.close()
        returncode, error, outcome, authenticated = (
            build_verify_module._read_runner_supervisor_control(
                controller.fileno(),
                launch_id=launch_id,
                writer_epoch_lease=None,
            )
        )
    finally:
        controller.close()
        peer.close()
    assert (returncode, error, outcome) == (0, None, "completed")
    assert authenticated == canonical_control
    assert (
        hashlib.sha256(authenticated).hexdigest()
        == hashlib.sha256(canonical_control).hexdigest()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="seqpacket requires Linux")
@pytest.mark.parametrize(
    ("supervisor_returncode", "expected_error"),
    [
        (-9, "supervisor exited without trusted settlement (returncode=-9)"),
        (0, "runner supervisor control record is unavailable"),
    ],
)
def test_terminal_read_failure_reaps_supervisor_before_classifying_authority(
    supervisor_returncode: int,
    expected_error: str,
) -> None:
    """Direct-child failure outranks EOF; rc zero cannot hide missing proof."""

    class _ExitedSupervisor:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.wait_timeouts: list[float] = []

        def wait(self, timeout: float) -> int:
            self.wait_timeouts.append(timeout)
            self.returncode = supervisor_returncode
            return supervisor_returncode

    process = _ExitedSupervisor()
    controller, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    control_read = controller.detach()
    peer.close()
    settlement_calls: list[str] = []
    transaction = build_verify_module._RunnerSettlementTransaction(
        process,
        control_read,
        launch_id="runner-nonzero-precedence",
        writer_epoch_lease=None,
        settlement_owner_ids=(),
        settlement_worker_ids=(),
        settlement_callback=settlement_calls.append,
    )

    transaction._settle_from_terminal(command_already_sent=True)

    with pytest.raises(
        build_verify_module._RunnerSettlementError,
        match=re.escape(expected_error),
    ) as caught:
        transaction.join()
    assert process.wait_timeouts == [build_verify_module._RUNNER_REAP_GRACE_S]
    assert settlement_calls == []
    if supervisor_returncode != 0:
        assert isinstance(
            caught.value.__cause__,
            build_verify_module._RunnerSettlementError,
        )
        assert "control record is unavailable" in str(caught.value.__cause__)
    else:
        assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "wait_failure",
    [
        subprocess.TimeoutExpired("runner-supervisor", 1),
        OSError("injected wait failure"),
    ],
    ids=("timeout", "oserror"),
)
def test_terminal_read_failure_rereads_status_published_by_failing_wait(
    wait_failure: BaseException,
) -> None:
    """A wait failure cannot hide a concurrently published nonzero status."""

    class _PublishingFailedWait:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self, timeout: float) -> int:
            assert timeout == build_verify_module._RUNNER_REAP_GRACE_S
            self.returncode = -9
            raise wait_failure

    process = _PublishingFailedWait()

    assert build_verify_module._observe_nonzero_runner_supervisor_exit(process) == -9


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_settlement_callback_precedes_timeout_translation(tmp_path: Path) -> None:
    settlement_calls: list[str] = []

    with pytest.raises(subprocess.TimeoutExpired):
        build_verify_module.run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            capture_output=True,
            timeout=0.05,
            settlement_callback=settlement_calls.append,
        )

    assert len(settlement_calls) == 1
    assert settlement_calls[0].startswith("sha256:")


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_settlement_callback_precedes_background_failure_translation(
    tmp_path: Path,
) -> None:
    settlement_calls: list[str] = []
    completed = build_verify_module.run_supervised_process(
        [
            sys.executable,
            "-c",
            (
                "import os,time\n"
                "child=os.fork()\n"
                "if child == 0:\n"
                " os.setsid()\n"
                " grandchild=os.fork()\n"
                " if grandchild == 0:\n"
                "  time.sleep(10)\n"
                "  os._exit(0)\n"
                " os._exit(0)\n"
                "os.waitpid(child, 0)\n"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=2,
        settlement_callback=settlement_calls.append,
    )

    assert completed.returncode == 2
    assert len(settlement_calls) == 1
    assert settlement_calls[0].startswith("sha256:")
    assert b"background descendants" in completed.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_settlement_digests_distinguish_success_timeout_and_background(
    tmp_path: Path,
) -> None:
    digests: dict[str, str] = {}
    normal = build_verify_module.run_supervised_process(
        [sys.executable, "-c", "raise SystemExit(0)"],
        cwd=tmp_path,
        capture_output=True,
        timeout=2,
        settlement_callback=lambda digest: digests.__setitem__(
            "success", digest
        ),
    )
    assert normal.returncode == 0

    with pytest.raises(subprocess.TimeoutExpired):
        build_verify_module.run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            capture_output=True,
            timeout=0.05,
            settlement_callback=lambda digest: digests.__setitem__(
                "timeout", digest
            ),
        )

    background = build_verify_module.run_supervised_process(
        [
            sys.executable,
            "-c",
            (
                "import os,time\n"
                "child=os.fork()\n"
                "if child == 0:\n"
                " os.setsid()\n"
                " grandchild=os.fork()\n"
                " if grandchild == 0:\n"
                "  time.sleep(10)\n"
                "  os._exit(0)\n"
                " os._exit(0)\n"
                "os.waitpid(child, 0)\n"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=2,
        settlement_callback=lambda digest: digests.__setitem__(
            "background", digest
        ),
    )
    assert background.returncode == 2

    assert set(digests) == {"success", "timeout", "background"}
    assert len(set(digests.values())) == 3
    assert all(
        len(digest) == len("sha256:") + 64
        and digest.startswith("sha256:")
        for digest in digests.values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_each_supervised_subprocess_gets_a_fresh_quiesced_epoch_generation(
    tmp_path: Path,
) -> None:
    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    generations: list[int] = []
    epoch_ids: list[str] = []

    for _attempt in range(2):
        completed = build_verify_module.run_supervised_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            cwd=tmp_path,
            capture_output=True,
            timeout=2,
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )
        assert completed.returncode == 0
        inspection = inspect_writer_epoch_family((storage,))
        assert inspection.state == "quiesced"
        assert inspection.generation is not None
        assert inspection.epoch_id is not None
        generations.append(inspection.generation)
        epoch_ids.append(inspection.epoch_id)
        settlement = storage.record["settlement"]
        assert settlement["kind"] == "process_supervisor"
        assert settlement["evidence_digest"] == evidence[-1]

    assert generations == [1, 2]
    assert len(set(epoch_ids)) == 2
    assert len(evidence) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="sealed memfd requires Linux")
def test_supervisor_uses_fixed_env_and_target_receives_project_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader/project controls can reach only the supervised target."""

    real_popen = build_verify_module.subprocess.Popen
    supervisor_launches: list[dict[str, object]] = []

    def observed_popen(command, *args, **kwargs):
        if (
            isinstance(command, (list, tuple))
            and len(command) >= 4
            and tuple(command[:3]) == ("/proc/self/exe", "-I", "-S")
            and str(command[3]).startswith("/proc/self/fd/")
        ):
            supervisor_launches.append(dict(kwargs))
            assert kwargs["env"] == build_verify_module.supervisor_environment()
            assert "LD_PRELOAD" not in kwargs["env"]
            assert "PYTHONPATH" not in kwargs["env"]
            assert len(kwargs["pass_fds"]) == 4
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(build_verify_module.subprocess, "Popen", observed_popen)
    target_env = dict(os.environ)
    target_env.update(
        {
            "LD_PRELOAD": "/definitely/not/a/crucible-library.so",
            "PYTHONPATH": str(tmp_path / "hostile-python-path"),
            "CRUCIBLE_TARGET_SENTINEL": "target-only-value",
        }
    )
    completed = build_verify_module.run_supervised_process(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import json,os; links=[]; "
                "[(links.append(os.readlink('/proc/self/fd/'+fd)) if "
                "os.path.exists('/proc/self/fd/'+fd) else None) "
                "for fd in os.listdir('/proc/self/fd')]; "
                "print(json.dumps({'sentinel': "
                "os.environ['CRUCIBLE_TARGET_SENTINEL'], 'pythonpath': "
                "os.environ['PYTHONPATH'], 'source_leaked': "
                "any('crucible-process-supervisor-source' in item "
                "for item in links)}))"
            ),
        ],
        cwd=tmp_path,
        env=target_env,
        capture_output=True,
        timeout=5,
    )

    assert completed.returncode == 0
    payload = json.loads(completed.stdout)
    assert payload == {
        "sentinel": "target-only-value",
        "pythonpath": str(tmp_path / "hostile-python-path"),
        "source_leaked": False,
    }
    assert supervisor_launches


@pytest.mark.skipif(sys.platform != "linux", reason="/proc/self/exe requires Linux")
def test_supervisor_launch_ignores_mutable_sys_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_interpreter = sys.executable
    malicious_marker = tmp_path / "mutable-interpreter-ran"
    mutable_interpreter = tmp_path / "mutable-python"
    mutable_interpreter.write_text(
        f"#!/bin/sh\nprintf compromised > {malicious_marker}\n",
        encoding="ascii",
    )
    mutable_interpreter.chmod(0o700)
    monkeypatch.setattr(
        build_verify_module.sys,
        "executable",
        str(mutable_interpreter),
    )
    completed = build_verify_module.run_supervised_process(
        [target_interpreter, "-c", "print('kernel-bound-supervisor')"],
        cwd=tmp_path,
        capture_output=True,
        timeout=2,
    )

    assert completed.returncode == 0
    assert completed.stdout == b"kernel-bound-supervisor\n"
    assert not malicious_marker.exists()
    first_env = build_verify_module.supervisor_environment()
    second_env = build_verify_module.supervisor_environment()
    assert first_env == second_env
    assert first_env is not second_env
    first_env["MUTATED"] = "only-this-copy"
    assert "MUTATED" not in second_env


@pytest.mark.skipif(sys.platform != "linux", reason="sealed memfd requires Linux")
def test_replaced_supervisor_path_cannot_replace_trusted_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from crucible.workflows._process_supervisor_contract import (
        capture_trusted_supervisor_source,
    )

    trusted_path = tmp_path / "trusted_supervisor.py"
    trusted_path.write_bytes(build_verify_module._RUNNER_SUPERVISOR_PATH.read_bytes())
    trusted_source = capture_trusted_supervisor_source(trusted_path)
    forged = tmp_path / "forged-ran"
    trusted_path.write_text(
        f"from pathlib import Path\nPath({str(forged)!r}).write_text('forged')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        build_verify_module,
        "acquire_trusted_supervisor_source",
        trusted_source.acquire,
    )

    completed = build_verify_module.run_supervised_process(
        [sys.executable, "-c", "print('trusted-target')"],
        cwd=tmp_path,
        capture_output=True,
        timeout=5,
    )

    assert completed.returncode == 0
    assert completed.stdout == b"trusted-target\n"
    assert not forged.exists()


@pytest.mark.parametrize(
    "environment",
    (
        {"BAD=KEY": "value"},
        {"GOOD_KEY": "bad\x00value"},
        {1: "value"},
        {"GOOD_KEY": 1},
        {"GOOD_KEY": "x" * (256 * 1024)},
    ),
    ids=(
        "equals-in-key",
        "nul-in-value",
        "non-string-key",
        "non-string-value",
        "oversized",
    ),
)
def test_supervised_target_environment_rejects_invalid_or_unbounded_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: dict,
) -> None:
    def forbidden_popen(*_args, **_kwargs):
        raise AssertionError("invalid target env must fail before supervisor spawn")

    monkeypatch.setattr(build_verify_module.subprocess, "Popen", forbidden_popen)
    with pytest.raises(ValueError):
        build_verify_module.run_supervised_process(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            timeout=1,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="sealed memfd requires Linux")
def test_supervised_run_uses_explicit_none_not_callable_truthiness(
    tmp_path: Path,
) -> None:
    class FalseyPath(os.PathLike[str]):
        def __bool__(self) -> bool:
            return False

        def __fspath__(self) -> str:
            return str(tmp_path)

    completed = build_verify_module.run_supervised_process(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        cwd=FalseyPath(),
        capture_output=True,
        timeout=5,
        text=True,
        encoding=None,
        errors=None,
    )
    assert completed.stdout.strip() == str(tmp_path)

    with pytest.raises(LookupError):
        build_verify_module.run_supervised_process(
            [sys.executable, "-c", "pass"],
            input="payload",
            cwd=tmp_path,
            capture_output=True,
            timeout=1,
            encoding="",
        )


@pytest.mark.parametrize("runner_name", ["ok", "ok_trailing_newline"])
def test_runner_log_preserves_stdout_with_exactly_one_stderr_delimiter(
    runner_pkg: str,
    tmp_path: Path,
    runner_name: str,
) -> None:
    log_dir = tmp_path / "logs"
    result = _run_runner(
        "build",
        f"{runner_pkg}.{runner_name}",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=log_dir,
    )

    assert result.returncode == 0
    expected_stdout = result.raw_output + (
        b"" if result.raw_output.endswith(b"\n") else b"\n"
    )
    log = (log_dir / "build.log").read_bytes()
    assert b"# stdout\n" + expected_stdout + b"# stderr\n" in log
    assert b"\n\n# stderr\n" not in log


def test_runner_stream_capture_is_bounded_and_does_not_deadlock(
    runner_pkg: str,
    tmp_path: Path,
):
    log_dir = tmp_path / "logs"
    started = time.monotonic()

    result = _run_runner(
        "verify",
        f"{runner_pkg}.oversized",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=log_dir,
    )

    assert time.monotonic() - started < 5
    assert result.returncode == 2
    assert b"stdout-head" in result.raw_output
    assert b"stdout-tail" in result.raw_output
    assert b"runner stream truncated" in result.raw_output
    expected_stdout_bytes = (
        len(b"stdout-head\n")
        + 9 * 1024 * 1024
        + len(b"\nstdout-tail\n")
    )
    assert (
        f"total_bytes={expected_stdout_bytes}".encode() in result.raw_output
    )
    assert b"stderr-head" in result.stderr
    assert b"stderr-tail" in result.stderr
    expected_stderr_bytes = (
        len(b"stderr-head\n")
        + 2 * 1024 * 1024
        + len(b"\nstderr-tail\n")
    )
    assert f"total_bytes={expected_stderr_bytes}".encode() in result.stderr
    assert b"stdout exceeded the bounded capture limit" in result.stderr
    assert len(result.raw_output) <= (
        build_verify_module._RUNNER_STDOUT_CAPTURE_BYTES
    )
    assert len(result.stderr) <= (
        build_verify_module._RUNNER_STDERR_CAPTURE_BYTES + 256
    )

    latest_log = (log_dir / "verify.log").read_bytes()
    assert b"stdout-head" in latest_log
    assert b"stdout-tail" in latest_log
    assert b"stderr-head" in latest_log
    assert b"stderr-tail" in latest_log
    assert len(latest_log) <= (
        build_verify_module._RUNNER_STDOUT_CAPTURE_BYTES
        + build_verify_module._RUNNER_STDERR_CAPTURE_BYTES
        + 2_048
    )


@pytest.mark.parametrize("runner_name", ["recursive_json", "invalid_utf8"])
def test_successful_runner_with_invalid_protocol_fails_closed(
    runner_pkg: str,
    tmp_path: Path,
    runner_name: str,
):
    result = _run_runner(
        "verify",
        f"{runner_pkg}.{runner_name}",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 2
    assert result.values == {}
    assert (
        b"successful runner stdout was not valid bounded UTF-8 JSON"
        in result.stderr
    )


def test_runner_last_line_json_protocol_remains_compatible(
    runner_pkg: str,
    tmp_path: Path,
):
    result = _run_runner(
        "verify",
        f"{runner_pkg}.log_then_json",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 0
    assert result.values == {"returncode": 0.0}
    assert result.raw_output.startswith(b"runner diagnostic\n")


def test_oversized_stderr_preserves_success_and_edge_diagnostics(
    runner_pkg: str,
    tmp_path: Path,
):
    result = _run_runner(
        "verify",
        f"{runner_pkg}.oversized_stderr",
        {"parameters": {}},
        cwd=str(tmp_path),
        timeout=5,
        log_dir=None,
    )

    assert result.returncode == 0
    assert result.values == {"returncode": 0.0}
    assert b"stderr-head" in result.stderr
    assert b"stderr-tail" in result.stderr
    assert b"runner stream truncated" in result.stderr
    assert len(result.stderr) <= (
        build_verify_module._RUNNER_STDERR_CAPTURE_BYTES
    )


def _force_stop_recorded_processes(*pid_paths: Path) -> None:
    """Best-effort test finalizer; never leak a failed fixture process."""

    targets: dict[int, psutil.Process] = {}
    for pid_path in pid_paths:
        try:
            root = psutil.Process(int(pid_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, psutil.NoSuchProcess):
            continue
        try:
            command = " ".join(root.cmdline())
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            continue
        if "stub_runners.stubborn" not in command:
            # The recorded PID exited and was reused; never kill the new owner.
            continue
        try:
            descendants = root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []
        for target in (*descendants, root):
            targets[target.pid] = target
    for target in tuple(targets.values()):
        try:
            target.kill()
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
    try:
        psutil.wait_procs(list(targets.values()), timeout=2)
    except (psutil.Error, OSError):
        pass


def test_runner_timeout_stops_and_reaps_cooperative_same_tree_descendant(
    runner_pkg: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    child_pid_path = tmp_path / "child.pid"
    child_term_path = tmp_path / "child.term"
    runner_pid_path = tmp_path / "runner.pid"
    runner_term_path = tmp_path / "runner.term"
    monkeypatch.setattr(
        build_verify_module,
        "_RUNNER_TERMINATION_GRACE_S",
        2.0,
    )
    for name in (
        "_RUNNER_KILL_GRACE_S",
        "_RUNNER_DRAIN_GRACE_S",
        "_RUNNER_REAP_GRACE_S",
    ):
        monkeypatch.setattr(build_verify_module, name, 0.5)

    try:
        result = _run_runner(
            "verify",
            f"{runner_pkg}.stubborn_tree",
            {
                "parameters": {
                    "child_pid_path": str(child_pid_path),
                    "child_term_path": str(child_term_path),
                    "runner_pid_path": str(runner_pid_path),
                    "runner_term_path": str(runner_term_path),
                },
            },
            cwd=str(tmp_path),
            timeout=1,
            log_dir=None,
        )

        assert result.returncode == 124
        assert b"runner-started" in result.raw_output
        assert b"runner-stderr" in result.stderr
        assert b"[timeout after 1s]" in result.stderr
        assert re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            result.process_settlement_digest or "",
        )
        assert runner_term_path.read_text(encoding="utf-8") == "term received\n"
        assert child_term_path.read_text(encoding="utf-8") == "term received\n"

        # The fixture runner polls and waits its child after cleanup kills it,
        # so both PIDs must disappear; a zombie is not accepted as "reaped."
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        runner_pid = int(runner_pid_path.read_text(encoding="utf-8"))
        assert not psutil.pid_exists(child_pid)
        assert not psutil.pid_exists(runner_pid)
    finally:
        _force_stop_recorded_processes(child_pid_path, runner_pid_path)


def test_runner_cleanup_waits_and_pipe_drains_are_strictly_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class _Pipe:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class _NeverExits:
        pid = 1 << 30

        def __init__(self) -> None:
            self.stdin = _Pipe()
            self.stdout = _Pipe()
            self.stderr = _Pipe()
            self.returncode = None
            self.communicate_timeouts: list[float | None] = []
            self.wait_timeouts: list[float | None] = []
            self.terminate_calls = 0
            self.kill_calls = 0

        def communicate(self, _input=None, *, timeout=None):
            self.communicate_timeouts.append(timeout)
            raise subprocess.TimeoutExpired(
                cmd=["never-exits"],
                timeout=timeout,
                output=b"partial stdout",
                stderr=b"partial stderr",
            )

        def poll(self):
            return None

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def wait(self, *, timeout=None):
            self.wait_timeouts.append(timeout)
            raise subprocess.TimeoutExpired(["never-exits"], timeout)

    fake = _NeverExits()
    monkeypatch.setattr(
        build_verify_module.subprocess,
        "Popen",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_ready",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        lambda *_args, **_kwargs: None,
    )
    for name in (
        "_RUNNER_TERMINATION_GRACE_S",
        "_RUNNER_KILL_GRACE_S",
        "_RUNNER_DRAIN_GRACE_S",
        "_RUNNER_REAP_GRACE_S",
    ):
        monkeypatch.setattr(build_verify_module, name, 0.01)

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        build_verify_module._run_runner_subprocess(
            ["never-exits"],
            config_json=b"{}",
            cwd=str(tmp_path),
            timeout=0.01,
            env={},
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5
    assert caught.value.stdout == b"partial stdout"
    assert caught.value.stderr == b"partial stderr"
    assert fake.terminate_calls == 1
    assert fake.kill_calls >= 2
    assert fake.communicate_timeouts
    assert all(timeout is not None for timeout in fake.communicate_timeouts)
    assert fake.wait_timeouts
    assert all(timeout is not None for timeout in fake.wait_timeouts)
    assert fake.stdin.closed and fake.stdout.closed and fake.stderr.closed


def test_timeout_output_merge_uses_cumulative_communicate_contract():
    assert build_verify_module._merge_captured(
        b"partial",
        b"partial plus drained",
    ) == b"partial plus drained"
    assert build_verify_module._merge_captured(
        b"partial",
        b" suffix-only adapter output",
    ) == b"partial suffix-only adapter output"


def test_bounded_cleanup_drain_does_not_swallow_keyboard_interrupt():
    class _InterruptedDrain:
        def communicate(self, *, timeout=None):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._bounded_communicate(
            _InterruptedDrain(),
            timeout=0.01,
            stdout=b"partial stdout",
            stderr=b"partial stderr",
        )


@pytest.mark.skipif(sys.platform != "linux", reason="seqpacket requires Linux")
@pytest.mark.parametrize("changed_authority", ["launch", "epoch"])
def test_runner_terminal_attestation_requires_exact_launch_and_epoch(
    changed_authority: str,
) -> None:
    controller, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    launch_id = "runner-exact-binding"
    record = {
        "schema_version": build_verify_module._RUNNER_SUPERVISOR_SCHEMA,
        "phase": "terminal",
        "launch_id": launch_id,
        "epoch": None,
        "outcome": "completed",
        "target_started": True,
        "returncode": 0,
        "settled": True,
        "error": None,
    }
    if changed_authority == "launch":
        record["launch_id"] = "runner-foreign-launch"
    else:
        record["epoch"] = {
            "epoch_id": "epoch:foreign",
            "generation": 1,
            "family_digest": "sha256:" + "0" * 64,
        }
    try:
        peer.send(json.dumps(record, sort_keys=True).encode("ascii"))
        peer.close()
        with pytest.raises(
            build_verify_module._RunnerSettlementError,
            match="did not prove complete descendant settlement",
        ):
            build_verify_module._read_runner_supervisor_control(
                controller.fileno(),
                launch_id=launch_id,
                writer_epoch_lease=None,
            )
    finally:
        controller.close()
        peer.close()


@pytest.mark.skipif(sys.platform != "linux", reason="seqpacket requires Linux")
def test_runner_terminal_attestation_rejects_an_extra_packet() -> None:
    controller, peer = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    launch_id = "runner-extra-packet"
    ready = {
        "schema_version": build_verify_module._RUNNER_SUPERVISOR_READY_SCHEMA,
        "phase": "ready",
        "launch_id": launch_id,
    }
    terminal = {
        "schema_version": build_verify_module._RUNNER_SUPERVISOR_SCHEMA,
        "phase": "terminal",
        "launch_id": launch_id,
        "epoch": None,
        "outcome": "completed",
        "target_started": True,
        "returncode": 0,
        "settled": True,
        "error": None,
    }
    try:
        peer.send(json.dumps(ready, sort_keys=True).encode("ascii"))
        peer.send(json.dumps(terminal, sort_keys=True).encode("ascii"))
        peer.send(b"{}")
        peer.close()
        build_verify_module._read_runner_supervisor_ready(
            controller.fileno(),
            launch_id=launch_id,
            timeout=0.0,
        )
        with pytest.raises(
            build_verify_module._RunnerSettlementError,
            match="multiple terminal control packets",
        ):
            build_verify_module._read_runner_supervisor_control(
                controller.fileno(),
                launch_id=launch_id,
                writer_epoch_lease=None,
            )
    finally:
        controller.close()
        peer.close()


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
@pytest.mark.parametrize("duplicate_command", ["start", "cancel"])
def test_supervisor_rejects_duplicate_transition_packets(
    duplicate_command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_send = build_verify_module._send_runner_supervisor_command
    duplicate_sent = threading.Event()

    def send_duplicate(*args, **kwargs):
        original_send(*args, **kwargs)
        if kwargs["command"] == "start":
            command = duplicate_command
            for _index in range(2 if command == "cancel" else 1):
                original_send(
                    args[0],
                    launch_id=kwargs["launch_id"],
                    command=command,
                    writer_epoch_lease=kwargs["writer_epoch_lease"],
                )
            duplicate_sent.set()

    def wait_until_duplicate(*_args, **_kwargs):
        assert duplicate_sent.wait(timeout=2)
        time.sleep(0.05)
        return b"", b"", False, False

    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        send_duplicate,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        wait_until_duplicate,
    )

    with pytest.raises(
        build_verify_module._RunnerSettlementError,
        match=(
            "control record is unavailable|did not prove complete descendant settlement"
        ),
    ):
        build_verify_module.run_supervised_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            capture_output=True,
            timeout=35,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_cancel_during_background_descendant_settlement_is_authenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first exact CANCEL remains legal after the direct target exits."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    target_marker = tmp_path / "settlement-target.pid"
    child_marker = tmp_path / "settlement-child.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    terminal_outcomes: list[str] = []
    real_terminal_reader = build_verify_module._read_runner_supervisor_control

    def capture_terminal_outcome(*args, **kwargs):
        terminal = real_terminal_reader(*args, **kwargs)
        terminal_outcomes.append(terminal[2])
        return terminal

    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_control",
        capture_terminal_outcome,
    )

    target = textwrap.dedent(
        """\
        import os, signal, sys, time
        from pathlib import Path

        Path(sys.argv[1]).write_text(str(os.getpid()), encoding="ascii")
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            Path(sys.argv[2]).write_text(str(os.getpid()), encoding="ascii")
            time.sleep(30)
            os._exit(0)
        deadline = time.monotonic() + 2
        while not Path(sys.argv[2]).exists():
            if time.monotonic() >= deadline:
                os._exit(3)
            time.sleep(0.005)
        """
    )

    def interrupt_during_settlement(process, *_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if target_marker.exists() and child_marker.exists():
                target_pid = int(target_marker.read_text(encoding="ascii"))
                child_pid = int(child_marker.read_text(encoding="ascii"))
                if (
                    not psutil.pid_exists(target_pid)
                    and psutil.pid_exists(child_pid)
                    and process.poll() is None
                ):
                    raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervisor did not enter descendant settlement")

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_during_settlement,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                target,
                str(target_marker),
                str(child_marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert not psutil.pid_exists(int(child_marker.read_text(encoding="ascii")))
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    assert terminal_outcomes == ["cancelled"]


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_controller_interrupt_before_ready_never_arms_or_starts_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation before READY cannot create mutation authority."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "pre-ready-target-started"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    original_read_ready = build_verify_module._read_runner_supervisor_ready
    ready_calls = 0

    def interrupt_first_ready(*args, **kwargs):
        nonlocal ready_calls
        ready_calls += 1
        if ready_calls == 1:
            raise KeyboardInterrupt
        return original_read_ready(*args, **kwargs)

    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_ready",
        interrupt_first_ready,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert ready_calls == 2
    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="supervisor signals require Linux")
def test_direct_supervisor_signal_after_ready_cannot_authorize_quiescence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """READY plus physical supervisor death is never a settlement receipt."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "direct-signal-target"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    real_popen = build_verify_module.subprocess.Popen
    launched: list[subprocess.Popen[bytes]] = []
    real_ready = build_verify_module._read_runner_supervisor_ready

    def capture_supervisor(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        launched.append(process)
        return process

    def kill_after_ready(*args, **kwargs):
        real_ready(*args, **kwargs)
        assert len(launched) == 1
        os.kill(launched[0].pid, signal.SIGTERM)
        launched[0].wait(timeout=3)

    monkeypatch.setattr(build_verify_module.subprocess, "Popen", capture_supervisor)
    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_ready",
        kill_after_ready,
    )

    with pytest.raises(
        (build_verify_module._RunnerSettlementError, OSError),
    ):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert evidence == []


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupted_partial_arm_settles_exact_epoch_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An interrupted root-first arm is retried and settled without START."""

    from crucible.workflows._writer_epoch import (
        WriterEpochArmError,
        inspect_writer_epoch_family,
    )

    marker = tmp_path / "arm-before-start-target"
    root = _MemoryWriterEpochStorage("runner-root")
    child = _MemoryWriterEpochStorage(
        "runner-child",
        parent_transaction_id="runner-root",
    )
    evidence: list[str] = []
    original_child_write = child.compare_and_write_writer_epoch_record
    interrupted = False

    def interrupt_child_arm(expected, replacement):
        nonlocal interrupted
        if not interrupted and replacement.get("state") == "armed":
            interrupted = True
            raise KeyboardInterrupt
        return original_child_write(expected, replacement)

    monkeypatch.setattr(
        child,
        "compare_and_write_writer_epoch_record",
        interrupt_child_arm,
    )

    with pytest.raises(WriterEpochArmError) as caught:
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(root, child),
        )

    assert isinstance(caught.value.error, KeyboardInterrupt)
    assert interrupted
    assert not marker.exists()
    inspection = inspect_writer_epoch_family((root, child))
    assert inspection.state == "quiesced"
    assert len(evidence) == 1
    assert root.record["settlement"]["evidence_digest"] == evidence[0]
    assert child.record == root.record


@pytest.mark.skipif(sys.platform != "linux", reason="pthread signals require Linux")
def test_pending_sigint_cannot_erase_partial_arm_retry_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT at unmask cannot replace the partial-arm lease."""

    from crucible.workflows._writer_epoch import (
        WriterEpochArmError,
        inspect_writer_epoch_family,
    )

    marker = tmp_path / "partial-arm-pending-sigint-target"
    root = _MemoryWriterEpochStorage("pending-root")
    child = _MemoryWriterEpochStorage(
        "pending-child",
        parent_transaction_id="pending-root",
    )
    evidence: list[str] = []
    original_child_write = child.compare_and_write_writer_epoch_record
    signal_sent = threading.Event()
    sender: threading.Thread | None = None
    main_thread_id = threading.get_ident()
    interrupted = False

    def fail_child_with_pending_sigint(expected, replacement):
        nonlocal interrupted, sender
        if not interrupted and replacement.get("state") == "armed":
            interrupted = True

            def queue_signal() -> None:
                signal.pthread_kill(main_thread_id, signal.SIGINT)
                signal_sent.set()

            sender = threading.Thread(target=queue_signal)
            sender.start()
            deadline = time.monotonic() + 2
            while not signal_sent.is_set() and time.monotonic() < deadline:
                time.sleep(0.001)
            assert signal_sent.is_set()
            raise OSError("injected child arm failure")
        return original_child_write(expected, replacement)

    monkeypatch.setattr(
        child,
        "compare_and_write_writer_epoch_record",
        fail_child_with_pending_sigint,
    )

    with pytest.raises(WriterEpochArmError) as caught:
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(root, child),
        )

    assert isinstance(caught.value.error, KeyboardInterrupt)
    assert interrupted and signal_sent.is_set()
    assert sender is not None
    sender.join(timeout=2)
    assert not sender.is_alive()
    assert not marker.exists()
    inspection = inspect_writer_epoch_family((root, child))
    assert inspection.state == "quiesced"
    assert len(evidence) == 1
    assert root.record["settlement"]["evidence_digest"] == evidence[0]
    assert child.record == root.record


@pytest.mark.skipif(sys.platform != "linux", reason="process signals require Linux")
def test_process_sigint_during_arm_retry_uses_prepared_lease_and_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process SIGINT cannot erase a lease retained before durable writes."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "mask-restore-target"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    original_retry = build_verify_module.retry_writer_epoch_arm
    signal_sent = threading.Event()
    sender: threading.Thread | None = None

    def retry_then_queue_sigint(lease):
        nonlocal sender
        original_retry(lease)

        def queue_signal() -> None:
            os.kill(os.getpid(), signal.SIGINT)
            signal_sent.set()

        sender = threading.Thread(target=queue_signal)
        sender.start()
        deadline = time.monotonic() + 2
        while not signal_sent.is_set() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert signal_sent.is_set()
        assert inspect_writer_epoch_family((storage,)).state == "armed"
        return lease

    monkeypatch.setattr(
        build_verify_module,
        "retry_writer_epoch_arm",
        retry_then_queue_sigint,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert sender is not None
    sender.join(timeout=2)
    assert not sender.is_alive()
    assert signal_sent.is_set()
    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    assert storage.record["settlement"]["evidence_digest"] == evidence[0]


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_slow_epoch_arm_does_not_consume_target_execution_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The target timeout starts at exact START, after durable arming."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "slow-arm-target-ran"
    storage = _MemoryWriterEpochStorage()
    original_retry = build_verify_module.retry_writer_epoch_arm
    first_retry = True

    def slow_first_retry(lease):
        nonlocal first_retry
        if first_retry:
            first_retry = False
            time.sleep(0.35)
        return original_retry(lease)

    monkeypatch.setattr(
        build_verify_module,
        "retry_writer_epoch_arm",
        slow_first_retry,
    )

    completed = build_verify_module._run_runner_subprocess(
        [
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(marker),
        ],
        config_json=b"",
        cwd=str(tmp_path),
        timeout=0.1,
        env=os.environ.copy(),
        writer_epoch_members=(storage,),
    )

    assert completed.returncode == 0
    assert marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"


@pytest.mark.skipif(sys.platform != "linux", reason="threads require Linux")
@pytest.mark.parametrize("native_worker_started", [False, True])
def test_settlement_transaction_start_interrupt_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_worker_started: bool,
) -> None:
    """An interrupted coordinator start creates no external authority."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    real_start = threading.Thread.start
    injected = False
    native_start_calls = 0

    def interrupt_start(thread):
        nonlocal injected, native_start_calls
        if thread.name != "crucible-runner-settlement":
            return real_start(thread)
        if not injected:
            injected = True
            if native_worker_started:
                native_start_calls += 1
                real_start(thread)
            raise KeyboardInterrupt
        native_start_calls += 1
        return real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", interrupt_start)

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert injected
    assert native_start_calls == int(native_worker_started)
    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    assert evidence == []


@pytest.mark.skipif(sys.platform != "linux", reason="threads require Linux")
def test_delayed_native_coordinator_bootstrap_aborts_without_external_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A KI-delayed native bootstrap later observes abort and exits safely."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    real_native_start = threading._start_new_thread
    bootstrap_gate = threading.Event()
    native_launched = threading.Event()
    native_exited = threading.Event()
    original_handlers = {
        member: signal.getsignal(member)
        for member in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    original_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())

    def interrupt_after_delayed_native_start(function, args, kwargs=None):
        def delayed_bootstrap() -> None:
            native_launched.set()
            bootstrap_gate.wait()
            try:
                function(*args, **(kwargs or {}))
            finally:
                native_exited.set()

        real_native_start(delayed_bootstrap, ())
        raise KeyboardInterrupt

    monkeypatch.setattr(
        threading,
        "_start_new_thread",
        interrupt_after_delayed_native_start,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert native_launched.wait(1)
    assert inspect_writer_epoch_family((storage,)).state == "never_armed"
    assert evidence == []
    assert {
        member: signal.getsignal(member)
        for member in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    } == original_handlers
    assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == original_mask

    bootstrap_gate.set()
    assert native_exited.wait(2)


@pytest.mark.skipif(sys.platform != "linux", reason="threads require Linux")
def test_interrupt_after_coordinator_start_return_aborts_idle_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KI after native start but before ready observation cannot leak worker."""

    storage = _MemoryWriterEpochStorage()
    real_is_set = threading.Event.is_set
    coordinator_worker: list[threading.Thread] = []
    interrupted = False

    def interrupt_ready_observation(event) -> bool:
        nonlocal interrupted
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (
            not interrupted
            and caller is not None
            and caller.f_code.co_name == "start"
            and real_is_set(event)
        ):
            interrupted = True
            coordinator_worker.extend(
                thread
                for thread in threading.enumerate()
                if thread.name == "crucible-runner-settlement"
            )
            raise KeyboardInterrupt
        return real_is_set(event)

    monkeypatch.setattr(threading.Event, "is_set", interrupt_ready_observation)

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            writer_epoch_members=(storage,),
        )

    assert interrupted
    assert coordinator_worker
    deadline = time.monotonic() + 2
    while any(worker.is_alive() for worker in coordinator_worker):
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_before_transaction_submission_aborts_without_epoch_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prepared zero-write lease is discarded if no worker claims its job."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    real_submit = build_verify_module._RunnerSettlementCoordinator.submit
    interrupted = False

    def interrupt_before_submit(coordinator, transaction):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return real_submit(coordinator, transaction)

    monkeypatch.setattr(
        build_verify_module._RunnerSettlementCoordinator,
        "submit",
        interrupt_before_submit,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            writer_epoch_members=(storage,),
        )

    assert interrupted
    assert inspect_writer_epoch_family((storage,)).state == "never_armed"


def test_unclaimed_descriptor_close_is_at_most_once_after_async_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An fd number reused after close cannot be closed by cleanup retry."""

    descriptor = os.open("/dev/null", os.O_RDONLY)
    replacement: list[int] = []
    real_close = os.close
    transaction = build_verify_module._RunnerSettlementTransaction(
        object(),
        descriptor,
        launch_id="runner-close-at-most-once",
        writer_epoch_lease=None,
        settlement_owner_ids=(),
        settlement_worker_ids=(),
        settlement_callback=None,
    )

    def close_then_interrupt(target: int) -> None:
        if target == descriptor and not replacement:
            real_close(target)
            replacement.append(os.open("/dev/null", os.O_RDONLY))
            assert replacement[0] == descriptor
            raise KeyboardInterrupt
        real_close(target)

    monkeypatch.setattr(build_verify_module.os, "close", close_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            transaction.close_unclaimed()
        transaction.close_unclaimed()
        os.fstat(replacement[0])
    finally:
        if replacement:
            real_close(replacement[0])


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_cancellation_observed_at_claim_never_starts_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A claimed pre-dispatch cancellation sends initial CANCEL, not START."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "pre-dispatch-target-started"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    real_submit = build_verify_module._RunnerSettlementCoordinator.submit

    def cancel_after_claim(coordinator, transaction):
        pending = real_submit(coordinator, transaction)
        assert pending is None
        return KeyboardInterrupt()

    monkeypatch.setattr(
        build_verify_module._RunnerSettlementCoordinator,
        "submit",
        cancel_after_claim,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_repeated_interrupts_before_start_wake_replace_start_with_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pre-delivery KIs cannot leak START before initial CANCEL."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "double-start-wake-target-started"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    commands: list[str] = []
    real_set = threading.Event.set
    real_send = build_verify_module._send_runner_supervisor_command
    interrupted = 0

    def interrupt_two_start_wakes(event) -> None:
        nonlocal interrupted
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (
            interrupted < 2
            and caller is not None
            and caller.f_code.co_name == "request_start"
        ):
            interrupted += 1
            raise KeyboardInterrupt
        real_set(event)

    def capture_send(*args, **kwargs):
        commands.append(kwargs["command"])
        return real_send(*args, **kwargs)

    monkeypatch.setattr(threading.Event, "set", interrupt_two_start_wakes)
    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        capture_send,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert interrupted == 2
    assert commands == ["cancel"]
    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_after_start_intent_before_wake_sends_only_initial_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undelivered START intent remains replaceable by cancellation."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "start-intent-gap-target-started"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    commands: list[str] = []
    real_set = threading.Event.set
    real_send = build_verify_module._send_runner_supervisor_command
    interrupted = False

    def interrupt_before_start_wake(event) -> None:
        nonlocal interrupted
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (
            not interrupted
            and caller is not None
            and caller.f_code.co_name == "request_start"
        ):
            interrupted = True
            raise KeyboardInterrupt
        real_set(event)

    def capture_send(*args, **kwargs):
        commands.append(kwargs["command"])
        return real_send(*args, **kwargs)

    monkeypatch.setattr(threading.Event, "set", interrupt_before_start_wake)
    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        capture_send,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert interrupted
    assert commands == ["cancel"]
    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_after_start_wake_claim_sends_followup_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wake that succeeds before raising KI cannot strand claimed START."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    commands: list[str] = []
    real_set = threading.Event.set
    real_send = build_verify_module._send_runner_supervisor_command
    interrupted = False

    def interrupt_after_claim(event) -> None:
        nonlocal interrupted
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        if (
            not interrupted
            and caller is not None
            and caller.f_code.co_name == "request_start"
        ):
            transaction = caller.f_locals["self"]
            real_set(event)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with transaction._command_lock:
                    if transaction._initial_command_claimed:
                        interrupted = True
                        raise KeyboardInterrupt
                time.sleep(0.001)
            pytest.fail("settlement worker did not claim START after its wake")
        real_set(event)

    def capture_send(*args, **kwargs):
        commands.append(kwargs["command"])
        return real_send(*args, **kwargs)

    monkeypatch.setattr(threading.Event, "set", interrupt_after_claim)
    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        capture_send,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert interrupted
    assert commands == ["start", "cancel"]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_repeated_interrupt_before_cancel_publication_finalizes_claimed_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper finalization publishes CANCEL and callback despite a second KI."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "double-pre-cancel-target-started"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    main_thread = threading.get_ident()
    real_retry = build_verify_module.retry_writer_epoch_arm
    real_claimed = build_verify_module._RunnerSettlementCoordinator.claimed_transaction
    arm_interrupted = False
    claim_interrupted = False

    def interrupt_main_arm(lease):
        nonlocal arm_interrupted
        if threading.get_ident() == main_thread and not arm_interrupted:
            arm_interrupted = True
            raise KeyboardInterrupt
        return real_retry(lease)

    def interrupt_first_cleanup_claim(coordinator):
        nonlocal claim_interrupted
        if arm_interrupted and not claim_interrupted:
            claim_interrupted = True
            raise KeyboardInterrupt
        return real_claimed(coordinator)

    monkeypatch.setattr(
        build_verify_module,
        "retry_writer_epoch_arm",
        interrupt_main_arm,
    )
    monkeypatch.setattr(
        build_verify_module._RunnerSettlementCoordinator,
        "claimed_transaction",
        interrupt_first_cleanup_claim,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert arm_interrupted and claim_interrupted
    assert not marker.exists()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
@pytest.mark.parametrize("interrupted_wake", ["submit", "start", "cancel"])
def test_interrupted_coordinator_wake_uses_published_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_wake: str,
) -> None:
    """KI between state publication and Event.set cannot strand the worker."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / f"interrupted-{interrupted_wake}-wake.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    wake_interrupted = False
    target_started = threading.Event()
    real_set = threading.Event.set

    def interrupt_selected_wake(event) -> None:
        nonlocal wake_interrupted
        frame = inspect.currentframe()
        caller = frame.f_back if frame is not None else None
        caller_name = caller.f_code.co_name if caller is not None else ""
        selected = {
            "submit": "submit",
            "start": "request_start",
            "cancel": "publish_cancel",
        }[interrupted_wake]
        if not wake_interrupted and caller_name == selected:
            wake_interrupted = True
            raise KeyboardInterrupt
        real_set(event)

    def interrupt_after_target_start(*_args, **_kwargs):
        if interrupted_wake != "cancel":
            return None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                target_started.set()
                os.kill(os.getpid(), signal.SIGINT)
                pytest.fail("SIGINT was not delivered")
            time.sleep(0.005)
        pytest.fail("supervised target did not start before cancellation")

    monkeypatch.setattr(threading.Event, "set", interrupt_selected_wake)
    if interrupted_wake == "cancel":
        monkeypatch.setattr(
            build_verify_module,
            "_bounded_runner_communicate",
            interrupt_after_target_start,
        )

    command = (
        [
            sys.executable,
            "-c",
            (
                "import os,pathlib,sys,time; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "time.sleep(30)"
            ),
            str(marker),
        ]
        if interrupted_wake == "cancel"
        else [sys.executable, "-c", "pass"]
    )
    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            command,
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35 if interrupted_wake == "cancel" else 5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert wake_interrupted
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    if target_started.is_set():
        assert not psutil.pid_exists(int(marker.read_text(encoding="ascii")))




@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_at_legacy_consume_callsite_cannot_erase_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal execution joins its pre-START transaction, never rereads the fd."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    legacy_consume_calls = 0

    def reject_legacy_consume(*_args, **_kwargs):
        nonlocal legacy_consume_calls
        legacy_consume_calls += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(
        build_verify_module,
        "_consume_runner_supervisor_settlement",
        reject_legacy_consume,
    )

    completed = build_verify_module._run_runner_subprocess(
        [sys.executable, "-c", "pass"],
        config_json=b"",
        cwd=str(tmp_path),
        timeout=5,
        env=os.environ.copy(),
        settlement_callback=evidence.append,
        writer_epoch_members=(storage,),
    )

    assert completed.returncode == 0
    assert legacy_consume_calls == 0
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_after_supervisor_exit_consumes_buffered_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CANCEL EPIPE cannot hide an already-buffered exact terminal."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    terminal_outcomes: list[str] = []
    real_terminal_reader = build_verify_module._read_runner_supervisor_control

    def capture_terminal_outcome(*args, **kwargs):
        terminal = real_terminal_reader(*args, **kwargs)
        terminal_outcomes.append(terminal[2])
        return terminal

    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_control",
        capture_terminal_outcome,
    )

    def interrupt_after_exit(process, *_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervisor did not exit before cancellation race")

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_after_exit,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    assert terminal_outcomes == ["completed"]


@pytest.mark.skipif(sys.platform != "linux", reason="process signals require Linux")
@pytest.mark.parametrize("signal_member", ["child", "root"])
def test_process_sigint_during_terminal_quiescence_is_deferred(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signal_member: str,
) -> None:
    """The settlement worker retains terminal authority through quiescence."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    root = _MemoryWriterEpochStorage("terminal-root")
    child = _MemoryWriterEpochStorage(
        "terminal-child",
        parent_transaction_id="terminal-root",
    )
    evidence: list[str] = []
    selected = child if signal_member == "child" else root
    original_write = selected.compare_and_write_writer_epoch_record
    signal_sent = threading.Event()
    main_thread_id = threading.get_ident()
    interrupted = False

    def signal_after_member_quiescence(expected, replacement):
        nonlocal interrupted
        result = original_write(expected, replacement)
        if not interrupted and replacement.get("state") == "quiesced":
            interrupted = True
            signal.pthread_kill(main_thread_id, signal.SIGINT)
            signal_sent.set()
            # Keep terminal quiescence in flight while the main thread gets a
            # scheduling window to observe and defer this exact process
            # signal; otherwise the test can leave the pytest.raises scope
            # before CPython services its pending handler.
            time.sleep(0.05)
        return result

    monkeypatch.setattr(
        selected,
        "compare_and_write_writer_epoch_record",
        signal_after_member_quiescence,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(root, child),
        )

    assert signal_sent.is_set()
    assert inspect_writer_epoch_family((root, child)).state == "quiesced"
    assert len(evidence) == 1
    assert root.record == child.record


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_controller_keyboard_interrupt_consumes_supervisor_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caught cancellation settles its tree before rollback is authorized."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "interrupt-target.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []

    def interrupt_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervised target did not start before cancellation")

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_after_target_start,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    target_pid = int(marker.read_text(encoding="ascii"))
    assert not psutil.pid_exists(target_pid)
    inspection = inspect_writer_epoch_family((storage,))
    assert inspection.state == "quiesced"
    assert len(evidence) == 1
    assert storage.record["settlement"]["evidence_digest"] == evidence[0]


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_retained_cancel_settles_before_outer_pipe_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipe-drain deadline cannot preempt supervisor-owned settlement."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "retained-cancel-target.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    settlement_published = threading.Event()
    real_bounded_communicate = build_verify_module._bounded_communicate

    def record_settlement(digest: str) -> None:
        evidence.append(digest)
        settlement_published.set()

    def require_settlement_before_drain(*args, **kwargs):
        assert settlement_published.is_set(), (
            "outer pipe drain started before retained settlement completed"
        )
        return real_bounded_communicate(*args, **kwargs)

    def interrupt_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervised target did not start before cancellation")

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_after_target_start,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_bounded_communicate",
        require_settlement_before_drain,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=record_settlement,
            writer_epoch_members=(storage,),
        )

    assert settlement_published.is_set()
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    assert not psutil.pid_exists(int(marker.read_text(encoding="ascii")))


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_repeated_interrupts_after_cancel_defer_until_same_transaction_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated KI after CANCEL cannot trigger physical supervisor teardown."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "repeated-interrupt-target.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    commands: list[str] = []
    physical_teardowns = 0
    join_interrupts = 0
    real_send = build_verify_module._send_runner_supervisor_command
    real_join = build_verify_module._RunnerSettlementTransaction.join
    real_teardown = build_verify_module._terminate_runner_tree

    def capture_send(*args, **kwargs):
        commands.append(kwargs["command"])
        return real_send(*args, **kwargs)

    def interrupt_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervised target did not start before cancellation")

    def repeatedly_interrupt_join(transaction, *args, **kwargs):
        nonlocal join_interrupts
        if "cancel" in commands and join_interrupts < 3:
            join_interrupts += 1
            raise KeyboardInterrupt
        return real_join(transaction, *args, **kwargs)

    def count_physical_teardown(*args, **kwargs):
        nonlocal physical_teardowns
        physical_teardowns += 1
        return real_teardown(*args, **kwargs)

    monkeypatch.setattr(
        build_verify_module,
        "_send_runner_supervisor_command",
        capture_send,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_after_target_start,
    )
    monkeypatch.setattr(
        build_verify_module._RunnerSettlementTransaction,
        "join",
        repeatedly_interrupt_join,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_terminate_runner_tree",
        count_physical_teardown,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert commands == ["start", "cancel"]
    assert join_interrupts == 3
    assert physical_teardowns == 0
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1
    assert not psutil.pid_exists(int(marker.read_text(encoding="ascii")))


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_interrupt_at_final_join_entry_retries_before_callback_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KI before normal-path join cannot race rollback or lose the callback."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []
    terminal_completed = threading.Event()
    real_settle = (
        build_verify_module._RunnerSettlementTransaction._settle_from_terminal
    )
    real_join = build_verify_module._RunnerSettlementTransaction.join
    join_entries = 0

    def observe_terminal_completion(transaction, *args, **kwargs):
        try:
            return real_settle(transaction, *args, **kwargs)
        finally:
            terminal_completed.set()

    def interrupt_first_completed_join(transaction, *args, **kwargs):
        nonlocal join_entries
        join_entries += 1
        if join_entries == 1:
            assert terminal_completed.wait(timeout=5)
            assert transaction.done
            assert evidence == []
            raise KeyboardInterrupt
        return real_join(transaction, *args, **kwargs)

    monkeypatch.setattr(
        build_verify_module._RunnerSettlementTransaction,
        "_settle_from_terminal",
        observe_terminal_completion,
    )
    monkeypatch.setattr(
        build_verify_module._RunnerSettlementTransaction,
        "join",
        interrupt_first_completed_join,
    )

    with pytest.raises(KeyboardInterrupt):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert join_entries >= 2
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_settlement_callback_failure_is_main_thread_and_at_most_once(
    tmp_path: Path,
) -> None:
    """Callback failure cannot repeat after the epoch is already quiescent."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    storage = _MemoryWriterEpochStorage()
    callback_threads: list[int] = []
    main_thread = threading.get_ident()

    def fail_callback(_digest: str) -> None:
        callback_threads.append(threading.get_ident())
        raise RuntimeError("injected settlement callback failure")

    with pytest.raises(
        RuntimeError,
        match="injected settlement callback failure",
    ):
        build_verify_module._run_runner_subprocess(
            [sys.executable, "-c", "pass"],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=5,
            env=os.environ.copy(),
            settlement_callback=fail_callback,
            writer_epoch_members=(storage,),
        )

    assert callback_threads == [main_thread]
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"


def test_replay_disabled_join_retains_fresh_interrupt_for_outer_boundary() -> None:
    """Cleanup-mode join cannot silently consume a newly arriving KI."""

    class _InterruptThenDone:
        def __init__(self) -> None:
            self.interrupted = False

        def is_set(self) -> bool:
            return self.interrupted

        def wait(self, _timeout: float) -> None:
            self.interrupted = True
            raise KeyboardInterrupt

    transaction = build_verify_module._RunnerSettlementTransaction(
        object(),
        -1,
        launch_id="runner-retained-interrupt",
        writer_epoch_lease=None,
        settlement_owner_ids=(),
        settlement_worker_ids=(),
        settlement_callback=None,
    )
    transaction._done = _InterruptThenDone()
    transaction._result.append((0, None, "completed", "sha256:" + "0" * 64))

    assert transaction.join(replay_interrupt=False) == (0, None, "completed")
    assert isinstance(transaction.take_deferred_interrupt(), KeyboardInterrupt)




@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_controller_keyboard_interrupt_without_authenticated_packet_stays_armed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation never converts rejected control evidence into authority."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "interrupt-rejected.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []

    def interrupt_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                raise KeyboardInterrupt
            time.sleep(0.005)
        pytest.fail("supervised target did not start before cancellation")

    def reject_control(_descriptor: int, **_kwargs):
        raise build_verify_module._RunnerSettlementError(
            "injected malformed supervisor control"
        )

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        interrupt_after_target_start,
    )
    monkeypatch.setattr(
        build_verify_module,
        "_read_runner_supervisor_control",
        reject_control,
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert not psutil.pid_exists(int(marker.read_text(encoding="ascii")))
    assert inspect_writer_epoch_family((storage,)).state == "armed"
    assert evidence == []
    assert isinstance(
        caught.value.__cause__,
        (build_verify_module._RunnerSettlementError, OSError),
    )
    assert any(
        "could not finalize authenticated settlement" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.skipif(sys.platform != "linux", reason="subreaper requires Linux")
def test_controller_outer_timeout_consumes_supervisor_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A controller-side timeout retains failure while settling the writer."""

    from crucible.workflows._writer_epoch import inspect_writer_epoch_family

    marker = tmp_path / "outer-timeout.pid"
    storage = _MemoryWriterEpochStorage()
    evidence: list[str] = []

    def timeout_after_target_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if marker.exists():
                raise subprocess.TimeoutExpired(
                    cmd=["injected-outer-timeout"],
                    timeout=0.01,
                    output=b"partial stdout",
                    stderr=b"partial stderr",
                )
            time.sleep(0.005)
        pytest.fail("supervised target did not start before timeout")

    monkeypatch.setattr(
        build_verify_module,
        "_bounded_runner_communicate",
        timeout_after_target_start,
    )

    with pytest.raises(subprocess.TimeoutExpired) as caught:
        build_verify_module._run_runner_subprocess(
            [
                sys.executable,
                "-c",
                (
                    "import os,pathlib,sys,time; "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "time.sleep(30)"
                ),
                str(marker),
            ],
            config_json=b"",
            cwd=str(tmp_path),
            timeout=35,
            env=os.environ.copy(),
            settlement_callback=evidence.append,
            writer_epoch_members=(storage,),
        )

    assert caught.value.stdout == b"partial stdout"
    assert caught.value.stderr == b"partial stderr"
    assert not psutil.pid_exists(int(marker.read_text(encoding="ascii")))
    assert inspect_writer_epoch_family((storage,)).state == "quiesced"
    assert len(evidence) == 1


def test_trusted_runner_cannot_be_shadowed_from_project_cwd(tmp_path: Path):
    shadow = tmp_path / "crucible" / "common" / "runners" / "lang"
    shadow.mkdir(parents=True)
    for package in (
        tmp_path / "crucible",
        tmp_path / "crucible" / "common",
        tmp_path / "crucible" / "common" / "runners",
        shadow,
    ):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "python.py").write_text(
        "import json\n"
        "print(json.dumps({'values': {'tests_passed': 999.0}}))\n",
        encoding="utf-8",
    )

    result = _run_runner(
        "verify",
        "crucible.common.runners.lang.python",
        {
            "parameters": {
                "test_command": [
                    sys.executable,
                    "-c",
                    "print('1 passed in 0.01s')",
                ],
                "working_dir": str(tmp_path),
            },
        },
        cwd=str(tmp_path),
        timeout=30,
        log_dir=None,
    )
    assert result.returncode == 0
    assert result.values["tests_passed"] == 1.0

    # The trusted polyglot dispatcher must preserve the same isolation when
    # it launches the language adapter as a second subprocess.
    polyglot_result = _run_runner(
        "verify",
        "crucible.common.runners.polyglot",
        {
                "parameters": {
                    "languages": ["python"],
                    "working_dir": str(tmp_path),
                    "per_language": {
                        "python": {
                        "test_command": [
                            sys.executable,
                            "-c",
                            "print('1 passed in 0.01s')",
                        ],
                        },
                },
            },
        },
        cwd=str(tmp_path),
        timeout=30,
        log_dir=None,
    )
    assert polyglot_result.returncode == 0
    assert polyglot_result.values["tests_passed"] == 1.0


def test_repeated_runner_logs_preserve_every_attempt(runner_pkg: str, tmp_path: Path):
    spec = _spec({"build": f"{runner_pkg}.ok"})
    log_dir = tmp_path / "logs"

    build_and_verify(spec, slice_id="s0", cwd=str(tmp_path), log_dir=log_dir)
    build_and_verify(spec, slice_id="s1", cwd=str(tmp_path), log_dir=log_dir)

    attempts = tuple(log_dir.glob("build_*.log"))
    assert len(attempts) == 2
    assert all(path.read_bytes() for path in attempts)
    assert (log_dir / "build.log").read_bytes() in {
        path.read_bytes() for path in attempts
    }


def test_build_failed_short_circuits(runner_pkg: str, tmp_path: Path):
    spec = _spec({
        "build": f"{runner_pkg}.fail",
        "verify": f"{runner_pkg}.ok",  # never reached
    })
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "build_failed"
    assert outcome.build is not None and not outcome.build.ok
    assert outcome.verify is None


def test_verify_failed(runner_pkg: str, tmp_path: Path):
    spec = _spec({
        "build": f"{runner_pkg}.ok",
        "verify": f"{runner_pkg}.fail",
    })
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "verify_failed"
    assert outcome.build is not None and outcome.build.ok
    assert outcome.verify is not None and not outcome.verify.ok


def test_partial_when_tests_failed_nonzero(runner_pkg: str, tmp_path: Path):
    spec = _spec({
        "build": f"{runner_pkg}.ok",
        "verify": f"{runner_pkg}.partial",
    })
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "partial"
    assert outcome.verify is not None
    assert outcome.verify.values.get("tests_failed", 0.0) == 1.0


def test_runner_missing_when_no_build_runner(tmp_path: Path):
    spec = _spec({})  # no runners at all
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "runner_missing"
    assert outcome.build is None and outcome.verify is None


def test_missing_dev_verify_settles_mixed_writer_epoch_family(
    tmp_path: Path,
) -> None:
    from crucible.workflows._writer_epoch import (
        WriterEpochBoundary,
        arm_writer_epoch_family,
        capture_writer_epoch_owner,
        inspect_writer_epoch_family,
        quiesce_writer_epoch_family,
        synchronous_callback_settlement_receipt,
    )

    root = _MemoryWriterEpochStorage("breach-root")
    prior = arm_writer_epoch_family(
        (root,),
        owner=capture_writer_epoch_owner("test", "prior"),
        boundary=WriterEpochBoundary(
            "test_prior",
            "prior",
            "synchronous_callback",
            (),
            ("prior-worker",),
        ),
    )
    quiesce_writer_epoch_family(
        prior,
        synchronous_callback_settlement_receipt(
            prior,
            settled_owner_ids=(),
            joined_worker_ids=("prior-worker",),
            evidence_digest="sha256:" + "1" * 64,
            outcome="succeeded",
        ),
    )
    checkpoint = _MemoryWriterEpochStorage(
        "breach-checkpoint",
        parent_transaction_id=root.transaction_id,
    )

    outcome = build_and_verify(
        _spec({"build": "runner-not-used-by-dev-verify"}),
        slice_id="breach_r0_pre_fix",
        cwd=str(tmp_path),
        gate="dev_verify",
        writer_epoch_members=(root, checkpoint),
    )

    assert outcome.classification == "success"
    inspection = inspect_writer_epoch_family((root, checkpoint))
    assert inspection.state == "quiesced"
    assert inspection.generation == prior.generation + 1


def test_success_with_build_only_when_verify_unconfigured(
    runner_pkg: str, tmp_path: Path,
):
    """A spec that declares build but not verify is considered successful
    when build returns 0 — the build runner is the contract."""
    spec = _spec({"build": f"{runner_pkg}.ok"})
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "success"
    assert outcome.verify is None


def test_verify_role_wins_over_test_when_both_declared(
    runner_pkg: str, tmp_path: Path,
):
    """When ``verify`` AND ``test`` are both declared, ``verify`` is run."""
    spec = _spec({
        "build": f"{runner_pkg}.ok",
        "verify": f"{runner_pkg}.ok",
        "test": f"{runner_pkg}.fail",  # would fail if it were run
    })
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "success"
    # The verify runner that actually ran:
    assert outcome.verify is not None
    assert outcome.verify.runner.endswith(".ok")


# ----------------------------------------------------------------------
# dev_verify gate
# ----------------------------------------------------------------------


def test_declared_verification_capability_is_trusted_and_versioned():
    assert trusted_runner_has_capability(
        _DEEPSWE_DEV_VERIFY_RUNNER,
        DECLARED_VERIFICATION_CHECKPOINT_V1,
    )
    assert not trusted_runner_has_capability(
        _SWEBENCH_DEV_VERIFY_RUNNER,
        DECLARED_VERIFICATION_CHECKPOINT_V1,
    )
    assert not trusted_runner_has_capability(
        "third_party.claims_support",
        DECLARED_VERIFICATION_CHECKPOINT_V1,
    )


def test_dev_verify_gate_success(runner_pkg: str, tmp_path: Path):
    """gate='dev_verify' runs the dev_verify runner, skips build, and
    populates the dev_verify field (not verify)."""
    spec = _spec({
        "build": f"{runner_pkg}.fail",       # must NOT run under dev_verify gate
        "verify": f"{runner_pkg}.fail",      # must NOT run under dev_verify gate
        "dev_verify": f"{runner_pkg}.ok",
    })
    outcome = build_and_verify(
        spec, slice_id="breach_r0", cwd=str(tmp_path),
        gate="dev_verify",
    )
    assert outcome.classification == "success"
    assert outcome.build is None         # build phase skipped
    assert outcome.verify is None        # legacy verify field unused
    assert outcome.dev_verify is not None and outcome.dev_verify.ok
    assert outcome.dev_verify.runner.endswith(".ok")


def test_dev_verify_without_capability_does_not_receive_declared_controls(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}

    def fake_runner(role, runner, payload, **_kwargs):
        captured.update(payload["parameters"])
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    directive = ImplementationDirective(
        slice_id="visible-tests",
        intent="exercise the product-visible suite",
        files_in_scope=("tests/test_visible.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )

    outcome = build_and_verify(
        _spec({"dev_verify": _SWEBENCH_DEV_VERIFY_RUNNER}),
        slice_id=directive.slice_id,
        cwd=str(tmp_path),
        slice_directive=directive,
        gate="dev_verify",
    )

    assert outcome.classification == "success"
    assert "declared_verification_surface" not in captured
    assert "accepted_declared_verification_reference" not in captured

    with pytest.raises(ValueError, match="does not have the trusted"):
        build_and_verify(
            _spec({"dev_verify": _SWEBENCH_DEV_VERIFY_RUNNER}),
            cwd=str(tmp_path),
            verification_directives=(directive,),
            gate="dev_verify",
        )


def test_dev_verify_transports_normalized_declared_surface(
    monkeypatch,
    tmp_path: Path,
):
    captured = {}

    def fake_runner(role, runner, payload, **_kwargs):
        captured.update(payload["parameters"])
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    directive = ImplementationDirective(
        slice_id="declared-tests",
        intent="add focused tests",
        files_in_scope=("tests\\test_existing.py", "src/./module.py"),
        creates_paths=("tests/test_new.py",),
        verify_mode="tests",
        blast_radius={"max_files": 3},
    )
    outcome = build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        slice_id=directive.slice_id,
        cwd=str(tmp_path),
        slice_directive=directive,
        gate="dev_verify",
    )

    assert outcome.classification == "success"
    assert captured["declared_verification_surface"] == {
        "schema_version": "crucible.declared-verification-surface.v1",
        "slices": [{
            "slice_id": "declared-tests",
            "files_in_scope": ["tests/test_existing.py", "src/module.py"],
            "creates_paths": ["tests/test_new.py"],
        }],
    }


def test_dev_verify_rejects_spoofed_or_unsafe_declared_surface(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        build_verify_module,
        "_run_runner",
        lambda *_args, **_kwargs: pytest.fail("unsafe runner must not execute"),
    )
    spoofed = trusted_runner_overrides({
        "declared_verification_surface": {
            "schema_version": "crucible.declared-verification-surface.v1",
            "slices": [],
        },
    })
    with pytest.raises(ValueError, match="workflow-owned"):
        build_and_verify(
            _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
            cwd=str(tmp_path),
            extra_parameters=spoofed,
            gate="dev_verify",
        )

    unsafe = ImplementationDirective(
        slice_id="unsafe",
        intent="escape",
        files_in_scope=("../outside.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    with pytest.raises(ValueError, match="repository-relative"):
        build_and_verify(
            _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
            cwd=str(tmp_path),
            slice_directive=unsafe,
            gate="dev_verify",
        )


def test_dev_verify_transports_final_multi_slice_union(monkeypatch, tmp_path):
    captured = {}

    def fake_runner(role, runner, payload, **_kwargs):
        captured.update(payload["parameters"])
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)

    def directive(slice_id, path):
        return ImplementationDirective(
            slice_id=slice_id,
            intent="test",
            files_in_scope=(),
            creates_paths=(path,),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        )

    directives = (
        directive("left", "tests/test_left.py"),
        directive("right", "tests/test_right.py"),
    )
    outcome = build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        slice_id="tier__final",
        cwd=str(tmp_path),
        verification_directives=directives,
        gate="dev_verify",
    )

    assert outcome.classification == "success"
    assert [
        item["slice_id"]
        for item in captured["declared_verification_surface"]["slices"]
    ] == ["left", "right"]


def test_t2_final_surface_reprobes_accepted_t1_declared_test(
    monkeypatch,
    tmp_path,
):
    ancestor_directive = ImplementationDirective(
        slice_id="t1-visible-test",
        intent="add visible regression coverage",
        files_in_scope=(),
        creates_paths=("tests/ancestor_contract.js",),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    current_directive = ImplementationDirective(
        slice_id="t2-production",
        intent="change production behavior that can regress the prior test",
        files_in_scope=("src/runtime.js",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    t1 = Tier(
        tier_id="t1",
        slices=(Slice(id="t1-visible-test", directive=ancestor_directive),),
        depends_on=(),
    )
    t2 = Tier(
        tier_id="t2",
        slices=(Slice(id="t2-production", directive=current_directive),),
        depends_on=("t1",),
    )
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(
            tiers=(t1, t2),
        ),
    )

    from crucible.workflows import _partition

    def partition_for_resume(_corpus, tier, *, spec):  # noqa: ARG001
        if tier.tier_id == "t1":
            return {"t1-visible-test": object()}, ()
        raise AssertionError("only accepted ancestors may be partitioned")

    monkeypatch.setattr(_partition, "partition_for_resume", partition_for_resume)
    monkeypatch.setattr(
        build_verify_module,
        "_tier_lifecycle_acceptance",
        lambda *_args, **_kwargs: (True, ""),
    )
    directives = build_verify_module.accepted_plan_verification_directives(
        object(),
        spec,
        current_tier_id="t2",
    )
    assert directives == (ancestor_directive, current_directive)

    captured = {}

    def fake_runner(role, runner, payload, **_kwargs):
        captured.update(payload["parameters"])
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=0,
            values={"returncode": 0.0},
            raw_output=b"{}",
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    outcome = build_and_verify(
        spec,
        slice_id="t2__final",
        cwd=str(tmp_path),
        verification_directives=directives,
        gate="dev_verify",
    )

    assert outcome.classification == "success"
    surfaces = captured["declared_verification_surface"]["slices"]
    assert [item["slice_id"] for item in surfaces] == [
        "t1-visible-test",
        "t2-production",
    ]
    assert surfaces[0]["creates_paths"] == ["tests/ancestor_contract.js"]
    from crucible.products.deepswe.runners import deepswe_dev_verify

    assert deepswe_dev_verify._declared_test_files(
        captured,
        "javascript",
    ) == ("tests/ancestor_contract.js",)


def test_cumulative_surface_rejects_partially_landed_formal_ancestor(
    monkeypatch,
):
    def directive(slice_id: str) -> ImplementationDirective:
        return ImplementationDirective(
            slice_id=slice_id,
            intent=f"implement {slice_id}",
            files_in_scope=(f"src/{slice_id}.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        )

    first = Slice(id="ancestor-done", directive=directive("ancestor-done"))
    second = Slice(
        id="ancestor-pending",
        directive=directive("ancestor-pending"),
    )
    ancestor = Tier(tier_id="ancestor", slices=(first, second), depends_on=())
    current_directive = directive("current")
    current = Tier(
        tier_id="current",
        slices=(Slice(id="current", directive=current_directive),),
        depends_on=("ancestor",),
    )
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(ancestor, current)),
    )

    from crucible.workflows import _partition

    def partition_for_resume(_corpus, tier, *, spec):  # noqa: ARG001
        assert tier.tier_id == "ancestor"
        return {"ancestor-done": object()}, (second,)

    monkeypatch.setattr(_partition, "partition_for_resume", partition_for_resume)

    with pytest.raises(
        ValueError,
        match=r"dependency 'ancestor' is only partially landed.*ancestor-pending",
    ):
        build_verify_module.accepted_plan_verification_directives(
            object(),
            spec,
            current_tier_id="current",
        )


def test_cumulative_surface_requires_latest_plan_bound_landed_lifecycle(
    tmp_path,
):
    from assay import Implementation, LifecycleEvent
    from assay.corpus.store import Corpus
    from crucible.workflows import _partition
    from crucible.workflows._run import exec_ctx_now

    ancestor_directive = ImplementationDirective(
        slice_id="ancestor",
        intent="land the ancestor",
        files_in_scope=("src/ancestor.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    current_directive = ImplementationDirective(
        slice_id="current",
        intent="consume the ancestor",
        files_in_scope=("src/current.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    ancestor_slice = Slice(id="ancestor", directive=ancestor_directive)
    ancestor = Tier(
        tier_id="ancestor",
        slices=(ancestor_slice,),
        depends_on=(),
    )
    current = Tier(
        tier_id="current",
        slices=(Slice(id="current", directive=current_directive),),
        depends_on=("ancestor",),
    )
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(ancestor, current)),
    )
    spec_dir = tmp_path / "corpus"
    spec_dir.mkdir()
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)
    implementation = Implementation(
        slice_id="ancestor",
        base_commit=spec.project_context.base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(b"diff", kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=("src/ancestor.py",),
        classification="success",
        execution=exec_ctx_now("test"),
        scope_escapes=(),
        artifacts={},
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        spec,
        ancestor,
        ancestor_slice,
        implementation,
    )
    corpus.commit(implementations=(implementation,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier ancestor: complete",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )

    directives = build_verify_module.accepted_plan_verification_directives(
        corpus,
        corpus.latest_spec(),
        current_tier_id="current",
    )
    assert directives == (ancestor_directive, current_directive)
    with pytest.raises(
        ValueError,
        match="do not share one lifecycle checkpoint",
    ):
        build_verify_module.accepted_predecessor_verification_reference(
            corpus,
            corpus.latest_spec(),
            current_tier_id="current",
        )

    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="abandoned",
            reason="tier ancestor: review_failed",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"predecessor 'ancestor' lacks plan-bound landed lifecycle",
    ):
        build_verify_module.accepted_plan_verification_directives(
            corpus,
            corpus.latest_spec(),
            current_tier_id="current",
        )


def test_tier_lifecycle_cannot_retroactively_target_a_future_spec(
    tmp_path: Path,
) -> None:
    """An old event cannot adopt Implementations visible only in its future."""
    from assay import Implementation, LifecycleEvent
    from assay.corpus.store import Corpus
    from crucible.workflows import _partition
    from crucible.workflows._run import exec_ctx_now

    directive = ImplementationDirective(
        slice_id="landed",
        intent="land once",
        files_in_scope=("src/landed.py",),
        creates_paths=(),
        verify_mode="skip",
        blast_radius={"max_files": 1},
    )
    slice_ = Slice(id="landed", directive=directive)
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    spec = replace(_spec({}), tier_dag=TierDAG(tiers=(tier,)))
    corpus_dir = tmp_path / "future-lifecycle-corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(spec)
    implementation = Implementation(
        slice_id="landed",
        base_commit=spec.project_context.base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(b"diff", kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=("src/landed.py",),
        classification="success",
        execution=exec_ctx_now("tier"),
        scope_escapes=(),
        artifacts={},
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        slice_,
        implementation,
    )
    corpus.commit(implementations=(implementation,))
    # This event is attached at v2 but falsely claims v3, which does not yet
    # exist. Once v3 appears, membership alone used to admit it retroactively.
    corpus.commit(lifecycle_events=(LifecycleEvent(
        kind="landed",
        reason="tier t0: complete",
        affected_versions=(3,),
        execution=exec_ctx_now("tier"),
    ),))
    corpus.put_spec(corpus.latest_spec().with_tier_dag(TierDAG(tiers=(tier,))))

    accepted, reason = build_verify_module._tier_lifecycle_acceptance(
        corpus,
        corpus.latest_spec(),
        tier,
        {"landed": implementation},
    )

    assert accepted is False
    assert "not first attached at its affected Spec" in reason


def test_cumulative_surface_includes_fully_landed_independent_predecessor(
    monkeypatch,
):
    predecessor_directive = ImplementationDirective(
        slice_id="independent-test",
        intent="establish a shared-workspace regression test",
        files_in_scope=(),
        creates_paths=("tests/test_independent.py",),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    current_directive = ImplementationDirective(
        slice_id="current-code",
        intent="change code later in canonical order",
        files_in_scope=("src/current.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    predecessor = Tier(
        tier_id="independent",
        slices=(
            Slice(id="independent-test", directive=predecessor_directive),
        ),
        depends_on=(),
    )
    current = Tier(
        tier_id="current",
        slices=(Slice(id="current-code", directive=current_directive),),
        depends_on=(),
    )
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(predecessor, current)),
    )

    from crucible.workflows import _partition

    def partition_for_resume(_corpus, tier, *, spec):  # noqa: ARG001
        assert tier.tier_id == "independent"
        return {"independent-test": object()}, ()

    monkeypatch.setattr(_partition, "partition_for_resume", partition_for_resume)
    monkeypatch.setattr(
        build_verify_module,
        "_tier_lifecycle_acceptance",
        lambda *_args, **_kwargs: (True, ""),
    )

    directives = build_verify_module.accepted_plan_verification_directives(
        object(),
        spec,
        current_tier_id="current",
    )

    assert directives == (predecessor_directive, current_directive)


def test_cumulative_surface_rejects_partial_independent_tier_without_checkpoint(
    monkeypatch,
):
    def directive(slice_id: str) -> ImplementationDirective:
        return ImplementationDirective(
            slice_id=slice_id,
            intent=f"implement {slice_id}",
            files_in_scope=(f"src/{slice_id}.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        )

    landed_slice = Slice(id="landed", directive=directive("landed"))
    pending_slice = Slice(id="pending", directive=directive("pending"))
    predecessor = Tier(
        tier_id="independent",
        slices=(landed_slice, pending_slice),
        depends_on=(),
    )
    current_directive = directive("current")
    current = Tier(
        tier_id="current",
        slices=(Slice(id="current", directive=current_directive),),
        depends_on=(),
    )
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(predecessor, current)),
    )

    from crucible.workflows import _partition

    def partition_for_resume(_corpus, tier, *, spec):  # noqa: ARG001
        assert tier.tier_id == "independent"
        return {"landed": object()}, (pending_slice,)

    monkeypatch.setattr(_partition, "partition_for_resume", partition_for_resume)

    with pytest.raises(
        ValueError,
        match=r"independent predecessor 'independent' is only partially landed",
    ):
        build_verify_module.accepted_plan_verification_directives(
            object(),
            spec,
            current_tier_id="current",
        )


def test_cumulative_surface_rejects_landed_later_canonical_tier(monkeypatch):
    def directive(slice_id: str) -> ImplementationDirective:
        return ImplementationDirective(
            slice_id=slice_id,
            intent=f"implement {slice_id}",
            files_in_scope=(f"src/{slice_id}.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        )

    current_directive = directive("current")
    later_slice = Slice(id="later", directive=directive("later"))
    current = Tier(
        tier_id="current",
        slices=(Slice(id="current", directive=current_directive),),
        depends_on=(),
    )
    later = Tier(tier_id="later", slices=(later_slice,), depends_on=())
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(current, later)),
    )

    from crucible.workflows import _partition

    monkeypatch.setattr(
        _partition,
        "partition_for_resume",
        lambda _corpus, tier, *, spec: (
            ({"later": object()}, ())
            if tier.tier_id == "later" else ({}, tuple(tier.slices))
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"landed slices in later canonical tier 'later'",
    ):
        build_verify_module.accepted_plan_verification_directives(
            object(),
            spec,
            current_tier_id="current",
        )
    assert build_verify_module.accepted_plan_verification_directives(
        object(),
        spec,
        current_tier_id="current",
        ignore_landed_later=True,
    ) == (current_directive,)


def test_direct_completed_tier_replay_rejects_landed_later_tier(monkeypatch):
    def directive(slice_id: str) -> ImplementationDirective:
        return ImplementationDirective(
            slice_id=slice_id,
            intent=f"implement {slice_id}",
            files_in_scope=(f"src/{slice_id}.py",),
            creates_paths=(),
            verify_mode="tests",
            blast_radius={"max_files": 1},
        )

    current_slice = Slice(id="current", directive=directive("current"))
    later_slice = Slice(id="later", directive=directive("later"))
    current = Tier(
        tier_id="current",
        slices=(current_slice,),
        depends_on=(),
    )
    later = Tier(tier_id="later", slices=(later_slice,), depends_on=())
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(current, later)),
    )

    from crucible.workflows import _partition

    def partition(_corpus, tier, *, spec):  # noqa: ARG001
        selected = current_slice if tier.tier_id == "current" else later_slice
        return {selected.id: object()}, ()

    monkeypatch.setattr(_partition, "partition_for_resume", partition)

    with pytest.raises(
        ValueError,
        match=r"landed slices in later canonical tier 'later'",
    ):
        build_verify_module.accepted_plan_verification_directives(
            object(),
            spec,
            current_tier_id="current",
        )


def test_accepted_checkpoint_rejects_tier_spec_and_surface_mismatch():
    from crucible.common.runners.declared_verification import (
        accepted_verification_reference_payload,
        surface_from_directives,
    )

    directive = ImplementationDirective(
        slice_id="accepted",
        intent="publish accepted outcome authority",
        files_in_scope=("src/accepted.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1},
    )
    reference = accepted_verification_reference_payload(
        predecessor_surface=surface_from_directives((directive,)),
        language="python",
        parser_kind="pytest",
        outcomes=(),
    )
    authority_hash = "sha256:" + "a" * 64
    encoded = build_verify_module.encode_accepted_verification_checkpoint(
        reference,
        tier_id="t0",
        spec_execution_authority_hash=authority_hash,
    )

    with pytest.raises(ValueError, match="tier mismatch"):
        build_verify_module.parse_accepted_verification_checkpoint(
            encoded,
            expected_tier_id="spoofed",
            expected_spec_execution_authority_hash=authority_hash,
            expected_verification_directives=(directive,),
        )
    with pytest.raises(ValueError, match="authority mismatch"):
        build_verify_module.parse_accepted_verification_checkpoint(
            encoded,
            expected_tier_id="t0",
            expected_spec_execution_authority_hash="sha256:" + "b" * 64,
            expected_verification_directives=(directive,),
        )
    with pytest.raises(ValueError, match="surface mismatch"):
        build_verify_module.parse_accepted_verification_checkpoint(
            encoded,
            expected_tier_id="t0",
            expected_spec_execution_authority_hash=authority_hash,
            expected_verification_directives=(replace(
                directive,
                files_in_scope=("src/different.py",),
            ),),
        )


def test_declared_reference_decode_recursion_is_typed_failure():
    recursive_json = b"[" * 50_000 + b"0" + b"]" * 50_000
    runner_result = build_verify_module.RunnerResult(
        role="dev_verify",
        runner=_DEEPSWE_DEV_VERIFY_RUNNER,
        returncode=0,
        values={},
        raw_output=recursive_json,
        stderr=b"",
        duration_s=0.0,
    )
    outcome = build_verify_module.BuildVerifyOutcome(
        classification="success",
        build=None,
        verify=None,
        dev_verify=runner_result,
    )

    with pytest.raises(ValueError, match="did not emit a JSON"):
        build_verify_module.declared_verification_reference_from_outcome(
            outcome,
            verification_directives=(),
        )


def test_checkpoint_decode_recursion_is_typed_failure():
    recursive_json = b"[" * 50_000 + b"0" + b"]" * 50_000

    with pytest.raises(ValueError, match="checkpoint is not JSON"):
        build_verify_module.parse_accepted_verification_checkpoint(
            recursive_json,
            expected_tier_id="t0",
            expected_spec_execution_authority_hash="sha256:" + "a" * 64,
            expected_verification_directives=(),
        )


def test_restart_lineage_ignores_old_later_tier_then_uses_replayed_checkpoint(
    tmp_path,
    monkeypatch,
):
    import hashlib

    from assay import Implementation, LifecycleEvent
    from assay.corpus.store import Corpus
    from crucible.common.runners.declared_verification import (
        accepted_verification_reference_payload,
        surface_from_directives,
    )
    from crucible.products.deepswe.runners.dev_verify_outcome import (
        parse_test_outcome,
    )
    from crucible.workflows import _partition
    from crucible.workflows._run import exec_ctx_now

    t0_directive = ImplementationDirective(
        slice_id="t0_slice",
        intent="establish focused predecessor coverage",
        files_in_scope=(),
        creates_paths=("tests/test_contract.py",),
        verify_mode="tests",
        blast_radius={"max_files": 1, "languages": ["python"]},
    )
    t1_directive = ImplementationDirective(
        slice_id="t1_slice",
        intent="consume the replayed predecessor",
        files_in_scope=("src/consumer.py",),
        creates_paths=(),
        verify_mode="tests",
        blast_radius={"max_files": 1, "languages": ["python"]},
    )
    t0_slice = Slice(id="t0_slice", directive=t0_directive)
    t1_slice = Slice(id="t1_slice", directive=t1_directive)
    t0 = Tier(tier_id="t0", slices=(t0_slice,), depends_on=())
    t1 = Tier(tier_id="t1", slices=(t1_slice,), depends_on=("t0",))
    spec = replace(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        tier_dag=TierDAG(tiers=(t0, t1)),
    )
    spec_dir = tmp_path / "corpus"
    spec_dir.mkdir()
    corpus = Corpus(spec_dir)
    corpus.put_spec(spec)
    command = (
        "cd /app && pytest tests/test_contract.py --maxfail=5 --tb=short -q"
    )

    def accepted_t0_implementation(*, tests_run: int) -> Implementation:
        parsed = parse_test_outcome(
            language="python",
            command=command,
            actual_returncode=0,
            stdout=f"{tests_run} passed in 0.01s",
            stderr="",
            parser_hint="pytest",
        )
        reference = accepted_verification_reference_payload(
            predecessor_surface=surface_from_directives((t0_directive,)),
            language="python",
            parser_kind="pytest",
            outcomes=((
                "tests/test_contract.py",
                hashlib.sha256(command.encode()).hexdigest(),
                parsed.to_payload(),
            ),),
        )
        checkpoint = build_verify_module.encode_accepted_verification_checkpoint(
            reference,
            tier_id="t0",
            spec_execution_authority_hash=(
                _partition.spec_execution_structure_hash(corpus.latest_spec())
            ),
        )
        checkpoint_ref = corpus.put_artifact(
            checkpoint,
            kind="accepted_verification_checkpoints",
        )
        implementation = Implementation(
            slice_id=t0_slice.id,
            base_commit=spec.project_context.base_commit,
            result_commit=None,
            diff_ref=corpus.put_artifact(b"t0", kind="diffs"),
            transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
            files_modified=("tests/test_contract.py",),
            classification="success",
            execution=exec_ctx_now("tier"),
            scope_escapes=(),
            artifacts={
                build_verify_module.ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT:
                checkpoint_ref,
            },
        )
        return _partition.attach_resume_identity(
            corpus,
            corpus.latest_spec(),
            t0,
            t0_slice,
            implementation,
        )

    corpus.commit(implementations=(accepted_t0_implementation(tests_run=1),))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )
    old_t1 = Implementation(
        slice_id=t1_slice.id,
        base_commit=spec.project_context.base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(b"old-t1", kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=("src/consumer.py",),
        classification="success",
        execution=exec_ctx_now("old_attempt"),
        scope_escapes=(),
        artifacts={},
    )
    old_t1 = _partition.attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        t1,
        t1_slice,
        old_t1,
    )
    corpus.commit(implementations=(old_t1,))

    with pytest.raises(ValueError, match="later canonical tier 't1'"):
        build_verify_module.accepted_plan_verification_directives(
            corpus,
            corpus.latest_spec(),
            current_tier_id="t0",
        )
    assert build_verify_module.accepted_plan_verification_directives(
        corpus,
        corpus.latest_spec(),
        current_tier_id="t0",
        ignore_landed_later=True,
    ) == (t0_directive,)

    # Model Architect's restart=True t0 replay on the fresh DeepSWE workspace.
    replayed_t0 = accepted_t0_implementation(tests_run=2)
    corpus.commit(implementations=(replayed_t0,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=exec_ctx_now("tier"),
        ),
    )

    checkpoint_ref = replayed_t0.artifacts[
        build_verify_module.ACCEPTED_VERIFICATION_CHECKPOINT_ARTIFACT
    ]
    observed_reads: list[tuple[str, int]] = []
    bounded_read = corpus.get_artifact_bounded

    def observe_bounded(ref, *, max_bytes):
        observed_reads.append((str(ref), max_bytes))
        return bounded_read(ref, max_bytes=max_bytes)

    def forbid_unbounded(*_args, **_kwargs):
        raise AssertionError("checkpoint replay used an unbounded artifact read")

    monkeypatch.setattr(corpus, "get_artifact_bounded", observe_bounded)
    monkeypatch.setattr(corpus, "get_artifact", forbid_unbounded)

    replayed = build_verify_module.accepted_predecessor_verification_reference(
        corpus,
        corpus.latest_spec(),
        current_tier_id="t1",
        ignore_landed_later=True,
    )
    assert replayed is not None
    assert replayed["outcomes"][0]["outcome"]["tests_run"] == 2
    assert (
        str(checkpoint_ref),
        build_verify_module._ACCEPTED_VERIFICATION_CHECKPOINT_MAX_BYTES,
    ) in observed_reads


def test_dev_verify_gate_failed_is_distinct_classification(
    runner_pkg: str, tmp_path: Path,
):
    """A failing dev_verify gate yields 'dev_verify_failed', NOT
    'verify_failed' — the round's adversarial tests caught the impl;
    the dataset gate did not run."""
    spec = _spec({"dev_verify": f"{runner_pkg}.fail"})
    outcome = build_and_verify(
        spec, slice_id="breach_r1", cwd=str(tmp_path),
        gate="dev_verify",
    )
    assert outcome.classification == "dev_verify_failed"
    assert outcome.dev_verify is not None and not outcome.dev_verify.ok
    assert outcome.verify is None


def test_dev_verify_infrastructure_evidence_is_non_retriable(
    monkeypatch,
    tmp_path: Path,
):
    def fake_runner(role, runner, payload, **_kwargs):
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=2,
            values={
                "returncode": 2.0,
                "dev_verify_baseline_unavailable": 1.0,
                "dev_verify_infrastructure_unavailable": 1.0,
            },
            raw_output=b'{"raw_output":"sealed dependency unavailable"}',
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    outcome = build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        cwd=str(tmp_path),
        gate="dev_verify",
    )

    assert outcome.classification == "dev_verify_unavailable"
    from crucible.workflows.tier import _is_model_retriable_build_failure

    assert not _is_model_retriable_build_failure(outcome)


def test_dev_verify_early_unavailable_flags_cross_subprocess_boundary(
    tmp_path: Path,
):
    outcome = build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        cwd=str(tmp_path),
        gate="dev_verify",
        extra_parameters={
            "deepswe_metadata": {
                "container_id": "unused-container",
                "language": "unsupported-language",
            },
        },
    )

    assert outcome.classification == "dev_verify_unavailable"
    assert outcome.dev_verify is not None
    assert outcome.dev_verify.returncode == 2
    assert outcome.dev_verify.values[
        "dev_verify_infrastructure_unavailable"
    ] == 1.0
    assert outcome.dev_verify.values[
        "dev_verify_language_unavailable"
    ] == 1.0


def test_actionable_rc2_without_infrastructure_evidence_remains_retriable(
    monkeypatch,
    tmp_path: Path,
):
    """A model-created compile/parser failure is not mislabeled as infra."""

    def fake_runner(role, runner, payload, **_kwargs):
        return build_verify_module.RunnerResult(
            role=role,
            runner=runner,
            returncode=2,
            values={
                "returncode": 2.0,
                "dev_verify_candidate_process_returncode": 1.0,
                "dev_verify_candidate_returncode": 2.0,
                "dev_verify_structurally_valid": 0.0,
                "dev_verify_candidate_incomparable": 1.0,
                "dev_verify_baseline_attempted": 0.0,
                "dev_verify_baseline_unavailable": 0.0,
                "dev_verify_infrastructure_unavailable": 0.0,
            },
            raw_output=b'{"raw_output":"src/index.ts: syntax error"}',
            stderr=b"",
            duration_s=0.0,
        )

    monkeypatch.setattr(build_verify_module, "_run_runner", fake_runner)
    outcome = build_and_verify(
        _spec({"dev_verify": _DEEPSWE_DEV_VERIFY_RUNNER}),
        cwd=str(tmp_path),
        gate="dev_verify",
    )

    assert outcome.classification == "dev_verify_failed"
    from crucible.workflows.tier import _is_model_retriable_build_failure

    assert _is_model_retriable_build_failure(outcome)


def test_dev_verify_gate_partial(runner_pkg: str, tmp_path: Path):
    """dev_verify gate honors the partial classification (rc 0 but
    tests_failed > 0)."""
    spec = _spec({"dev_verify": f"{runner_pkg}.partial"})
    outcome = build_and_verify(
        spec, slice_id="breach_r2", cwd=str(tmp_path),
        gate="dev_verify",
    )
    assert outcome.classification == "partial"
    assert outcome.dev_verify is not None
    assert outcome.dev_verify.values.get("tests_failed", 0.0) == 1.0


def test_dev_verify_gate_missing_runner_is_success(tmp_path: Path):
    """A spec with no dev_verify runner collapses to success under the
    dev_verify gate (build skipped, no gate to run) — same contract as
    an absent verify runner."""
    spec = _spec({"build": "stub_runners.ok"})  # no dev_verify entry
    outcome = build_and_verify(
        spec, slice_id="breach_r0", cwd=str(tmp_path),
        gate="dev_verify",
    )
    assert outcome.classification == "success"
    assert outcome.build is None
    assert outcome.dev_verify is None


def test_verify_gate_default_unchanged(runner_pkg: str, tmp_path: Path):
    """The default gate='verify' path still runs build + verify and
    never touches the dev_verify runner."""
    spec = _spec({
        "build": f"{runner_pkg}.ok",
        "verify": f"{runner_pkg}.ok",
        "dev_verify": f"{runner_pkg}.fail",  # must NOT run under verify gate
    })
    outcome = build_and_verify(spec, slice_id="s0", cwd=str(tmp_path))
    assert outcome.classification == "success"
    assert outcome.dev_verify is None
    assert outcome.verify is not None and outcome.verify.ok


def test_build_and_verify_rejects_unknown_gate(runner_pkg: str, tmp_path: Path):
    spec = _spec({"build": f"{runner_pkg}.ok"})
    with pytest.raises(ValueError, match="gate must be"):
        build_and_verify(spec, slice_id="s0", cwd=str(tmp_path), gate="bogus")


# ----------------------------------------------------------------------
# Path validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_path", [
    "a; rm -rf /",     # shell meta
    "../../etc/passwd",  # filesystem traversal via slash
    "no spaces allowed",
    "trailing.dot.",
    ".leading_dot",
    "1starts_with_digit",
    "",
])
def test_validate_runner_path_rejects_bad_inputs(bad_path: str):
    with pytest.raises(ValueError, match="dotted Python identifier"):
        _validate_runner_path(bad_path)


@pytest.mark.parametrize("good_path", [
    "crucible.products.swebench.runners.swebench_pytest",
    "stub_runners.ok",
    "single_module",
])
def test_validate_runner_path_accepts_good_inputs(good_path: str):
    _validate_runner_path(good_path)  # no raise
