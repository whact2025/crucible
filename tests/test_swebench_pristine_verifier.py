# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Hermetic contracts for patch-only pristine SWE-bench verification."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from crucible.products.swebench import pristine_verifier as verifier
from crucible.products.swebench.dataset import SwebenchInstance
from crucible.products.swebench.instance_container import InstanceContainerSpec


_BASE_COMMIT = "a" * 40
_IMAGE_DIGEST = "registry.example/repo@sha256:" + "b" * 64
_OTHER_IMAGE_DIGEST = "registry.example/repo@sha256:" + "c" * 64
_SOURCE_PATCH = b"""\
diff --git a/src/widget.py b/src/widget.py
index 1111111..2222222 100644
--- a/src/widget.py
+++ b/src/widget.py
@@ -1 +1 @@
-VALUE = 0
+VALUE = 1
"""


def _instance() -> SwebenchInstance:
    return SwebenchInstance(
        instance_id="example__project-1",
        repo="example/project",
        base_commit=_BASE_COMMIT,
        patch="GOLD_PATCH_MUST_NOT_LEAK",
        test_patch="HIDDEN_TEST_PATCH_MUST_NOT_LEAK",
        problem_statement="Fix the public widget behavior.",
        fail_to_pass=("HIDDEN_FAIL_TO_PASS_MUST_NOT_LEAK",),
        pass_to_pass=("HIDDEN_PASS_TO_PASS_MUST_NOT_LEAK",),
        version="1.0",
    )


def _runner_result(
    *,
    runner_rc: int = 0,
    candidate_rc: int = 0,
    comparison_mode: str = "baseline_match",
    preexisting: int = 0,
    new: int = 0,
    worsened: int = 0,
    candidate_failed: int = 0,
    candidate_errored: int = 0,
    baseline_unavailable: int = 0,
) -> subprocess.CompletedProcess[bytes]:
    payload = {
        "values": {
            "returncode": float(runner_rc),
            "python_returncode": float(runner_rc),
            "visible_test_candidate_returncode": float(candidate_rc),
            "visible_test_preexisting_failures": float(preexisting),
            "visible_test_new_failures": float(new),
            "visible_test_worsened_failures": float(worsened),
            "visible_test_candidate_tests_failed": float(candidate_failed),
            "visible_test_candidate_tests_errored": float(candidate_errored),
            "visible_test_baseline_unavailable": float(baseline_unavailable),
        },
        "raw_output": (
            "candidate output\n"
            "[crucible clean-base comparison]\n"
            f"mode={comparison_mode}"
        ),
    }
    return subprocess.CompletedProcess(
        ["trusted-runner"],
        runner_rc,
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
    )


class _Quota:
    def __init__(self, harness: "_Harness") -> None:
        self._harness = harness

    def __enter__(self) -> "_Quota":
        self._harness.events.append("quota_enter")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._harness.events.append("quota_exit")
        return False

    def retire_kernel_lease(self) -> None:
        self._harness.events.append("quota_retire")


class _Harness:
    """Realistic fake for the verifier's explicitly hookable boundaries."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.workspace = tmp_path / "fresh-verifier-workspace"
        self.author_workspace = tmp_path / "AUTHOR_WORKSPACE_MUST_NOT_CROSS"
        self.events: list[str] = []
        self.setup_call: dict[str, Any] = {}
        self.runner_call: dict[str, Any] = {}
        self.apply_check_rc = 0
        self.apply_rc = 0
        self.diff_rc = 1
        self.staged_paths = b"src/widget.py\0"
        self.actual_image_digest = _IMAGE_DIGEST
        self.container_absent = True
        self.runner_result = _runner_result()
        self.runner_error: BaseException | None = None
        self.setup_error: BaseException | None = None
        self.patch = _SOURCE_PATCH
        self.patch_sha256 = hashlib.sha256(self.patch).hexdigest()
        self._install()

    def _install(self) -> None:
        harness = self

        self.monkeypatch.setattr(
            verifier,
            "verify_immutable_image",
            lambda digest: harness.events.append("verify_image"),
        )
        self.monkeypatch.setattr(
            verifier,
            "reconcile_orphans",
            lambda scope: harness.events.append("reconcile") or (),
        )

        @contextlib.contextmanager
        def fake_repo_workspace(instance, *, cache_dir=None):
            harness.events.append("workspace_enter")
            assert instance is _instance_ref[0]
            assert cache_dir == harness.tmp_path / "cache"
            harness.workspace.mkdir(parents=True)
            (harness.workspace / ".git").mkdir()
            try:
                yield harness.workspace
            finally:
                shutil.rmtree(harness.workspace, ignore_errors=True)
                harness.events.append("workspace_exit")

        # The context manager verifies identity against the instance used by
        # ``verify`` without storing hidden dataset fields in the fake.
        _instance_ref[:] = [_instance()]
        self.instance = _instance_ref[0]
        self.monkeypatch.setattr(verifier, "repo_workspace", fake_repo_workspace)

        def fake_quota(paths, *, limit_bytes, on_exceeded, require_kernel):
            harness.events.append("quota_create")
            assert tuple(paths) == (harness.workspace.resolve(),)
            assert limit_bytes > 0
            assert callable(on_exceeded)
            assert require_kernel is True
            return _Quota(harness)

        self.monkeypatch.setattr(verifier, "DirectoryGrowthQuota", fake_quota)

        def fake_resolve(instance):
            harness.events.append("resolve_spec")
            assert instance is harness.instance
            return InstanceContainerSpec(
                image="mutable.example/repo:latest",
                install_command="python -m pip install -e .",
            )

        self.monkeypatch.setattr(verifier, "resolve_container_spec", fake_resolve)

        def fake_setup(workspace, container_name, spec, *, labels=None):
            harness.events.append("setup")
            harness.setup_call = {
                "workspace": Path(workspace),
                "container_name": container_name,
                "spec": spec,
                "labels": dict(labels or {}),
            }
            if harness.setup_error is not None:
                raise harness.setup_error
            return "verifier-container"

        self.monkeypatch.setattr(verifier, "setup_instance_container", fake_setup)

        def fake_container_image(container_id):
            harness.events.append("container_image")
            assert container_id == "verifier-container"
            return harness.actual_image_digest

        self.monkeypatch.setattr(
            verifier, "container_immutable_image", fake_container_image
        )

        def fake_container_git(container_id, arguments, *, timeout=30):
            harness.events.append("container_head")
            assert container_id == "verifier-container"
            assert arguments == ["rev-parse", "--verify", "HEAD^{commit}"]
            assert timeout > 0
            return subprocess.CompletedProcess(
                ["docker", "exec"], 0, stdout=(_BASE_COMMIT + "\n").encode(),
                stderr=b"",
            )

        self.monkeypatch.setattr(verifier, "_run_container_git", fake_container_git)

        def fake_git(
            workspace,
            arguments,
            *,
            input_bytes=None,
            timeout=verifier._GIT_TIMEOUT_S,
        ):
            assert Path(workspace) == harness.workspace.resolve()
            if arguments == ["rev-parse", "--verify", "HEAD^{commit}"]:
                harness.events.append("host_head")
                return subprocess.CompletedProcess(
                    ["git", *arguments],
                    0,
                    stdout=(_BASE_COMMIT + "\n").encode(),
                    stderr=b"",
                )
            if arguments[:3] == ["apply", "--check", "--index"]:
                harness.events.append("apply_check")
                assert input_bytes == harness.patch
                return subprocess.CompletedProcess(
                    ["git", *arguments],
                    harness.apply_check_rc,
                    stdout=b"",
                    stderr=(
                        b"patch does not apply"
                        if harness.apply_check_rc else b""
                    ),
                )
            if arguments[:2] == ["apply", "--index"]:
                harness.events.append("apply")
                assert input_bytes == harness.patch
                return subprocess.CompletedProcess(
                    ["git", *arguments],
                    harness.apply_rc,
                    stdout=b"",
                    stderr=b"malformed patch" if harness.apply_rc else b"",
                )
            if arguments == ["diff", "--quiet", "HEAD", "--"]:
                harness.events.append("diff_check")
                return subprocess.CompletedProcess(
                    ["git", *arguments], harness.diff_rc, stdout=b"", stderr=b""
                )
            if arguments == [
                "diff", "--cached", "--name-only", "-z", "HEAD", "--"
            ]:
                harness.events.append("staged_paths")
                return subprocess.CompletedProcess(
                    ["git", *arguments],
                    0,
                    stdout=harness.staged_paths,
                    stderr=b"",
                )
            raise AssertionError(f"unexpected Git invocation: {arguments!r}")

        self.monkeypatch.setattr(verifier, "_run_git", fake_git)

        def fake_runner(workspace, payload, *, timeout):
            harness.events.append("runner")
            harness.runner_call = {
                "workspace": Path(workspace),
                "payload": payload,
                "timeout": timeout,
            }
            if harness.runner_error is not None:
                raise harness.runner_error
            return harness.runner_result

        self.monkeypatch.setattr(verifier, "_invoke_runner", fake_runner)

        def fake_teardown(container_id, image=None):
            harness.events.append("teardown")
            assert container_id == "verifier-container"
            assert image is None

        self.monkeypatch.setattr(
            verifier, "teardown_instance_container", fake_teardown
        )

        def fake_absent(container_id):
            harness.events.append("container_absent")
            assert container_id == "verifier-container"
            return harness.container_absent

        self.monkeypatch.setattr(verifier, "_container_is_absent", fake_absent)

    def verify(self) -> verifier.PristineVerificationReport:
        return verifier.verify_source_patch(
            self.instance,
            source_patch=self.patch,
            expected_sha256=self.patch_sha256,
            expected_image_digest=_IMAGE_DIGEST,
            repo_cache_dir=self.tmp_path / "cache",
            scope=str(self.tmp_path / "spec"),
            timeout=90,
        )


# A tiny mutable cell lets the context-manager assertion use the exact object
# that the harness passes without capturing an incompletely initialized self.
_instance_ref: list[SwebenchInstance] = []


@pytest.mark.parametrize(
    "expected_sha256",
    ["0" * 64, "A" * 64, "not-a-sha256"],
)
def test_hash_invariant_fails_before_workspace_or_docker(
    monkeypatch: pytest.MonkeyPatch,
    expected_sha256: str,
) -> None:
    patch = _SOURCE_PATCH

    def unexpected(*args, **kwargs):
        pytest.fail("hash mismatch must fail before workspace or Docker setup")

    for name in (
        "verify_immutable_image",
        "reconcile_orphans",
        "repo_workspace",
        "resolve_container_spec",
        "setup_instance_container",
        "_run_git",
        "_invoke_runner",
    ):
        monkeypatch.setattr(verifier, name, unexpected)

    report = verifier.verify_source_patch(
        _instance(),
        source_patch=patch,
        expected_sha256=expected_sha256,
        expected_image_digest=_IMAGE_DIGEST,
        repo_cache_dir=None,
        scope="/spec",
    )

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.patch_sha256 == hashlib.sha256(patch).hexdigest()
    assert report.apply_check_rc is None
    assert report.runner_rc is None
    assert report.cleanup_status == "not_started"
    assert "hash" in report.reason


def test_container_absence_polls_through_async_docker_rm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    returncodes = iter((0, 1))
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, next(returncodes), stdout=b"", stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(verifier.time, "sleep", lambda seconds: None)

    assert verifier._container_is_absent("container-1", timeout_s=1.0)
    assert calls == [
        ["docker", "inspect", "container-1"],
        ["docker", "inspect", "container-1"],
    ]


def test_validated_source_paths_accepts_normalized_source_paths() -> None:
    assert verifier._validated_source_paths(
        b"src/widget.py\0pkg/data.json\0"
    ) == ("src/widget.py", "pkg/data.json")


@pytest.mark.parametrize(
    "payload",
    [
        b"tests/test_widget.py\0",
        b"/etc/passwd\0",
        b"src/../secret.py\0",
        b"src/\xff.py\0",
        b"src/widget.py\0src/widget.py\0",
        b"src/widget.py",
    ],
    ids=(
        "standard-test-path",
        "absolute-path",
        "parent-traversal",
        "non-utf8",
        "duplicate",
        "missing-terminal-nul",
    ),
)
def test_validated_source_paths_rejects_unsafe_or_ambiguous_payloads(
    payload: bytes,
) -> None:
    assert verifier._validated_source_paths(payload) is None


def test_setup_precedes_patch_apply_and_runner_receives_public_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.PASSED
    assert report.cleanup_status == "complete"
    assert harness.events.index("setup") < harness.events.index("apply_check")
    assert harness.events.index("apply_check") < harness.events.index("apply")
    assert harness.events.index("apply") < harness.events.index("runner")
    assert harness.events.index("runner") < harness.events.index("teardown")
    assert harness.setup_call["workspace"] == harness.workspace.resolve()
    assert harness.author_workspace != harness.setup_call["workspace"]
    assert harness.setup_call["spec"].image == _IMAGE_DIGEST
    labels = harness.setup_call["labels"]
    assert labels["io.crucible.phase"] == "visible-verifier"
    assert labels["io.crucible.candidate-patch-sha256"] == harness.patch_sha256
    assert "visible-verifier" in harness.setup_call["container_name"]

    runner_payload = harness.runner_call["payload"]
    metadata = runner_payload["parameters"]["swebench_metadata"]
    assert metadata == {
        "instance_id": harness.instance.instance_id,
        "repo": harness.instance.repo,
        "version": harness.instance.version,
        "base_commit": harness.instance.base_commit,
        "container_id": "verifier-container",
        "image_digest": _IMAGE_DIGEST,
        "candidate_patch_sha256": harness.patch_sha256,
    }
    rendered = json.dumps(runner_payload, sort_keys=True)
    assert str(harness.author_workspace) not in rendered
    for secret in (
        harness.instance.patch,
        harness.instance.test_patch,
        *harness.instance.fail_to_pass,
        *harness.instance.pass_to_pass,
    ):
        assert secret not in rendered
    assert not harness.workspace.exists()


@pytest.mark.parametrize(
    ("apply_check_rc", "apply_rc", "expected_check", "expected_apply"),
    [
        (1, 0, 1, None),
        (0, 2, 0, 2),
    ],
)
def test_apply_failures_skip_runner_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    apply_check_rc: int,
    apply_rc: int,
    expected_check: int,
    expected_apply: int | None,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.apply_check_rc = apply_check_rc
    harness.apply_rc = apply_rc

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.PATCH_APPLY_FAILED
    assert report.apply_check_rc == expected_check
    assert report.apply_rc == expected_apply
    assert report.runner_rc is None
    assert "runner" not in harness.events
    assert "teardown" in harness.events
    assert "container_absent" in harness.events
    assert not harness.workspace.exists()


def test_staged_test_path_is_infrastructure_failure_and_never_runs_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.staged_paths = b"src/widget.py\0tests/test_widget.py\0"

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.apply_check_rc == 0
    assert report.apply_rc == 0
    assert report.runner_rc is None
    assert "runner" not in harness.events
    assert "source-only changed-path policy" in report.raw_output_tail
    assert "teardown" in harness.events
    assert "container_absent" in harness.events
    assert report.cleanup_status == "complete"
    assert not harness.workspace.exists()


@pytest.mark.parametrize(
    (
        "runner_result",
        "expected_status",
        "expected_candidate_rc",
        "expected_mode",
    ),
    [
        (
            _runner_result(),
            verifier.PristineVerificationStatus.PASSED,
            0,
            "baseline_match",
        ),
        (
            _runner_result(
                candidate_rc=1,
                comparison_mode="preexisting_only",
                preexisting=2,
                candidate_failed=2,
            ),
            verifier.PristineVerificationStatus.PASSED,
            1,
            "preexisting_only",
        ),
        (
            _runner_result(
                runner_rc=1,
                candidate_rc=1,
                comparison_mode="regressed",
                new=1,
                candidate_failed=1,
            ),
            verifier.PristineVerificationStatus.TEST_FAILED,
            1,
            "regressed",
        ),
        (
            _runner_result(
                runner_rc=2,
                candidate_rc=2,
                comparison_mode="regressed",
                worsened=1,
                candidate_errored=1,
            ),
            verifier.PristineVerificationStatus.TEST_FAILED,
            2,
            "regressed",
        ),
        (
            _runner_result(
                baseline_unavailable=1,
                comparison_mode="unavailable",
            ),
            verifier.PristineVerificationStatus.INFRA_FAILED,
            0,
            "unavailable",
        ),
    ],
)
def test_runner_json_classifies_pass_test_and_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner_result: subprocess.CompletedProcess[bytes],
    expected_status: verifier.PristineVerificationStatus,
    expected_candidate_rc: int,
    expected_mode: str,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.runner_result = runner_result

    report = harness.verify()

    assert report.status is expected_status
    assert report.runner_rc == runner_result.returncode
    assert report.candidate_rc == expected_candidate_rc
    assert report.comparison_mode == expected_mode
    assert report.cleanup_status == "complete"


def test_malformed_runner_json_is_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.runner_result = subprocess.CompletedProcess(
        ["trusted-runner"], 0, stdout=b"{not valid json", stderr=b"protocol log"
    )

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.runner_rc == 0
    assert report.candidate_rc is None
    assert "malformed JSON" in report.reason
    assert "protocol log" in report.raw_output_tail
    assert "teardown" in harness.events
    assert not harness.workspace.exists()


def test_runner_timeout_is_infrastructure_failure_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.runner_error = subprocess.TimeoutExpired(
        cmd=["trusted-runner"], timeout=90, output=b"partial output"
    )

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.runner_rc is None
    assert "timed out" in report.reason
    assert "partial output" in report.raw_output_tail
    assert "teardown" in harness.events
    assert "container_absent" in harness.events
    assert not harness.workspace.exists()


def test_container_image_digest_mismatch_is_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.actual_image_digest = _OTHER_IMAGE_DIGEST

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.image_digest == _IMAGE_DIGEST
    assert report.apply_check_rc is None
    assert report.runner_rc is None
    assert "digest mismatch" in report.raw_output_tail
    assert "apply_check" not in harness.events
    assert "runner" not in harness.events
    assert "teardown" in harness.events
    assert not harness.workspace.exists()


def test_setup_failure_releases_workspace_and_quota(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.setup_error = RuntimeError("container setup failed")

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.cleanup_status == "complete"
    assert "teardown" not in harness.events
    assert "workspace_exit" in harness.events
    assert "quota_exit" in harness.events
    assert "quota_retire" in harness.events
    assert not harness.workspace.exists()


def test_leaked_container_overrides_a_passing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    harness.container_absent = False

    report = harness.verify()

    assert report.status is verifier.PristineVerificationStatus.INFRA_FAILED
    assert report.cleanup_status == "failed"
    assert "container leaked" in report.reason
    assert "container leaked" in report.raw_output_tail


def test_structured_payload_is_complete_round_trips_and_contains_no_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    report = harness.verify()
    payload = report.to_payload()

    assert set(payload) == {
        "schema",
        "status",
        "reason",
        "patch_sha256",
        "patch_bytes",
        "base_commit",
        "image_digest",
        "apply_check_rc",
        "apply_rc",
        "runner_rc",
        "candidate_rc",
        "comparison_mode",
        "preexisting_failures",
        "new_failures",
        "worsened_failures",
        "duration_s",
        "cleanup_status",
        "raw_output_tail",
    }
    assert payload["schema"] == "crucible.swebench.pristine-verification.v1"
    assert payload["status"] == "passed"
    assert payload["patch_sha256"] == harness.patch_sha256
    assert payload["patch_bytes"] == len(harness.patch)
    assert verifier.PristineVerificationReport.from_payload(payload) == report

    rendered = json.dumps(payload, sort_keys=True)
    assert str(harness.author_workspace) not in rendered
    assert str(harness.workspace) not in rendered
    for hidden_name in ("test_patch", "fail_to_pass", "pass_to_pass"):
        assert hidden_name not in rendered
    for secret in (
        harness.instance.patch,
        harness.instance.test_patch,
        *harness.instance.fail_to_pass,
        *harness.instance.pass_to_pass,
    ):
        assert secret not in rendered


def test_structured_payload_rejects_unknown_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _Harness(tmp_path, monkeypatch)
    payload = harness.verify().to_payload()
    payload["author_workspace"] = "/host/secret"

    with pytest.raises(ValueError, match="unknown=.*author_workspace"):
        verifier.PristineVerificationReport.from_payload(payload)
