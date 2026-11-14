# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source-only SWE-bench prediction capture regressions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from crucible.products.swebench.dataset import SwebenchInstance
from crucible.products.swebench.paths import is_standard_test_path
from crucible.workflows.swebench import (
    _capture_workspace_diff_report,
    _filter_corpus_patch_to_source,
)


def _git(repo: Path, *args: str, input_bytes: bytes | None = None):
    return subprocess.run(
        ["git", *args], cwd=repo, input=input_bytes, capture_output=True,
        check=True,
    )


def _instance() -> SwebenchInstance:
    return SwebenchInstance(
        instance_id="example__project-1",
        repo="example/project",
        base_commit="a" * 40,
        patch="",
        test_patch="",
        problem_statement="x",
        fail_to_pass=(),
        pass_to_pass=(),
    )


@pytest.mark.parametrize(
    "path",
    [
        "tests/test_widget.py",
        "src/test/java/org/example/WidgetTest.java",
        "pkg/test_widget.py",
        "pkg/widget_test.py",
        "pkg/widget_tests.py",
        "pkg/widget_test.go",
        "pkg/widget.spec.ts",
        "pkg/widget.test.js",
        "spec/widget_spec.rb",
        "auth_tests/helpers.py",
        "conftest.py",
    ],
)
def test_standard_test_path_recognizes_structural_conventions(path: str):
    assert is_standard_test_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "contest.py",
        "src/contest.py",
        "src/latest.py",
        "src/testing_utils.py",
        "src/testament.py",
        "src/Contest.java",
        "docs/testing.md",
    ],
)
def test_standard_test_path_avoids_test_substring_false_positives(path: str):
    assert not is_standard_test_path(path)


def test_capture_filters_all_test_change_shapes_and_patch_applies(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    (repo / "src" / "module.py").write_text("VALUE = 0\n")
    (repo / "src" / "contest.py").write_text("RESULT = 0\n")
    (repo / "tests" / "test_modified.py").write_text("assert False\n")
    (repo / "tests" / "test_deleted.py").write_text("assert False\n")
    (repo / "tests" / "test_renamed.py").write_text("assert False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    # Source edits and a false-positive filename survive.
    (repo / "src" / "module.py").write_text("VALUE = 1\n")
    (repo / "src" / "contest.py").write_text("RESULT = 1\n")
    (repo / "src" / "new_helper.py").write_text("HELPER = True\n")
    # Tracked modification, tracked deletion, tracked rename, and untracked
    # addition exercise every test-path change shape.
    (repo / "tests" / "test_modified.py").write_text("assert True\n")
    (repo / "tests" / "test_deleted.py").unlink()
    _git(
        repo, "mv", "tests/test_renamed.py",
        "tests/test_renamed_destination.py",
    )
    (repo / "tests" / "test_new.py").write_text("assert True\n")

    capture = _capture_workspace_diff_report(repo, _instance())
    patch = capture.source_patch.decode()

    assert capture.provenance == "workspace"
    assert capture.raw_diff_bytes > len(capture.source_patch) > 0
    assert capture.stripped_test_diff_bytes > 0
    assert set(capture.stripped_test_paths) == {
        "tests/test_deleted.py",
        "tests/test_modified.py",
        "tests/test_new.py",
        "tests/test_renamed.py",
        "tests/test_renamed_destination.py",
    }
    assert "src/module.py" in patch
    assert "src/contest.py" in patch
    assert "src/new_helper.py" in patch
    assert "tests/" not in patch

    # Filtering only affects the emitted patch: visible tests remain readable
    # and runnable in the authoring checkout.
    assert (repo / "tests" / "test_modified.py").exists()
    assert (repo / "tests" / "test_new.py").exists()
    assert (repo / "tests" / "test_renamed_destination.py").exists()

    clean = tmp_path / "clean"
    _git(tmp_path, "clone", "-q", str(repo), str(clean))
    check = subprocess.run(
        ["git", "apply", "--check", "-"], cwd=clean,
        input=capture.source_patch, capture_output=True, check=False,
    )
    assert check.returncode == 0, check.stderr.decode(errors="replace")


def test_capture_test_only_changes_has_empty_source_patch(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_only.py").write_text("assert False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tests" / "test_only.py").write_text("assert True\n")

    capture = _capture_workspace_diff_report(repo, _instance())

    assert capture.source_patch == b""
    assert capture.raw_diff_bytes > 0
    assert capture.stripped_test_paths == ("tests/test_only.py",)


@pytest.mark.parametrize(
    ("old_path", "new_path"),
    [
        ("src/module.py", "tests/test_module.py"),
        ("tests/test_module.py", "src/module.py"),
    ],
)
def test_cross_boundary_rename_is_filtered_as_one_semantic_entry(
    tmp_path: Path,
    old_path: str,
    new_path: str,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    old = repo / old_path
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / new_path).parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "mv", old_path, new_path)

    capture = _capture_workspace_diff_report(repo, _instance())

    assert capture.source_patch == b""
    assert capture.raw_diff_bytes > 0
    assert set(capture.stripped_test_paths) == {old_path, new_path}


@pytest.mark.parametrize(
    ("source_path", "destination_path"),
    [
        ("src/module.py", "tests/test_module.py"),
        ("tests/test_module.py", "src/module.py"),
    ],
)
def test_cross_boundary_copy_is_filtered_as_one_semantic_entry(
    tmp_path: Path,
    source_path: str,
    destination_path: str,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / source_path
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    destination = repo / destination_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())

    capture = _capture_workspace_diff_report(repo, _instance())

    assert capture.source_patch == b""
    assert capture.raw_diff_bytes > 0
    assert set(capture.stripped_test_paths) == {
        source_path,
        destination_path,
    }


def test_corpus_fallback_patch_is_reparsed_through_temporary_index(
    tmp_path: Path,
):
    dirty = tmp_path / "dirty"
    dirty.mkdir()
    _git(dirty, "init", "-q")
    _git(dirty, "config", "user.email", "test@example.com")
    _git(dirty, "config", "user.name", "Test")
    (dirty / "src").mkdir()
    (dirty / "tests").mkdir()
    (dirty / "src" / "module.py").write_text("VALUE = 0\n")
    (dirty / "tests" / "test_module.py").write_text("assert False\n")
    _git(dirty, "add", ".")
    _git(dirty, "commit", "-q", "-m", "base")
    (dirty / "src" / "module.py").write_text("VALUE = 1\n")
    (dirty / "tests" / "test_module.py").write_text("assert True\n")
    candidate = _git(dirty, "diff", "--binary", "HEAD").stdout

    clean = tmp_path / "clean"
    _git(tmp_path, "clone", "-q", str(dirty), str(clean))
    capture = _filter_corpus_patch_to_source(clean, _instance(), candidate)
    patch = capture.source_patch.decode()

    assert capture.provenance == "corpus_fallback"
    assert capture.raw_diff_bytes == len(candidate)
    assert capture.stripped_test_paths == ("tests/test_module.py",)
    assert capture.source_paths == ("src/module.py",)
    assert "src/module.py" in patch
    assert "tests/test_module.py" not in patch
    _git(clean, "apply", "--check", "-", input_bytes=capture.source_patch)


def test_test_only_outcome_has_specific_soft_failure(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_only.py").write_text("assert False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    (repo / "tests" / "test_only.py").write_text("assert True\n")

    class FakeImplementation:
        cost_usd = 0.0
        files_modified = ("tests/test_only.py",)
        classification = "success"
        diff_ref = None

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.classification == "soft_fail_test_only_changes"
    assert outcome.model_patch == ""
    assert outcome.files_modified == ()
    assert outcome.raw_workspace_diff_bytes > 0
    assert outcome.diff_provenance == "workspace"
    assert outcome.stripped_test_paths == ("tests/test_only.py",)
    assert outcome.candidate_capture is None
    assert outcome.candidate_eligibility.status == "ineligible"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.NO_NONEMPTY_CANDIDATE,
    )
    assert outcome.submission_decision.submitted is False


def test_filtered_workspace_salvages_prior_source_diff(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    source = repo / "src" / "module.py"
    test = repo / "tests" / "test_module.py"
    source.write_text("VALUE = 0\n")
    test.write_text("assert False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    source.write_text("VALUE = 1\n")
    stored_source_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 0\n")
    test.write_text("assert True\n")

    class FakeImplementation:
        slice_id = "source-fix"
        cost_usd = 0.0
        files_modified = ("src/module.py", "tests/test_module.py")
        classification = "success"
        diff_ref = "stored-source"

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "stored-source"
            return stored_source_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.classification == "success"
    assert "src/module.py" in outcome.model_patch
    assert "tests/test_module.py" not in outcome.model_patch
    assert outcome.files_modified == ("src/module.py",)
    assert outcome.stripped_test_paths == ("tests/test_module.py",)
    assert outcome.diff_provenance == "workspace+corpus_fallback"
    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.origin == "corpus_fallback"
    assert outcome.candidate_capture.capture_status == "source_filtered"
    assert outcome.candidate_capture.files_modified == ("src/module.py",)
    assert outcome.candidate_capture.diff_ref == (
        f"sha256:{outcome.candidate_capture.sha256}"
    )
    assert outcome.candidate_eligibility.status == "eligible"
    assert outcome.submission_decision.submitted is True
    assert outcome.submission_decision.override_used is False
    assert outcome.submission_decision.submitted_patch_ref == (
        outcome.candidate_capture.diff_ref
    )


def test_failed_corpus_fallback_is_retained_but_not_submitted_by_default(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    source.write_text("VALUE = 1\n")
    stored_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 0\n")

    class FakeImplementation:
        slice_id = "known-failing"
        cost_usd = 0.0
        files_modified = ("module.py",)
        classification = "dev_verify_failed"
        diff_ref = "stored-failed-patch"

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "stored-failed-patch"
            return stored_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)

    suppressed = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=1, workspace=repo,
    )
    assert suppressed.classification == "dev_verify_failed"
    assert suppressed.model_patch == ""
    assert suppressed.diff_bytes == 0
    assert suppressed.files_modified == ()
    assert suppressed.raw_workspace_diff_bytes == len(stored_patch)
    assert "corpus_fallback_suppressed_dev_verify_failed" in (
        suppressed.diff_provenance
    )
    assert suppressed.candidate_capture is not None
    assert suppressed.candidate_capture.diff_bytes == len(stored_patch)
    assert suppressed.candidate_capture.files_modified == ("module.py",)
    assert suppressed.candidate_capture.producer_slice_id == "known-failing"
    assert (
        suppressed.candidate_capture.producer_classification
        == "dev_verify_failed"
    )
    assert suppressed.candidate_eligibility.status == "ineligible"
    assert (
        workflow.CandidateEligibilityBlocker
        .CANDIDATE_CLASSIFICATION_NOT_SUCCESS
        in suppressed.candidate_eligibility.blockers
    )
    assert suppressed.submission_decision.policy == "verified_success_only"
    assert suppressed.submission_decision.submitted is False
    assert suppressed.submission_decision.submitted_patch_ref is None
    summary_metadata = workflow._prediction_metadata_summary(suppressed)
    assert summary_metadata["candidate_capture"]["diff_ref"] == (
        suppressed.candidate_capture.diff_ref
    )
    assert summary_metadata["candidate_eligibility"]["status"] == "ineligible"
    assert summary_metadata["submission_decision"]["submitted"] is False

    best_effort = workflow._spec_to_outcome(
        FakeCorpus(),
        _instance(),
        architect_rc=1,
        workspace=repo,
        submit_any_nonempty_best_effort=True,
    )
    assert best_effort.classification == "dev_verify_failed"
    assert "module.py" in best_effort.model_patch
    assert best_effort.diff_bytes == len(stored_patch)
    assert best_effort.files_modified == ("module.py",)
    assert best_effort.candidate_eligibility.status == "ineligible"
    assert best_effort.submission_decision.policy == (
        "best_effort_any_nonempty"
    )
    assert best_effort.submission_decision.submitted is True
    assert best_effort.submission_decision.override_requested is True
    assert best_effort.submission_decision.override_used is True
    assert best_effort.submission_decision.submitted_patch_ref == (
        best_effort.candidate_capture.diff_ref
    )


def test_live_candidate_binds_exact_success_not_later_failed_retry(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    source.write_text("VALUE = 1\n")
    successful_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 2\n")
    failed_retry_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    # The retry failed verification and its checkpoint restored the earlier
    # verified candidate, while retaining the failed record for accounting.
    source.write_text("VALUE = 1\n")

    class FakeImplementation:
        def __init__(
            self,
            slice_id: str,
            classification: str,
            diff_ref: str,
            cost_usd: float,
        ):
            self.slice_id = slice_id
            self.classification = classification
            self.diff_ref = diff_ref
            self.cost_usd = cost_usd
            self.files_modified = ("module.py",)

    records = {
        "verified": FakeImplementation(
            "verified-candidate", "success", "verified-diff", 0.25,
        ),
        "failed-retry": FakeImplementation(
            "failed-retry", "dev_verify_failed", "failed-diff", 0.75,
        ),
    }
    patches = {
        "verified-diff": successful_patch,
        "failed-diff": failed_retry_patch,
    }

    class FakeSpec:
        implementation_refs = ("verified", "failed-retry")
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            return records[ref]

        def get_artifact(self, ref):
            return patches[ref]

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=1, workspace=repo,
    )

    # Run failure and spend remain terminal facts, but neither is allowed to
    # relabel the different live candidate's producer.
    assert outcome.classification == "architect_rc_1"
    assert outcome.cost_usd == pytest.approx(1.0)
    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.source_patch == successful_patch
    assert outcome.candidate_capture.producer_slice_id == "verified-candidate"
    assert outcome.candidate_capture.producer_classification == "success"
    assert outcome.candidate_eligibility.status == "ineligible"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.ARCHITECT_FAILED,
        workflow.CandidateEligibilityBlocker
        .TERMINAL_CLASSIFICATION_NOT_SUCCESS,
    )
    assert outcome.submission_decision.submitted is False
    assert outcome.model_patch == ""
    assert "suppressed_architect_rc_1" in outcome.diff_provenance


def test_later_same_candidate_gate_failure_remains_authoritative(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    source.write_text("VALUE = 1\n")
    live_patch = _git(repo, "diff", "--binary", "HEAD").stdout

    class FakeImplementation:
        def __init__(
            self,
            slice_id: str,
            classification: str,
            cost_usd: float,
        ):
            self.slice_id = slice_id
            self.classification = classification
            self.diff_ref = f"{slice_id}-diff"
            self.cost_usd = cost_usd
            self.files_modified = ("module.py",)

    records = {
        "verified": FakeImplementation(
            "verified-candidate", "success", 0.25,
        ),
        "failed": FakeImplementation(
            "same-candidate-failure", "dev_verify_failed", 0.75,
        ),
    }

    class FakeSpec:
        implementation_refs = ("verified", "failed")
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            return records[ref]

        def get_artifact(self, ref):
            assert ref in {
                "verified-candidate-diff",
                "same-candidate-failure-diff",
            }
            return live_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=1, workspace=repo,
    )

    assert outcome.classification == "dev_verify_failed"
    assert outcome.cost_usd == pytest.approx(1.0)
    assert outcome.candidate_capture is not None
    assert (
        outcome.candidate_capture.producer_slice_id
        == "same-candidate-failure"
    )
    assert (
        outcome.candidate_capture.producer_classification
        == "dev_verify_failed"
    )
    assert outcome.candidate_eligibility.status == "ineligible"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.ARCHITECT_FAILED,
        workflow.CandidateEligibilityBlocker
        .TERMINAL_CLASSIFICATION_NOT_SUCCESS,
        workflow.CandidateEligibilityBlocker
        .CANDIDATE_CLASSIFICATION_NOT_SUCCESS,
    )
    assert outcome.submission_decision.submitted is False
    assert outcome.model_patch == ""


def test_later_different_success_does_not_override_exact_failed_producer(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    source.write_text("VALUE = 1\n")
    failed_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 2\n")
    different_success_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 1\n")

    class FakeImplementation:
        def __init__(
            self,
            slice_id: str,
            classification: str,
            diff_ref: str,
            cost_usd: float,
        ):
            self.slice_id = slice_id
            self.classification = classification
            self.diff_ref = diff_ref
            self.cost_usd = cost_usd
            self.files_modified = ("module.py",)

    records = {
        "failed": FakeImplementation(
            "exact-failed-candidate",
            "dev_verify_failed",
            "failed-diff",
            0.25,
        ),
        "different-success": FakeImplementation(
            "different-success",
            "success",
            "different-success-diff",
            0.75,
        ),
    }
    patches = {
        "failed-diff": failed_patch,
        "different-success-diff": different_success_patch,
    }

    class FakeSpec:
        implementation_refs = ("failed", "different-success")
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            return records[ref]

        def get_artifact(self, ref):
            return patches[ref]

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=1, workspace=repo,
    )

    assert outcome.classification == "dev_verify_failed"
    assert outcome.cost_usd == pytest.approx(1.0)
    assert outcome.candidate_capture is not None
    assert (
        outcome.candidate_capture.producer_slice_id
        == "exact-failed-candidate"
    )
    assert (
        outcome.candidate_capture.producer_classification
        == "dev_verify_failed"
    )
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.ARCHITECT_FAILED,
        workflow.CandidateEligibilityBlocker
        .TERMINAL_CLASSIFICATION_NOT_SUCCESS,
        workflow.CandidateEligibilityBlocker
        .CANDIDATE_CLASSIFICATION_NOT_SUCCESS,
    )
    assert outcome.submission_decision.submitted is False


def test_unmatched_live_candidate_is_unknown_and_not_submitted(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    source.write_text("VALUE = 1\n")
    stored_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    source.write_text("VALUE = 2\n")

    class FakeImplementation:
        slice_id = "different-success"
        classification = "success"
        diff_ref = "different-diff"
        cost_usd = 0.5
        files_modified = ("module.py",)

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "different-diff"
            return stored_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.classification == "success"
    assert outcome.cost_usd == pytest.approx(0.5)
    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.producer_slice_id is None
    assert outcome.candidate_capture.producer_classification is None
    assert outcome.candidate_eligibility.status == "unknown"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.CANDIDATE_PROVENANCE_UNBOUND,
    )
    assert outcome.submission_decision.submitted is False
    assert outcome.model_patch == ""
    assert "suppressed_candidate_provenance_unbound" in (
        outcome.diff_provenance
    )


def test_live_candidate_matches_source_normalized_stored_diff(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "tests").mkdir()
    source = repo / "module.py"
    test = repo / "tests" / "test_module.py"
    source.write_text("VALUE = 0\n")
    test.write_text("assert False\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")

    source.write_text("VALUE = 1\n")
    test.write_text("assert True\n")
    stored_patch = _git(repo, "diff", "--binary", "HEAD").stdout
    # The live source candidate is identical after the standard test-path
    # filter, despite the stored producer containing additional test bytes.
    test.write_text("assert False\n")

    class FakeImplementation:
        slice_id = "source-producer"
        classification = "success"
        diff_ref = "source-and-test-diff"
        cost_usd = 0.25
        files_modified = ("module.py", "tests/test_module.py")

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "source-and-test-diff"
            return stored_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.producer_slice_id == "source-producer"
    assert outcome.candidate_capture.producer_classification == "success"
    assert outcome.candidate_eligibility.status == "eligible"
    assert outcome.submission_decision.submitted is True
    assert "module.py" in outcome.model_patch
    assert "tests/test_module.py" not in outcome.model_patch


def test_later_verified_repair_supersedes_historical_gate_failure(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    source.write_text("VALUE = 1\n")
    repaired_patch = _git(repo, "diff", "--binary", "HEAD").stdout

    class FakeImplementation:
        def __init__(self, classification: str, diff_ref: str):
            self.slice_id = diff_ref
            self.cost_usd = 0.0
            self.files_modified = ("module.py",)
            self.classification = classification
            self.diff_ref = diff_ref

    records = {
        "failed": FakeImplementation("dev_verify_failed", "failed-diff"),
        "repaired": FakeImplementation("success", "repaired-diff"),
    }

    class FakeSpec:
        implementation_refs = ("failed", "repaired")
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            return records[ref]

        def get_artifact(self, ref):
            assert ref in {"failed-diff", "repaired-diff"}
            return repaired_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.classification == "success"
    assert outcome.diff_bytes == len(repaired_patch)
    assert "module.py" in outcome.model_patch
    assert outcome.candidate_eligibility.status == "eligible"
    assert outcome.submission_decision.submitted is True


def test_downstream_failure_suppresses_earlier_successful_live_diff(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    source.write_text("VALUE = 1\n")
    live_patch = _git(repo, "diff", "--binary", "HEAD").stdout

    class FakeImplementation:
        slice_id = "initial-success"
        cost_usd = 0.0
        files_modified = ("module.py",)
        classification = "success"
        diff_ref = "successful-diff"

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "successful-diff"
            return live_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=1, workspace=repo,
    )

    assert outcome.classification == "architect_rc_1"
    assert outcome.model_patch == ""
    assert outcome.diff_bytes == 0
    assert outcome.files_modified == ()
    assert "suppressed_architect_rc_1" in outcome.diff_provenance
    assert outcome.candidate_capture is not None
    assert outcome.candidate_capture.origin == "workspace"
    assert outcome.candidate_eligibility.status == "ineligible"
    assert (
        workflow.CandidateEligibilityBlocker.ARCHITECT_FAILED
        in outcome.candidate_eligibility.blockers
    )
    assert outcome.submission_decision.submitted is False


def test_incomplete_terminal_record_set_suppresses_live_diff(
    tmp_path: Path,
    monkeypatch,
):
    from crucible.workflows import swebench as workflow

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    source = repo / "module.py"
    source.write_text("VALUE = 0\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    source.write_text("VALUE = 1\n")
    live_patch = _git(repo, "diff", "--binary", "HEAD").stdout

    class FakeImplementation:
        slice_id = "known-success"
        cost_usd = 0.0
        files_modified = ("module.py",)
        classification = "success"
        diff_ref = "successful-diff"

    class FakeSpec:
        implementation_refs = ("impl", "missing-terminal-record")
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            if ref == "missing-terminal-record":
                raise FileNotFoundError(ref)
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "successful-diff"
            return live_patch

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0, workspace=repo,
    )

    assert outcome.classification == "success"
    assert outcome.cost_incomplete is True
    assert outcome.model_patch == ""
    assert outcome.diff_bytes == 0
    assert "suppressed_incomplete_record_rollup" in outcome.diff_provenance
    assert outcome.candidate_capture is not None
    assert outcome.candidate_eligibility.status == "unknown"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.RECORD_ROLLUP_INCOMPLETE,
    )
    assert outcome.submission_decision.submitted is False

    best_effort = workflow._spec_to_outcome(
        FakeCorpus(),
        _instance(),
        architect_rc=0,
        workspace=repo,
        submit_any_nonempty_best_effort=True,
    )
    assert best_effort.model_patch
    assert best_effort.candidate_eligibility.status == "unknown"
    assert best_effort.submission_decision.submitted is True
    assert best_effort.submission_decision.override_used is True



@pytest.mark.parametrize(
    "stored_patch",
    [
        pytest.param(b"stored bytes without a Git patch", id="opaque"),
        pytest.param(
            b"diff --git a/module.py b/module.py\n"
            b"--- a/module.py\n+++ b/module.py\n"
            b"@@ -1 +1 @@\n-old\n+new\n",
            id="git-diff",
        ),
    ],
)
def test_unbound_stored_candidate_cannot_be_submitted_with_override(
    tmp_path: Path,
    monkeypatch,
    stored_patch: bytes,
):
    from crucible.workflows import swebench as workflow

    class FakeImplementation:
        cost_usd = 0.25
        files_modified = ("module.py",)
        classification = "success"
        diff_ref = "stored-diff"
        slice_id = "s0"

    class FakeSpec:
        implementation_refs = ("impl",)
        verdict_refs = ()

    class FakeCorpus:
        def latest_spec(self):
            return FakeSpec()

        def get_record(self, ref, *, as_type):
            assert ref == "impl"
            return FakeImplementation()

        def get_artifact(self, ref):
            assert ref == "stored-diff"
            return stored_patch

        def put_artifact(self, *args, **kwargs):
            pytest.fail("unbound candidate was published")

    monkeypatch.setattr(workflow, "Implementation", FakeImplementation)
    outcome = workflow._spec_to_outcome(
        FakeCorpus(), _instance(), architect_rc=0,
        workspace=tmp_path / "missing-workspace",
        submit_any_nonempty_best_effort=True,
    )

    assert outcome.model_patch == ""
    assert outcome.diff_bytes == 0
    assert outcome.files_modified == ()
    assert outcome.cost_usd == pytest.approx(0.25)
    assert outcome.raw_workspace_diff_bytes == len(stored_patch)
    assert "corpus_fallback_workspace_unavailable" in outcome.diff_provenance
    assert outcome.candidate_capture is None
    assert outcome.candidate_eligibility.status == "unknown"
    assert outcome.candidate_eligibility.blockers == (
        workflow.CandidateEligibilityBlocker.CAPTURE_FAILED,
    )
    assert outcome.submission_decision.override_requested is True
    assert outcome.submission_decision.override_used is False
    assert outcome.submission_decision.submitted is False
    assert outcome.submission_decision.submitted_patch_ref is None
