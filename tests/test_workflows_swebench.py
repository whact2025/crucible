# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for the SWE-bench Verified adapter (workflow + dataset + predictions).

Covers deterministic dataset and prediction plumbing plus the live cascade
with hermetic authoring environments and independently leased scripted agents.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ._review_v2_adapter import adapt_scripted_review_response

from crucible.products.swebench.dataset import (
    SwebenchInstance,
    SwebenchInstanceError,
    filter_instances,
    load_dataset,
)
from crucible.products.swebench.predictions import (
    Prediction,
    append_prediction_jsonl,
    to_record,
    write_predictions_json,
    write_predictions_jsonl,
)
from crucible.workflows import _outcome


def _prepare_swebench_parent(path: Path) -> Path:
    path.mkdir(parents=True)
    _outcome.begin_run(path, workflow="swebench")
    return path


@pytest.fixture(autouse=True)
def _authoring_sandbox_stub(monkeypatch):
    """Live workflow tests get a container identity without Docker."""
    from crucible.workflows import swebench as workflow

    @contextlib.contextmanager
    def lifecycle(*args, **kwargs):
        yield {
            "container_id": "test-authoring-container",
            "image_digest": "test/image@sha256:" + "a" * 64,
        }

    monkeypatch.setattr(workflow, "_docker_available", lambda: True)
    monkeypatch.setattr(workflow, "_check_disk_headroom", lambda: None)
    monkeypatch.setattr(workflow, "_reconcile_orphans", lambda scope: ())
    monkeypatch.setattr(workflow, "_instance_env_lifecycle", lifecycle)
    # Unit-level workflow tests use synthetic repositories and no Docker. The
    # pristine verifier has its own hermetic suite; dedicated integration tests
    # below override this identity stub to assert worker ordering and policy.
    def passed_pristine_verifier(instance, outcome, **kwargs):
        if not outcome.model_patch:
            return outcome
        candidate, eligibility, submission = (
            workflow._effective_prediction_metadata(outcome)
        )
        patch = outcome.model_patch.encode("utf-8")
        report = workflow.PristineVerificationReport(
            schema="crucible.swebench.pristine-verification.v1",
            status=workflow.PristineVerificationStatus.PASSED,
            reason="unit-test pristine verifier pass",
            patch_sha256=hashlib.sha256(patch).hexdigest(),
            patch_bytes=len(patch),
            base_commit=instance.base_commit,
            image_digest=kwargs["image_digest"],
            apply_check_rc=0,
            apply_rc=0,
            runner_rc=0,
            candidate_rc=0,
            comparison_mode="baseline_match",
            preexisting_failures=0,
            new_failures=0,
            worsened_failures=0,
            duration_s=0.1,
            cleanup_status="complete",
            raw_output_tail="",
        )
        return workflow.dataclasses.replace(
            outcome,
            candidate_capture=candidate,
            candidate_eligibility=eligibility,
            submission_decision=submission,
            pristine_verification=report,
        )

    monkeypatch.setattr(
        workflow, "_verify_submitted_candidate", passed_pristine_verifier,
    )


# ----------------------------------------------------------------------
# Shared multi-role scripted backend
# ----------------------------------------------------------------------
#
# The per-instance worker runs the full architect cascade — plan → critic →
# implement. Tests that drive the workflow end-to-end must
# script responses for each role. This helper identifies the call type
# from invocation.system and returns a canned response per role, so
# tests can monkeypatch one resolver and let it serve all three.
#
# The default runs three independent planners, their adversarial reviews, one
# comparative selection, and implementation. Container-backed smoke tests cover
# the breach/review-on path.

_DEFAULT_DAG_JSON = json.dumps({
    "tiers": [{
        "tier_id": "t0",
        "depends_on": [],
        "slices": [{
            "id": "s0",
            "directive": {
                "slice_id": "s0",
                "intent": "edit seed.py to satisfy the issue",
                "files_in_scope": ["seed.py"],
                "creates_paths": [],
                "verify_mode": "skip",
                "blast_radius": {"max_files": 1, "languages": ["python"]},
            },
        }],
    }],
})

def _passing_review(system_prompt: str) -> str:
    """Return a pass response covering the sources declared by the rubric.

    The production rubric contract renders its exact required-source set into
    the system prompt.  Keeping the test double responsive to that contract
    avoids silently teaching fixtures the old artifact-only protocol.
    """

    marker = "Required source coverage includes: "
    rendered = system_prompt.split(marker, 1)[1].split(".", 1)[0]
    sources = [item.strip() for item in rendered.split(",") if item.strip()]
    return json.dumps({
        "schema_version": "assay.review.v2",
        "decision": "pass",
        "reasoning": "complete and valid",
        "coverage": {
            "complete": True,
            "sources_reviewed": sources,
            "omissions": [],
        },
        "findings": [],
    })


class _MultiRoleScriptedBackend:
    """Drop-in stub for the SWE-bench architect cascade.

    Dispatches on ``invocation.system`` to return planner JSON, critic
    pass-verdict JSON, or "DONE" for the implementer (with optional
    side-effect callback to mutate the workspace before assay.implement
    captures the diff).

    Public counters: ``counters["planner"|"critic"|"implementer"|"other"]``
    track per-role calls so tests can assert on the expected sequence.
    """

    name = "stub"
    model = "stub-model"

    def __init__(
        self,
        *,
        on_implement=None,
        dag_json=None,
        cost_per_call=0.0,
    ):
        self.calls: list = []
        self.counters: dict[str, int] = {
            "planner": 0, "critic": 0, "judge": 0,
            "implementer": 0, "selector": 0, "other": 0,
        }
        self._on_implement = on_implement
        self._dag_json = dag_json or _DEFAULT_DAG_JSON
        self._cost_per_call = cost_per_call

    def attach_toolset(self, toolset, *, context_budget=None):
        pass

    def close(self):
        pass

    def _planner_response(self, invocation) -> str:
        """Bind canned plan slices to the invocation's actual root intent."""

        payload = json.loads(self._dag_json)
        intent = None
        for line in invocation.user.splitlines():
            try:
                catalog = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(catalog, dict) or "intent" not in catalog:
                continue
            candidate = catalog["intent"]
            if not isinstance(candidate, str) or not candidate.strip():
                raise AssertionError(
                    "planner prompt has an invalid authoritative root intent"
                )
            intent = candidate
            break
        if intent is None:
            raise AssertionError(
                "planner prompt is missing its authoritative root intent"
            )
        for tier in payload["tiers"]:
            for slice_ in tier["slices"]:
                slice_["directive"].setdefault(
                    "requirement_refs", [intent],
                )
        return json.dumps(payload)

    def invoke(self, invocation):
        from assay import (
            BackendExecutionEvent,
            BackendExecutionTrace,
            ExecutionCompleteness,
            InvocationResult,
            Message,
            Usage,
            UsagePricingEvidence,
            UsagePricingSegment,
        )
        self.calls.append(invocation)
        system = (invocation.system or "").lower()
        # Check the most-distinctive phrase first. The critic's
        # system prompt mentions "planner" multiple times (it's
        # judging the planner's output) so a naive "planner in
        # system" check matches both — order matters.
        if "compare complete competing plans" in system:
            from .test_plan_selection import _Backend

            self.counters["selector"] += 1
            final = _Backend().invoke(invocation).final_message
        elif "you are the judge" in system or "judge for plan" in system:
            self.counters["judge"] += 1
            final = _passing_review(invocation.system or "")
        elif "you are the critic" in system or "critic for plan" in system:
            self.counters["critic"] += 1
            final = _passing_review(invocation.system or "")
        elif "you are a planner" in system or "planner for the" in system:
            self.counters["planner"] += 1
            final = self._planner_response(invocation)
        elif (
            "implementation agent" in system
            or "software engineer fixing a github issue" in system
        ):
            # Match BOTH assay's default _IMPLEMENTER_SYSTEM_PROMPT
            # ("implementation agent ...") AND the swebench-threaded
            # SWEBENCH_SYSTEM_PROMPT ("software engineer fixing a
            # GitHub issue ..."). The swebench worker now opts into the
            # latter via impl_system_prompt; without this branch the
            # implementer call would fall through to the no-op `else`
            # and never fire the workspace-mutation callback.
            self.counters["implementer"] += 1
            if self._on_implement is not None:
                self._on_implement()
            final = "DONE"
        else:
            self.counters["other"] += 1
            final = "DONE"
        final = adapt_scripted_review_response(final, invocation)
        return InvocationResult(
            final_message=final,
            transcript=(
                Message(role="user", content=invocation.user),
                Message(role="assistant", content=final),
            ),
            usage=Usage(
                input_tokens=10, output_tokens=10,
                cache_creation_tokens=0, cache_read_tokens=0,
                cost_usd=self._cost_per_call,
            ),
            pricing_evidence=UsagePricingEvidence(
                segments=(UsagePricingSegment(
                    input_tokens=10, output_tokens=10,
                    cache_creation_tokens=0, cache_read_tokens=0,
                    cost_usd=self._cost_per_call,
                ),),
                complete=True,
            ),
            stop_reason="end_turn",
            completeness=ExecutionCompleteness(
                token_usage_complete=True,
                cost_complete=True,
                terminal_complete=True,
                trace_complete=True,
            ),
            execution_trace=BackendExecutionTrace(
                backend=self.name,
                requested_model=self.model,
                events=(
                    BackendExecutionEvent(
                        sequence=0,
                        kind="terminal",
                        name="end_turn",
                    ),
                ),
            ),
        )


def _patch_backend_resolver(monkeypatch, backend):
    """Lease distinct scripted agents through the cascade's backend pool."""
    import copy

    # Planner, critic, and judge must be independent backend objects. A
    # shallow copy preserves the test's shared counters/callbacks while
    # giving each routed role a distinct identity.
    def _resolve(*args, **kwargs):
        return copy.copy(backend)

    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role", _resolve,
    )


def _patch_smoke_check(monkeypatch):
    """Keep workflow plumbing tests independent of local lint tools."""
    from assay.types.smoke import SmokeResult

    monkeypatch.setattr(
        "crucible.workflows.tier.smoke_check",
        lambda directive, *, cwd, timeout=120, **_kwargs: SmokeResult(
            passed=True,
            language="python",
            log_excerpt="",
            duration_ms=0,
        ),
    )


# ----------------------------------------------------------------------
# Dataset loader
# ----------------------------------------------------------------------


def test_parse_args_accepts_test_only_resume_target(tmp_path: Path):
    from crucible.workflows.swebench import _parse_args

    parsed = _parse_args([
        "--spec", str(tmp_path / "spec"),
        "--dataset", str(tmp_path / "dataset.json"),
        "--rerun-status", "soft_fail_test_only_changes",
    ])

    assert parsed.rerun_status == ["soft_fail_test_only_changes"]
    assert parsed.planner_count == 3
    diagnostic = _parse_args([
        "--spec", str(tmp_path / "spec"),
        "--dataset", str(tmp_path / "dataset.json"),
        "--planner-count", "1",
    ])
    assert diagnostic.planner_count == 1


def _minimal_record(instance_id: str = "django__django-15347", **overrides) -> dict:
    """Build a record with all REQUIRED fields filled, overridable."""
    base = {
        "instance_id": instance_id,
        "repo": "django/django",
        "base_commit": "0123456789abcdef0123456789abcdef01234567",
        "patch": "diff --git a/foo.py b/foo.py\n",
        "test_patch": "diff --git a/test_foo.py b/test_foo.py\n",
        "problem_statement": "Fix the bug",
        "FAIL_TO_PASS": ["tests.test_foo::test_bug"],
        "PASS_TO_PASS": [],
    }
    base.update(overrides)
    return base


def test_checkpoint_fingerprint_binds_all_backend_runtime_recipes(monkeypatch):
    from crucible.workflows import swebench as wf

    crucible_root = Path(wf.__file__).resolve().parents[1]
    observed: set[str] = set()
    revisions: dict[str, str] = {}

    def _digest(path):
        candidate = Path(path).resolve()
        relative = (
            candidate.relative_to(crucible_root).as_posix()
            if candidate.is_relative_to(crucible_root)
            else candidate.as_posix()
        )
        observed.add(relative)
        return {
            "name": candidate.name,
            "sha256": revisions.get(relative, "0" * 64),
            "size": 1,
        }

    monkeypatch.setattr(wf, "file_digest", _digest)
    monkeypatch.setattr(
        wf,
        "file_set_fingerprint",
        lambda *_args, **_kwargs: "sha256:rubrics",
    )
    monkeypatch.setattr(
        wf,
        "image_verification_policy_identity",
        lambda: {"schema": "test-image-policy.v1"},
    )
    inst = SwebenchInstance(
        instance_id="org__repo-1",
        repo="org/repo",
        base_commit="0" * 40,
        patch="",
        test_patch="",
        problem_statement="fix it",
        fail_to_pass=(),
        pass_to_pass=(),
    )

    def _fingerprint(planner_count: int = 3) -> str:
        return wf._swe_checkpoint_fingerprint(
            inst,
            routing_source_sha256=None,
            max_turns=10,
            child_budget_usd=10.0,
            planner_count=planner_count,
            include_breach=True,
            include_review=True,
            enable_build_verify=True,
            evaluate_inline=True,
            eval_timeout=60,
            routed_backend="codex",
            routed_model="gpt-6-astra",
            image_digest="test/image@sha256:" + "a" * 64,
        )

    baseline = _fingerprint()
    assert _fingerprint(1) != baseline
    expected_backend_modules = {
        f"backends/{name}"
        for name in (
            "__init__.py",
            "_classify.py",
            "_claude_session_transport.py",
            "_physical_telemetry.py",
            "_sdk_debug.py",
            "capabilities.py",
            "claude.py",
            "codex.py",
            "codex_official.py",
            "codex_official_worker.py",
            "config.py",
            "deepseek.py",
            "evidence.py",
            "features.py",
            "kimi.py",
            "limits.py",
            "openai_compat.py",
            "pricing.py",
            "roles.py",
            "tool_authority.py",
            "trusted_codex.py",
        )
    }
    expected_worker_recipes = {
        f"backends/{project}/{relative}"
        for project, package in (
            ("codex_worker", "crucible_codex_worker"),
            ("deepseek_worker", "crucible_deepseek_worker"),
            ("kimi_worker", "crucible_kimi_worker"),
        )
        for relative in (
            ".python-version",
            "pyproject.toml",
            "uv.lock",
            f"src/{package}/__init__.py",
            f"src/{package}/__main__.py",
        )
    }
    expected_shared_runtime = {
        "common/authority_io.py",
        "common/process_settlement.py",
        "tools/__init__.py",
        "tools/_dispatch_budget.py",
        "tools/_isolated_mutation_host.py",
        "tools/_mcp_bridge.py",
        "tools/_mcp_http_server.py",
        "workflows/_backends.py",
        "workflows/_backend_recipe.py",
        "workflows/_benchmark_planning.py",
        "workflows/_plan_competition.py",
        "workflows/_plan_selection.py",
        "workflows/_plan_source_snapshot.py",
        "workflows/architect.py",
        "workflows/plan.py",
        "workflows/_process_supervisor_contract.py",
        "workflows/_process_supervisor.py",
        ".python-version",
        "pyproject.toml",
        "uv.lock",
    }
    assert (
        expected_backend_modules | expected_worker_recipes | expected_shared_runtime
    ) <= observed

    for relative in (
        "common/authority_io.py",
        "common/process_settlement.py",
        "backends/claude.py",
        "backends/codex.py",
        "backends/codex_official.py",
        "backends/deepseek.py",
        "backends/kimi.py",
        "backends/openai_compat.py",
        "backends/features.py",
        "backends/codex_worker/uv.lock",
        "backends/deepseek_worker/src/crucible_deepseek_worker/__main__.py",
        "backends/kimi_worker/pyproject.toml",
        "tools/__init__.py",
        "tools/_isolated_mutation_host.py",
        "workflows/_backend_recipe.py",
        "workflows/_benchmark_planning.py",
        "workflows/_plan_competition.py",
        "workflows/_plan_selection.py",
        "workflows/_plan_source_snapshot.py",
        "workflows/architect.py",
        "workflows/plan.py",
        "workflows/_process_supervisor_contract.py",
        "workflows/_process_supervisor.py",
        ".python-version",
        "uv.lock",
    ):
        revisions[relative] = "1" * 64
        assert _fingerprint() != baseline, relative
        revisions.clear()


def test_load_dataset_json_array(tmp_path: Path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([_minimal_record()]))
    instances = load_dataset(path)
    assert len(instances) == 1
    assert instances[0].instance_id == "django__django-15347"
    assert instances[0].repo == "django/django"
    assert instances[0].repo_org == "django"
    assert instances[0].repo_name == "django"
    assert instances[0].fail_to_pass == ("tests.test_foo::test_bug",)


def test_load_dataset_jsonl(tmp_path: Path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        json.dumps(_minimal_record("a__b-1")) + "\n"
        + json.dumps(_minimal_record("c__d-2")) + "\n"
    )
    instances = load_dataset(path)
    assert [i.instance_id for i in instances] == ["a__b-1", "c__d-2"]


def test_load_dataset_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.json")


def test_load_dataset_empty_file_returns_empty_tuple(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text("")
    assert load_dataset(path) == ()


def test_load_dataset_rejects_missing_required_field(tmp_path: Path):
    rec = _minimal_record()
    del rec["base_commit"]
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([rec]))
    with pytest.raises(SwebenchInstanceError) as exc:
        load_dataset(path)
    assert "base_commit" in str(exc.value)


def test_load_dataset_rejects_empty_instance_id(tmp_path: Path):
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([_minimal_record(instance_id="")]))
    with pytest.raises(SwebenchInstanceError):
        load_dataset(path)


def test_load_dataset_rejects_repo_without_slash(tmp_path: Path):
    """A record where ``repo`` doesn't match ``<org>/<name>`` shape
    fails at load time with a clear message — downstream code splits
    on ``/`` to build GitHub URLs + cache paths, so silent acceptance
    would degrade into a confusing error far from the source."""
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([_minimal_record(repo="just-a-name")]))
    with pytest.raises(SwebenchInstanceError) as exc:
        load_dataset(path)
    assert "repo" in str(exc.value)
    assert "<org>/<name>" in str(exc.value)


def test_load_dataset_handles_stringified_lists(tmp_path: Path):
    """SWE-bench parquet exports sometimes JSON-encode FAIL_TO_PASS /
    PASS_TO_PASS as a string. Coerce gracefully."""
    rec = _minimal_record()
    rec["FAIL_TO_PASS"] = '["test_a", "test_b"]'
    path = tmp_path / "ds.json"
    path.write_text(json.dumps([rec]))
    instances = load_dataset(path)
    assert instances[0].fail_to_pass == ("test_a", "test_b")


def test_load_dataset_rejects_non_list_object_at_top(tmp_path: Path):
    """A JSON file starting with { (the canonical Parquet-row-dict shape
    rather than a list-of-records) must surface a clear error."""
    path = tmp_path / "ds.json"
    path.write_text(json.dumps({"instance_id": "x"}))
    with pytest.raises(SwebenchInstanceError) as exc:
        load_dataset(path)
    assert "must be a list" in str(exc.value)


def test_load_dataset_skips_blank_jsonl_lines(tmp_path: Path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        json.dumps(_minimal_record("a-1")) + "\n"
        "\n"
        + json.dumps(_minimal_record("a-2")) + "\n"
    )
    instances = load_dataset(path)
    assert len(instances) == 2


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------


def _three_instances() -> tuple[SwebenchInstance, SwebenchInstance, SwebenchInstance]:
    return (
        SwebenchInstance(
            instance_id="django__django-1", repo="django/django",
            base_commit="a" * 40, patch="", test_patch="",
            problem_statement="", fail_to_pass=(), pass_to_pass=(),
        ),
        SwebenchInstance(
            instance_id="sympy__sympy-1", repo="sympy/sympy",
            base_commit="b" * 40, patch="", test_patch="",
            problem_statement="", fail_to_pass=(), pass_to_pass=(),
        ),
        SwebenchInstance(
            instance_id="django__django-2", repo="django/django",
            base_commit="c" * 40, patch="", test_patch="",
            problem_statement="", fail_to_pass=(), pass_to_pass=(),
        ),
    )


def test_filter_no_args_returns_all():
    inst = _three_instances()
    assert filter_instances(inst) == inst


def test_filter_by_instance_id():
    inst = _three_instances()
    out = filter_instances(inst, instance_ids=["django__django-2"])
    assert [i.instance_id for i in out] == ["django__django-2"]


def test_filter_by_repo():
    inst = _three_instances()
    out = filter_instances(inst, repos=["django/django"])
    assert {i.instance_id for i in out} == {
        "django__django-1", "django__django-2",
    }


def test_filter_limit_caps_after_other_filters():
    inst = _three_instances()
    out = filter_instances(inst, repos=["django/django"], limit=1)
    assert len(out) == 1
    # Limit applies after repo filter — only django instances.
    assert out[0].repo == "django/django"


def test_filter_empty_whitelist_is_no_filter():
    """Empty list (not None) means "no filter applied" — matches the
    CLI convention where ``--instance-id`` is repeated zero times."""
    inst = _three_instances()
    out = filter_instances(inst, instance_ids=[], repos=[])
    assert out == inst


# ----------------------------------------------------------------------
# Predictions emitter
# ----------------------------------------------------------------------


def test_to_record_shape():
    p = Prediction(
        instance_id="django__django-1",
        model_patch="diff --git a/foo b/foo\n",
        model_name_or_path="aws/anthropic/bedrock-claude-opus-4-8",
    )
    assert to_record(p) == {
        "instance_id": "django__django-1",
        "model_patch": "diff --git a/foo b/foo\n",
        "model_name_or_path": "aws/anthropic/bedrock-claude-opus-4-8",
    }


def test_write_predictions_jsonl_round_trips(tmp_path: Path):
    preds = [
        Prediction("a-1", "patch-a", "aws/anthropic/bedrock-claude-opus-4-8"),
        Prediction("b-2", "", "aws/anthropic/bedrock-claude-opus-4-8"),
    ]
    out = tmp_path / "predictions.jsonl"
    count = write_predictions_jsonl(preds, out)
    assert count == 2

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["instance_id"] == "a-1"
    assert parsed[1]["model_patch"] == ""


def test_write_predictions_jsonl_creates_parent_dir(tmp_path: Path):
    out = tmp_path / "nested" / "dir" / "predictions.jsonl"
    write_predictions_jsonl([Prediction("a", "p", "m")], out)
    assert out.exists()


def test_write_predictions_jsonl_preserves_checkpoint_on_failed_consolidation(
    tmp_path: Path,
):
    """A failed final render must not truncate incremental checkpoints."""
    out = tmp_path / "predictions.jsonl"
    checkpoint = json.dumps(to_record(Prediction("done", "patch", "m"))) + "\n"
    out.write_text(checkpoint, encoding="utf-8")

    def broken_predictions():
        yield Prediction("next", "partial", "m")
        raise RuntimeError("render interrupted")

    with pytest.raises(RuntimeError, match="render interrupted"):
        write_predictions_jsonl(broken_predictions(), out)

    assert out.read_text(encoding="utf-8") == checkpoint
    assert not list(tmp_path.glob(".tmp_predictions.jsonl_*"))


def test_concurrent_partial_prediction_appends_never_interleave(
    tmp_path: Path, monkeypatch,
):
    """One logical record stays atomic even when os.write is partial."""

    from concurrent.futures import ThreadPoolExecutor
    import os

    out = tmp_path / "predictions.jsonl"
    original_write = os.write

    def partial_write(fd, data):
        return original_write(fd, data[:127])

    monkeypatch.setattr(os, "write", partial_write)
    predictions = [
        Prediction(f"instance-{index}", str(index) * 20_000, "model")
        for index in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tuple(pool.map(lambda item: append_prediction_jsonl(item, out), predictions))

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert {record["instance_id"] for record in records} == {
        item.instance_id for item in predictions
    }
    assert len(records) == len(predictions)


def test_write_predictions_json_array_form(tmp_path: Path):
    preds = [Prediction("a-1", "patch-a", "m"), Prediction("b-2", "", "m")]
    out = tmp_path / "predictions.json"
    write_predictions_json(preds, out)
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)
    assert [p["instance_id"] for p in parsed] == ["a-1", "b-2"]


# ----------------------------------------------------------------------
# Workflow dispatch
# ----------------------------------------------------------------------


def _setup_corpus(tmp_path: Path):
    from assay import Spec, ProjectContext
    from assay.corpus.store import Corpus

    spec_dir = tmp_path / "corpus" / "swebench-test"
    spec_dir.mkdir(parents=True)
    corpus = Corpus(spec_dir)
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="swebench", repo_url="x",
            base_commit="a" * 40,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="", constraints=(),
        done_criteria=(), plan_policy=None,
    )
    corpus.put_spec(spec)
    return corpus


def _seed_terminal_prediction(
    root_corpus,
    inst: SwebenchInstance,
    *,
    classification: str,
    patch: str = "",
    model: str = "stub-prior",
):
    """Persist the production per-instance terminal resume marker."""
    from assay.corpus.store import Corpus
    from crucible.backends.config import (
        load_routing,
        routing_fingerprint,
        routing_source_fingerprint,
    )
    from crucible.products.swebench.spec_synthesis import (
        synthesize_starter_spec,
    )
    from crucible.workflows import swebench as wf

    corpus = Corpus(
        root_corpus.spec_dir / "instances" / inst.instance_id / "corpus",
    )
    wf._prepare_instance_corpus(
        corpus, synthesize_starter_spec(inst), instance_id=inst.instance_id,
    )
    routing = load_routing(spec_dir=root_corpus.spec_dir)
    route = routing.for_role("implementer")
    image_digest = "test/image@sha256:" + "a" * 64
    payload = wf._PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(root_corpus.spec_dir),
        backend_config_path=None,
        max_turns=90,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir="",
        parent_run_dir=str(root_corpus.spec_dir / "unused-parent"),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        eval_timeout=7200,
        routed_backend=route.backend or "codex",
        routed_model=route.model or "unknown",
        routing=routing,
        routing_fingerprint=routing_fingerprint(routing),
        routing_source_sha256=routing_source_fingerprint(routing),
    )
    checkpoint_fingerprint = wf._pool_checkpoint_fingerprint(
        payload, image_digest=image_digest,
    )
    pristine_report = None
    if patch:
        patch_bytes = patch.encode("utf-8")
        pristine_report = wf.PristineVerificationReport(
            schema="crucible.swebench.pristine-verification.v1",
            status=wf.PristineVerificationStatus.PASSED,
            reason="test fixture passed pristine verification",
            patch_sha256=hashlib.sha256(patch_bytes).hexdigest(),
            patch_bytes=len(patch_bytes),
            base_commit=inst.base_commit,
            image_digest=image_digest,
            apply_check_rc=0,
            apply_rc=0,
            runner_rc=0,
            candidate_rc=0,
            comparison_mode="baseline_match",
            preexisting_failures=0,
            new_failures=0,
            worsened_failures=0,
            duration_s=0.1,
            cleanup_status="complete",
            raw_output_tail="",
        )
    wf._persist_surfaced_prediction(
        corpus,
        inst,
        wf.InstanceOutcome(
            instance_id=inst.instance_id,
            classification=classification,
            model_patch=patch,
            cost_usd=0.0,
            diff_bytes=len(patch.encode()),
            files_modified=("x",) if patch else (),
            pristine_verification=pristine_report,
        ),
        backend_name="stub",
        model=model,
        checkpoint_fingerprint=checkpoint_fingerprint,
        image_digest=image_digest,
        assigned_budget_usd=payload.budget_usd,
    )


def _default_backend_config(tmp_path: Path) -> Path:
    """Pin default-routing tests against workspace-local operator config."""
    path = tmp_path / "default_backends.yaml"
    path.write_text(
        "schema_version: crucible.backend-routing.v1\n"
        "default:\n"
        "  backend: claude\n"
        "  model: aws/anthropic/bedrock-claude-opus-4-8\n"
    )
    return path


def test_workflow_dispatch_dry_run(tmp_path: Path, monkeypatch):
    """End-to-end dry-run: workflow loads dataset, applies filters,
    emits a zero-patch predictions file under the run dir, and
    finalizes the run handle."""
    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record("django__django-1"),
        _minimal_record("django__django-2"),
        _minimal_record("sympy__sympy-1", repo="sympy/sympy"),
    ]))

    from crucible.workflows.swebench import main
    backend_config = _default_backend_config(tmp_path)
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--backend-config", str(backend_config),
        "--repo", "django/django",
        "--dry-run",
    ])
    assert rc == 0

    # Locate the run dir + predictions file the workflow created.
    runs = list((corpus.spec_dir / "runs").iterdir())
    assert len(runs) == 1
    preds_path = runs[0] / "predictions.jsonl"
    assert preds_path.exists()
    lines = preds_path.read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert {p["instance_id"] for p in parsed} == {
        "django__django-1", "django__django-2",
    }
    # Dry-run produces zero-patch predictions for every filtered instance.
    assert all(p["model_patch"] == "" for p in parsed)
    assert all(p["model_name_or_path"] == "aws/anthropic/bedrock-claude-opus-4-8" for p in parsed)

    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["mode"] == "dry_run"
    assert summary["filtered_instances"] == 2
    assert summary["total_instances"] == 3

    from crucible.backends.config import (
        load_routing,
        routing_feature_fingerprint,
        routing_fingerprint,
        routing_source_fingerprint,
    )
    routing = load_routing(
        config_path=backend_config,
        spec_dir=corpus.spec_dir,
    )
    manifest = json.loads((runs[0] / "run_manifest.json").read_text())
    provenance_inputs = manifest["provenance"]["inputs"]
    assert provenance_inputs["backend_routing"] == routing_fingerprint(routing)
    assert provenance_inputs["backend_features"] == (
        routing_feature_fingerprint(routing)
    )
    assert provenance_inputs["backend_routing_source_sha256"] == (
        routing_source_fingerprint(routing)
    )
    assert provenance_inputs["backend_routing"] != (
        f"sha256:{routing.source_sha256}"
    )


def test_workflow_dispatch_list_only_emits_no_predictions(
    tmp_path: Path, capsys,
):
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--list",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "a-1" in out
    # --list path doesn't open a run dir.
    runs = list((corpus.spec_dir / "runs").iterdir()) if \
        (corpus.spec_dir / "runs").exists() else []
    assert runs == []


def test_workflow_dispatch_missing_dataset(tmp_path: Path):
    corpus = _setup_corpus(tmp_path)
    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(tmp_path / "nonexistent.json"),
        "--dry-run",
    ])
    assert rc == 2
    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "dataset_unavailable"
    assert result["status"] == "dataset_unavailable"
    assert summary["duration_s"] > 0
    assert result["duration_s"] == summary["duration_s"]


def test_implement_empty_dataset_is_no_work_not_success(tmp_path: Path):
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "empty.json"
    dataset_path.write_text("[]", encoding="utf-8")

    from crucible.workflows.swebench import main

    assert main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
    ]) == 1
    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "no_work"
    assert summary["filtered_instances"] == 0
    assert summary["duration_s"] > 0
    assert result["status"] == "no_work"
    assert result["duration_s"] == summary["duration_s"]
    assert result["error_kind"] == "insufficient_input"


def test_implement_summary_separates_cascade_from_official_evaluation(
    tmp_path: Path, monkeypatch,
):
    from crucible.workflows import swebench as module

    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record("a-1"),
        _minimal_record("a-2"),
    ]))
    outcomes = [
        module.InstanceOutcome(
            instance_id="a-1",
            classification="success",
            model_patch="patch one",
            cost_usd=0.0,
            diff_bytes=9,
            files_modified=("a.py",),
            eval_status="unresolved",
        ),
        module.InstanceOutcome(
            instance_id="a-2",
            classification="hard_fail_verify",
            model_patch="patch two",
            cost_usd=0.0,
            diff_bytes=9,
            files_modified=("b.py",),
            eval_status="resolved",
        ),
    ]
    monkeypatch.setattr(module, "_docker_available", lambda: True)
    monkeypatch.setattr(module, "_check_disk_headroom", lambda: None)
    monkeypatch.setattr(
        module,
        "_run_implement_serial",
        lambda *args, **kwargs: (outcomes, 0.0, False, False),
    )

    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--backend-config", str(_default_backend_config(tmp_path)),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-evaluate-inline",
    ]) == 0

    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["cascade_success_count"] == 1
    assert "success_count" not in summary
    assert summary["resolved_count"] == 1
    assert summary["eval_status_counts"] == {"resolved": 1, "unresolved": 1}
    assert [row["eval_status"] for row in summary["per_instance"]] == [
        "unresolved", "resolved",
    ]


def test_workflow_dispatch_filters_match_nothing_returns_1(tmp_path: Path):
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))
    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--repo", "nonexistent/repo",
        "--dry-run",
    ])
    assert rc == 1
    run_dir = next((corpus.spec_dir / "runs").iterdir())
    assert json.loads((run_dir / "summary.json").read_text())[
        "status"
    ] == "no_instances_matched"


def test_workflow_dispatch_no_mode_selected(tmp_path: Path):
    """Without --list / --dry-run / --implement, the workflow refuses
    to act and emits a clear "no_mode_selected" status."""
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))
    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
    ])
    assert rc == 1
    runs = list((corpus.spec_dir / "runs").iterdir())
    assert len(runs) == 1
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["mode"] == "no_mode_selected"


def test_implement_container_preflight_terminal_has_summary_and_result(
    tmp_path: Path, monkeypatch,
):
    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))
    monkeypatch.setattr(
        "crucible.workflows.swebench._docker_available",
        lambda: False,
    )
    from crucible.workflows.swebench import main

    assert main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--backend-config", str(_default_backend_config(tmp_path)),
        "--implement", "--budget-usd-per-instance", "10",
    ]) == 2

    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == "container_runtime_required"
    assert result["status"] == "container_runtime_required"
    assert (run_dir / "events.jsonl").is_file()


def test_implement_requires_bounded_competition_before_authoring(
    tmp_path: Path, monkeypatch,
):
    from crucible.workflows import swebench as workflow

    corpus = _setup_corpus(tmp_path)
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([_minimal_record("a-1")]))
    monkeypatch.setattr(
        workflow, "_docker_available",
        lambda: pytest.fail("unbounded planning must fail before authoring"),
    )
    assert workflow.main([
        "--spec", str(corpus.spec_dir), "--dataset", str(dataset),
        "--backend-config", str(_default_backend_config(tmp_path)),
        "--implement",
    ]) == 2
    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "planning_policy_invalid"
    assert summary["total_cost_usd"] == 0.0
    assert not (corpus.spec_dir / "instances").exists()
    payload = workflow._PoolPayload(
        instance=load_dataset(dataset)[0],
        workspace_root_spec_dir=str(corpus.spec_dir),
        backend_config_path=None,
        max_turns=1,
        budget_usd=None,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "logs"),
        parent_run_dir=str(run_dir),
    )
    with pytest.raises(ValueError, match="positive finite total budget"):
        workflow._pool_worker(payload)
    assert not (corpus.spec_dir / "instances").exists()


def test_implement_rejects_removed_prompt_template_override(tmp_path: Path):
    """Authoring prompts are workflow-owned in the greenfield CLI."""

    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))
    template = tmp_path / "prompt.txt"
    template.write_text("operator-controlled prompt")
    from crucible.workflows.swebench import main

    with pytest.raises(SystemExit) as exc:
        main([
            "--spec", str(corpus.spec_dir),
            "--dataset", str(dataset_path),
            "--backend-config", str(_default_backend_config(tmp_path)),
            "--implement", "--budget-usd-per-instance", "10",
            "--prompt-template-file", str(template),
        ])
    assert exc.value.code == 2
    runs_dir = corpus.spec_dir / "runs"
    assert not runs_dir.exists() or not any(runs_dir.iterdir())


@pytest.mark.parametrize(
    ("spent", "accounting_incomplete", "expected_status"),
    [
        (2.0, False, "budget_exhausted"),
        (0.5, True, "cost_incomplete"),
    ],
)
def test_implement_never_completes_with_overspend_or_incomplete_cost(
    tmp_path: Path,
    monkeypatch,
    spent: float,
    accounting_incomplete: bool,
    expected_status: str,
):
    from crucible.workflows import swebench as module

    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([_minimal_record("a-1")]))
    outcome = module.InstanceOutcome(
        instance_id="a-1",
        classification="success",
        model_patch="patch",
        cost_usd=spent,
        diff_bytes=5,
        files_modified=("x",),
        cost_incomplete=accounting_incomplete,
    )
    monkeypatch.setattr(
        module,
        "_run_implement_serial",
        lambda *args, **kwargs: (
            [outcome], spent, False, accounting_incomplete,
        ),
    )

    assert module.main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--backend-config", str(_default_backend_config(tmp_path)),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-build-verify",
        "--no-evaluate-inline",
        "--budget-usd", "1",
    ]) == 1

    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    assert summary["status"] == expected_status
    assert result["status"] == expected_status


def test_accounting_halt_reports_usage_transport_and_unreached_reason(
    tmp_path: Path,
    monkeypatch,
    caplog,
):
    from crucible.workflows import swebench as module

    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record("a-1"),
        _minimal_record("a-2"),
    ]))
    outcome = module.InstanceOutcome(
        instance_id="a-1",
        classification="hard_fail_no_implementation",
        model_patch="",
        cost_usd=0.0,
        diff_bytes=0,
        files_modified=(),
        cost_incomplete=True,
        input_tokens=2468,
        output_tokens=135,
        planning_backend_calls=4,
        planning_transport_failure=True,
    )

    def _run_serial(filtered, *, costs, **kwargs):
        costs.record(
            role=outcome.instance_id,
            usage=outcome,
            backend_name="openai_compat",
            model="nvidia/zai-org/glm-5.2",
        )
        return [outcome], 0.0, False, True

    monkeypatch.setattr(module, "_run_implement_serial", _run_serial)

    with caplog.at_level("WARNING"):
        assert module.main([
            "--spec", str(corpus.spec_dir),
            "--dataset", str(dataset_path),
            "--backend-config", str(_default_backend_config(tmp_path)),
            "--implement", "--budget-usd-per-instance", "10",
            "--no-build-verify",
            "--no-evaluate-inline",
        ]) == 1

    run_dir = next((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["status"] == "cost_incomplete"
    assert summary["unreached_instances"] == 1
    assert summary["unreached_reason"] == "accounting_incomplete"
    assert summary["transport_failure_instances"] == 1
    assert summary["costs"]["total_input_tokens"] == 2468
    assert summary["costs"]["total_output_tokens"] == 135
    assert summary["costs"]["planning_backend_calls"] == 4
    assert summary["costs"]["remaining_usd"] == 0.0
    assert summary["costs"]["accounting_debt"] is True
    assert summary["costs"]["accounting_debt_by_role"] == {
        "a-1": ["cost_complete", "token_usage_complete"],
    }
    messages = [record.getMessage() for record in caplog.records]
    assert any("due to incomplete cost accounting" in msg for msg in messages)
    assert not any("due to budget halt" in msg for msg in messages)


@pytest.mark.parametrize(
    ("workers", "expected_ids"),
    [
        (1, ("a-1",)),
        (2, ("a-1", "a-2")),
    ],
)
def test_instance_rollup_debt_closes_serial_and_parallel_cost_authority(
    tmp_path: Path,
    monkeypatch,
    workers: int,
    expected_ids: tuple[str, ...],
) -> None:
    from crucible.common.cost import CostAccumulator
    from crucible.workflows import swebench as module

    instances = tuple(
        SwebenchInstance(
            instance_id=instance_id,
            repo="example/repo",
            base_commit="0" * 40,
            patch="",
            test_patch="",
            problem_statement="fix it",
            fail_to_pass=(),
            pass_to_pass=(),
        )
        for instance_id in ("a-1", "a-2")
    )
    outcomes = {
        "a-1": module.InstanceOutcome(
            instance_id="a-1",
            classification="hard_fail_no_implementation",
            model_patch="",
            cost_usd=0.25,
            diff_bytes=0,
            files_modified=(),
            cost_incomplete=True,
            input_tokens=100,
            output_tokens=10,
        ),
        "a-2": module.InstanceOutcome(
            instance_id="a-2",
            classification="success",
            model_patch="",
            cost_usd=0.10,
            diff_bytes=0,
            files_modified=(),
            input_tokens=50,
            output_tokens=5,
        ),
    }
    monkeypatch.setattr(
        module,
        "_pool_worker",
        lambda payload: outcomes[payload.instance.instance_id],
    )
    routing = SimpleNamespace(
        for_role=lambda _role: SimpleNamespace(
            backend="claude",
            model="test-model",
        )
    )
    args = SimpleNamespace(
        max_concurrent_instances=workers,
        spec=tmp_path / "corpus",
        backend_config=None,
        max_turns=3,
        planner_count=3,
        budget_usd_per_instance=None,
        repo_cache_dir=None,
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        eval_timeout=30,
        rerun=False,
        rerun_status=(),
        submit_any_nonempty_best_effort=False,
        budget_usd=10.0,
    )
    events: list[dict] = []
    costs = CostAccumulator()

    completed, spent, halted, incomplete = module._run_implement_parallel(
        instances,
        args=args,
        run_dir=tmp_path / "run",
        handle=SimpleNamespace(write_event=events.append),
        predictions_path=tmp_path / "predictions.jsonl",
        routing=routing,
        costs=costs,
    )

    assert tuple(outcome.instance_id for outcome in completed) == expected_ids
    assert spent == pytest.approx(
        sum(outcomes[instance_id].cost_usd for instance_id in expected_ids)
    )
    assert halted is False
    assert incomplete is True
    cost_summary = costs.to_dict()
    assert cost_summary["total_usd"] == pytest.approx(spent)
    assert cost_summary["remaining_usd"] == 0.0
    assert cost_summary["accounting_debt"] is True
    assert cost_summary["accounting_debt_by_role"] == {
        "a-1": ["cost_complete", "token_usage_complete"],
    }


# ----------------------------------------------------------------------
# --implement live path (mocked backend + workspace)
# ----------------------------------------------------------------------


def _init_git_repo(path: Path) -> str:
    """Initialize a minimal git repo with one seed commit, return SHA."""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=path, check=True,
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    (path / "seed.py").write_text("def f(): return 0\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"], cwd=path, check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path, check=True, capture_output=True, text=True,
    ).stdout.strip()


def test_implement_e2e_with_stub_backend_and_fake_workspace(
    tmp_path: Path, monkeypatch,
):
    """End-to-end --implement through the full plan → critic → implement
    cascade: stub backend serves all three roles (multi-role dispatch
    on ``invocation.system``), mutates seed.py during the implementer
    call, workflow captures the diff into predictions JSONL."""
    import contextlib as _contextlib

    monkeypatch.delenv("CRUCIBLE_BACKEND_CONFIG", raising=False)

    corpus = _setup_corpus(tmp_path)

    # Pre-prepare a git working tree at HEAD = base_commit.
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "fake__repo-1",
            repo="fake/repo",
            base_commit=base,
            problem_statement="Make f() return 42 instead of 0.",
            FAIL_TO_PASS=["tests::test_f"],
            PASS_TO_PASS=[],
        )
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 42\n")
        ),
        cost_per_call=0.01,
    )
    _patch_backend_resolver(monkeypatch, backend)
    _patch_smoke_check(monkeypatch)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--backend-config", str(_default_backend_config(tmp_path)),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    assert backend.counters["planner"] == 3
    assert backend.counters["critic"] == 3
    assert backend.counters["judge"] == 3
    assert backend.counters["selector"] == 1
    assert backend.counters["implementer"] == 1

    runs = list((corpus.spec_dir / "runs").iterdir())
    assert len(runs) == 1
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["mode"] == "implement"
    from crucible.workflows._benchmark_planning import benchmark_planning_identity

    assert summary["planning"] == benchmark_planning_identity(3)
    assert summary["total_cost_usd"] == pytest.approx(0.11)
    assert summary["processed_instances"] == 1
    assert summary["nonempty_patch_count"] == 1
    assert summary["per_instance"][0]["instance_id"] == "fake__repo-1"
    assert summary["per_instance"][0]["classification"] == "success"
    assert summary["submission_policy"] == "verified_success_only"
    candidate = summary["per_instance"][0]["candidate_capture"]
    assert candidate["diff_ref"] == f"sha256:{candidate['sha256']}"
    assert candidate["diff_bytes"] > 0
    assert candidate["files_modified"] == ["seed.py"]
    assert candidate["origin"] == "workspace"
    assert candidate["capture_status"] == "source_filtered"
    assert summary["per_instance"][0]["candidate_eligibility"] == {
        "blockers": [],
        "status": "eligible",
    }
    submission = summary["per_instance"][0]["submission_decision"]
    assert submission == {
        "override_requested": False,
        "override_used": False,
        "policy": "verified_success_only",
        "submitted": True,
        "submitted_patch_ref": candidate["diff_ref"],
    }
    assert summary["duration_s"] > 0.0
    result = json.loads((runs[0] / "result.json").read_text())
    assert result["duration_s"] == summary["duration_s"]

    # Predictions file carries the captured diff.
    preds_path = runs[0] / "predictions.jsonl"
    parsed = json.loads(preds_path.read_text(encoding="utf-8").strip())
    assert parsed["instance_id"] == "fake__repo-1"
    assert "return 42" in parsed["model_patch"]
    # Model attribution comes from the explicitly isolated routing config's
    # implementer role, independent of any workspace-local operator config.
    assert parsed["model_name_or_path"] == "aws/anthropic/bedrock-claude-opus-4-8"


# --evaluate-predictions tests deleted alongside the CLI flag — the
# standalone batch-eval phase is gone now that --implement runs the
# official harness inline before image teardown. Inline-eval coverage
# lives in the tests below.


def test_evaluate_prediction_inline_skips_empty_patch(tmp_path: Path):
    """``_evaluate_prediction_inline`` short-circuits on empty
    model_patch — the harness would report ``resolved=False`` for
    a no-op prediction and the docker run cost is wasted. Skipping
    is the right shape; the prediction is still on disk in
    predictions.jsonl for the cascade's outcome record."""
    from crucible.workflows.swebench import (
        InstanceOutcome, _evaluate_prediction_inline,
    )

    corpus = _setup_corpus(tmp_path)
    inst = SwebenchInstance(
        instance_id="x-1", repo="x/y", base_commit="0" * 40,
        problem_statement="", patch="", test_patch="",
        fail_to_pass=(), pass_to_pass=(), version="1.0",
        environment_setup_commit=None,
    )
    outcome = InstanceOutcome(
        instance_id="x-1", classification="success",
        model_patch="", cost_usd=0.0, diff_bytes=0,
        files_modified=(),
    )
    result = _evaluate_prediction_inline(
        inst, outcome, "any-model",
        instance_corpus=corpus,
        instance_log_dir=tmp_path / "log",
    )
    assert result is None


def test_evaluate_prediction_inline_invokes_harness_with_single_instance_scope(
    tmp_path: Path, monkeypatch,
):
    """The inline call MUST pass ``instance_ids=[<this_id>]`` to the
    harness — without this scope, the harness would try to evaluate
    every record in the predictions.jsonl which is exactly what the
    inline path is meant to avoid. Also: ``max_workers=1`` (inline is
    one-at-a-time per call) so we don't over-parallelize on a single
    instance."""
    from crucible.products.swebench.eval import (
        EvalReport, InstanceEvalReport,
    )
    from crucible.workflows.swebench import (
        InstanceOutcome, _evaluate_prediction_inline,
    )

    corpus = _setup_corpus(tmp_path)
    inst = SwebenchInstance(
        instance_id="django__django-1", repo="django/django",
        base_commit="0" * 40, problem_statement="", patch="",
        test_patch="", fail_to_pass=(), pass_to_pass=(), version="1.0",
        environment_setup_commit=None,
    )
    outcome = InstanceOutcome(
        instance_id="django__django-1", classification="success",
        model_patch="diff --git a/x b/x\nindex 1..2 100644\n--- a/x\n+++ b/x\n",
        cost_usd=0.0, diff_bytes=10,
        files_modified=("x",),
    )

    captured: dict = {}

    def fake_run_evaluation(
        predictions_path, instance_ids, *, run_id, model, output_dir,
        max_workers, **kwargs,
    ):
        captured["predictions_path"] = predictions_path
        captured["instance_ids"] = list(instance_ids)
        captured["max_workers"] = max_workers
        output_dir.mkdir(parents=True, exist_ok=True)
        return EvalReport(
            run_id=run_id, model=model, output_dir=output_dir,
            per_instance=(
                InstanceEvalReport(
                    instance_id=inst.instance_id,
                    resolved=True, patch_applied=True,
                    fail_to_pass_passed=2, fail_to_pass_failed=0,
                    pass_to_pass_passed=5, pass_to_pass_failed=0,
                    raw_report={"resolved": True},
                ),
            ),
        )

    monkeypatch.setattr(
        "crucible.workflows.swebench.run_evaluation", fake_run_evaluation,
    )

    log_dir = tmp_path / "log"
    result = _evaluate_prediction_inline(
        inst, outcome, "aws/anthropic/bedrock-claude-opus-4-8",
        instance_corpus=corpus,
        instance_log_dir=log_dir,
    )
    assert result is not None
    assert result.resolved is True
    # The harness was scoped to just this instance — not the whole
    # predictions.jsonl.
    assert captured["instance_ids"] == ["django__django-1"]
    # max_workers=1: inline is one-at-a-time per call, the
    # parallelism is at the swebench worker layer above.
    assert captured["max_workers"] == 1
    # Predictions JSONL was written next to the harness output, with
    # exactly this one prediction.
    preds_text = captured["predictions_path"].read_text()
    rec = json.loads(preds_text.strip())
    assert rec["instance_id"] == "django__django-1"
    assert rec["model_name_or_path"] == "aws/anthropic/bedrock-claude-opus-4-8"


def test_evaluate_prediction_inline_swallows_harness_failure(
    tmp_path: Path, monkeypatch,
):
    """When the harness raises (docker daemon hiccup, image evicted
    between setup + eval, etc.), the inline path returns None and
    logs — the cascade's diff is already persisted to the corpus +
    predictions.jsonl; recovery is via ``run_evaluation`` against the
    frozen predictions file. The implement phase doesn't get marked
    as failed just because eval crashed."""
    from crucible.products.swebench.eval import SwebenchEvalError
    from crucible.workflows.swebench import (
        InstanceOutcome, _evaluate_prediction_inline,
    )

    corpus = _setup_corpus(tmp_path)
    inst = SwebenchInstance(
        instance_id="x-1", repo="x/y", base_commit="0" * 40,
        problem_statement="", patch="", test_patch="",
        fail_to_pass=(), pass_to_pass=(), version="1.0",
        environment_setup_commit=None,
    )
    outcome = InstanceOutcome(
        instance_id="x-1", classification="success",
        model_patch="diff --git a/x b/x\n", cost_usd=0.0, diff_bytes=10,
        files_modified=("x",),
    )

    def raises(*args, **kwargs):
        raise SwebenchEvalError("docker daemon unreachable")

    monkeypatch.setattr(
        "crucible.workflows.swebench.run_evaluation", raises,
    )

    result = _evaluate_prediction_inline(
        inst, outcome, "any-model",
        instance_corpus=corpus,
        instance_log_dir=tmp_path / "log",
    )
    assert result is None


def test_latest_nonempty_diff_candidate_salvages_from_corpus(tmp_path):
    """The newest stored non-empty candidate retains its producer identity."""
    from crucible.workflows.swebench import _latest_nonempty_diff_candidate
    from assay.corpus.store import Corpus
    from assay import (
        Implementation, ExecutionContext, ProjectContext,
    )
    from assay.types.spec import Spec
    import datetime as _dt

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)

    def _exec():
        return ExecutionContext(
            timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
            machine="t", cwd="/t", toolchain={}, backend="codex",
            model="gpt-5.5", recipe="test",
            recipe_commit=None, gpu_model=None, run_id="r",
        )

    def _impl(slice_id: str, classification: str, diff_bytes: bytes):
        diff_ref = corpus.put_artifact(
            diff_bytes, kind="diffs", name_hint=f"{slice_id}.diff",
        ) if diff_bytes else corpus.put_artifact(
            b"", kind="diffs", name_hint=f"{slice_id}.diff",
        )
        return Implementation(
            slice_id=slice_id,
            base_commit="0" * 40,
            result_commit=None,
            diff_ref=diff_ref,
            transcript_ref=corpus.put_artifact(
                b"{}", kind="transcripts", name_hint=f"{slice_id}.t",
            ),
            files_modified=("f.py",) if diff_bytes else (),
            classification=classification,
            execution=_exec(),
            scope_escapes=(),
            cost_usd=0.0,
            artifacts={},
        )

    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="https://x/y", base_commit="0" * 40,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None,
        intent="t", constraints=(), done_criteria=(), plan_policy=None,
    )
    # Build 4 implementations: success(big), empty, success(small), empty.
    # The salvage walks in reverse — latest non-empty (small=625B)
    # wins, NOT the older success(big) record.
    impls = [
        _impl("phase1", "verify_failed", b"INITIAL_DIFF_3469B" * 100),
        _impl("breach_r0", "hard_fail_empty_diff", b""),
        _impl("breach_r1", "success", b"REFINED_DIFF_625B"),
        _impl("breach_r2", "hard_fail_empty_diff", b""),
    ]
    corpus.put_spec(spec)
    cur_spec = spec
    for impl in impls:
        cur_spec = cur_spec.with_implementation(impl)
        corpus.put_record(impl, kind="implementations")
        corpus.put_spec(cur_spec)
    cur_spec = corpus.latest_spec()

    salvaged = _latest_nonempty_diff_candidate(corpus, cur_spec, "x-1")
    # The salvage walks reverse: r2 is empty (skip), r1 has the
    # 625B refined diff — that wins. Phase1's bigger but older diff
    # is correctly NOT picked because r1 supersedes it.
    assert salvaged is not None
    assert salvaged.patch == b"REFINED_DIFF_625B"
    assert salvaged.classification == "success"
    assert salvaged.slice_id == "breach_r1"


def test_latest_nonempty_diff_candidate_returns_none_when_no_diffs(tmp_path):
    """No stored implementation diff yields no candidate."""
    from crucible.workflows.swebench import _latest_nonempty_diff_candidate
    from assay.corpus.store import Corpus
    from assay.types.spec import Spec
    from assay import ProjectContext

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    spec = Spec(
        version=0, parent=None, root=None, schema_version=1,
        project_context=ProjectContext(
            project="t", repo_url="https://x/y", base_commit="0" * 40,
            branch=None, submodule_path=None, runners={},
        ),
        execution=None, intent="t", constraints=(), done_criteria=(),
        plan_policy=None,
    )
    # No implementation_refs at all.
    assert _latest_nonempty_diff_candidate(corpus, spec, "x-1") is None


def test_replace_outcome_eval_status_cascade_aborted_wins():
    """When ``cascade_aborted=True``, the sentinel is
    ``"cascade_aborted_partial"`` regardless of other inputs —
    distinct from skipped_disabled / skipped_empty / harness_failed
    so post-sweep inspection attributes lost work to mid-flight
    cascade abort (budget cap, persistence-veto, transport error) vs.
    other causes. Pin the priority so a future refactor that re-orders
    the branches doesn't accidentally collapse this case into one of
    the other skipped sentinels."""
    from crucible.products.swebench.eval import InstanceEvalReport
    from crucible.workflows.swebench import (
        InstanceOutcome, _replace_outcome_eval_status,
    )

    outcome = InstanceOutcome(
        instance_id="x-1", classification="success",
        model_patch="diff --git ...", cost_usd=5.10,
        diff_bytes=200, files_modified=("x.py",),
    )
    # Cascade aborted, eval not run.
    out = _replace_outcome_eval_status(
        outcome, evaluate_inline_requested=False,
        eval_report=None, cascade_aborted=True,
    )
    assert out.eval_status == "cascade_aborted_partial"

    # Cascade abort dominates even if a (stale) eval report exists.
    fake_report = InstanceEvalReport(
        instance_id="x-1", resolved=True, patch_applied=True,
        raw_report={"resolved": True},
    )
    out = _replace_outcome_eval_status(
        outcome, evaluate_inline_requested=True,
        eval_report=fake_report, cascade_aborted=True,
    )
    assert out.eval_status == "cascade_aborted_partial"


def test_assay_budget_exhausted_is_not_an_exception_class():
    """Source-invariant: ``assay.BudgetExhausted`` is a ``@dataclass``,
    NOT an Exception subclass — it's attached to
    ``InvocationResult.error`` by backends, never raised.

    ``workflows/swebench.py`` MUST NOT contain ``except BudgetExhausted``
    clauses. Prior code had two such clauses (parallel + serial paths);
    both were no-ops on the happy path and crashed with
    ``TypeError: catching classes that do not inherit from BaseException``
    on any real exception from ``run_architect``, silently losing
    partial cascade work. Observed on pylint-4551 (recovery sweep
    2026-05-24): 10 valid Implementation refs in corpus, prediction
    still empty because the broken except clause crashed before
    ``_spec_to_outcome``'s salvage path could run.

    Two-pronged invariant:
      1. ``assay.BudgetExhausted`` is NOT a BaseException subclass —
         catching it directly is always a Python-level bug.
      2. The string ``except BudgetExhausted`` MUST NOT appear in
         ``workflows/swebench.py``.
    """
    from assay import BudgetExhausted

    assert isinstance(BudgetExhausted, type), (
        "BudgetExhausted must be a class for this invariant to apply"
    )
    assert not issubclass(BudgetExhausted, BaseException), (
        "If BudgetExhausted is ever made an Exception subclass, this "
        "invariant should be revisited — the narrow ``except BudgetExhausted`` "
        "clauses removed from swebench.py could in principle come back."
    )

    swebench_path = Path(__file__).resolve().parent.parent / "workflows" / "swebench.py"
    # Scan only non-comment lines so the regression comments in this
    # file's docstrings don't trigger the invariant.
    code_lines = []
    for raw in swebench_path.read_text(encoding="utf-8").splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        code_lines.append(raw)
    code_only = "\n".join(code_lines)
    assert "except BudgetExhausted" not in code_only, (
        f"{swebench_path.name} contains ``except BudgetExhausted`` in "
        "live code — this is a no-op on the happy path and crashes with "
        "TypeError on any real exception. Use ``except Exception`` "
        "instead, with a salvage path that captures partial workspace state."
    )


def test_replace_outcome_eval_status_decision_table():
    """The decision table that derives ``eval_status`` sentinels.
    Pinning each case explicitly so a future refactor that collapses
    branches surfaces here rather than silently producing the wrong
    sentinel in a sweep summary."""
    from crucible.products.swebench.eval import InstanceEvalReport
    from crucible.workflows.swebench import (
        InstanceOutcome, _replace_outcome_eval_status,
    )

    def _outcome(model_patch: str = "diff --git ...") -> InstanceOutcome:
        return InstanceOutcome(
            instance_id="x-1", classification="success",
            model_patch=model_patch, cost_usd=0.0, diff_bytes=10,
            files_modified=("x",),
        )

    # Disabled.
    out = _replace_outcome_eval_status(_outcome(), False, None)
    assert out.eval_status == "skipped_disabled"

    # Empty patch.
    out = _replace_outcome_eval_status(_outcome(""), True, None)
    assert out.eval_status == "skipped_empty"

    # Harness raised (helper returned None despite non-empty patch).
    out = _replace_outcome_eval_status(_outcome(), True, None)
    assert out.eval_status == "harness_failed"

    # Resolved (all green).
    resolved_report = InstanceEvalReport(
        instance_id="x-1", resolved=True, patch_applied=True,
        raw_report={"resolved": True},
    )
    out = _replace_outcome_eval_status(_outcome(), True, resolved_report)
    assert out.eval_status == "resolved"

    # Patch applied but tests failed → "unresolved".
    unresolved_report = InstanceEvalReport(
        instance_id="x-1", resolved=False, patch_applied=True,
        raw_report={"resolved": False},
    )
    out = _replace_outcome_eval_status(_outcome(), True, unresolved_report)
    assert out.eval_status == "unresolved"

    # Patch failed to apply at all → "patch_apply_failed" (distinct
    # remediation from "unresolved": implementer produced bad-format
    # output rather than a wrong fix).
    apply_failed_report = InstanceEvalReport(
        instance_id="x-1", resolved=False, patch_applied=False,
        raw_report={"resolved": False, "patch_successfully_applied": False},
    )
    out = _replace_outcome_eval_status(_outcome(), True, apply_failed_report)
    assert out.eval_status == "patch_apply_failed"


def _pristine_policy_fixture(status, *, best_effort=False):
    from crucible.workflows import swebench as workflow

    patch = b"diff --git a/x.py b/x.py\n"
    sha256 = hashlib.sha256(patch).hexdigest()
    candidate = workflow.PredictionCandidateCapture(
        source_patch=patch,
        diff_ref=f"sha256:{sha256}",
        sha256=sha256,
        diff_bytes=len(patch),
        files_modified=("x.py",),
        origin=workflow.CandidateOrigin.WORKSPACE,
        capture_status=workflow.CandidateCaptureStatus.SOURCE_FILTERED,
        producer_slice_id="source",
        producer_classification="success",
    )
    outcome = workflow.InstanceOutcome(
        instance_id="x-1",
        classification="success",
        model_patch=patch.decode(),
        cost_usd=1.0,
        diff_bytes=len(patch),
        files_modified=("x.py",),
        diff_provenance="workspace",
        candidate_capture=candidate,
        candidate_eligibility=workflow.PredictionCandidateEligibility(
            status=workflow.CandidateEligibilityStatus.ELIGIBLE,
        ),
        submission_decision=workflow.PredictionSubmissionDecision(
            policy=(
                workflow.PredictionSubmissionPolicy.BEST_EFFORT_ANY_NONEMPTY
                if best_effort
                else workflow.PredictionSubmissionPolicy.VERIFIED_SUCCESS_ONLY
            ),
            submitted=True,
            override_requested=best_effort,
            override_used=False,
            submitted_patch_ref=f"sha256:{sha256}",
        ),
    )
    report = workflow.PristineVerificationReport(
        schema="crucible.swebench.pristine-verification.v1",
        status=status,
        reason="test report",
        patch_sha256=sha256,
        patch_bytes=len(patch),
        base_commit="a" * 40,
        image_digest="test/image@sha256:" + "b" * 64,
        apply_check_rc=(
            1
            if status
            == workflow.PristineVerificationStatus.PATCH_APPLY_FAILED
            else 0
        ),
        apply_rc=(
            None
            if status
            == workflow.PristineVerificationStatus.PATCH_APPLY_FAILED
            else 0
        ),
        runner_rc=(
            0
            if status == workflow.PristineVerificationStatus.PASSED
            else 1
            if status == workflow.PristineVerificationStatus.TEST_FAILED
            else None
        ),
        candidate_rc=(
            0
            if status == workflow.PristineVerificationStatus.PASSED
            else 1
            if status == workflow.PristineVerificationStatus.TEST_FAILED
            else None
        ),
        comparison_mode=(
            "baseline_match"
            if status == workflow.PristineVerificationStatus.PASSED
            else "regressed"
            if status == workflow.PristineVerificationStatus.TEST_FAILED
            else "unavailable"
        ),
        preexisting_failures=0,
        new_failures=(
            1
            if status == workflow.PristineVerificationStatus.TEST_FAILED
            else 0
        ),
        worsened_failures=0,
        duration_s=0.5,
        cleanup_status="complete",
        raw_output_tail="",
    )
    return workflow, outcome, report


def test_pristine_verification_pass_preserves_submission():
    from crucible.workflows import swebench as workflow

    workflow, outcome, report = _pristine_policy_fixture(
        workflow.PristineVerificationStatus.PASSED,
    )
    verified = workflow._apply_pristine_verification(outcome, report)

    assert verified.model_patch == outcome.model_patch
    assert verified.classification == "success"
    assert verified.pristine_verification is report
    assert verified.submission_decision.submitted is True


@pytest.mark.parametrize(
    ("status", "classification", "eligibility", "blocker"),
    (
        (
            "patch_apply_failed",
            "hard_fail_pristine_patch_apply",
            "ineligible",
            "pristine_verify_patch_apply_failed",
        ),
        (
            "test_failed",
            "hard_fail_pristine_test",
            "ineligible",
            "pristine_verify_test_failed",
        ),
        (
            "infra_failed",
            "hard_fail_pristine_infra",
            "unknown",
            "pristine_verify_infra_failed",
        ),
    ),
)
def test_pristine_verification_failures_suppress_default_submission(
    status,
    classification,
    eligibility,
    blocker,
):
    from crucible.workflows import swebench as workflow

    enum_status = workflow.PristineVerificationStatus(status)
    workflow, outcome, report = _pristine_policy_fixture(enum_status)
    failed = workflow._apply_pristine_verification(outcome, report)

    assert failed.model_patch == ""
    assert failed.diff_bytes == 0
    assert failed.files_modified == ()
    assert failed.classification == classification
    assert failed.candidate_capture is outcome.candidate_capture
    assert failed.candidate_eligibility.status == eligibility
    assert blocker in failed.candidate_eligibility.blockers
    assert failed.submission_decision.submitted is False
    assert failed.submission_decision.submitted_patch_ref is None
    assert failed.pristine_verification is report
    evaluated = workflow._replace_outcome_eval_status(failed, True, None)
    assert evaluated.eval_status == f"skipped_pristine_{status}"


def test_pristine_verification_failure_requires_explicit_best_effort_override():
    from crucible.workflows import swebench as workflow

    workflow, outcome, report = _pristine_policy_fixture(
        workflow.PristineVerificationStatus.TEST_FAILED,
        best_effort=True,
    )
    overridden = workflow._apply_pristine_verification(outcome, report)

    assert overridden.model_patch == outcome.model_patch
    assert overridden.classification == "success"
    assert overridden.candidate_eligibility.status == "ineligible"
    assert overridden.submission_decision.submitted is True
    assert overridden.submission_decision.override_requested is True
    assert overridden.submission_decision.override_used is True
    assert "best_effort" in overridden.diff_provenance


def test_evaluate_prediction_inline_stamps_measurements_to_corpus(
    tmp_path: Path, monkeypatch,
):
    """The inline helper must call ``_stamp_eval_measurements`` so the
    dashboard's per-instance view surfaces the harness verdict alongside
    the cascade's slice classifications. Without this stamp the
    verdict lives only in summary.json / report.json and the corpus
    is silent."""
    from crucible.products.swebench.eval import (
        EvalReport, InstanceEvalReport,
    )
    from crucible.workflows.swebench import (
        InstanceOutcome, _evaluate_prediction_inline,
    )

    corpus = _setup_corpus(tmp_path)
    inst = SwebenchInstance(
        instance_id="x-1", repo="x/y", base_commit="0" * 40,
        problem_statement="", patch="", test_patch="",
        fail_to_pass=(), pass_to_pass=(), version="1.0",
        environment_setup_commit=None,
    )
    outcome = InstanceOutcome(
        instance_id="x-1", classification="success",
        model_patch="diff --git a/x b/x\n", cost_usd=0.0, diff_bytes=10,
        files_modified=("x",),
    )

    evaluation_calls: list[str] = []

    def fake_run_evaluation(
        predictions_path, instance_ids, *, run_id, model, output_dir,
        **kwargs,
    ):
        evaluation_calls.append(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        return EvalReport(
            run_id=run_id, model=model, output_dir=output_dir,
            per_instance=(
                InstanceEvalReport(
                    instance_id="x-1",
                    resolved=True, patch_applied=True,
                    fail_to_pass_passed=1, fail_to_pass_failed=0,
                    pass_to_pass_passed=2, pass_to_pass_failed=0,
                    raw_report={"resolved": True},
                ),
            ),
        )

    monkeypatch.setattr(
        "crucible.workflows.swebench.run_evaluation", fake_run_evaluation,
    )

    _evaluate_prediction_inline(
        inst, outcome, "m",
        instance_corpus=corpus,
        instance_log_dir=tmp_path / "log",
    )
    # Both the instance bin and aggregate bin are content-addressed and
    # attached to the immutable Spec lineage, not merely left in SQLite.
    refs = list(corpus.list_artifacts(kind="measurements"))
    latest = corpus.latest_spec()
    assert len(refs) == 2
    assert set(latest.measurement_refs) == set(refs)
    assert len(latest.lifecycle_event_refs) == 1
    # Per-instance and aggregate bins are one indivisible evaluation fact set.
    assert latest.version == 1
    assert evaluation_calls == ["inline_x-1"]


def test_inline_eval_fires_before_container_teardown(
    tmp_path: Path, monkeypatch,
):
    """The load-bearing assumption of the inline-eval design is that
    ``run_evaluation`` runs WHILE the per-instance image is still
    cached locally — i.e., BEFORE ``teardown_instance_container``'s
    ``docker image rm`` fires. If a future refactor reorders the
    with-block exit so teardown precedes the eval call, the cache-
    reuse premise silently breaks and every inline call re-pulls a
    multi-GB image.

    Strategy: run the cascade with build_verify ENABLED (so the
    ``_instance_env_lifecycle`` docker setup + teardown path actually
    fires) but mock both the setup + teardown so the test doesn't
    need a real docker daemon. Spy on both ``run_evaluation`` and
    ``_teardown_instance_container`` and assert eval-first. The
    prior shape of this test passed vacuously when --no-build-verify
    skipped the teardown path entirely.
    """
    import contextlib as _contextlib

    from crucible.products.swebench.eval import (
        EvalReport, InstanceEvalReport,
    )

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="fix it",
            FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    # Backend that produces a non-empty diff (so inline eval doesn't
    # short-circuit on skipped_empty).
    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 2\n")
        ),
    )
    _patch_backend_resolver(monkeypatch, backend)

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )

    # Force the docker container path inside ``_instance_env_lifecycle``
    # without needing a real daemon. ``_docker_available`` says yes,
    # the spec resolver yields a fake spec, and setup returns a fake
    # container id. Teardown is spied so we can pin order.
    call_order: list[str] = []
    from crucible.workflows import swebench as swebench_workflow
    default_pristine_verifier = (
        swebench_workflow._verify_submitted_candidate
    )

    def spy_pristine_verifier(instance, outcome, **kwargs):
        call_order.append("pristine_verification")
        return default_pristine_verifier(instance, outcome, **kwargs)

    monkeypatch.setattr(
        "crucible.workflows.swebench._verify_submitted_candidate",
        spy_pristine_verifier,
    )

    @_contextlib.contextmanager
    def ordered_lifecycle(*args, **kwargs):
        try:
            yield {
                "container_id": "fake_container_id_abc123",
                "image_digest": "fake/img@sha256:" + "b" * 64,
            }
        finally:
            call_order.append("teardown_instance_container")

    monkeypatch.setattr(
        "crucible.workflows.swebench._instance_env_lifecycle",
        ordered_lifecycle,
    )

    from crucible.products.swebench.instance_container import (
        InstanceContainerSpec,
    )

    monkeypatch.setattr(
        "crucible.workflows.swebench._docker_available", lambda: True,
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench._resolve_container_spec_impl",
        lambda inst: InstanceContainerSpec(
            image="fake/img:latest", install_command="true",
        ),
    )

    def fake_setup_container(workspace, container_name, spec):
        return "fake_container_id_abc123"

    monkeypatch.setattr(
        "crucible.workflows.swebench._setup_instance_container_impl",
        fake_setup_container,
    )

    def spy_teardown_container(*args, **kwargs):
        call_order.append("teardown_instance_container")

    monkeypatch.setattr(
        "crucible.workflows.swebench._teardown_instance_container",
        spy_teardown_container,
    )

    # Stub the pre-check ``docker rm -f <name>`` that runs before
    # setup — it's an unrelated cleanup call and would otherwise add
    # noise to call_order.
    import subprocess as _subprocess
    _real_run = _subprocess.run

    def filtered_run(cmd, *a, **kw):
        # Pass-through everything except the docker rm precheck.
        if (isinstance(cmd, list) and len(cmd) >= 3
                and cmd[0] == "docker" and cmd[1] == "rm"):
            class _FakeCP:
                returncode = 0
                stdout = ""
                stderr = ""
            return _FakeCP()
        return _real_run(cmd, *a, **kw)

    monkeypatch.setattr(_subprocess, "run", filtered_run)

    def spy_run_evaluation(*args, **kwargs):
        call_order.append("run_evaluation")
        return EvalReport(
            run_id=kwargs.get("run_id", "x"),
            model=kwargs.get("model", "m"),
            output_dir=kwargs["output_dir"],
            per_instance=(InstanceEvalReport(
                instance_id="a-1",
                resolved=True, patch_applied=True,
                raw_report={"resolved": True},
            ),),
        )

    monkeypatch.setattr(
        "crucible.workflows.swebench.run_evaluation", spy_run_evaluation,
    )

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach",
        "--no-include-review",
        # This test isolates eval-before-teardown ordering. Its fake container
        # cannot execute the visible-test gate, so explicitly retain the
        # resulting non-empty best-effort patch for the inline-eval assertion.
        "--submit-any-nonempty-best-effort",
        # build_verify default ON — drives the docker setup +
        # teardown path that the eval-before-teardown invariant
        # actually applies to.
    ])
    assert rc in (0, 1)

    # Both calls MUST have fired — the docker container path was
    # forced via the monkeypatches above, so teardown is non-optional.
    assert "run_evaluation" in call_order, (
        f"inline eval never fired; calls observed: {call_order!r}"
    )
    assert "pristine_verification" in call_order
    assert "teardown_instance_container" in call_order, (
        f"teardown never fired despite forced docker container path; "
        f"calls observed: {call_order!r}"
    )
    verifier_idx = call_order.index("pristine_verification")
    eval_idx = call_order.index("run_evaluation")
    teardown_idx = call_order.index("teardown_instance_container")
    assert verifier_idx < eval_idx < teardown_idx, (
        "pristine verification and inline eval must both run before "
        "teardown_instance_container, in that order "
        f"(image cache reuse premise); call_order={call_order!r}"
    )


def test_implement_writes_predictions_incrementally(
    tmp_path: Path, monkeypatch,
):
    """A multi-instance --implement run must persist each prediction
    as the instance finishes, not just at end-of-loop. Snapshot the
    predictions file from inside the implementer call — first call
    sees 0 lines, second sees 1."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x",
            FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
        _minimal_record(
            "a-2", repo="a/r", base_commit=base,
            problem_statement="x",
            FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    snapshots: list[int] = []
    preds_target = tmp_path / "preds.jsonl"
    counter = {"n": 0}

    def _on_implement():
        counter["n"] += 1
        (repo / "seed.py").write_text(f"def f(): return {counter['n']}\n")
        # Snapshot the predictions file state from inside the
        # implementer call. The serial loop appends AFTER the
        # per-instance run_architect returns, so the first snapshot
        # sees 0 lines (truncated-at-start file exists empty); the
        # second sees 1 line (a-1's prediction was appended after
        # its run_architect returned).
        if preds_target.exists():
            snapshots.append(len([
                l for l in preds_target.read_text().splitlines()
                if l.strip()
            ]))
        else:
            snapshots.append(0)

    backend = _MultiRoleScriptedBackend(on_implement=_on_implement)

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
        "--predictions-out", str(preds_target),
    ])
    assert rc == 0
    assert snapshots == [0, 1]
    # Final file has both predictions.
    final_lines = [
        l for l in preds_target.read_text().splitlines() if l.strip()
    ]
    assert len(final_lines) == 2


# SWE-bench specificity enters through the architect cascade, product prompt,
# and dedicated planner/Breaker rubrics; tests exercise the current composed
# path rather than a direct single-call implementer path.


def test_pool_worker_runs_one_instance_in_process(tmp_path: Path, monkeypatch):
    """The pool worker is a top-level callable; invoke it directly
    (in-process) with mocked repo_workspace + multi-role backend to
    assert it returns a well-formed InstanceOutcome. This exercises
    the same code path workers run under ThreadPoolExecutor without
    paying for the fork."""
    import contextlib as _contextlib
    from crucible.workflows.swebench import (
        InstanceOutcome,
        _architect_operation_id,
        _pool_worker,
        _PoolPayload,
    )

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 99\n")
        ),
        cost_per_call=0.02,
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)
    _patch_smoke_check(monkeypatch)

    inst = SwebenchInstance(
        instance_id="fake-1", repo="fake/repo",
        base_commit=base, patch="", test_patch="",
        problem_statement="Make f return 99", fail_to_pass=(),
        pass_to_pass=(),
    )
    parent_run = _prepare_swebench_parent(tmp_path / "runs" / "swebench-parent")
    payload = _PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(corpus.spec_dir),
        backend_config_path=str(_default_backend_config(tmp_path)), max_turns=10,
        budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "worker_log"),
        parent_run_dir=str(parent_run),
        include_breach=False,
        include_review=False,
        enable_build_verify=False,
        evaluate_inline=False,
        routed_model="aws/anthropic/bedrock-claude-opus-4-8",
    )
    outcome = _pool_worker(payload)
    assert isinstance(outcome, InstanceOutcome)
    assert outcome.instance_id == "fake-1"
    assert outcome.classification == "success"
    assert "return 99" in outcome.model_patch
    assert outcome.diff_bytes > 0
    operation_id = _architect_operation_id(inst.instance_id)
    child = _outcome.child_outcome(parent_run, operation_id)
    child_run = _outcome.child_run_dir(parent_run, operation_id)
    assert child.workflow == "architect"
    assert child_run.parent.parent == (
        corpus.spec_dir / "instances" / inst.instance_id / "corpus"
    )
    # All three architect roles fired (planner / critic may fire
    # more than once across plan's internal critique loop).
    assert backend.counters["planner"] >= 1
    assert backend.counters["critic"] >= 1
    assert backend.counters["implementer"] == 1


def test_pool_worker_workspace_error_returns_outcome(
    tmp_path: Path, monkeypatch,
):
    """A clone failure inside the worker doesn't propagate; the worker
    returns an InstanceOutcome with hard_fail_workspace_error so the
    parent can fold it into the per-instance summary."""
    import contextlib as _contextlib
    from crucible.products.swebench.repo_workspace import RepoWorkspaceError
    from crucible.workflows.swebench import (
        _PoolPayload, _pool_worker,
    )

    corpus = _setup_corpus(tmp_path)

    @_contextlib.contextmanager
    def failing_workspace(instance, *, cache_dir=None, timeout=600):
        raise RepoWorkspaceError("clone exploded")
        yield

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", failing_workspace,
    )
    # No backend call expected — the workspace error fires before any
    # architect-cascade invocation.
    _patch_backend_resolver(monkeypatch, _MultiRoleScriptedBackend())

    inst = SwebenchInstance(
        instance_id="bad-1", repo="bad/repo",
        base_commit="a" * 40, patch="", test_patch="",
        problem_statement="x", fail_to_pass=(), pass_to_pass=(),
    )
    payload = _PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(corpus.spec_dir),
        backend_config_path=None, max_turns=1, budget_usd=10.0,
        repo_cache_dir=None,
        log_dir=str(tmp_path / "worker_log"),
        parent_run_dir=str(tmp_path / "unused-parent"),
        include_breach=False,
        include_review=False,
    )
    outcome = _pool_worker(payload)
    assert outcome.classification == "hard_fail_workspace_error"
    assert outcome.model_patch == ""


def test_implement_max_concurrent_1_uses_serial_path(
    tmp_path: Path, monkeypatch,
):
    """A one-worker run uses the same leased worker protocol as fan-out."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)

    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "fake-1", repo="fake/repo", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows import swebench as swebench_workflow

    worker_calls: list[str] = []
    real_pool_worker = swebench_workflow._pool_worker

    def tracked_pool_worker(payload):
        worker_calls.append(payload.instance.instance_id)
        return real_pool_worker(payload)

    monkeypatch.setattr(
        "crucible.workflows.swebench._pool_worker", tracked_pool_worker,
    )

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--max-concurrent-instances", "1",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    assert worker_calls == ["fake-1"]
    # Serial path fired the implementer exactly once.
    assert backend.counters["implementer"] == 1


def test_implement_resumability_skips_already_implemented(
    tmp_path: Path, monkeypatch,
):
    """A second --implement run skips instances that already have an
    Implementation record in the corpus. Their prior model_patch is
    surfaced into the new predictions.jsonl so downstream eval sees
    the full set."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
        _minimal_record(
            "a-2", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    _seed_terminal_prediction(
        corpus,
        load_dataset(dataset_path)[0],
        classification="success",
        patch=(
            "diff --git a/x b/x\n--- a/x\n+++ b/x\n"
            "@@ -1 +1 @@\n-old\n+new from prior run\n"
        ),
    )

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
        cost_per_call=0.01,
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    # Only a-2 should have triggered an implementer call; a-1 was
    # skipped via the resumability gate.
    assert backend.counters["implementer"] == 1
    assert backend.counters["planner"] >= 1
    assert backend.counters["critic"] >= 1

    # Predictions file contains BOTH a-1 (resurfaced from prior diff)
    # and a-2 (freshly implemented).
    runs = list((corpus.spec_dir / "runs").iterdir())
    # _setup_corpus leaves spec_dir/runs in an unknown initial state;
    # find the most recent run dir created by THIS workflow invocation.
    preds_path = max(runs).joinpath("predictions.jsonl")
    parsed = [
        json.loads(line) for line in preds_path.read_text().strip().splitlines()
    ]
    by_id = {p["instance_id"]: p for p in parsed}
    assert "a-1" in by_id
    assert "a-2" in by_id
    assert "new from prior run" in by_id["a-1"]["model_patch"]


# _PoolPayload carries execution inputs rather than a workflow Spec.
# Per-instance Specs are synthesized inside the worker via
# crucible.products.swebench.spec_synthesis.synthesize_starter_spec
# and persisted to a per-instance Corpus at <root>/instances/<id>/
# corpus/. The pickle test therefore verifies only the current payload.


def test_pool_payload_is_pickleable(tmp_path: Path):
    """``_PoolPayload`` survives pickle round-trip (matters for
    ProcessPool fan-out; ThreadPool is the default but the payload
    must remain shippable for future process-mode use)."""
    import pickle
    from crucible.products.swebench.dataset import SwebenchInstance
    from crucible.workflows.swebench import _PoolPayload

    inst = SwebenchInstance(
        instance_id="a-1", repo="a/r",
        base_commit="0" * 40, patch="", test_patch="",
        problem_statement="x", fail_to_pass=(), pass_to_pass=(),
    )
    payload = _PoolPayload(
        instance=inst,
        workspace_root_spec_dir=str(tmp_path / "c"),
        backend_config_path=None, max_turns=1, budget_usd=10.0,
        repo_cache_dir=None, log_dir=str(tmp_path / "l"),
        parent_run_dir=str(tmp_path / "unused-parent"),
        include_breach=True, include_review=True,
    )
    rt = pickle.loads(pickle.dumps(payload))
    assert rt.instance.instance_id == "a-1"
    assert rt.workspace_root_spec_dir == str(tmp_path / "c")
    assert rt.include_breach is True
    assert rt.include_review is True


def test_architect_cost_uses_saga_identity_not_unrelated_runs(tmp_path: Path):
    """Valid and partial sibling runs cannot steal child cost attribution."""

    from assay.corpus.runs import RunResult
    from assay.corpus.store import Corpus
    from crucible.workflows import swebench as wf
    from crucible.workflows._run import open_run_with_capture

    parent = _prepare_swebench_parent(tmp_path / "parent" / "swebench-run")
    child_corpus = Corpus(tmp_path / "instance-corpus")
    operation_id = wf._architect_operation_id("django__django-1")
    with _outcome.child_operation(
        parent, operation_id=operation_id, workflow="architect",
    ):
        with open_run_with_capture(child_corpus, "architect") as handle:
            handle.write_summary({
                "status": "complete",
                "authoritative_spent_usd": 1.25,
                "transport_failure": True,
                "costs": {
                    "total_usd": 1.25,
                    "total_input_tokens": 1234,
                    "total_output_tokens": 56,
                    "planning_backend_calls": 7,
                    "rollup_incomplete": False,
                },
            })
            handle.finalize(RunResult(status="complete", cost_usd=1.25))

    # A later fully-terminal sibling and a partially-created directory are
    # both plausible under timestamp scanning, but neither is saga-bound.
    with open_run_with_capture(child_corpus, "architect") as unrelated:
        unrelated.write_summary({
            "status": "complete",
            "authoritative_spent_usd": 999.0,
            "costs": {"total_usd": 999.0, "rollup_incomplete": False},
        })
        unrelated.finalize(RunResult(status="complete", cost_usd=999.0))
    partial = child_corpus.spec_dir / "runs" / "zzzz_architect_partial"
    partial.mkdir()
    partial.joinpath("summary.json").write_text(json.dumps({
        "authoritative_spent_usd": 777.0,
        "costs": {"total_usd": 777.0, "rollup_incomplete": False},
    }))

    cost, incomplete = wf._architect_cost_for_operation(parent, operation_id)

    assert cost == pytest.approx(1.25)
    assert incomplete is False
    assert wf._architect_usage_for_operation(parent, operation_id) == (
        1234,
        56,
        7,
        True,
    )


def test_concurrent_architect_children_keep_distinct_saga_identity(tmp_path: Path):
    """Threaded instance children cannot cross-attribute terminal spend."""

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from assay.corpus.runs import RunResult
    from assay.corpus.store import Corpus
    from crucible.workflows import swebench as wf
    from crucible.workflows._run import open_run_with_capture

    parent = _prepare_swebench_parent(tmp_path / "parent" / "swebench-run")
    barrier = threading.Barrier(2)

    def publish(instance_id: str, cost: float) -> tuple[str, float]:
        operation_id = wf._architect_operation_id(instance_id)
        child_corpus = Corpus(tmp_path / "instances" / instance_id / "corpus")
        with _outcome.child_operation(
            parent, operation_id=operation_id, workflow="architect",
        ):
            barrier.wait(timeout=5)
            with open_run_with_capture(child_corpus, "architect") as handle:
                handle.write_summary({
                    "status": "complete",
                    "authoritative_spent_usd": cost,
                    "costs": {
                        "total_usd": cost,
                        "rollup_incomplete": False,
                    },
                })
                handle.finalize(RunResult(status="complete", cost_usd=cost))
        return operation_id, cost

    with ThreadPoolExecutor(max_workers=2) as executor:
        published = tuple(executor.map(
            lambda item: publish(*item),
            (("instance-a", 1.0), ("instance-b", 2.0)),
        ))

    for operation_id, expected_cost in published:
        cost, incomplete = wf._architect_cost_for_operation(
            parent, operation_id,
        )
        assert cost == pytest.approx(expected_cost)
        assert incomplete is False


def test_resumability_retries_hard_failed_instances_by_default(
    tmp_path: Path, monkeypatch,
):
    """Prior Implementation records classified ``hard_fail_*`` are
    retried on the next --implement run. Only ``success`` and
    ``soft_fail_no_changes`` records are skipped — the operator
    paid for the hard-fail attempt but got no real result, so
    re-attempt is the right default."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    _seed_terminal_prediction(
        corpus, load_dataset(dataset_path)[0],
        classification="hard_fail_timeout",
        model="aws/anthropic/bedrock-claude-opus-4-8",
    )

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 99\n")
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    # Despite a prior hard_fail_timeout record, a-1 is retried — the
    # implementer fires exactly once.
    assert backend.counters["implementer"] == 1


def test_resumability_skips_soft_fail_no_changes(
    tmp_path: Path, monkeypatch,
):
    """``soft_fail_no_changes`` is a legitimate terminal state under
    verify_mode=skip — the agent declined the instance with no fix.
    Re-attempting won't change the outcome, so skip it (matches the
    ``success`` policy)."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    _seed_terminal_prediction(
        corpus, load_dataset(dataset_path)[0],
        classification="soft_fail_no_changes",
        model="aws/anthropic/bedrock-claude-opus-4-8",
    )

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend()
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    # soft_fail_no_changes is skip-worthy; no architect-cascade call.
    assert sum(backend.counters.values()) == 0


def test_resumed_predictions_preserve_prior_model_attribution(
    tmp_path: Path, monkeypatch,
):
    """A skipped instance's prior model_patch surfaces with the
    model that ORIGINALLY produced it, not the current run's routed
    model. Cross-model comparisons depend on this field; the
    prior bug attributed yesterday's patches to today's model
    silently."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    _seed_terminal_prediction(
        corpus, load_dataset(dataset_path)[0],
        classification="success",
        patch=(
            "diff --git a/y b/y\n--- a/y\n+++ b/y\n"
            "@@ -1 +1 @@\n-q\n+r\n"
        ),
        model="claude-opus-4-6-OLD",
    )

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, _MultiRoleScriptedBackend())

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0

    runs = sorted((corpus.spec_dir / "runs").iterdir())
    preds_path = runs[-1] / "predictions.jsonl"
    parsed = json.loads(preds_path.read_text().strip())
    # The skipped a-1 surfaces its ORIGINAL model attribution.
    assert parsed["instance_id"] == "a-1"
    assert parsed["model_name_or_path"] == "claude-opus-4-6-OLD"


def test_unreached_instances_emit_zero_patch_with_sentinel_model(
    tmp_path: Path, monkeypatch,
):
    """When --budget-usd halts the run mid-loop, instances that
    were never reached emit zero-patch entries with a model name
    suffixed ``/unreached`` so postmortem can attribute the drop."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
        _minimal_record(
            "a-2", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
        _minimal_record(
            "a-3", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    # Greenfield instances always run independently routed
    # Planner→Critic→Judge→Implementer. At $0.125/call each completed
    # instance costs $0.50. The $1 total therefore admits exactly two
    # instances and marks the third unreached.
    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
        cost_per_call=0.125,
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--budget-usd", "1.0", "--planner-count", "1",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    # exit 1 because total budget exhausted.
    assert rc == 1

    runs = sorted((corpus.spec_dir / "runs").iterdir())
    preds_path = runs[-1] / "predictions.jsonl"
    parsed = [
        json.loads(line) for line in preds_path.read_text().strip().splitlines()
    ]
    by_id = {p["instance_id"]: p for p in parsed}
    # All three instances appear in the predictions file.
    assert set(by_id.keys()) == {"a-1", "a-2", "a-3"}
    # At least one is unreached (sentinel model suffix).
    unreached = [
        p for p in parsed
        if p["model_name_or_path"].endswith("/unreached")
    ]
    assert len(unreached) >= 1
    # Unreached entries carry empty patches.
    assert all(p["model_patch"] == "" for p in unreached)


def test_implement_rerun_flag_redoes_existing(tmp_path: Path, monkeypatch):
    """--rerun bypasses the resumability skip, re-implementing
    instances that already have a recorded Implementation and child run."""
    import contextlib as _contextlib

    from assay.corpus.store import Corpus

    from crucible.workflows._run import open_run_with_capture
    from crucible.workflows.swebench import _architect_operation_id

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    prior_patch = (
        "diff --git a/x b/x\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/x\n"
        "@@ -0,0 +1 @@\n"
        "+old\n"
    )
    _seed_terminal_prediction(
        corpus,
        load_dataset(dataset_path)[0],
        classification="success",
        patch=prior_patch,
    )

    # Reproduce the real rerun shape: the per-instance Corpus already has a
    # terminal Architect child bound to a previous SWE-bench parent.  Keep the
    # Corpus path relative so reconciliation must canonicalize it exactly as
    # ``begin_run`` did when the old child was first published.
    monkeypatch.chdir(tmp_path)
    relative_spec = corpus.spec_dir.relative_to(tmp_path)
    relative_root_corpus = Corpus(relative_spec)
    relative_instance_corpus = Corpus(
        relative_spec / "instances" / "a-1" / "corpus",
    )
    with open_run_with_capture(
        relative_root_corpus, "swebench",
    ) as prior_parent:
        with _outcome.child_operation(
            prior_parent.run_dir,
            operation_id=_architect_operation_id("a-1"),
            workflow="architect",
        ):
            with open_run_with_capture(
                relative_instance_corpus, "architect",
            ) as prior_child:
                prior_child.write_summary({"status": "complete"})
    prior_child_dir = prior_child.run_dir.resolve()
    prior_manifest_stat = (prior_child_dir / "run_manifest.json").stat()

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
        cost_per_call=0.01,
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(relative_spec),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10", "--rerun",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    # With --rerun, a-1 IS re-implemented even though it had a prior
    # record — the implementer fires exactly once.
    assert backend.counters["implementer"] == 1
    # Reconciliation neither reuses nor replaces the old terminal child.
    current_manifest_stat = (prior_child_dir / "run_manifest.json").stat()
    assert current_manifest_stat.st_ino == prior_manifest_stat.st_ino
    assert current_manifest_stat.st_mtime_ns == prior_manifest_stat.st_mtime_ns


def test_implement_synthesizes_default_spec_when_absent(
    tmp_path: Path, monkeypatch,
):
    """If the operator hasn't created a spec.yaml under
    <spec>/versions/, the workflow synthesizes a minimal default
    rather than refusing to run."""
    import contextlib as _contextlib
    from assay.corpus.store import Corpus

    # Fresh corpus dir with NO spec yet.
    corpus_dir = tmp_path / "corpus" / "no-spec"
    corpus_dir.mkdir(parents=True)
    Corpus(corpus_dir)  # init; doesn't auto-seed a spec

    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 2\n")
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    # A default spec was persisted as part of the workflow run.
    assert Corpus(corpus_dir).latest_spec() is not None


def test_implement_handles_workspace_error_gracefully(
    tmp_path: Path, monkeypatch,
):
    """A clone/checkout failure for one instance lands as
    hard_fail_workspace_error with an empty patch; the workflow
    continues to subsequent instances (here: just the one)."""
    from crucible.products.swebench.repo_workspace import RepoWorkspaceError

    corpus = _setup_corpus(tmp_path)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record("bad__repo-1", repo="bad/repo")
    ]))

    import contextlib as _contextlib

    @_contextlib.contextmanager
    def failing_workspace(instance, *, cache_dir=None, timeout=600):
        raise RepoWorkspaceError("simulated clone failure")
        yield  # unreachable; satisfies the contextmanager contract

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", failing_workspace,
    )
    # Backend never called; workspace error short-circuits.
    _patch_backend_resolver(monkeypatch, _MultiRoleScriptedBackend())

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    # Workspace errors don't halt the run; the per-instance outcome
    # records the failure and the workflow exits 0 (or 1 only on
    # total-budget exhaustion).
    assert rc == 0
    runs = list((corpus.spec_dir / "runs").iterdir())
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert summary["processed_instances"] == 1
    assert summary["per_instance"][0]["classification"] == "hard_fail_workspace_error"
    assert summary["nonempty_patch_count"] == 0


# ----------------------------------------------------------------------
# Cascade-kwarg wiring (integration)
#
# These tests verify the wiring of cascade kwargs from swebench →
# run_architect → run_plan / run_tier → breach. Each asserts on a
# specific load-bearing data-flow assertion that a backend-level mock
# alone wouldn't cover: the toolset attachment, the {spec_intent}
# placeholder threading, the breaker-rubric pool-resolution.
# ----------------------------------------------------------------------


def test_planner_backend_receives_source_toolset_with_enable_source_tools(
    tmp_path: Path, monkeypatch,
):
    """SWE-bench dispatches ``run_architect(enable_source_tools=True)``,
    which must reach the planner backend's ``attach_toolset(...)``. Native
    host tools stay disabled; repository reads go through the scoped MCP
    source profile."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="x", FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    captured_toolsets: list = []
    captured_planner_tools: list = []

    class _ToolCapturingBackend(_MultiRoleScriptedBackend):
        def attach_toolset(self, toolset, *, context_budget=None):
            captured_toolsets.append(toolset)

        def invoke(self, invocation):
            # Record planner invocations' tools surface; critic /
            # implementer invocations are skipped.
            system = (invocation.system or "").lower()
            if "you are a planner" in system or "planner for the" in system:
                captured_planner_tools.append(invocation.tools)
            return super().invoke(invocation)

    backend = _ToolCapturingBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0

    # The planner backend received a non-None toolset.
    assert captured_toolsets, "attach_toolset was never called"
    assert any(t is not None for t in captured_toolsets), (
        "attach_toolset was always called with None; "
        "expected build_source_tools(ctx) for the planner"
    )
    # Native host tools remain disabled. The attached, workspace-rooted
    # MCP source profile is the planner's only repository surface.
    assert captured_planner_tools, "planner role was never invoked"
    for tools in captured_planner_tools:
        assert tools in (None, ()), (
            "planner unexpectedly received native host tools: "
            f"{tools!r}"
        )


def test_swebench_plan_critic_receives_spec_intent_placeholder(
    tmp_path: Path, monkeypatch,
):
    """``swebench_plan_critic.yaml`` references ``{spec_intent}``;
    swebench dispatches the issue's ``problem_statement`` through
    ``run_plan -> assay.plan -> review(context={"spec_intent": ...})``
    so the rubric template can format successfully. Asserts the
    critic's user_message contains the verbatim issue text."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    issue_text = "marker-issue-72c91a — fix the bogus default in foo()"
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement=issue_text,
            FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    captured_critic_messages: list[str] = []

    class _CriticCapturingBackend(_MultiRoleScriptedBackend):
        def invoke(self, invocation):
            system = (invocation.system or "").lower()
            if (
                "you are the critic" in system
                or "critic for plan" in system
            ):
                captured_critic_messages.append(invocation.user or "")
            return super().invoke(invocation)

    backend = _CriticCapturingBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
    )
    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    _patch_backend_resolver(monkeypatch, backend)

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-include-breach", "--no-include-review",
        "--no-build-verify", "--no-evaluate-inline",
    ])
    assert rc == 0
    assert captured_critic_messages, "critic was never invoked"
    # The plan_critic's user_message embeds the issue text via the
    # {spec_intent} placeholder. Without context= threading, the
    # rubric.user_template.format would raise KeyError at review()
    # time and the workflow would never complete.
    assert any(issue_text in msg for msg in captured_critic_messages), (
        f"critic user_message never contained the issue text "
        f"{issue_text!r} (got: {captured_critic_messages!r})"
    )


def test_breach_pool_resolves_swebench_impl_breaker_role(
    tmp_path: Path, monkeypatch,
):
    """SWE-bench dispatches ``breaker_rubric="swebench_impl_breaker"``
    through tier -> breach_workflow. workflows/breach.py resolves the
    Breaker backend via ``pool.get(args.breaker_rubric)`` so the
    rubric name doubles as the routing role — operator routing
    overrides in ``products/swebench/backends.yaml`` actually take
    effect. Asserts ``resolve_for_role`` was queried for the
    swebench-specific role name."""
    import contextlib as _contextlib

    corpus = _setup_corpus(tmp_path)
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    base = _init_git_repo(repo)
    dataset_path = tmp_path / "ds.json"
    dataset_path.write_text(json.dumps([
        _minimal_record(
            "a-1", repo="a/r", base_commit=base,
            problem_statement="fix the bug",
            FAIL_TO_PASS=[], PASS_TO_PASS=[],
        ),
    ]))

    @_contextlib.contextmanager
    def fake_workspace(instance, *, cache_dir=None, timeout=600):
        yield repo

    backend = _MultiRoleScriptedBackend(
        on_implement=lambda: (
            (repo / "seed.py").write_text("def f(): return 1\n")
        ),
    )
    requested_roles: list[str] = []

    def _capturing_resolver(
        role, *args, routing=None, log_dir=None, cwd=None,
        config_path=None, spec_dir=None, **kw,
    ):
        import copy
        requested_roles.append(role)
        return copy.copy(backend)

    monkeypatch.setattr(
        "crucible.workflows.swebench.repo_workspace", fake_workspace,
    )
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        _capturing_resolver,
    )

    from crucible.workflows.swebench import main
    rc = main([
        "--spec", str(corpus.spec_dir),
        "--dataset", str(dataset_path),
        "--implement", "--budget-usd-per-instance", "10",
        "--no-build-verify", "--no-evaluate-inline",
        # Breach + review ON so the breaker role is actually
        # resolved during this run.
    ])
    # rc may be 0 (everything fine) or 1 (some downstream phase
    # non-success); the contract under test is the pool-role lookup,
    # not the verdict shape.
    assert rc in (0, 1)
    # The swebench-specialized breaker rubric was used as the pool
    # role key; without M3's rubric-as-role plumbing this lookup
    # would never fire.
    assert "swebench_impl_breaker" in requested_roles, (
        f"breach phase did not resolve swebench_impl_breaker role; "
        f"requested roles: {requested_roles!r}"
    )


# ----------------------------------------------------------------------
# Safety: _capture_workspace_diff_report guards against wrong-cwd resolution
# ----------------------------------------------------------------------


def test_capture_workspace_diff_report_refuses_when_git_walks_up_to_parent_repo(
    tmp_path: Path,
):
    """When the per-instance worktree's ``.git`` file is missing /
    corrupted, ``git -C <workspace>`` silently walks up to find a
    parent repo's ``.git``. For SWE-bench runs that parent is the
    outer crucible-host repo — capturing its diff would emit
    operator-uncommitted state + submodule pointers as the
    leaderboard prediction. The safety gate verifies
    ``rev-parse --show-toplevel`` resolves to the workspace path
    itself; mismatch → empty bytes + ERROR log.
    """
    import subprocess
    from crucible.workflows.swebench import _capture_workspace_diff_report
    from crucible.products.swebench.dataset import SwebenchInstance

    # Outer "parent" repo with uncommitted state — what git would
    # capture if it walked up.
    parent = tmp_path / "parent_repo"
    parent.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=parent, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=parent, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=parent, check=True)
    (parent / "outer.txt").write_text("seed\n")
    subprocess.run(["git", "add", "outer.txt"], cwd=parent, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=parent, check=True)
    # Operator-uncommitted state — what the bug would have leaked.
    (parent / "outer.txt").write_text("modified by operator\n")
    (parent / "submodule_ptr").write_text("0123abcd\n")
    subprocess.run(["git", "add", "submodule_ptr"], cwd=parent, check=True)

    # Workspace as a SUBDIRECTORY of parent, with NO valid .git file —
    # simulates a corrupted worktree where git -C <subdir> walks up
    # to find parent's .git.
    workspace = parent / "corrupted_worktree"
    workspace.mkdir()
    (workspace / "astropy_file.py").write_text("def f(): pass\n")
    # Note: no .git file or directory in workspace/ — git will walk up.

    inst = SwebenchInstance(
        instance_id="astropy__astropy-13453", repo="astropy/astropy",
        base_commit="a" * 40, patch="", test_patch="",
        problem_statement="x", fail_to_pass=(), pass_to_pass=(),
    )
    diff_bytes = _capture_workspace_diff_report(workspace, inst).source_patch
    assert diff_bytes == b"", (
        "safety gate must refuse capture when git walks up to a "
        "parent repo; got diff of "
        f"{len(diff_bytes)} bytes (would have leaked parent state)"
    )
    # The escape would have shown outer.txt and submodule_ptr in the
    # diff — sanity check that we DON'T see those.
    assert b"outer.txt" not in diff_bytes
    assert b"submodule_ptr" not in diff_bytes


def test_capture_workspace_diff_report_succeeds_when_workspace_is_canonical_root(
    tmp_path: Path,
):
    """Positive case: a workspace whose ``.git`` resolves to itself
    captures its own diff cleanly."""
    import subprocess
    from crucible.workflows.swebench import _capture_workspace_diff_report
    from crucible.products.swebench.dataset import SwebenchInstance

    workspace = tmp_path / "clean_workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=workspace, check=True,
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=workspace, check=True)
    (workspace / "seed.py").write_text("def f(): return 0\n")
    subprocess.run(["git", "add", "seed.py"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=workspace, check=True)
    # Uncommitted change — should appear in the diff.
    (workspace / "seed.py").write_text("def f(): return 42\n")

    inst = SwebenchInstance(
        instance_id="astropy__astropy-13453", repo="astropy/astropy",
        base_commit="a" * 40, patch="", test_patch="",
        problem_statement="x", fail_to_pass=(), pass_to_pass=(),
    )
    diff_bytes = _capture_workspace_diff_report(workspace, inst).source_patch
    assert diff_bytes != b"", "valid workspace must produce a non-empty diff"
    assert b"return 42" in diff_bytes
    assert b"seed.py" in diff_bytes


def test_is_install_artifact_classification():
    """Unit-level coverage for the install-artifact path classifier.

    Build / dist / cache prefixes match; ``*.egg-info`` matches at any
    depth. Real source paths (``requests/models.py``, ``tests/...``,
    ``src/foo.py``) do not.
    """
    from crucible.workflows.swebench import _is_install_artifact

    artifact_paths = [
        "build/lib/requests/__init__.py",
        "build/lib/requests/models.py",
        "dist/foo-1.0.tar.gz",
        ".tox/py39/lib/python3.9/site-packages/foo.py",
        ".pytest_cache/v/cache/lastfailed",
        ".mypy_cache/3.11/foo.json",
        ".ruff_cache/CACHEDIR.TAG",
        "node_modules/lodash/index.js",
        "requests.egg-info/PKG-INFO",
        "src/pkg.egg-info/SOURCES.txt",
    ]
    for p in artifact_paths:
        assert _is_install_artifact(p), f"{p!r} should classify as install artifact"

    real_paths = [
        "requests/models.py",
        "tests/test_requests.py",
        "src/foo/bar.py",
        "docs/build/html/index.html",  # docs build NOT a python build
        "setup.py",
        "README.md",
        "buildtools.py",  # doesn't start with "build/"
        "distutils_compat.py",  # doesn't start with "dist/"
    ]
    for p in real_paths:
        assert not _is_install_artifact(p), f"{p!r} should NOT classify as install artifact"


def test_capture_workspace_diff_report_excludes_install_artifacts(tmp_path: Path):
    """End-to-end: pip install -e .-style ``build/lib/`` leftovers
    must be filtered out of the captured diff, while real source-tree
    edits (and untracked source files) survive.

    Reproduces the psf__requests-1142 failure mode where an 8-line
    edit in ``requests/models.py`` shipped as an 872 KB prediction
    because pip's editable install left build/lib/requests/* behind.
    """
    import subprocess as _sp
    from crucible.workflows.swebench import _capture_workspace_diff_report
    from crucible.products.swebench.dataset import SwebenchInstance

    workspace = tmp_path / "with_install_artifacts"
    workspace.mkdir()
    _sp.run(["git", "init", "-q"], cwd=workspace, check=True)
    _sp.run(["git", "config", "user.email", "t@t"], cwd=workspace, check=True)
    _sp.run(["git", "config", "user.name", "t"], cwd=workspace, check=True)
    (workspace / "requests").mkdir()
    (workspace / "requests" / "models.py").write_text("def f(): return 0\n")
    _sp.run(["git", "add", "."], cwd=workspace, check=True)
    _sp.run(["git", "commit", "-q", "-m", "init"], cwd=workspace, check=True)

    # Real edit to a tracked file — must survive.
    (workspace / "requests" / "models.py").write_text("def f(): return 42\n")

    # Untracked install-artifacts — must be filtered.
    (workspace / "build" / "lib" / "requests").mkdir(parents=True)
    (workspace / "build" / "lib" / "requests" / "__init__.py").write_text(
        "# stale build artifact\n"
    )
    (workspace / "build" / "lib" / "requests" / "models.py").write_text(
        "def stale(): pass\n"
    )
    (workspace / "dist").mkdir()
    (workspace / "dist" / "requests-1.1.tar.gz").write_bytes(b"\x00binary\x00")
    (workspace / "requests.egg-info").mkdir()
    (workspace / "requests.egg-info" / "PKG-INFO").write_text("Name: requests\n")

    # Untracked real source addition — must survive (it's a model edit).
    (workspace / "requests" / "new_helper.py").write_text("def helper(): return 1\n")

    inst = SwebenchInstance(
        instance_id="psf__requests-1142", repo="psf/requests",
        base_commit="0" * 40, patch="", test_patch="",
        problem_statement="x", fail_to_pass=(), pass_to_pass=(),
    )
    diff_bytes = _capture_workspace_diff_report(workspace, inst).source_patch
    diff_text = diff_bytes.decode("utf-8", errors="replace")

    # Real edits survive.
    assert "requests/models.py" in diff_text, (
        "tracked-file model edit must appear in diff"
    )
    assert "return 42" in diff_text
    assert "requests/new_helper.py" in diff_text, (
        "untracked real-source file must appear in diff"
    )

    # Install artifacts are filtered.
    assert "build/lib/requests" not in diff_text, (
        "build/lib/requests/* must NOT appear in diff (install artifact leak)"
    )
    assert "dist/requests-1.1.tar.gz" not in diff_text, (
        "dist/* must NOT appear in diff"
    )
    assert "requests.egg-info" not in diff_text, (
        "*.egg-info/* must NOT appear in diff"
    )


# ----------------------------------------------------------------------
# Cost-accounting regression — verdict.cost_usd MUST be summed
# alongside Implementation.cost_usd in _spec_to_outcome.
# ----------------------------------------------------------------------
#
# The pre-fix code only summed Implementation.cost_usd, missing the
# planner critic/judge, breach Breaker/Judge, and Phase-4 review
# verdict costs. With Codex routing (where Implementation.cost_usd is
# already 0 because the SDK didn't surface tokens), this manifested as
# outcome.cost_usd=$0 across a 13-minute run, leaving the
# --budget-usd cap effectively infinite. Fix lives at
# crucible/workflows/swebench.py:_spec_to_outcome.


def test_spec_to_outcome_sums_both_implementation_and_verdict_costs(
    tmp_path: Path,
):
    from assay import ExecutionContext, Implementation
    from assay.types.findings import ReviewCoverage, Verdict
    from crucible.workflows.swebench import _spec_to_outcome

    corpus = _setup_corpus(tmp_path)
    spec = corpus.latest_spec()
    base = "0" * 40

    repo = tmp_path / "fake_repo"
    repo.mkdir()
    _init_git_repo(repo)

    # One implementer record at $0.40.
    impl = Implementation(
        slice_id="swebench/x-1",
        base_commit=base, result_commit=None,
        diff_ref=corpus.put_artifact(b"", kind="diffs", name_hint="x.diff"),
        transcript_ref=corpus.put_artifact(
            b"[]", kind="transcripts", name_hint="x.json",
        ),
        files_modified=("y.py",),
        scope_escapes=(),
        classification="success",
        execution=ExecutionContext(
            timestamp="2026-05-23T00:00:00+00:00",
            machine="t", cwd="/", toolchain={},
            backend="codex", model="gpt-5.5",
            recipe="swebench", recipe_commit=None,
            gpu_model=None, run_id=None,
        ),
        cost_usd=0.40,
    )
    impl_ref = corpus.put_record(impl, kind="implementations")
    spec = spec.with_implementation(impl)
    corpus.put_spec(spec)

    # Three verdicts (critic + judge + review) totalling $1.10.
    for rubric, kind, cost in (
        ("swebench_plan_critic", "critic", 0.30),
        ("plan_judge", "judge", 0.45),
        ("diff_scope", "review", 0.35),
    ):
        verdict = Verdict(
            rubric=rubric,
            decision="pass",
            findings=(),
            artifact_ref=corpus.put_artifact(
                b"x", kind="reviewed_artifacts", name_hint=rubric,
            ),
            artifact_commit=None,
            reasoning="",
            coverage=ReviewCoverage(
                complete=True,
                sources_reviewed=("artifact",),
                omissions=(),
            ),
            backend_name="codex",
            transcript_ref=corpus.put_artifact(
                b"[]", kind="transcripts", name_hint=f"{rubric}.json",
            ),
            cost_usd=cost,
            execution=None,
            kind=kind,
        )
        corpus.put_record(verdict, kind="verdicts")
        spec = spec.with_verdict(verdict)
        corpus.put_spec(spec)

    inst = SwebenchInstance(
        instance_id="x-1", repo="x/x", base_commit=base,
        patch="", test_patch="",
        problem_statement="", fail_to_pass=(), pass_to_pass=(),
    )
    outcome = _spec_to_outcome(corpus, inst, architect_rc=0, workspace=repo)

    # 0.40 (impl) + 0.30 + 0.45 + 0.35 (verdicts) = 1.50.
    # Pre-fix this returned 0.40 — only implementer summed, verdict
    # spend was silently dropped.
    assert outcome.cost_usd == pytest.approx(1.50)


def test_spec_to_outcome_zero_when_no_records(tmp_path: Path):
    """Sanity: outcome.cost_usd is 0 for a spec with no implementations
    or verdicts. Confirms the verdict-sum loop tolerates the empty
    spec.verdict_refs case."""
    from crucible.workflows.swebench import _spec_to_outcome

    corpus = _setup_corpus(tmp_path)
    inst = SwebenchInstance(
        instance_id="empty-1", repo="x/x", base_commit="0" * 40,
        patch="", test_patch="",
        problem_statement="", fail_to_pass=(), pass_to_pass=(),
    )
    # architect_rc=0 + no impls → soft_fail_no_changes, cost 0.
    outcome = _spec_to_outcome(corpus, inst, architect_rc=0, workspace=None)
    assert outcome.cost_usd == 0.0
    assert outcome.classification == "soft_fail_no_changes"
