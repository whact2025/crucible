# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tier Phase 1 slices that end without a verdict.

A slice whose smoke gate could not run, whose working-tree delta could not be
captured, or whose implementer was cancelled was not judged. The tier still
restores the slice transaction and halts, but it reports ``slice_inconclusive``
or ``cancelled`` instead of the permanent ``slice_failed``. A judged failure,
an unproven restore, a write of the model outside the slice scope, and a gate
that changed the candidate stay ``slice_failed``.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from assay import (
    BackendExecutionEvent,
    BackendExecutionTrace,
    ExecutionCompleteness,
    Implementation,
    InvocationResult,
    ProjectContext,
    Spec,
    Usage,
)
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG
from assay.types.invocation import ExternalCancellation
from assay.types.smoke import SmokeResult

from crucible.common.policies import BreachPolicy


_COMPLETE_EXECUTION = ExecutionCompleteness(
    token_usage_complete=True,
    cost_complete=True,
    terminal_complete=True,
    trace_complete=True,
)
_COMPLETE_TRACE = BackendExecutionTrace(
    backend="stub-unjudged",
    requested_model="stub-model",
    events=(BackendExecutionEvent(sequence=0, kind="terminal"),),
)
_SLICE_FILES = ("a.py", "b.py")


class _WritingBackend:
    """Write the next slice's in-scope file, then return a scripted terminal.

    The result carries complete accounting evidence: an unjudged slice whose
    usage is unknown ends ``incomplete_accounting`` before the per-slice halt
    (``_UnknownUsageBackend``).
    """

    name = "stub-unjudged"
    model = "stub-model"
    completeness = _COMPLETE_EXECUTION

    def __init__(self, repo: Path, *, error: object | None = None) -> None:
        self.repo = repo
        self.error = error
        self.calls = 0
        self.returned = False

    def attach_toolset(self, toolset, *, context_budget=None) -> None:
        pass

    def close(self) -> None:
        pass

    def invoke(self, invocation) -> InvocationResult:
        self.calls += 1
        target = self.repo / _SLICE_FILES[self.calls - 1]
        target.write_text(f"candidate {self.calls}\n", encoding="utf-8")
        self.returned = True
        return InvocationResult(
            final_message="done" if self.error is None else "",
            transcript=(),
            usage=Usage(input_tokens=1, output_tokens=1, cost_usd=0.0),
            stop_reason="end_turn" if self.error is None else "error",
            error=self.error,
            completeness=self.completeness,
            execution_trace=_COMPLETE_TRACE,
        )


class _UnknownUsageBackend(_WritingBackend):
    """The same terminal, but the provider reported no usable accounting."""

    completeness = ExecutionCompleteness(
        token_usage_complete=False,
        cost_complete=False,
        terminal_complete=True,
        trace_complete=True,
    )


def _init_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tier-unjudged@example.invalid"],
        cwd=path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tier Unjudged"],
        cwd=path,
        check=True,
    )
    for name in _SLICE_FILES:
        (path / name).write_text(f"baseline {name}\n", encoding="utf-8")
    subprocess.run(["git", "add", *_SLICE_FILES], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _spec(base_commit: str, *, slice_count: int) -> Spec:
    slices = tuple(
        Slice(
            id=f"s{index}",
            directive=ImplementationDirective(
                slice_id=f"s{index}",
                intent=f"change {_SLICE_FILES[index]}",
                files_in_scope=(_SLICE_FILES[index],),
                creates_paths=(),
                verify_mode="tests",
                blast_radius={"max_files": 1},
            ),
        )
        for index in range(slice_count)
    )
    return Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier_unjudged_outcomes",
            repo_url="https://example.invalid/tier-unjudged.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise slices that end without a verdict",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
    )


def _stub_smoke(
    monkeypatch,
    *,
    infrastructure_failure: bool,
    mutate: Path | None = None,
) -> list[str]:
    """Replace the smoke primitive with one failing result."""

    calls: list[str] = []

    def smoke(directive, *, cwd, timeout=120, **_kwargs):
        calls.append(directive.slice_id)
        if mutate is not None:
            mutate.write_text("rewritten by the gate\n", encoding="utf-8")
        return SmokeResult(
            passed=False,
            language="python",
            log_excerpt=(
                "python: required tool 'ruff' is not installed"
                if infrastructure_failure
                else "a.py:1:1: SyntaxError"
            ),
            duration_ms=1,
            infrastructure_failure=infrastructure_failure,
        )

    monkeypatch.setattr("crucible.workflows.tier.smoke_check", smoke)
    return calls


def _stub_build_verify(monkeypatch) -> list[str]:
    """Record any build/verify dispatch; none of these slices may reach it."""

    calls: list[str] = []

    def build_and_verify(*args, **kwargs):
        calls.append(kwargs.get("slice_id", "?"))
        raise AssertionError("build/verify ran for a slice the gate stopped")

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        build_and_verify,
    )
    return calls


def _stub_breach(monkeypatch, repo: Path) -> list[str]:
    """Observe the Breach seed, settle the child's writer epoch, then stop."""

    from crucible.workflows import _mutation_journal
    from crucible.workflows import tier as tier_workflow

    seeds: list[str] = []

    def breach_main(argv, *, fresh_lineage, development_budget_ledger):
        seeds.append((repo / "a.py").read_text(encoding="utf-8"))
        parent = _mutation_journal._require_live_parent_transaction(  # noqa: SLF001
            argv[argv.index("--parent-mutation-transaction-id") + 1]
        )
        tier_workflow._build_verify._settle_no_process_writer_epoch(
            _mutation_journal.live_writer_epoch_family(parent),
            phase="test_unjudged_breach_seed_observation",
            succeeded=False,
        )
        raise RuntimeError("stop after observing the Breach seed")

    monkeypatch.setattr(
        tier_workflow.breach_workflow,
        "validate_breach_observation_runner",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(tier_workflow.breach_workflow, "main", breach_main)
    return seeds


def _run(
    tmp_path: Path,
    monkeypatch,
    *,
    slice_count: int = 1,
    error: object | None = None,
    backend_type=_WritingBackend,
    **run_kwargs,
):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    if not (repo / ".git").exists():
        base_commit = _init_repo(repo)
    else:
        base_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    spec_dir = tmp_path / "corpus" / "spec1"
    spec_dir.mkdir(parents=True)
    Corpus(spec_dir).put_spec(_spec(base_commit, slice_count=slice_count))
    backend = backend_type(repo, error=error)
    monkeypatch.setattr(
        "crucible.workflows._backends.resolve_for_role",
        lambda role, *args, **kwargs: backend,
    )
    from crucible.workflows.tier import run_tier

    run_kwargs.setdefault("skip_smoke_check", True)
    run_kwargs.setdefault("skip_build_verify", False)
    run_kwargs.setdefault("breach_policy", BreachPolicy(enabled=False))
    rc = run_tier(
        spec=spec_dir,
        tier_id="t0",
        cwd=str(repo),
        skip_mutation_scope=True,
        skip_smoke_bench=True,
        skip_review=True,
        **run_kwargs,
    )
    run_dir = sorted((spec_dir / "runs").glob("*tier*"))[-1]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    return rc, summary, outcome, backend, repo, Corpus(spec_dir)


def _latest_implementation(corpus: Corpus) -> Implementation:
    spec = corpus.latest_spec()
    assert spec is not None
    return corpus.get_record(spec.implementation_refs[-1], as_type=Implementation)


# ---------------------------------------------------------------------------
# The halt status
# ---------------------------------------------------------------------------


def _record(classification: str, *, scope_escapes: tuple[str, ...] = ()):
    return SimpleNamespace(
        classification=classification,
        scope_escapes=scope_escapes,
    )


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("smoke_unavailable", "slice_inconclusive"),
        ("dev_verify_unavailable", "slice_inconclusive"),
        ("hard_fail_capture", "slice_inconclusive"),
        ("hard_fail_cancellation", "cancelled"),
        ("hard_fail_smoke", "slice_failed"),
        ("dev_verify_failed", "slice_failed"),
        ("verify_failed", "slice_failed"),
        ("hard_fail_max_turns", "slice_failed"),
        ("hard_fail_timeout", "slice_failed"),
        ("hard_fail_gate_mutation", "slice_failed"),
        ("hard_fail_scope_escape", "slice_failed"),
        ("hard_fail_mutation_scope", "slice_failed"),
    ],
)
def test_slice_halt_status_separates_unjudged_cancelled_and_judged(
    classification: str,
    expected: str,
) -> None:
    from crucible.workflows.tier import _slice_halt_status

    assert _slice_halt_status(_record(classification), restored=True) == expected
    # A status that is not permanent claims the workspace is back at its
    # baseline. Without that proof every slice halt stays ``slice_failed``.
    assert (
        _slice_halt_status(_record(classification), restored=False)
        == "slice_failed"
    )


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        # A gate that could not run is the only writer after the candidate
        # passed its own scope validation: paths it left outside the slice
        # are gate debris, restored with the transaction.
        ("smoke_unavailable", "slice_inconclusive"),
        ("dev_verify_unavailable", "slice_inconclusive"),
        # Before any gate the model is the only writer: an escape is
        # deterministic evidence against the slice.
        ("hard_fail_capture", "slice_failed"),
        ("hard_fail_cancellation", "slice_failed"),
    ],
)
def test_slice_halt_status_scope_escape_guard_follows_the_writer(
    classification: str,
    expected: str,
) -> None:
    from crucible.workflows.tier import _slice_halt_status

    escaped = _record(classification, scope_escapes=("outside.py",))

    assert _slice_halt_status(escaped, restored=True) == expected
    assert _slice_halt_status(escaped, restored=False) == "slice_failed"


# ---------------------------------------------------------------------------
# Smoke gate
# ---------------------------------------------------------------------------


def test_smoke_gate_that_could_not_run_ends_slice_inconclusive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    smoke_calls = _stub_smoke(monkeypatch, infrastructure_failure=True)
    gate_calls = _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
    )

    assert rc == 1
    assert summary["status"] == "slice_inconclusive"
    assert outcome["detail_status"] == "slice_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "smoke_unavailable"
    assert row["smoke_passed"] is False
    assert row["smoke_infrastructure_failure"] is True
    assert row["build_verify"] is None
    # The Phase 1 event of the slice carries the flag as the summary row does.
    run_dir = sorted((corpus.spec_dir / "runs").glob("*tier*"))[-1]
    slice_events = [
        event
        for event in map(
            json.loads,
            (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines(),
        )
        if event.get("phase") == 1
        and event.get("slice_id") == "s0"
        and "classification" in event
    ]
    assert len(slice_events) == 1
    assert slice_events[0]["classification"] == "smoke_unavailable"
    assert slice_events[0]["smoke_infrastructure_failure"] is True
    # No retry of any kind is keyed on the flag.
    assert smoke_calls == ["s0"]
    assert gate_calls == []
    assert backend.calls == 1
    # Fail-closed behaviour is unchanged: the slice transaction is restored.
    assert summary["rollback_applied"] is True
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    assert summary["phase_3_breach"] is None
    record = _latest_implementation(corpus)
    assert record.classification == "smoke_unavailable"
    assert b"ruff" in corpus.get_artifact(record.artifacts["smoke_log"])


def test_smoke_failure_in_the_source_stays_slice_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _stub_smoke(monkeypatch, infrastructure_failure=False)
    gate_calls = _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
    )

    assert rc == 1
    assert summary["status"] == "slice_failed"
    assert outcome["terminal_class"] == "permanent_failure"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "hard_fail_smoke"
    assert row["smoke_passed"] is False
    assert row["smoke_infrastructure_failure"] is False
    assert gate_calls == []
    assert backend.calls == 1
    assert summary["rollback_applied"] is True
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    assert "smoke_log" in _latest_implementation(corpus).artifacts


def test_flagged_smoke_gate_that_changed_source_stays_slice_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A gate that rewrote the candidate is a failure whatever it reports."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    _stub_smoke(
        monkeypatch,
        infrastructure_failure=True,
        mutate=repo / "a.py",
    )
    _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, _corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
    )

    assert rc == 1
    assert summary["status"] == "slice_failed"
    assert outcome["terminal_class"] == "permanent_failure"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "hard_fail_gate_mutation"
    # The classification no longer says so, the row still does.
    assert row["smoke_infrastructure_failure"] is True
    assert backend.calls == 1
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"


def test_flagged_smoke_result_is_never_a_breach_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Benchmark single-slice shape: unjudged work is not handed to Breach."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    _stub_smoke(monkeypatch, infrastructure_failure=True)
    _stub_build_verify(monkeypatch)
    breach_seeds = _stub_breach(monkeypatch, repo)

    rc, summary, outcome, backend, repo, _corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert breach_seeds == []
    assert rc == 1
    assert summary["status"] == "slice_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    assert summary["phase_1_slices"][0]["classification"] == "smoke_unavailable"
    assert summary["phase_3_breach"] is None
    assert summary["rollback_applied"] is True
    assert backend.calls == 1
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"


def test_unflagged_smoke_failure_still_seeds_breach(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The benchmark deferral of a judged smoke failure is unchanged."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    _stub_smoke(monkeypatch, infrastructure_failure=False)
    _stub_build_verify(monkeypatch)
    breach_seeds = _stub_breach(monkeypatch, repo)

    rc, summary, _outcome, backend, repo, _corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
        extra_parameters={"swebench_metadata": {}},
        breach_policy=BreachPolicy(enabled=True),
    )

    assert breach_seeds == ["candidate 1\n"]
    assert rc == 1
    assert summary["status"] == "breach_exception"
    assert summary["phase_1_slices"][0]["classification"] == "hard_fail_smoke"
    assert backend.calls == 1
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"


# ---------------------------------------------------------------------------
# Cancelled implementer
# ---------------------------------------------------------------------------


def test_cancelled_implementer_ends_cancelled_and_stops_the_tier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gate_calls = _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, corpus = _run(
        tmp_path,
        monkeypatch,
        slice_count=2,
        error=ExternalCancellation(
            cause="operator",
            message="implementer cancelled",
        ),
    )

    assert rc == 1
    assert summary["status"] == "cancelled"
    assert outcome["detail_status"] == "cancelled"
    assert outcome["terminal_class"] == "cancelled"
    assert [row["slice_id"] for row in summary["phase_1_slices"]] == ["s0"]
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_cancellation"
    )
    # Unlike a transport failure the tier halts: no later slice is attempted,
    # and the cancelled call gets no recovery turn.
    assert backend.calls == 1
    assert summary["transport_failure_slices"] == 0
    assert gate_calls == []
    assert summary["phase_3_breach"] is None
    assert summary["rollback_applied"] is True
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "baseline b.py\n"
    assert _latest_implementation(corpus).classification == (
        "hard_fail_cancellation"
    )


def test_cancelled_implementer_with_unknown_usage_ends_incomplete_accounting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Accounting debt keeps its precedence over the unjudged statuses."""

    _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, _corpus = _run(
        tmp_path,
        monkeypatch,
        slice_count=2,
        error=ExternalCancellation(
            cause="operator",
            message="implementer cancelled",
        ),
        backend_type=_UnknownUsageBackend,
    )

    assert rc == 1
    assert summary["status"] == "incomplete_accounting"
    assert outcome["detail_status"] == "incomplete_accounting"
    assert summary["costs"]["accounting_debt"] is True
    assert summary["phase_1_slices"][0]["classification"] == (
        "hard_fail_cancellation"
    )
    assert backend.calls == 1
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    assert (repo / "b.py").read_text(encoding="utf-8") == "baseline b.py\n"


# ---------------------------------------------------------------------------
# Uncapturable delta
# ---------------------------------------------------------------------------


def test_uncapturable_slice_delta_ends_slice_inconclusive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # ``assay.primitives`` re-exports the ``implement`` function under the
    # module's own name, so resolve the module explicitly.
    assay_implement = importlib.import_module("assay.primitives.implement")
    gate_calls = _stub_build_verify(monkeypatch)
    real_diff_bytes = assay_implement._git_diff_bytes  # noqa: SLF001
    state: dict[str, object] = {"backend": None, "failed": 0}

    def diff_bytes(*args, **kwargs):
        backend = state["backend"]
        # ``assay.implement`` calls this helper directly; the tier's own
        # captures go through ``capture_workspace_delta`` and always pass
        # ``max_diff_bytes``. Fail only the primitive's post-invocation capture.
        if (
            backend is not None
            and backend.returned
            and "max_diff_bytes" not in kwargs
            and not state["failed"]
        ):
            state["failed"] = 1
            raise assay_implement.GitCaptureError(
                "capture Git diff failed: scratch index unavailable"
            )
        return real_diff_bytes(*args, **kwargs)

    monkeypatch.setattr(assay_implement, "_git_diff_bytes", diff_bytes)

    class _Backend(_WritingBackend):
        def __init__(self, repo: Path, *, error: object | None = None) -> None:
            super().__init__(repo, error=error)
            state["backend"] = self

    rc, summary, outcome, backend, repo, corpus = _run(
        tmp_path,
        monkeypatch,
        backend_type=_Backend,
    )

    assert state["failed"] == 1
    assert rc == 1
    assert summary["status"] == "slice_inconclusive"
    assert outcome["detail_status"] == "slice_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "hard_fail_capture"
    assert row["smoke_infrastructure_failure"] is None
    assert gate_calls == []
    assert backend.calls == 1
    assert summary["phase_3_breach"] is None
    assert summary["rollback_applied"] is True
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    record = _latest_implementation(corpus)
    assert record.classification == "hard_fail_capture"
    assert record.files_modified == ()
    assert "capture_error" in record.artifacts


# ---------------------------------------------------------------------------
# The halt guards in a tier run
# ---------------------------------------------------------------------------


def _report_slice_restore_unproven(monkeypatch, *, slice_id: str) -> dict[str, int]:
    """Let the slice journal restore, then report that restore as failed once."""

    from crucible.workflows import _mutation_journal

    journal_type = _mutation_journal.WorkspaceMutationJournal
    real_restore = journal_type.restore
    state = {"depth": 0, "unproven": 0}

    def restore(self, **kwargs):
        # ``restore`` re-enters itself under the workspace lock; the tier
        # reads the answer of the outermost call only.
        state["depth"] += 1
        try:
            restored = real_restore(self, **kwargs)
        finally:
            state["depth"] -= 1
        if (
            state["depth"] == 0
            and self.slice_id == slice_id
            and not state["unproven"]
        ):
            state["unproven"] = 1
            return False
        return restored

    monkeypatch.setattr(journal_type, "restore", restore)
    return state


def _stub_unavailable_gate(monkeypatch, *, debris: Path) -> list[str]:
    """Replace build/verify with a gate that could not run and left a file."""

    from crucible.workflows import _build_verify
    from tests.test_tier_build_progress_retry import _dev_verify_unavailable

    calls: list[str] = []

    def build_and_verify(*args, **kwargs):
        calls.append(kwargs.get("slice_id", "?"))
        debris.write_text("left by the gate\n", encoding="utf-8")
        _build_verify._settle_no_process_writer_epoch(  # noqa: SLF001
            kwargs.get("writer_epoch_members"),
            phase="test_unjudged_unavailable_gate_stub",
            succeeded=False,
        )
        return _dev_verify_unavailable()

    monkeypatch.setattr(
        "crucible.workflows.tier._build_verify.build_and_verify",
        build_and_verify,
    )
    return calls


class _EscapingBackend(_WritingBackend):
    """Also write a path no slice declares before returning the terminal."""

    def invoke(self, invocation) -> InvocationResult:
        (self.repo / "outside.py").write_text("escaped\n", encoding="utf-8")
        return super().invoke(invocation)


def test_unproven_restore_keeps_an_unjudged_slice_slice_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Without a proven restore no status may claim a clean baseline."""

    _stub_smoke(monkeypatch, infrastructure_failure=True)
    gate_calls = _stub_build_verify(monkeypatch)
    restores = _report_slice_restore_unproven(monkeypatch, slice_id="s0")

    rc, summary, outcome, backend, repo, _corpus = _run(
        tmp_path,
        monkeypatch,
        skip_smoke_check=False,
    )

    assert restores["unproven"] == 1
    assert rc == 1
    assert summary["rollback_applied"] is False
    assert summary["status"] == "slice_failed"
    assert outcome["detail_status"] == "slice_failed"
    assert outcome["terminal_class"] == "permanent_failure"
    # The row still names what happened to the slice.
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "smoke_unavailable"
    assert row["smoke_infrastructure_failure"] is True
    assert gate_calls == []
    assert backend.calls == 1
    assert summary["phase_3_breach"] is None
    # The bytes were restored; only the proof was withheld.
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"


def test_unavailable_gate_that_left_a_file_outside_scope_stays_inconclusive(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """What a gate left behind is not evidence against the slice."""

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    gate_calls = _stub_unavailable_gate(monkeypatch, debris=repo / "debris.txt")

    rc, summary, outcome, backend, repo, corpus = _run(tmp_path, monkeypatch)

    assert rc == 1
    assert summary["status"] == "slice_inconclusive"
    assert outcome["detail_status"] == "slice_inconclusive"
    assert outcome["terminal_class"] == "inconclusive"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "dev_verify_unavailable"
    # The gate ran once on a candidate that had passed its own scope check;
    # no model retry is keyed on its result.
    assert len(gate_calls) == 1
    assert backend.calls == 1
    # The file is recorded on the slice and removed with its transaction.
    record = _latest_implementation(corpus)
    assert record.classification == "dev_verify_unavailable"
    assert record.scope_escapes == ("debris.txt",)
    assert summary["rollback_applied"] is True
    assert not (repo / "debris.txt").exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
    assert summary["phase_3_breach"] is None


def test_cancelled_implementer_that_wrote_outside_scope_stays_slice_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Where the model was the only writer, an escape is evidence against it."""

    gate_calls = _stub_build_verify(monkeypatch)

    rc, summary, outcome, backend, repo, corpus = _run(
        tmp_path,
        monkeypatch,
        error=ExternalCancellation(
            cause="operator",
            message="implementer cancelled",
        ),
        backend_type=_EscapingBackend,
    )

    assert rc == 1
    assert summary["status"] == "slice_failed"
    assert outcome["detail_status"] == "slice_failed"
    assert outcome["terminal_class"] == "permanent_failure"
    row = summary["phase_1_slices"][0]
    assert row["classification"] == "hard_fail_cancellation"
    assert gate_calls == []
    assert backend.calls == 1
    record = _latest_implementation(corpus)
    assert record.classification == "hard_fail_cancellation"
    assert record.scope_escapes == ("outside.py",)
    # The escape is restored like the in-scope write.
    assert summary["rollback_applied"] is True
    assert not (repo / "outside.py").exists()
    assert (repo / "a.py").read_text(encoding="utf-8") == "baseline a.py\n"
