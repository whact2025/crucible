# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Model-authored competition plans carry canonical scope paths in every record."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace

import pytest

from assay import ExecutionCompleteness, ReviewCoverage, TierDAG, Verdict
from assay.canonical import hash_record
from assay.corpus.store import Corpus
from assay.primitives.plan import _run_planner, _serialize_tier_dag

from crucible.rubrics import load
from crucible.workflows import _plan_competition as competition
from crucible.workflows import _plan_selection as selection
from crucible.workflows import architect, plan
from crucible.workflows._plan_validation import build_pre_review_candidate_validator
from tests import test_plan_competition as harness
from tests.test_plan_review_response_retry import _Backend as ReplyBackend
from tests.test_plan_selection import _Backend as SelectorBackend, _candidate, _spec


def _written(name, files, creates=()):
    """Strictly decode a one-slice plan; its scope paths stay as written."""
    return plan._decode_review_candidate(json.dumps({
        "tiers": [{
            "tier_id": "t0", "depends_on": [], "slices": [{
                "id": name, "directive": {
                    "slice_id": name, "intent": "implement the thing",
                    "requirement_refs": ["implement the thing"],
                    "files_in_scope": list(files), "creates_paths": list(creates),
                    "verify_mode": "tests",
                    "blast_radius": {"max_files": 8, "languages": ["python"]},
                },
            }],
        }],
    }).encode())


def _scope(dag):
    directive = dag.tiers[0].slices[0].directive
    return directive.files_in_scope, directive.creates_paths


def _author(monkeypatch, default, **by_name):
    """Let every fake draft, and the synthesis ``combined``, write this scope."""

    def authored(name="s0"):
        return _written(name, *by_name.get(name, default))

    monkeypatch.setattr(harness, "_dag", authored)


def _nested_project(root):
    """Give the project root a real directory named like the project."""
    (root / "demo/src").mkdir(parents=True)
    (root / "demo/src/thing.py").write_text("value = 2\n")


def _compete(repo, *, pool=None, **kwargs):
    _workspace, root, corpus, spec = repo
    pool = pool or harness.Pool()
    run_dir = corpus.spec_dir / "runs/competition"
    outcome = competition.run_plan_competition(
        spec,
        pool=pool,
        corpus=corpus,
        cwd=str(root),
        run_dir=run_dir,
        planner_role="plan_planner",
        critic_role="plan_critic",
        judge_role="plan_judge",
        critic_rubric=load("plan_critic"),
        judge_rubric=load("plan_judge"),
        budget_usd=10,
        **kwargs,
    )
    report = json.loads((run_dir / "plan_competition.json").read_text())
    return pool, outcome, report


def _judged(corpus, ref):
    return plan._decode_review_candidate(corpus.get_artifact(ref))


def _finding_ids(corpus, row):
    return {
        finding.id
        for ref in row["verdict_refs"]
        for finding in corpus.get_record(ref, as_type=Verdict).findings
    }


def _plan_once(monkeypatch, tmp_path, spec, root, reply):
    """Run single-track planning on one Planner reply with a passing review."""
    corpus = Corpus(tmp_path / "single-track")

    def passing_review(artifact, rubric, **kwargs):
        return Verdict(
            rubric=rubric.name, decision="pass", findings=(),
            artifact_ref=corpus.put_artifact(artifact, kind="reviewed_artifacts"),
            artifact_commit=None, reasoning="fixture review",
            coverage=ReviewCoverage(
                complete=True, sources_reviewed=("artifact",), omissions=(),
                artifact_chars_total=len(artifact),
                artifact_chars_reviewed=len(artifact),
            ),
            backend_name="fixture", transcript_ref=None, cost_usd=0,
            execution=None, kind=kwargs["kind"],
            cache_creation_tokens=0, cache_read_tokens=0,
            execution_completeness=ExecutionCompleteness(
                token_usage_complete=True, cost_complete=True,
                terminal_complete=True, trace_complete=True,
            ),
        )

    with monkeypatch.context() as patch:
        patch.setattr(plan, "strict_review", passing_review)
        return plan.run_plan(
            spec,
            corpus=corpus,
            cwd=str(root),
            planner_backend=ReplyBackend(reply),
            critic_backend=ReplyBackend(),
            judge_backend=ReplyBackend(),
            critic_rubric=load("plan_critic"),
            judge_rubric=load("plan_judge"),
            candidate_validator=build_pre_review_candidate_validator(
                spec=spec, project_root=root
            ),
            max_candidate_validation_repairs=0,
        )


@pytest.mark.parametrize(
    "project,submodule,nested",
    [
        pytest.param("demo", None, False, id="no-same-named-directory"),
        pytest.param("demo", None, True, id="real-same-named-directory"),
        pytest.param("deepswe/acme/demo", None, False, id="namespaced-project"),
        pytest.param("demo", "project", False, id="submodule-over-project"),
        pytest.param("", None, False, id="empty-project-name"),
    ],
)
def test_authored_scope_equals_assay_planner_parse(
    tmp_path, monkeypatch, project, submodule, nested
):
    """One rule: the competition decode equals Assay's single-track parse."""
    spec = _spec()
    spec = replace(
        spec,
        project_context=replace(
            spec.project_context, project=project, submodule_path=submodule
        ),
    )
    name = submodule or project or "demo"
    root = tmp_path / "root"
    root.mkdir()
    if nested:
        (root / name).mkdir(parents=True)
    # The process runs where the name is a directory; only the root counts.
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / name).mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    reply = _serialize_tier_dag(_written(
        "s0",
        [
            f"{name}/src/thing.py", "./src/other.py", "src\\win.py",
            "/etc/passwd", "demo/src/named.py",
        ],
        [
            f"{name}/{name}/twice.py", f".\\{name}\\src\\new_mod.py",
            "C:\\Windows\\system.ini",
        ],
    ))

    def assay_parse(project_root):
        dag, _transcript, error, _result = _run_planner(
            spec, ReplyBackend(reply.decode()), Corpus(tmp_path / "corpus"),
            (), (), 0, 5, None, project_root=project_root,
        )
        assert error is None and dag is not None
        return dag

    for project_root in (str(root), None):
        authored = selection._decode_authored_dag(reply, spec, project_root)
        assert hash_record(authored) == hash_record(assay_parse(project_root))
        assert hash_record(authored) != hash_record(
            plan._decode_review_candidate(reply)
        )


def test_authored_scope_never_reads_the_process_cwd(tmp_path, monkeypatch):
    spec = _spec()
    reply = _serialize_tier_dag(
        _written("s0", ["demo/src/thing.py"], ["demo/src/new_mod.py"])
    )
    kept = (("demo/src/thing.py",), ("demo/src/new_mod.py",))
    stripped = (("src/thing.py",), ("src/new_mod.py",))
    with_directory, without_directory = tmp_path / "with", tmp_path / "without"
    (with_directory / "demo").mkdir(parents=True)
    without_directory.mkdir()

    for process_cwd in (with_directory, without_directory):
        monkeypatch.chdir(process_cwd)
        decode = selection._decode_authored_dag
        assert _scope(decode(reply, spec, str(with_directory))) == kept
        assert _scope(decode(reply, spec, str(without_directory))) == stripped
        assert _scope(decode(reply, spec, None)) == stripped


@pytest.mark.parametrize("field", ["files_in_scope", "creates_paths"])
@pytest.mark.parametrize("written", ["demo/", "./", ".\\", "./demo/"])
def test_scope_path_empty_after_canonicalization_is_refused(
    tmp_path, written, field
):
    scope = ([written], []) if field == "files_in_scope" else ([], [written])
    reply = _serialize_tier_dag(_written("s0", *scope))

    with pytest.raises(ValueError, match="empty after canonicalization") as refusal:
        selection._decode_authored_dag(reply, _spec(), str(tmp_path))

    assert "'s0'" in str(refusal.value) and repr(written) in str(refusal.value)


def test_nonempty_canonical_spellings_of_the_project_root_are_not_refused(tmp_path):
    """The refusal is emptiness, not a judgement that a path names the root."""
    spec = _spec()

    def canonical(written):
        reply = _serialize_tier_dag(_written("s0", [written], [written]))
        return _scope(selection._decode_authored_dag(reply, spec, str(tmp_path)))

    assert canonical(".") == ((".",), (".",))
    assert canonical("demo/.") == ((".",), (".",))
    assert canonical("demo") == (("demo",), ("demo",))
    (tmp_path / "demo").mkdir()
    assert canonical("demo/") == (("demo/",), ("demo/",))


def test_only_the_authored_decode_rewrites_scope(tmp_path):
    """Stored and operator plans keep their bytes; the rule is not idempotent."""
    written = (("./src/thing.py", "src\\x.py"), ("demo/demo/new_mod.py",))
    raw = _serialize_tier_dag(_written("s0", *written))

    for decode in (plan._decode_review_candidate, selection._decode_dag):
        stored = decode(raw)
        assert _scope(stored) == written
        assert _serialize_tier_dag(stored) == raw

    authored = selection._decode_authored_dag(raw, _spec(), str(tmp_path))
    assert _scope(authored) == (("src/thing.py", "src/x.py"), ("demo/new_mod.py",))
    reread = selection._decode_dag(_serialize_tier_dag(authored))
    assert hash_record(reread) == hash_record(authored)


def test_selection_rejects_a_relative_project_root_before_any_call(
    tmp_path, monkeypatch
):
    corpus = Corpus(tmp_path / "corpus")
    (tmp_path / "project").mkdir()
    monkeypatch.chdir(tmp_path)
    backend = SelectorBackend()

    with pytest.raises(ValueError, match="project_root must be an absolute path"):
        selection.select_plan_candidate(
            _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
            project_root="project",
        )

    assert not backend.invocations


def test_empty_synthesis_scope_is_repaired_through_the_response_contract(tmp_path):
    corpus = Corpus(tmp_path)

    def synthesis(files):
        return {
            "mode": "synthesize", "selected_id": None, "selected_digest": None,
            "synthesized_dag": json.loads(
                _serialize_tier_dag(_written("combined", files))
            ),
        }

    backend = SelectorBackend(synthesis(["demo/"]), synthesis(["demo/src/thing.py"]))

    result = selection.select_plan_candidate(
        _spec(), (_candidate(corpus, "a"),), backend=backend, corpus=corpus,
    )

    assert result.error is None and result.mode == "synthesize"
    assert len(backend.invocations) == 2
    assert _scope(result.candidate_dag) == (("src/thing.py",), ())
    stored = json.loads(corpus.get_artifact(result.selection_ref))
    proposal = stored["response"]["synthesized_dag"]["tiers"][0]["slices"][0]
    assert proposal["directive"]["files_in_scope"] == ["demo/src/thing.py"]
    judged = stored["candidate_dag"]["tiers"][0]["slices"][0]["directive"]
    assert judged["files_in_scope"] == ["src/thing.py"]
    assert stored["candidate_digest"] == hash_record(result.candidate_dag)


@pytest.mark.parametrize(
    "written,canonical",
    [
        pytest.param(
            (["demo/src/thing.py"], []), (("src/thing.py",), ()),
            id="existing-file",
        ),
        pytest.param(
            (["src/thing.py"], ["demo/src/new_mod.py"]),
            (("src/thing.py",), ("src/new_mod.py",)),
            id="created-file",
        ),
    ],
)
def test_prefixed_draft_scope_is_canonical_in_every_record(
    tmp_path, monkeypatch, written, canonical
):
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, written)

    pool, outcome, report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == canonical
    for row in report["candidates"]:
        # The draft stays stored as the model wrote it; the row describes
        # the plan that was judged.
        assert _scope(_judged(corpus, row["raw_ref"])) == (
            tuple(written[0]), tuple(written[1]),
        )
        judged = _judged(corpus, row["candidate_ref"])
        assert _scope(judged) == canonical
        assert row["digest"] == hash_record(judged)
    (selected,) = (
        row for row in report["candidates"] if row["id"] == report["selected_id"]
    )
    assert selected["digest"] == hash_record(outcome.spec.tier_dag)
    bundle = json.loads(corpus.get_artifact(report["comparison"]["bundle_ref"]))
    compared = [
        item["dag"]["tiers"][0]["slices"][0]["directive"]
        for item in bundle["candidates"]
    ]
    assert [
        (tuple(item["files_in_scope"]), tuple(item["creates_paths"]))
        for item in compared
    ] == [canonical] * 3
    review_prompts = [
        invocation.user
        for backend in pool.backends if backend.label not in pool.finished
        for invocation in backend.invocations
    ]
    assert len(review_prompts) == 7
    assert not any("demo/src/" in prompt for prompt in review_prompts)


@pytest.mark.parametrize(
    "nested", [False, True],
    ids=["no-same-named-directory", "real-same-named-directory"],
)
def test_draft_scope_follows_the_project_root_from_any_process_cwd(
    tmp_path, monkeypatch, nested
):
    repo = harness._repo(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    if nested:
        _nested_project(repo[1])
        elsewhere.mkdir()
    else:
        # The process runs where the name is a directory; the root has none.
        (elsewhere / "demo").mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    _author(monkeypatch, (["./demo/src/thing.py"], ["./demo/src/new_mod.py"]))

    _pool, outcome, _report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == (
        (("demo/src/thing.py",), ("demo/src/new_mod.py",))
        if nested
        else (("src/thing.py",), ("src/new_mod.py",))
    )


def test_repeated_name_under_a_real_same_named_directory_is_left_for_the_gate(
    tmp_path, monkeypatch
):
    repo = harness._repo(tmp_path)
    _workspace, root, corpus, _ = repo
    _nested_project(root)
    (root / "src/other.py").write_text("other = 1\n")
    _author(monkeypatch, (["./demo/src/thing.py", "./demo/src/other.py"], []))

    pool, outcome, report = _compete(repo)

    assert not outcome.accepted
    assert outcome.status == "plan_competition_inconclusive"
    assert len(pool.order) == 3  # Blocked before any paid review.
    for row in report["candidates"]:
        assert row["status"] == "validate_blocked"
        found = _finding_ids(corpus, row)
        assert f"missing_files_in_scope__{row['id']}__demo/src/other.py" in found
        assert not any(item.endswith("demo/src/thing.py") for item in found)


def test_absolute_draft_scope_is_kept_for_the_gate(tmp_path, monkeypatch):
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, (["/etc/passwd", "C:\\Windows\\system.ini"], []))

    pool, outcome, report = _compete(repo)

    assert not outcome.accepted
    assert outcome.status == "plan_competition_inconclusive"
    assert len(pool.order) == 3
    for row in report["candidates"]:
        assert row["status"] == "validate_blocked"
        prefix = f"scope_path_escape__{row['id']}__files_in_scope__"
        assert {
            prefix + "/etc/passwd", prefix + "C:/Windows/system.ini",
        } <= _finding_ids(corpus, row)


def test_backslash_and_dot_slash_draft_scope_is_normalized(tmp_path, monkeypatch):
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, (["./src/thing.py"], []), plan_1=(["src\\thing.py"], []))

    _pool, outcome, report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == (("src/thing.py",), ())
    for row in report["candidates"]:
        assert row["accepted"] is True, row
        assert _scope(_judged(corpus, row["candidate_ref"])) == (
            ("src/thing.py",), (),
        )


def test_gates_judge_the_canonical_form_of_a_created_path(tmp_path, monkeypatch):
    """``demo/src/thing.py`` names the existing file; creating it is refused."""
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, (["src/thing.py"], ["demo/src/thing.py"]))

    pool, outcome, report = _compete(repo)

    assert not outcome.accepted
    assert outcome.status == "plan_competition_inconclusive"
    assert len(pool.order) == 3
    for row in report["candidates"]:
        assert row["status"] == "validate_blocked"
        assert (
            f"creates_path_already_exists__{row['id']}__src/thing.py"
            in _finding_ids(corpus, row)
        )


def test_caller_transform_and_validator_receive_canonical_scope(
    tmp_path, monkeypatch
):
    """The rule runs first, so a caller's transform and gate cannot disagree."""
    repo = harness._repo(tmp_path)
    _workspace, root, _corpus, spec = repo
    _author(monkeypatch, (["./src/thing.py"], ["demo/src/new_mod.py"]))
    gate = build_pre_review_candidate_validator(spec=spec, project_root=root)
    seen = {"transform": [], "validator": []}

    def transform(dag):
        seen["transform"].append(_scope(dag))
        return dag

    def validator(dag):
        seen["validator"].append(_scope(dag))
        return gate(dag)

    _pool, outcome, _report = _compete(
        repo, candidate_transform=transform, candidate_validator=validator
    )

    canonical = (("src/thing.py",), ("src/new_mod.py",))
    assert outcome.accepted, (outcome.status, outcome.reason)
    assert len(seen["transform"]) >= 3 and set(seen["transform"]) == {canonical}
    assert len(seen["validator"]) >= 3 and set(seen["validator"]) == {canonical}


def test_read_only_reference_gate_sees_canonical_write_scope(tmp_path, monkeypatch):
    repo = harness._repo(tmp_path)
    _workspace, root, corpus, _ = repo
    other = root / "src/other.py"
    other.write_text("other = 1\n")
    _author(monkeypatch, (["./src/thing.py"], []))
    pool = harness.Pool(
        reference={
            "reference_id": "other",
            "path": "src/other.py",
            "sha256": "sha256:" + hashlib.sha256(other.read_bytes()).hexdigest(),
            "start_line": 1,
            "end_line": 1,
        }
    )

    _pool, outcome, report = _compete(repo, pool=pool)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == (("src/thing.py",), ())
    for row in report["candidates"]:
        assert "read_only_source_invalid" not in {
            finding.category
            for ref in row["verdict_refs"]
            for finding in corpus.get_record(ref, as_type=Verdict).findings
        }


@pytest.mark.parametrize(
    "case",
    ["no-same-named-directory", "real-same-named-directory", "submodule"],
)
def test_same_reply_gives_the_same_plan_at_planner_count_one_and_three(
    tmp_path, monkeypatch, case
):
    repo = harness._repo(tmp_path, submodule=case == "submodule")
    _workspace, root, _corpus, spec = repo
    if case == "submodule":
        written = (["project/src/thing.py"], ["demo/keep.py", "project/project/twice.py"])
        canonical = (("src/thing.py",), ("demo/keep.py", "project/twice.py"))
    elif case == "real-same-named-directory":
        _nested_project(root)
        written = (["./demo/src/thing.py"], ["demo/demo/twice.py", "demo\\src\\new_mod.py"])
        canonical = (
            ("demo/src/thing.py",), ("demo/demo/twice.py", "demo/src/new_mod.py"),
        )
    else:
        written = (["demo/src/thing.py"], ["demo/demo/twice.py", "./src/new_mod.py"])
        canonical = (("src/thing.py",), ("demo/twice.py", "src/new_mod.py"))
    reply = _written("s0", *written)
    monkeypatch.setattr(harness, "_dag", lambda name="s0": reply)

    single = _plan_once(
        monkeypatch, tmp_path, spec, root, _serialize_tier_dag(reply).decode()
    )
    _pool, compared, _report = _compete(repo)

    assert single.accepted, (single.status, single.reason)
    assert compared.accepted, (compared.status, compared.reason)
    assert _scope(single.spec.tier_dag) == canonical
    assert hash_record(compared.spec.tier_dag) == hash_record(single.spec.tier_dag)


@pytest.mark.parametrize(
    "nested,written,canonical",
    [
        pytest.param(
            False, (["demo/src/thing.py"], []), (("src/thing.py",), ()),
            id="existing-file",
        ),
        pytest.param(
            False, (["src/thing.py"], ["demo/src/new_mod.py"]),
            (("src/thing.py",), ("src/new_mod.py",)),
            id="created-file",
        ),
        pytest.param(
            True, (["./demo/src/thing.py"], ["./demo/src/new_mod.py"]),
            (("demo/src/thing.py",), ("demo/src/new_mod.py",)),
            id="real-same-named-directory",
        ),
    ],
)
def test_synthesized_scope_is_canonical_before_its_fresh_gates(
    tmp_path, monkeypatch, nested, written, canonical
):
    repo = harness._repo(tmp_path)
    _workspace, root, corpus, _ = repo
    if nested:
        _nested_project(root)
    _author(monkeypatch, (["src/thing.py"], []), combined=written)

    pool, outcome, report = _compete(repo, pool=harness.Pool(synthesis=True))

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == canonical
    assert [label for label, _ in pool.order[-2:]] == [
        "synthesis_critic", "synthesis_judge",
    ]
    judged = _judged(corpus, report["synthesis"]["candidate_ref"])
    assert _scope(judged) == canonical
    stored = json.loads(corpus.get_artifact(report["comparison"]["selection_ref"]))
    proposal = stored["response"]["synthesized_dag"]["tiers"][0]["slices"][0]
    assert (
        proposal["directive"]["files_in_scope"],
        proposal["directive"]["creates_paths"],
    ) == written
    assert stored["candidate_digest"] == hash_record(judged)


def test_synthesis_copying_a_judged_path_is_canonicalized_again(
    tmp_path, monkeypatch
):
    """A synthesis is new model text, so the rule runs on it as on a draft.

    The comparison reads candidates in their judged form. A path it copies
    from one still loses a leading project-name segment when the root holds
    no such directory.
    """
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(
        monkeypatch,
        (["src/thing.py"], ["demo/demo/x.py"]),
        combined=(["src/thing.py"], ["demo/x.py"]),
    )

    _pool, outcome, report = _compete(repo, pool=harness.Pool(synthesis=True))

    assert outcome.accepted, (outcome.status, outcome.reason)
    for row in report["candidates"]:
        assert _scope(_judged(corpus, row["candidate_ref"]))[1] == ("demo/x.py",)
    assert _scope(outcome.spec.tier_dag)[1] == ("x.py",)
    judged = _judged(corpus, report["synthesis"]["candidate_ref"])
    assert hash_record(judged) == hash_record(outcome.spec.tier_dag)


def test_empty_canonical_draft_scope_is_an_invalid_candidate(tmp_path, monkeypatch):
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, (["src/thing.py"], []), plan_2=(["demo/"], []))

    pool, outcome, report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    rows = {row["id"]: row for row in report["candidates"]}
    refused = rows["plan_2"]
    assert refused["status"] == "invalid_candidate"
    assert "empty after canonicalization" in refused["reason"]
    assert "'demo/'" in refused["reason"]
    # Nothing was judged, and no record says so.
    assert "candidate_ref" not in refused and "digest" not in refused
    assert _scope(_judged(corpus, refused["raw_ref"])) == (("demo/",), ())
    assert not any(backend.label.startswith("plan_2_") for backend in pool.backends)
    assert rows["plan_1"]["accepted"] is True and rows["plan_3"]["accepted"] is True


def test_all_drafts_with_empty_canonical_scope_end_inconclusive(
    tmp_path, monkeypatch
):
    repo = harness._repo(tmp_path)
    _author(monkeypatch, (["./"], []))

    pool, outcome, report = _compete(repo)

    assert not outcome.accepted and outcome.spec.tier_dag is None
    assert outcome.status == "plan_competition_inconclusive"
    assert [row["status"] for row in report["candidates"]] == [
        "invalid_candidate"
    ] * 3
    assert len(pool.order) == 3


def test_empty_canonical_synthesis_scope_exhausts_the_bounded_response_repair(
    tmp_path, monkeypatch
):
    repo = harness._repo(tmp_path)
    corpus = repo[2]
    _author(monkeypatch, (["src/thing.py"], []), combined=(["demo/"], []))

    pool, outcome, report = _compete(repo, pool=harness.Pool(synthesis=True))

    assert not outcome.accepted and outcome.spec.tier_dag is None
    assert outcome.status == "plan_competition_inconclusive"
    labels = [label for label, _ in pool.order]
    assert labels.count("comparison") == 3
    assert not any(label.startswith("synthesis") for label in labels)
    assert report["comparison"]["mode"] == "none" and report["comparison"]["error"]
    assert "synthesis" not in report
    verdict = corpus.get_record(report["comparison"]["verdict_ref"], as_type=Verdict)
    retries = json.loads(
        corpus.get_artifact(verdict.artifacts["review_retry_manifest"])
    )
    assert [
        "empty after canonicalization" in (attempt["validation_error"] or "")
        for attempt in retries["attempts"]
    ] == [True] * 3


def test_prefixed_dot_scope_is_judged_in_its_canonical_form(tmp_path, monkeypatch):
    """``demo/.`` becomes ``.``: nonempty, so it is judged, not refused here."""
    repo = harness._repo(tmp_path)
    _author(monkeypatch, (["demo/."], []))

    _pool, outcome, _report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert _scope(outcome.spec.tier_dag) == ((".",), ())


def test_default_plan_cli_publishes_canonical_scope_for_a_submodule(
    tmp_path, monkeypatch
):
    workspace, _root, corpus, _ = harness._repo(tmp_path, submodule=True)
    _author(monkeypatch, (["project/src/thing.py"], ["demo/keep.py"]))
    pool = harness.Pool()

    @contextmanager
    def backend_pool(*args, **kwargs):
        yield pool

    monkeypatch.setattr(plan, "backend_pool", backend_pool)

    assert plan.main([
        "--spec", str(corpus.spec_dir), "--cwd", str(workspace),
        "--budget-usd", "10", "--enable-source-tools",
    ]) == 0
    assert _scope(corpus.latest_spec().tier_dag) == (
        ("src/thing.py",), ("demo/keep.py",),
    )


def test_plan_cli_canonicalizes_drafts_and_keeps_operator_candidate_scope(
    tmp_path, monkeypatch
):
    """The same plan bytes: rewritten as a draft, not as an operator file."""
    written = (["./src/thing.py"], ["demo/src/new_mod.py"])
    candidate = tmp_path / "candidate.json"
    candidate.write_bytes(_serialize_tier_dag(_written("s0", *written)))
    _author(monkeypatch, written)
    pools = []

    @contextmanager
    def backend_pool(*args, **kwargs):
        pools.append(harness.Pool())
        yield pools[-1]

    monkeypatch.setattr(plan, "backend_pool", backend_pool)
    published = {}
    for source in ("drafts", "operator"):
        workspace, _root, corpus, _ = harness._repo(tmp_path / source)
        argv = [
            "--spec", str(corpus.spec_dir), "--cwd", str(workspace),
            "--budget-usd", "10",
        ]
        if source == "operator":
            argv += ["--review-candidate", str(candidate)]
        assert plan.main(argv) == 0, source
        published[source] = corpus.latest_spec().tier_dag

    assert _scope(published["drafts"]) == (("src/thing.py",), ("src/new_mod.py",))
    assert _scope(published["operator"]) == (tuple(written[0]), tuple(written[1]))
    assert hash_record(published["operator"]) == hash_record(
        plan._decode_review_candidate(candidate.read_bytes())
    )


def test_default_architect_publishes_canonical_synthesis(tmp_path, monkeypatch):
    _workspace, root, corpus, _ = harness._repo(tmp_path, submodule=True)
    _author(
        monkeypatch,
        (["src/thing.py"], []),
        combined=(["project/src/thing.py"], []),
    )
    pool = harness.Pool(synthesis=True, exhaust=True)

    @contextmanager
    def backend_pool(*args, **kwargs):
        yield pool

    monkeypatch.setattr(architect, "backend_pool", backend_pool)
    monkeypatch.setattr(
        architect, "run_tier", lambda **kw: pytest.fail("no tier budget remains")
    )

    assert architect.run_architect(
        spec=corpus.spec_dir, cwd=str(root), budget_usd=10,
        enable_source_tools=True,
    ) == 1
    published = corpus.latest_spec().tier_dag
    assert published is not None
    assert published.tiers[0].slices[0].id == "combined"
    assert _scope(published) == (("src/thing.py",), ())


def _tiered(*tiers):
    """Join one-slice plans into one plan: one argument per tier, in order."""
    return TierDAG(tiers=tuple(
        replace(
            one_slice_plans[0].tiers[0],
            tier_id=f"t{index}",
            depends_on=(f"t{index - 1}",) if index else (),
            slices=tuple(each.tiers[0].slices[0] for each in one_slice_plans),
        )
        for index, one_slice_plans in enumerate(tiers)
    ))


def _scopes(dag):
    return [
        (slice_.id, slice_.directive.files_in_scope, slice_.directive.creates_paths)
        for tier in dag.tiers
        for slice_ in tier.slices
    ]


def test_every_slice_of_every_tier_is_canonical_as_in_assay_planner_parse(
    tmp_path, monkeypatch
):
    """Two tiers, three slices: Assay's parse, the decode and a default run agree."""
    repo = harness._repo(tmp_path)
    _workspace, root, corpus, spec = repo
    (root / "src/other.py").write_text("other = 1\n")
    written = _tiered(
        [
            _written("a", ["./src/thing.py"]),
            _written("b", ["demo/src/other.py"], ["demo\\src\\new_b.py"]),
        ],
        [_written("c", ["demo/src/thing.py"], ["./demo/src/new_c.py"])],
    )
    reply = _serialize_tier_dag(written)
    canonical = [
        ("a", ("src/thing.py",), ()),
        ("b", ("src/other.py",), ("src/new_b.py",)),
        ("c", ("src/thing.py",), ("src/new_c.py",)),
    ]

    parsed, _transcript, error, _result = _run_planner(
        spec, ReplyBackend(reply.decode()), Corpus(tmp_path / "corpus"),
        (), (), 0, 5, None, project_root=str(root),
    )
    assert error is None and _scopes(parsed) == canonical

    monkeypatch.setattr(harness, "_dag", lambda name="s0": written)
    _pool, outcome, report = _compete(repo)

    assert outcome.accepted, (outcome.status, outcome.reason)
    assert hash_record(outcome.spec.tier_dag) == hash_record(parsed)
    for row in report["candidates"]:
        assert _scopes(_judged(corpus, row["raw_ref"])) == _scopes(written)
        assert _scopes(_judged(corpus, row["candidate_ref"])) == canonical
        assert row["digest"] == hash_record(parsed)
    authored = selection._decode_authored_dag(reply, spec, str(root))
    assert hash_record(authored) == hash_record(parsed)
