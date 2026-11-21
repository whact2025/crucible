# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for ``crucible.workflows._reaper.reap_orphaned_dirt``.

Each test fixtures a fresh Git repository. Mechanical cases use a Corpus-shaped
mock with real resume-identity payloads and lifecycle/CAS behavior; authority,
corruption, concurrency, and crash-window cases use a real Assay Corpus.

Coverage includes the legacy mechanics plus lifecycle/chronology authority:
- Empty corpus → empty report, nothing touched.
- Crashed-rollback case: hard_fail Implementation with diff present on
  disk → reaped via git apply -R.
- Skip-worthy Implementation (success) with diff present → diff stays
  on disk (not in failure class, no reap).
- Crashed-rollback case where diff doesn't apply in reverse (tree
  drifted) → recorded in crashed_rollbacks_skipped, no exception.
- Scope-escape file deleted from any Implementation regardless of
  classification.
- Scope-escape file missing → skipped silently.
- Scope-escape on a non-empty directory → refused and the outer transaction restores.
- Cumulative later failure without an exact journal fails closed.
- Multiple slices, mixed classifications → correct subset reaped.
- Implementation with unresolvable diff_ref → captured in failures, no crash.
- Spec reference order wins over hostile artifact enumeration/timestamps.
- Exact cross-slice Tier landing protects both diff and scope-escape paths.
- A newer cumulative failure restores the exact landed preimage through its journal.
- Unattached legacy records are diagnostic-only without journal chronology.
- Corrupt attached chronology and journal failure both block every write.
- Current-plan identity rejects a reused old-plan slice ID.
- Multi-candidate failure and a concurrent Spec advance restore atomically.
- A crash after cleanup CAS preserves the accepted settlement on restart.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assay import ContentHash, ExecutionContext, Implementation, ProjectContext, Spec
from assay.canonical import hash_record, serialize
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)


def _commit(root: Path, msg: str = "step") -> None:
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-qm", msg], cwd=str(root), check=True)


def _capture_diff(root: Path) -> bytes:
    return subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=str(root),
        check=True,
        capture_output=True,
    ).stdout


def _mock_impl(
    slice_id: str,
    classification: str,
    diff_ref: str,
    *,
    ts: str = "2026-05-21T10:00:00+00:00",
    scope_escapes: tuple[str, ...] = (),
) -> Implementation:
    return Implementation(
        slice_id=slice_id,
        base_commit="0" * 40,
        result_commit=None,
        diff_ref=diff_ref,
        transcript_ref="sha256:" + "0" * 64,
        files_modified=(),
        classification=classification,
        execution=ExecutionContext(
            timestamp=ts,
            machine="reaper-test",
            cwd=".",
            toolchain={},
            backend=None,
            model=None,
            recipe="test",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
        scope_escapes=scope_escapes,
    )


def _mock_corpus(
    impls: list[Implementation],
    diffs: dict[str, bytes],
    *,
    cwd: Path,
    attached_refs: list[str] | None = None,
    artifact_refs: list[str] | None = None,
) -> MagicMock:
    from crucible.workflows._partition import _resume_identity_payload

    corpus = MagicMock()
    corpus.spec_dir = cwd.parent / f"{cwd.name}-mock-corpus"
    corpus.spec_dir.mkdir(exist_ok=True)
    versions_dir = corpus.spec_dir / "versions"
    versions_dir.mkdir(exist_ok=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    aliases = {
        str(ref)
        for ref in (
            *(impl.diff_ref for impl in impls),
            *(attached_refs or ()),
            *(artifact_refs or ()),
        )
    }
    record_ref_by_alias = {
        alias: ContentHash(
            "sha256:"
            + hashlib.sha256(
                f"reaper-record:{alias}".encode("utf-8")
            ).hexdigest()
        )
        for alias in aliases
    }
    alias_by_record_ref = {
        str(record_ref): alias
        for alias, record_ref in record_ref_by_alias.items()
    }
    refs = [record_ref_by_alias[str(impl.diff_ref)] for impl in impls]
    attached_refs = list(
        refs
        if attached_refs is None
        else (record_ref_by_alias[str(ref)] for ref in attached_refs)
    )
    artifact_refs = list(
        refs
        if artifact_refs is None
        else (record_ref_by_alias[str(ref)] for ref in artifact_refs)
    )
    corpus.list_artifacts.side_effect = lambda *a, **kw: iter(list(artifact_refs))
    slice_ids = tuple(dict.fromkeys(impl.slice_id for impl in impls))
    slices = tuple(
        Slice(
            id=slice_id,
            directive=ImplementationDirective(
                slice_id=slice_id,
                intent=f"exercise reaper authority for {slice_id}",
                files_in_scope=("anchor.txt",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )
        for slice_id in slice_ids
    )
    spec = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="reaper-test",
            repo_url="https://example.invalid/reaper.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise reaper authority",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(
            tiers=(Tier(tier_id="t0", slices=slices, depends_on=()),),
        ),
        implementation_refs=tuple(attached_refs),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    )
    tier = spec.tier_dag.tiers[0]
    slice_by_id = {slice_.id: slice_ for slice_ in tier.slices}
    artifacts: dict[str, bytes] = dict(diffs)
    bound_impls: list[Implementation] = []
    for index, impl in enumerate(impls):
        identity_ref = ContentHash(
            "sha256:" + hashlib.sha256(f"identity-{index}".encode()).hexdigest()
        )
        bound = dataclasses.replace(
            impl,
            base_commit=base_commit,
            artifacts={**dict(impl.artifacts), "resume_identity": identity_ref},
        )
        payload = _resume_identity_payload(
            spec,
            tier,
            slice_by_id[bound.slice_id],
            bound,
        )
        artifacts[str(identity_ref)] = json.dumps(
            payload,
            sort_keys=True,
        ).encode("utf-8")
        bound_impls.append(bound)

    by_ref: dict[str, object] = {
        str(record_ref_by_alias[str(impl.diff_ref)]): impl
        for impl in bound_impls
    }
    versions = {spec.version: spec}

    def _write_spec_manifest(value: Spec) -> None:
        digest = hash_record(value).split(":", 1)[-1]
        (versions_dir / f"v{value.version}_{digest[:12]}.json").write_bytes(
            serialize(value)
        )

    _write_spec_manifest(spec)

    def _get_record(ref, as_type):
        del as_type
        rendered = str(ref)
        try:
            return by_ref[rendered]
        except KeyError as exc:
            raise KeyError(alias_by_record_ref.get(rendered, rendered)) from exc

    corpus.get_record.side_effect = _get_record

    def _get_artifact(ref):
        return artifacts[str(ref)]

    corpus.get_artifact.side_effect = _get_artifact

    def _get_artifact_bounded(ref, *, max_bytes):
        payload = _get_artifact(ref)
        if len(payload) > max_bytes:
            raise ValueError(
                f"artifact {ref} exceeds bounded read limit {max_bytes}"
            )
        return payload

    corpus.get_artifact_bounded.side_effect = _get_artifact_bounded

    def _put_artifact(blob, *, kind, name_hint=None):
        del kind, name_hint
        ref = ContentHash("sha256:" + hashlib.sha256(blob).hexdigest())
        artifacts[str(ref)] = blob
        return ref

    corpus.put_artifact.side_effect = _put_artifact
    corpus.latest_spec.return_value = spec

    @contextmanager
    def _locked_spec_authority():
        yield corpus.latest_spec()

    corpus.locked_spec_authority.side_effect = _locked_spec_authority

    class _RetainedSpecAuthority:
        def __init__(self) -> None:
            self.active = False

        def __enter__(self):
            self.active = True
            return corpus.latest_spec()

        def retry_cleanup(self) -> None:
            self.active = False

    corpus.retained_spec_authority.side_effect = _RetainedSpecAuthority

    def _get_spec(version):
        latest = corpus.latest_spec()
        if latest is not None and latest.version == version:
            return latest
        return versions[version]

    corpus.get_spec.side_effect = _get_spec

    def _get_spec_with_digest(version):
        manifests = tuple(versions_dir.glob(f"v{version}_*.json"))
        if len(manifests) != 1:
            raise RuntimeError(
                f"expected exactly one persisted manifest for v{version}, "
                f"found {len(manifests)}"
            )
        payload = manifests[0].read_bytes()
        digest = ContentHash(
            "sha256:" + hashlib.sha256(payload).hexdigest()
        )
        return _get_spec(version), digest

    corpus.get_spec_with_digest.side_effect = _get_spec_with_digest

    def _commit(*, lifecycle_event_factory, transform, **kwargs):
        del kwargs
        latest = corpus.latest_spec()
        version = latest.version + 1
        event = lifecycle_event_factory(version)
        event_ref = ContentHash(hash_record(event))
        by_ref[str(event_ref)] = event
        candidate = dataclasses.replace(
            latest,
            version=version,
            parent=ContentHash(hash_record(latest)),
            root=(
                latest.root
                if latest.root is not None
                else ContentHash(hash_record(latest))
            ),
            lifecycle_event_refs=(*latest.lifecycle_event_refs, event_ref),
        )
        candidate = transform(candidate)
        versions[version] = candidate
        _write_spec_manifest(candidate)
        corpus.latest_spec.return_value = candidate
        return SimpleNamespace(
            spec=candidate,
            refs=SimpleNamespace(lifecycle_event_refs=(event_ref,)),
        )

    corpus.commit.side_effect = _commit
    return corpus


def _real_setup(tmp_path: Path) -> tuple[Path, object, Tier, Slice, str]:
    from assay.corpus.store import Corpus

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")
    _commit(repo, "base")
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    slice_ = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="exercise lifecycle-ordered reaping",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(
        Spec(
            version=0,
            parent=None,
            root=None,
            schema_version=1,
            project_context=ProjectContext(
                project="reaper-real-test",
                repo_url="https://example.invalid/reaper.git",
                base_commit=base_commit,
                branch=None,
                submodule_path=None,
                runners={},
            ),
            execution=None,
            intent="exercise lifecycle-ordered reaping",
            constraints=(),
            done_criteria=(),
            plan_policy=None,
            depends_on=(),
            tier_dag=TierDAG(tiers=(tier,)),
            implementation_refs=(),
            verdict_refs=(),
            measurement_refs=(),
            lifecycle_event_refs=(),
        )
    )
    return repo, corpus, tier, slice_, base_commit


def _real_multi_setup(
    tmp_path: Path,
    *,
    tiers: tuple[Tier, ...],
    files: dict[str, str],
):
    from assay.corpus.store import Corpus

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    for relative, content in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _commit(repo, "base")
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="reaper-multi-test",
            repo_url="https://example.invalid/reaper.git",
            base_commit=base_commit,
            branch=None,
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="exercise chronological reaper authority",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
        depends_on=(),
        tier_dag=TierDAG(tiers=tiers),
        implementation_refs=(),
        verdict_refs=(),
        measurement_refs=(),
        lifecycle_event_refs=(),
    ))
    return repo, corpus, base_commit


def _real_impl(
    corpus,
    *,
    diff: bytes,
    classification: str,
    base_commit: str,
    timestamp: str,
    scope_escapes: tuple[str, ...] = (),
    slice_id: str = "s0",
    files_modified: tuple[str, ...] = ("foo.txt",),
) -> Implementation:
    return Implementation(
        slice_id=slice_id,
        base_commit=base_commit,
        result_commit=None,
        diff_ref=corpus.put_artifact(
            diff,
            kind="diffs",
            name_hint=f"s0-{classification}.diff",
        ),
        transcript_ref=corpus.put_artifact(
            b"[]",
            kind="transcripts",
            name_hint=f"s0-{classification}.json",
        ),
        files_modified=files_modified,
        classification=classification,
        execution=ExecutionContext(
            timestamp=timestamp,
            machine="reaper-test",
            cwd=".",
            toolchain={},
            backend=None,
            model=None,
            recipe="tier",
            recipe_commit=None,
            gpu_model=None,
            run_id="tier-test",
        ),
        scope_escapes=scope_escapes,
    )


def _land_with_rejected_handoff_identity(
    corpus,
    tier: Tier,
    slice_: Slice,
    implementation: Implementation,
) -> tuple[Implementation, str]:
    """Fixture the historical Tier writer/reader identity split."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import _resume_identity_payload

    handoff = dataclasses.replace(
        implementation,
        execution=dataclasses.replace(
            implementation.execution,
            recipe="handoff",
        ),
    )
    payload = _resume_identity_payload(
        corpus.latest_spec(),
        tier,
        slice_,
        handoff,
        tool_authority_digest="sha256:" + "a" * 64,
    )
    identity_ref = corpus.put_artifact(
        json.dumps(payload, sort_keys=True).encode("utf-8"),
        kind="resume_identities",
        name_hint="rejected-handoff.json",
    )
    handoff = dataclasses.replace(
        handoff,
        artifacts={
            **(handoff.artifacts or {}),
            "resume_identity": identity_ref,
        },
    )
    committed = corpus.commit(
        implementations=(handoff,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=dataclasses.replace(
                handoff.execution,
                recipe="tier",
                run_id="tier-test",
            ),
        ),
    )
    return handoff, str(committed.refs.implementation_refs[0])


def _promote_current_head_as_base(corpus, repo: Path) -> str:
    current_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        project_context=dataclasses.replace(
            latest.project_context,
            base_commit=current_base,
        ),
    ))
    return current_base


def _land_and_advance_base(
    tmp_path: Path,
    *,
    commit_landing: bool,
):
    """Create one old landing, then advance Git/Spec with or without it."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity

    landed_slice = Slice(
        id="landed",
        directive=ImplementationDirective(
            slice_id="landed",
            intent="land shared path before a base advance",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = dataclasses.replace(
        landed_slice,
        id="failed",
        directive=dataclasses.replace(
            landed_slice.directive,
            slice_id="failed",
            intent="fail on the shared path after a base advance",
        ),
    )
    landed_tier = Tier(tier_id="t0", slices=(landed_slice,), depends_on=())
    failed_tier = Tier(
        tier_id="t1",
        slices=(failed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, old_base = _real_multi_setup(
        tmp_path,
        tiers=(landed_tier, failed_tier),
        files={"foo.txt": "base\n", "unrelated.txt": "u0\n"},
    )
    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    landed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=old_base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="landed",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        landed,
    )
    landed_commit = corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    (repo / "unrelated.txt").write_text("u1\n", encoding="utf-8")
    add_command = (
        ["git", "add", "-A"]
        if commit_landing
        else ["git", "add", "--", "unrelated.txt"]
    )
    subprocess.run(
        add_command,
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "advance base"],
        cwd=repo,
        check=True,
    )
    current_base = _promote_current_head_as_base(corpus, repo)
    return (
        repo,
        corpus,
        failed_tier,
        failed_slice,
        current_base,
        str(landed_commit.refs.implementation_refs[0]),
    )


def test_empty_corpus_noop(tmp_path: Path) -> None:
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path)
    corpus = _mock_corpus([], {}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.reaped_anything is False
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert report.failures == ()


def test_oversized_reaper_diff_is_rejected_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diff above its bounded-read ceiling cannot reach Git cleanup."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("failed bytes\n", encoding="utf-8")
    diff = _capture_diff(repo)
    implementation = _real_impl(
        corpus,
        diff=diff,
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    implementation = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        implementation,
    )
    corpus.commit(implementations=(implementation,))
    before_diff = _capture_diff(repo)
    before_version = corpus.latest_spec().version
    reverse_calls = 0

    def forbidden_reverse(*_args, **_kwargs):
        nonlocal reverse_calls
        reverse_calls += 1
        raise AssertionError("oversized diff reached mutation")

    monkeypatch.setattr(_reaper, "_MAX_REAPER_DIFF_BYTES", len(diff) - 1)
    monkeypatch.setattr(_reaper, "_git_apply_reverse", forbidden_reverse)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any(
        "bounded Implementation diff artifact is unreadable" in failure
        for failure in report.failures
    ), report.failures
    assert reverse_calls == 0
    assert _capture_diff(repo) == before_diff
    assert corpus.latest_spec().version == before_version


@pytest.mark.parametrize("budget_kind", ("count", "bytes"))
def test_reaper_aggregate_artifact_budget_blocks_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_kind: str,
) -> None:
    """Multiple bounded diffs still share one finite invocation budget."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    first_slice = Slice(
        id="first",
        directive=ImplementationDirective(
            slice_id="first",
            intent="fail on first.txt",
            files_in_scope=("first.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    second_slice = Slice(
        id="second",
        directive=ImplementationDirective(
            slice_id="second",
            intent="fail on second.txt",
            files_in_scope=("second.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    tier = Tier(
        tier_id="t0",
        slices=(first_slice, second_slice),
        depends_on=(),
    )
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files={"first.txt": "first base\n", "second.txt": "second base\n"},
    )
    (repo / "first.txt").write_text("first failed\n", encoding="utf-8")
    first_diff = _capture_diff(repo)
    (repo / "first.txt").write_text("first base\n", encoding="utf-8")
    (repo / "second.txt").write_text("second failed\n", encoding="utf-8")
    second_diff = _capture_diff(repo)
    (repo / "second.txt").write_text("second base\n", encoding="utf-8")
    first = _real_impl(
        corpus,
        diff=first_diff,
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="first",
        files_modified=("first.txt",),
    )
    second = _real_impl(
        corpus,
        diff=second_diff,
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:01+00:00",
        slice_id="second",
        files_modified=("second.txt",),
    )
    spec = corpus.latest_spec()
    first = attach_resume_identity(corpus, spec, tier, first_slice, first)
    second = attach_resume_identity(corpus, spec, tier, second_slice, second)
    corpus.commit(implementations=(first, second))
    (repo / "first.txt").write_text("first failed\n", encoding="utf-8")
    (repo / "second.txt").write_text("second failed\n", encoding="utf-8")
    before_diff = _capture_diff(repo)
    before_version = corpus.latest_spec().version
    reverse_calls = 0

    def forbidden_reverse(*_args, **_kwargs):
        nonlocal reverse_calls
        reverse_calls += 1
        raise AssertionError("aggregate-exhausted diff reached mutation")

    if budget_kind == "count":
        monkeypatch.setattr(_reaper, "_MAX_REAPER_ARTIFACTS", 1)
        expected = "aggregate count budget exhausted"
    else:
        monkeypatch.setattr(
            _reaper,
            "_MAX_REAPER_ARTIFACT_AGGREGATE_BYTES",
            len(first_diff) + len(second_diff) - 1,
        )
        expected = "aggregate byte budget exhausted"
    monkeypatch.setattr(_reaper, "_git_apply_reverse", forbidden_reverse)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any(expected in failure for failure in report.failures), report.failures
    assert reverse_calls == 0
    assert _capture_diff(repo) == before_diff
    assert corpus.latest_spec().version == before_version


def test_reaper_landed_resume_identity_uses_bounded_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical lifecycle acceptance cannot read an oversized identity."""

    from assay import LifecycleEvent
    from crucible.workflows import _partition, _reaper

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("landed bytes\n", encoding="utf-8")
    diff = _capture_diff(repo)
    implementation = _real_impl(
        corpus,
        diff=diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        implementation,
    )
    oversized_ref = corpus.put_artifact(
        b"x" * (_partition._RESUME_IDENTITY_MAX_BYTES + 1),
        kind="resume_identities",
    )
    implementation = dataclasses.replace(
        implementation,
        artifacts={
            **dict(implementation.artifacts or {}),
            "resume_identity": oversized_ref,
        },
    )
    corpus.commit(implementations=(implementation,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=f"tier {tier.tier_id}: complete",
            affected_versions=(version,),
            execution=implementation.execution,
        ),
    )
    before_diff = _capture_diff(repo)
    before_version = corpus.latest_spec().version
    observed: list[tuple[str, int]] = []
    reverse_calls = 0
    real_bounded = corpus.get_artifact_bounded

    def observe_bounded(ref, *, max_bytes):
        observed.append((str(ref), max_bytes))
        return real_bounded(ref, max_bytes=max_bytes)

    def forbid_unbounded(*_args, **_kwargs):
        raise AssertionError("reaper used unbounded artifact read")

    def forbid_reverse(*_args, **_kwargs):
        nonlocal reverse_calls
        reverse_calls += 1
        raise AssertionError("oversized resume identity reached Git mutation")

    monkeypatch.setattr(corpus, "get_artifact_bounded", observe_bounded)
    monkeypatch.setattr(corpus, "get_artifact", forbid_unbounded)
    monkeypatch.setattr(_reaper, "_git_apply_reverse", forbid_reverse)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures
    assert (str(oversized_ref), _partition._RESUME_IDENTITY_MAX_BYTES) in observed
    assert reverse_calls == 0
    assert _capture_diff(repo) == before_diff
    assert corpus.latest_spec().version == before_version


@pytest.mark.parametrize("budget_kind", ("count", "bytes"))
def test_reaper_landed_identity_shares_aggregate_budget_with_diff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    budget_kind: str,
) -> None:
    """Lifecycle identity and diff reads consume the same reaper budget."""

    from assay import LifecycleEvent
    from crucible.workflows import _partition, _reaper

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("landed bytes\n", encoding="utf-8")
    diff = _capture_diff(repo)
    implementation = _real_impl(
        corpus,
        diff=diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    implementation = _partition.attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        implementation,
    )
    identity_ref = implementation.artifacts["resume_identity"]
    identity = corpus.get_artifact_bounded(
        identity_ref,
        max_bytes=_partition._RESUME_IDENTITY_MAX_BYTES,
    )
    corpus.commit(implementations=(implementation,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=f"tier {tier.tier_id}: complete",
            affected_versions=(version,),
            execution=implementation.execution,
        ),
    )
    before_diff = _capture_diff(repo)
    before_version = corpus.latest_spec().version
    reverse_calls = 0

    if budget_kind == "count":
        monkeypatch.setattr(_reaper, "_MAX_REAPER_ARTIFACTS", 1)
        expected = "aggregate count budget exhausted"
    else:
        monkeypatch.setattr(
            _reaper,
            "_MAX_REAPER_ARTIFACT_AGGREGATE_BYTES",
            len(identity) + len(diff) - 1,
        )
        expected = "aggregate byte budget exhausted"

    def forbid_unbounded(*_args, **_kwargs):
        raise AssertionError("reaper used unbounded artifact read")

    def forbid_reverse(*_args, **_kwargs):
        nonlocal reverse_calls
        reverse_calls += 1
        raise AssertionError("aggregate-exhausted identity reached Git mutation")

    monkeypatch.setattr(corpus, "get_artifact", forbid_unbounded)
    monkeypatch.setattr(_reaper, "_git_apply_reverse", forbid_reverse)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert any(expected in failure for failure in report.failures), report.failures
    assert reverse_calls == 0
    assert _capture_diff(repo) == before_diff
    assert corpus.latest_spec().version == before_version


@pytest.mark.parametrize(
    ("ref_field", "expected_message"),
    (
        (
            "implementation_refs",
            "Spec lineage repeats attached Implementation ref",
        ),
        (
            "lifecycle_event_refs",
            "Spec lineage repeats attached LifecycleEvent ref",
        ),
    ),
)
def test_duplicate_attached_ref_blocks_recovery(
    tmp_path: Path,
    ref_field: str,
    expected_message: str,
) -> None:
    """A duplicate content ref cannot rewrite append-order chronology."""
    from assay import LifecycleEvent
    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    repo, corpus, _tier, _slice, base = _real_setup(tmp_path)
    if ref_field == "implementation_refs":
        implementation = _real_impl(
            corpus,
            diff=b"",
            classification="success",
            base_commit=base,
            timestamp="2026-08-06T00:00:00+00:00",
            files_modified=(),
        )
        committed = corpus.commit(implementations=(implementation,))
        duplicate_ref = committed.refs.implementation_refs[0]
    else:
        committed = corpus.commit(
            lifecycle_event_factory=lambda version: LifecycleEvent(
                kind="failed",
                reason="duplicate chronology fixture",
                affected_versions=(version,),
                execution=exec_ctx_now("test"),
            ),
        )
        duplicate_ref = committed.refs.lifecycle_event_refs[0]
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        **{
            ref_field: (*getattr(latest, ref_field), duplicate_ref),
        },
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(expected_message in failure for failure in report.failures)


def test_direct_latest_manifest_cannot_reorder_lifecycle_authority(
    tmp_path: Path,
) -> None:
    """Read-time lineage validation rejects a canonical-file reorder."""
    from assay import LifecycleEvent
    from assay.canonical import serialize
    from assay.corpus.store import Corpus

    from crucible.workflows._reaper import reap_orphaned_dirt
    from crucible.workflows._run import exec_ctx_now

    repo, corpus, _tier, _slice, _base = _real_setup(tmp_path)
    for label in ("first", "second"):
        corpus.commit(
            lifecycle_event_factory=lambda version, label=label: LifecycleEvent(
                kind="failed",
                reason=f"chronology {label}",
                affected_versions=(version,),
                execution=exec_ctx_now("test"),
            ),
        )
    latest = corpus.latest_spec()
    corrupted = dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        lifecycle_event_refs=tuple(reversed(latest.lifecycle_event_refs)),
    )
    corrupted_hash = hash_record(corrupted)
    manifest = (
        Path(corpus.spec_dir)
        / "versions"
        / f"v{corrupted.version}_{corrupted_hash.split(':', 1)[-1][:12]}.json"
    )
    manifest.write_bytes(serialize(corrupted))
    reloaded = Corpus(corpus.spec_dir)
    before = (repo / "foo.txt").read_bytes()

    report = reap_orphaned_dirt(reloaded, repo)

    assert any(
        "rewrites append-only lifecycle_event_refs history" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_bytes() == before


def test_old_manifest_rewrite_cannot_launder_cleanup_authority(
    tmp_path: Path,
) -> None:
    """Every historical manifest retains its own content/hash identity."""
    from assay.corpus.store import ConcurrentSpecError, Corpus
    from crucible.workflows._partition import attach_resume_identity

    repo, corpus, tier, slice_, base = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("accepted-a\n", encoding="utf-8")
    diff = _capture_diff(repo)
    success = _real_impl(
        corpus,
        diff=diff,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    success = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        success,
    )
    committed = corpus.commit(implementations=(success,))

    # Store a different, unattached record whose reverse diff would erase the
    # accepted live bytes. Then rewrite old v1 in place, retaining its original
    # hash-derived filename, to substitute that failure as cleanup authority.
    failure = dataclasses.replace(
        success,
        classification="hard_fail_max_turns",
        execution=dataclasses.replace(
            success.execution,
            timestamp="2026-08-06T00:01:00+00:00",
        ),
    )
    failure_ref = corpus.put_record(failure, kind="implementations")
    v1 = committed.spec
    v1_manifest = next(
        (Path(corpus.spec_dir) / "versions").glob("v1_*.json")
    )
    laundered_v1 = dataclasses.replace(
        v1,
        implementation_refs=(failure_ref,),
    )
    v1_manifest.write_bytes(serialize(laundered_v1))

    # Assay now validates each historical manifest's canonical filename while
    # loading the full chain. The corrupted v1 is rejected before another
    # writer or Crucible's reaper can derive mutation authority from it.
    reloaded = Corpus(corpus.spec_dir)
    with pytest.raises(
        ConcurrentSpecError,
        match="content does not match its exact canonical version/hash filename",
    ):
        reloaded.latest_spec()

    assert (repo / "foo.txt").read_text(encoding="utf-8") == "accepted-a\n"


def test_crashed_rollback_reaped(tmp_path: Path) -> None:
    """hard_fail Implementation with dirt on disk → git apply -R."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    # Simulate the killed implementer's dirt: foo.txt modified on disk.
    (tmp_path / "foo.txt").write_text("dirty crash content")
    crash_diff = _capture_diff(tmp_path)
    impl = _mock_impl("s0", "hard_fail_max_turns", "sha256:s0diff")
    corpus = _mock_corpus([impl], {"sha256:s0diff": crash_diff}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.crashed_rollbacks_reaped == ("s0",)
    assert report.cleanup_settlement_ref is not None
    assert (tmp_path / "foo.txt").read_text() == "original"

    repeated = reap_orphaned_dirt(corpus, tmp_path)
    assert repeated.failures == ()
    assert repeated.crashed_rollbacks_reaped == ()
    assert any("already settled" in item for item in repeated.legacy_mutations_skipped)


@pytest.mark.parametrize(
    "classification",
    (
        "dev_verify_failed",
        "dev_verify_unavailable",
        "runner_missing",
        "smoke_unavailable",
    ),
)
def test_tier_build_outcome_failure_is_reaped(
    tmp_path: Path,
    classification: str,
) -> None:
    """Tier's terminal gate classifications remain rollback authorities."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original", encoding="utf-8")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("dirty", encoding="utf-8")
    impl = _mock_impl("s0", classification, "sha256:s0diff")
    corpus = _mock_corpus(
        [impl],
        {"sha256:s0diff": _capture_diff(tmp_path)},
        cwd=tmp_path,
    )

    report = reap_orphaned_dirt(corpus, tmp_path)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ("s0",)
    assert (tmp_path / "foo.txt").read_text(encoding="utf-8") == "original"


def test_direct_reaper_holds_workspace_lock_through_recovery_and_cas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _reaper
    from crucible.workflows._mutation_journal import _LOCK_STATE

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("base", encoding="utf-8")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("dirty", encoding="utf-8")
    impl = _mock_impl("s0", "hard_fail_max_turns", "sha256:s0diff")
    corpus = _mock_corpus(
        [impl],
        {"sha256:s0diff": _capture_diff(tmp_path)},
        cwd=tmp_path,
    )
    observed: list[str] = []
    original_recovery = _reaper._recover_mutation_journals_locked

    def _recovery(*args, **kwargs):
        assert getattr(_LOCK_STATE, "held", {})
        observed.append("recovery")
        return original_recovery(*args, **kwargs)

    original_commit = corpus.commit.side_effect

    def _commit_locked(*args, **kwargs):
        assert getattr(_LOCK_STATE, "held", {})
        observed.append("commit")
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(
        _reaper,
        "_recover_mutation_journals_locked",
        _recovery,
    )
    corpus.commit.side_effect = _commit_locked

    report = _reaper.reap_orphaned_dirt(corpus, tmp_path)

    assert report.failures == ()
    assert observed == ["recovery", "commit"]


def test_skip_worthy_diff_left_alone(tmp_path: Path) -> None:
    """success-class Implementation with dirt on disk is left as-is."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("success-landed content")
    diff = _capture_diff(tmp_path)
    impl = _mock_impl("s0", "success", "sha256:s0diff")
    corpus = _mock_corpus([impl], {"sha256:s0diff": diff}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.crashed_rollbacks_reaped == ()
    # The success Implementation's diff stays on disk.
    assert (tmp_path / "foo.txt").read_text() == "success-landed content"


def test_scope_escape_tracked_file_reverted_not_deleted(tmp_path: Path) -> None:
    """Regression: a TRACKED file recorded as a scope_escape must be
    reverted to HEAD, NOT unlinked. The prior reaper unconditionally
    deleted scope-escape paths, removing legitimate tracked sources (an
    under-scoped D5 cleanup recorded tracked rtx.* headers as escapes)
    and breaking the build with missing-file errors."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "tracked.h").write_text("// canonical content\n")
    _commit(tmp_path, "initial")
    # Simulate the rogue out-of-scope edit left in the tree.
    (tmp_path / "tracked.h").write_text("// rogue out-of-scope edit\n")
    impl = _mock_impl(
        "s_d5",
        "hard_fail_scope_escape",
        "sha256:empty",
        scope_escapes=("tracked.h",),
    )
    corpus = _mock_corpus([impl], {"sha256:empty": b""}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    # Still present AND reverted to HEAD — not deleted.
    assert (tmp_path / "tracked.h").exists()
    assert (tmp_path / "tracked.h").read_text() == "// canonical content\n"
    assert "tracked.h" in report.scope_escapes_deleted


def test_scope_escape_untracked_file_deleted(tmp_path: Path) -> None:
    """An UNtracked scope-escape file is still unlinked (unchanged)."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.h").write_text("anchor\n")
    _commit(tmp_path, "initial")
    (tmp_path / "rogue_untracked.cpp").write_text("// rogue untracked\n")
    impl = _mock_impl(
        "s_d5",
        "hard_fail_scope_escape",
        "sha256:empty",
        scope_escapes=("rogue_untracked.cpp",),
    )
    corpus = _mock_corpus([impl], {"sha256:empty": b""}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert not (tmp_path / "rogue_untracked.cpp").exists()
    assert "rogue_untracked.cpp" in report.scope_escapes_deleted


def test_diff_does_not_apply_in_reverse(tmp_path: Path) -> None:
    """A stale diff that doesn't match the tree → recorded as skipped."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    # Capture a diff against a DIFFERENT modification, then revert.
    (tmp_path / "foo.txt").write_text("ghost content")
    stale_diff = _capture_diff(tmp_path)
    (tmp_path / "foo.txt").write_text("original")  # tree no longer matches
    impl = _mock_impl("s0", "hard_fail_max_turns", "sha256:s0diff")
    corpus = _mock_corpus([impl], {"sha256:s0diff": stale_diff}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.crashed_rollbacks_reaped == ()
    assert report.crashed_rollbacks_skipped == ("s0",)


def test_scope_escape_file_deleted(tmp_path: Path) -> None:
    """Any Implementation's scope_escapes entry that exists → deleted."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path)
    (tmp_path / "rogue.txt").write_text("written outside scope")
    impl = _mock_impl(
        "s0",
        "success",
        "sha256:s0diff",
        scope_escapes=("rogue.txt",),
    )
    corpus = _mock_corpus([impl], {"sha256:s0diff": b""}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert "rogue.txt" in report.scope_escapes_deleted
    assert not (tmp_path / "rogue.txt").exists()


def test_scope_escape_missing_file_skipped(tmp_path: Path) -> None:
    """scope_escapes entry pointing at a file that doesn't exist → no-op."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path)
    impl = _mock_impl(
        "s0",
        "hard_fail_max_turns",
        "sha256:s0diff",
        scope_escapes=("never_existed.txt",),
    )
    corpus = _mock_corpus([impl], {"sha256:s0diff": b""}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.scope_escapes_deleted == ()
    assert report.failures == ()


def test_scope_escape_nonempty_dir_left_in_place(tmp_path: Path) -> None:
    """Non-empty dir in scope_escapes refuses to recurse."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path)
    (tmp_path / "rogue_dir").mkdir()
    (tmp_path / "rogue_dir" / "child").write_text("contents")
    impl = _mock_impl(
        "s0",
        "success",
        "sha256:s0diff",
        scope_escapes=("rogue_dir",),
    )
    corpus = _mock_corpus([impl], {"sha256:s0diff": b""}, cwd=tmp_path)
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.scope_escapes_deleted == ()  # not deleted
    assert (tmp_path / "rogue_dir" / "child").exists()


def test_cumulative_newer_failure_without_journal_fails_closed(
    tmp_path: Path,
) -> None:
    """A later Assay diff is cumulative, not an attempt-local rollback."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("dirty crash content")
    crash_diff = _capture_diff(tmp_path)
    old_success = _mock_impl(
        "s0",
        "success",
        "sha256:old",
        ts="2026-05-21T10:00:00+00:00",
    )
    new_fail = _mock_impl(
        "s0",
        "hard_fail_max_turns",
        "sha256:new",
        ts="2026-05-21T12:00:00+00:00",
    )
    corpus = _mock_corpus(
        [old_success, new_fail],
        {"sha256:old": b"", "sha256:new": crash_diff},
        cwd=tmp_path,
    )
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.crashed_rollbacks_reaped == ()
    assert any("cumulative" in failure for failure in report.failures)
    assert (tmp_path / "foo.txt").read_text() == "dirty crash content"


def test_spec_order_and_exact_landing_outrank_hostile_artifact_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A different slice's older failure cannot reap accepted shared bytes."""

    from assay import LifecycleEvent
    from assay.corpus.store import Corpus
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")
    (repo / "old.txt").write_text("old scope\n", encoding="utf-8")
    _commit(repo, "base")
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="older failed mutation",
            files_in_scope=("old.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    landed_slice = Slice(
        id="landed",
        directive=ImplementationDirective(
            slice_id="landed",
            intent="accept shared bytes",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    landed_tier = Tier(tier_id="t1", slices=(landed_slice,), depends_on=())
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus = Corpus(corpus_dir)
    corpus.put_spec(
        Spec(
            version=0,
            parent=None,
            root=None,
            schema_version=1,
            project_context=ProjectContext(
                project="reaper-overlap-test",
                repo_url="https://example.invalid/reaper.git",
                base_commit=base_commit,
                branch=None,
                submodule_path=None,
                runners={},
            ),
            execution=None,
            intent="protect a different slice's accepted shared path",
            constraints=(),
            done_criteria=(),
            plan_policy=None,
            depends_on=(),
            tier_dag=TierDAG(tiers=(old_tier, landed_tier)),
            implementation_refs=(),
            verdict_refs=(),
            measurement_refs=(),
            lifecycle_event_refs=(),
        )
    )
    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    reverse_applicable = _capture_diff(repo)
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")
    older_failure = _real_impl(
        corpus,
        diff=reverse_applicable,
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2099-01-01T00:00:00+00:00",
        scope_escapes=("foo.txt",),
        slice_id="old",
    )
    older_failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        older_failure,
    )
    failed_commit = corpus.commit(implementations=(older_failure,))

    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    success = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2000-01-01T00:00:00+00:00",
        slice_id="landed",
    )
    success = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        success,
    )
    landed_commit = corpus.commit(
        implementations=(success,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t1: complete",
            affected_versions=(version,),
            execution=dataclasses.replace(
                success.execution,
                recipe="tier",
                run_id="tier-test",
            ),
        ),
    )
    failure_ref = failed_commit.refs.implementation_refs[0]
    success_ref = landed_commit.refs.implementation_refs[0]
    monkeypatch.setattr(
        corpus,
        "list_artifacts",
        lambda *, kind: iter((success_ref, failure_ref)),
    )
    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert any("superseded" in item for item in report.legacy_mutations_skipped)
    assert any("protected" in item for item in report.legacy_mutations_skipped)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_newer_failure_cannot_reverse_older_landed_path(
    tmp_path: Path,
) -> None:
    """A newer cumulative failure needs its journal to restore old authority."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    landed_slice = Slice(
        id="landed",
        directive=ImplementationDirective(
            slice_id="landed",
            intent="land shared path",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = dataclasses.replace(
        landed_slice,
        id="failed",
        directive=dataclasses.replace(
            landed_slice.directive,
            slice_id="failed",
            intent="later failed shared-path attempt",
        ),
    )
    landed_tier = Tier(tier_id="t0", slices=(landed_slice,), depends_on=())
    failed_tier = Tier(tier_id="t1", slices=(failed_slice,), depends_on=("t0",))
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(landed_tier, failed_tier),
        files={"foo.txt": "base\n"},
    )

    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    landed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="landed",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        landed,
    )
    corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    (repo / "foo.txt").write_text("failed\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "positive Implementation" in failure and "not live" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "failed\n"


@pytest.mark.parametrize("stale_tool_authority", [False, True])
def test_cumulative_failure_matching_older_landed_postimage_is_non_mutating(
    tmp_path: Path,
    stale_tool_authority: bool,
) -> None:
    """A cumulative failure cannot make an accepted postimage its mutation."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    landed_slice = Slice(
        id="landed",
        directive=ImplementationDirective(
            slice_id="landed",
            intent="land the dependency postimage",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = dataclasses.replace(
        landed_slice,
        id="failed",
        directive=dataclasses.replace(
            landed_slice.directive,
            slice_id="failed",
            intent="record a later cumulative failure without local writes",
        ),
    )
    landed_tier = Tier(tier_id="t0", slices=(landed_slice,), depends_on=())
    failed_tier = Tier(
        tier_id="t1",
        slices=(failed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(landed_tier, failed_tier),
        files={"foo.txt": "base\n"},
    )

    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    shared_diff = _capture_diff(repo)
    landed = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="landed",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        landed,
    )
    corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    recorded_tool_digest = "sha256:" + "a" * 64
    failed = _real_impl(
        corpus,
        diff=shared_diff,
        classification="hard_fail_implementer_blocked",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
        tool_authority_digest=recorded_tool_digest,
    )
    corpus.commit(implementations=(failed,))
    version_before = corpus.latest_spec().version

    report = reap_orphaned_dirt(
        corpus,
        repo,
        tool_authority_digests={
            "failed": (
                "sha256:" + "b" * 64
                if stale_tool_authority
                else recorded_tool_digest
            ),
        },
    )

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert corpus.latest_spec().version == version_before
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"
    if stale_tool_authority:
        assert any(
            "no unprotected exact postimage is present" in diagnostic
            for diagnostic in report.legacy_mutations_skipped
        )
    else:
        assert any(
            "matches exact surviving path authority" in diagnostic
            for diagnostic in report.legacy_mutations_skipped
        )


def test_stale_cumulative_failure_with_different_postimage_still_blocks(
    tmp_path: Path,
) -> None:
    """Older authority cannot cover a stale failure's different object ID."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    landed_slice = Slice(
        id="landed",
        directive=ImplementationDirective(
            slice_id="landed",
            intent="land the dependency postimage",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = dataclasses.replace(
        landed_slice,
        id="failed",
        directive=dataclasses.replace(
            landed_slice.directive,
            slice_id="failed",
            intent="record a distinct stale failure postimage",
        ),
    )
    landed_tier = Tier(tier_id="t0", slices=(landed_slice,), depends_on=())
    failed_tier = Tier(
        tier_id="t1",
        slices=(failed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(landed_tier, failed_tier),
        files={"foo.txt": "base\n"},
    )

    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    landed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="landed",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        landed,
    )
    corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    (repo / "foo.txt").write_text("failed\n", encoding="utf-8")
    failed_diff = _capture_diff(repo)
    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=failed_diff,
        classification="hard_fail_implementer_blocked",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
        tool_authority_digest="sha256:" + "a" * 64,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(
        corpus,
        repo,
        tool_authority_digests={"failed": "sha256:" + "b" * 64},
    )

    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_unrelated_base_advance_does_not_absorb_dirty_landing(
    tmp_path: Path,
) -> None:
    """HEAD advancement alone cannot retire old dirty-worktree authority."""

    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    (
        repo,
        corpus,
        failed_tier,
        failed_slice,
        current_base,
        _landed_ref,
    ) = _land_and_advance_base(tmp_path, commit_landing=False)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=current_base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert not any(
        "absorbed into current base" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_unrelated_base_advance_without_later_record_blocks_startup(
    tmp_path: Path,
) -> None:
    """Unabsorbed historical dirt fails even before a new failure exists."""

    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, *_rest = _land_and_advance_base(
        tmp_path,
        commit_landing=False,
    )

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_fresh_current_plan_recapture_supersedes_unabsorbed_landing(
    tmp_path: Path,
) -> None:
    """A newer exact retained postimage can supersede old landed dirt."""

    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    (
        repo,
        corpus,
        _failed_tier,
        _failed_slice,
        current_base,
        landed_ref,
    ) = _land_and_advance_base(tmp_path, commit_landing=False)
    prior_spec = corpus.latest_spec()
    prior_tier = next(
        tier for tier in prior_spec.tier_dag.tiers if tier.tier_id == "t0"
    )
    prior_slice = next(
        slice_ for slice_ in prior_tier.slices if slice_.id == "landed"
    )
    current_slice = dataclasses.replace(
        prior_slice,
        directive=dataclasses.replace(
            prior_slice.directive,
            intent="fresh current-base recapture under revised authority",
        ),
    )
    current_tier = dataclasses.replace(
        prior_tier,
        slices=tuple(
            current_slice if slice_.id == "landed" else slice_
            for slice_ in prior_tier.slices
        ),
    )
    corpus.put_spec(dataclasses.replace(
        prior_spec,
        version=prior_spec.version + 1,
        parent=ContentHash(hash_record(prior_spec)),
        tier_dag=TierDAG(tiers=tuple(
            current_tier if tier.tier_id == "t0" else tier
            for tier in prior_spec.tier_dag.tiers
        )),
    ))
    current_spec = corpus.latest_spec()
    current_tier = next(
        tier for tier in current_spec.tier_dag.tiers if tier.tier_id == "t0"
    )
    current_slice = next(
        slice_ for slice_ in current_tier.slices if slice_.id == "landed"
    )
    recaptured = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=current_base,
        timestamp="2026-08-06T00:02:00+00:00",
        slice_id="landed",
    )
    recaptured = attach_resume_identity(
        corpus,
        current_spec,
        current_tier,
        current_slice,
        recaptured,
    )
    recaptured_commit = corpus.commit(implementations=(recaptured,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"
    assert any(
        landed_ref in diagnostic
        and "no unprotected exact postimage is present" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert any(
        str(recaptured_commit.refs.implementation_refs[0]) in diagnostic
        and "is retained" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )


def test_committed_landing_is_absorbed_by_new_base_before_cleanup(
    tmp_path: Path,
) -> None:
    """An exact old postimage in the new base no longer blocks recovery."""

    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    (
        repo,
        corpus,
        failed_tier,
        failed_slice,
        current_base,
        landed_ref,
    ) = _land_and_advance_base(tmp_path, commit_landing=True)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"
    (repo / "foo.txt").write_text("failed\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=current_base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ("failed",)
    assert any(
        landed_ref in diagnostic
        and "absorbed into current base" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_absorption_uses_latest_ref_for_overlapping_landed_slices(
    tmp_path: Path,
) -> None:
    """Cumulative same-tier records compose by authoritative ref order."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    slice_a = Slice(
        id="a",
        directive=ImplementationDirective(
            slice_id="a",
            intent="write the first accepted shared-path state",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    slice_b = dataclasses.replace(
        slice_a,
        id="b",
        directive=dataclasses.replace(
            slice_a.directive,
            slice_id="b",
            intent="supersede the shared path in the same tier",
        ),
    )
    failed_slice = dataclasses.replace(
        slice_a,
        id="failed",
        directive=dataclasses.replace(
            slice_a.directive,
            slice_id="failed",
            intent="leave a current-base failure on the shared path",
        ),
    )
    landed_tier = Tier(
        tier_id="t0",
        slices=(slice_a, slice_b),
        depends_on=(),
    )
    failed_tier = Tier(
        tier_id="t1",
        slices=(failed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, old_base = _real_multi_setup(
        tmp_path,
        tiers=(landed_tier, failed_tier),
        files={"foo.txt": "base\n"},
    )

    landed_refs: list[str] = []
    for index, (slice_, content) in enumerate((
        (slice_a, "a\n"),
        (slice_b, "b\n"),
    )):
        (repo / "foo.txt").write_text(content, encoding="utf-8")
        implementation = _real_impl(
            corpus,
            diff=_capture_diff(repo),
            classification="success",
            base_commit=old_base,
            timestamp=f"2026-08-06T00:0{index}:00+00:00",
            slice_id=slice_.id,
        )
        implementation = attach_resume_identity(
            corpus,
            corpus.latest_spec(),
            landed_tier,
            slice_,
            implementation,
        )
        committed = corpus.commit(
            implementations=(implementation,),
            **(
                {
                    "lifecycle_event_factory": lambda version, execution=(
                        implementation.execution
                    ): LifecycleEvent(
                        kind="landed",
                        reason="tier t0: complete",
                        affected_versions=(version,),
                        execution=execution,
                    )
                }
                if slice_.id == "b"
                else {}
            ),
        )
        landed_refs.append(str(committed.refs.implementation_refs[0]))

    _commit(repo, "integrate final shared-path state")
    current_base = _promote_current_head_as_base(corpus, repo)
    (repo / "foo.txt").write_text("failed\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=current_base,
        timestamp="2026-08-06T00:02:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ("failed",)
    assert all(
        any(ref in diagnostic for diagnostic in report.legacy_mutations_skipped)
        for ref in landed_refs
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "b\n"


@pytest.mark.parametrize(
    "shared_path",
    (True, False),
    ids=("overlapping-path", "disjoint-paths"),
)
def test_absorption_composes_landed_events_across_base_generations(
    tmp_path: Path,
    shared_path: bool,
) -> None:
    """Global lifecycle order composes old bases before one exact proof."""

    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    first_path = "foo.txt" if shared_path else "a.txt"
    second_path = "foo.txt" if shared_path else "b.txt"

    def _slice(slice_id: str, path: str) -> Slice:
        return Slice(
            id=slice_id,
            directive=ImplementationDirective(
                slice_id=slice_id,
                intent=f"land {path} in one base generation",
                files_in_scope=(path,),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={"max_files": 1},
            ),
        )

    first_slice = _slice("first", first_path)
    second_slice = _slice("second", second_path)
    failed_slice = _slice("failed", second_path)
    first_tier = Tier(tier_id="t0", slices=(first_slice,), depends_on=())
    second_tier = Tier(
        tier_id="t1",
        slices=(second_slice,),
        depends_on=("t0",),
    )
    failed_tier = Tier(
        tier_id="t2",
        slices=(failed_slice,),
        depends_on=("t1",),
    )
    initial_files = (
        {"foo.txt": "base\n"}
        if shared_path
        else {"a.txt": "a-base\n", "b.txt": "b-base\n"}
    )
    repo, corpus, first_base = _real_multi_setup(
        tmp_path,
        tiers=(first_tier, second_tier, failed_tier),
        files=initial_files,
    )

    (repo / first_path).write_text("first\n", encoding="utf-8")
    first = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=first_base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="first",
        files_modified=(first_path,),
    )
    first = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        first_tier,
        first_slice,
        first,
    )
    first_commit = corpus.commit(
        implementations=(first,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=first.execution,
        ),
    )
    _commit(repo, "integrate first generation")
    second_base = _promote_current_head_as_base(corpus, repo)

    (repo / second_path).write_text("second\n", encoding="utf-8")
    second = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=second_base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="second",
        files_modified=(second_path,),
    )
    second = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        second_tier,
        second_slice,
        second,
    )
    second_commit = corpus.commit(
        implementations=(second,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t1: complete",
            affected_versions=(version,),
            execution=second.execution,
        ),
    )
    _commit(repo, "integrate second generation")
    current_base = _promote_current_head_as_base(corpus, repo)

    (repo / second_path).write_text("failed\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=current_base,
        timestamp="2026-08-06T00:02:00+00:00",
        slice_id="failed",
        files_modified=(second_path,),
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    landed_refs = {
        str(first_commit.refs.implementation_refs[0]),
        str(second_commit.refs.implementation_refs[0]),
    }
    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ("failed",)
    assert all(
        any(ref in diagnostic for diagnostic in report.legacy_mutations_skipped)
        for ref in landed_refs
    )
    assert (repo / first_path).read_text(encoding="utf-8") == (
        "second\n" if shared_path else "first\n"
    )
    assert (repo / second_path).read_text(encoding="utf-8") == "second\n"


def test_failed_scope_escape_cannot_erase_retained_prefix(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    prefix_slice = Slice(
        id="prefix",
        directive=ImplementationDirective(
            slice_id="prefix",
            intent="retain prefix bytes",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = Slice(
        id="failed",
        directive=ImplementationDirective(
            slice_id="failed",
            intent="record a sibling scope escape",
            files_in_scope=("b.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    tier = Tier(tier_id="t0", slices=(prefix_slice, failed_slice), depends_on=())
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files={"a.txt": "a-base\n", "b.txt": "b-base\n"},
    )
    (repo / "a.txt").write_text("a-retained\n", encoding="utf-8")
    retained = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="prefix",
        files_modified=("a.txt",),
    )
    retained = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        prefix_slice,
        retained,
    )
    corpus.commit(implementations=(retained,))
    escaped = _real_impl(
        corpus,
        diff=b"",
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
        files_modified=(),
        scope_escapes=("a.txt",),
    )
    escaped = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        failed_slice,
        escaped,
    )
    corpus.commit(implementations=(escaped,))

    report = reap_orphaned_dirt(corpus, repo)

    assert any("retained postimage" in failure for failure in report.failures)
    assert report.scope_escapes_deleted == ()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "a-retained\n"


def test_journal_backed_newer_failure_restores_exact_landed_preimage(
    tmp_path: Path,
) -> None:
    """Cumulative failure evidence is settled only by its exact journal."""

    from assay import LifecycleEvent
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("landed\n", encoding="utf-8")
    success = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2099-01-01T00:00:00+00:00",
    )
    success = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        success,
    )
    landed_commit = corpus.commit(
        implementations=(success,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=success.execution,
        ),
    )
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "newer-failure" / "mutations",
        slice_id="s0",
        scope=("foo.txt",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "foo.txt").write_text("failed\n", encoding="utf-8")
    newer_failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2000-01-01T00:00:00+00:00",
    )
    newer_failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        newer_failure,
    )
    journal.prepare(newer_failure, keep=False)
    failed_commit = corpus.commit(implementations=(newer_failure,))
    failure_ref = failed_commit.refs.implementation_refs[0]
    assert failure_ref != landed_commit.refs.implementation_refs[0]
    _unregister_live(journal.transaction_id)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ("s0",)
    assert report.crashed_rollbacks_reaped == ()
    assert any(failure_ref in item for item in report.legacy_mutations_skipped)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "landed\n"


def test_recovered_newer_sibling_does_not_invalidate_historical_landing(
    tmp_path: Path,
) -> None:
    """A negative later A record cannot erase sibling B's landed authority."""
    from assay import LifecycleEvent
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    slice_a = Slice(
        id="a",
        directive=ImplementationDirective(
            slice_id="a",
            intent="land a",
            files_in_scope=("a.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    slice_b = Slice(
        id="b",
        directive=ImplementationDirective(
            slice_id="b",
            intent="land foo",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    slice_c = dataclasses.replace(
        slice_b,
        id="c",
        directive=dataclasses.replace(
            slice_b.directive,
            slice_id="c",
            intent="fail over landed foo",
        ),
    )
    tier0 = Tier(tier_id="t0", slices=(slice_a, slice_b), depends_on=())
    tier1 = Tier(tier_id="t1", slices=(slice_c,), depends_on=("t0",))
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier0, tier1),
        files={"a.txt": "a-base\n", "foo.txt": "foo-base\n"},
    )

    (repo / "a.txt").write_text("a-landed\n", encoding="utf-8")
    impl_a = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="a",
        files_modified=("a.txt",),
    )
    impl_a = attach_resume_identity(
        corpus, corpus.latest_spec(), tier0, slice_a, impl_a
    )
    corpus.commit(implementations=(impl_a,))

    (repo / "foo.txt").write_text("foo-landed\n", encoding="utf-8")
    impl_b = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="b",
        files_modified=("foo.txt",),
    )
    impl_b = attach_resume_identity(
        corpus, corpus.latest_spec(), tier0, slice_b, impl_b
    )
    corpus.commit(
        implementations=(impl_b,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=dataclasses.replace(impl_b.execution, recipe="tier"),
        ),
    )

    (repo / "foo.txt").write_text("foo-failed\n", encoding="utf-8")
    impl_c = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:02:00+00:00",
        slice_id="c",
        files_modified=("foo.txt",),
    )
    impl_c = attach_resume_identity(
        corpus, corpus.latest_spec(), tier1, slice_c, impl_c
    )
    corpus.commit(implementations=(impl_c,))

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "newer-a" / "mutations",
        slice_id="a",
        scope=("a.txt",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "a.txt").write_text("a-failed\n", encoding="utf-8")
    newer_a = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:03:00+00:00",
        slice_id="a",
        files_modified=("a.txt",),
    )
    newer_a = attach_resume_identity(
        corpus, corpus.latest_spec(), tier0, slice_a, newer_a
    )
    journal.prepare(newer_a, keep=False)
    corpus.commit(implementations=(newer_a,))
    _unregister_live(journal.transaction_id)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_recovered == ("a",)
    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "positive Implementation" in failure and "not live" in failure
        for failure in report.failures
    )
    assert (repo / "a.txt").read_text(encoding="utf-8") == "a-landed\n"
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "foo-failed\n"


def test_later_failure_cannot_erase_earlier_unproved_postimage(
    tmp_path: Path,
) -> None:
    """A cumulative legacy diff cannot cross an unproved slice preimage."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    pending_slice = Slice(
        id="pending",
        directive=ImplementationDirective(
            slice_id="pending",
            intent="retain pending gate bytes",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    failed_slice = dataclasses.replace(
        pending_slice,
        id="failed",
        directive=dataclasses.replace(
            pending_slice.directive,
            slice_id="failed",
            intent="later cumulative failure",
        ),
    )
    pending_tier = Tier(tier_id="t0", slices=(pending_slice,), depends_on=())
    failed_tier = Tier(
        tier_id="t1",
        slices=(failed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(pending_tier, failed_tier),
        files={"foo.txt": "base\n"},
    )

    (repo / "foo.txt").write_text("pending bytes\n", encoding="utf-8")
    pending = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="pending_gates",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="pending",
    )
    pending = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        pending_tier,
        pending_slice,
        pending,
    )
    corpus.commit(implementations=(pending,))

    (repo / "foo.txt").write_text("failed bytes\n", encoding="utf-8")
    failed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="failed",
    )
    failed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        failed_tier,
        failed_slice,
        failed,
    )
    corpus.commit(implementations=(failed,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "slice failed: failure diff overlaps earlier unproved "
        "Implementation evidence" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "failed bytes\n"


def test_recovered_later_failure_is_not_surviving_path_authority(
    tmp_path: Path,
) -> None:
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="old failed bytes",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    later_slice = dataclasses.replace(
        old_slice,
        id="later",
        directive=dataclasses.replace(
            old_slice.directive,
            slice_id="later",
            intent="later journal-backed failure",
        ),
    )
    tier = Tier(tier_id="t0", slices=(old_slice, later_slice), depends_on=())
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files={"foo.txt": "base\n"},
    )
    (repo / "foo.txt").write_text("old dirt\n", encoding="utf-8")
    old = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="old",
    )
    old = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        old_slice,
        old,
    )
    corpus.commit(implementations=(old,))

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "later" / "mutations",
        slice_id="later",
        scope=("foo.txt",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "foo.txt").write_text("later dirt\n", encoding="utf-8")
    later = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="later",
    )
    later = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        later_slice,
        later,
    )
    journal.prepare(later, keep=False)
    corpus.commit(implementations=(later,))
    _unregister_live(journal.transaction_id)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_recovered == ("later",)
    assert report.crashed_rollbacks_reaped == ("old",)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "base\n"


def test_cleanup_publish_then_raise_reconciles_attached_lifecycle(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    original_commit = corpus.commit

    def _publish_then_raise(*args, **kwargs):
        original_commit(*args, **kwargs)
        raise OSError("lost commit acknowledgement")

    monkeypatch.setattr(corpus, "commit", _publish_then_raise)
    first = reap_orphaned_dirt(corpus, repo)

    assert first.failures == ()
    assert first.crashed_rollbacks_reaped == ("s0",)
    assert first.cleanup_lifecycle_ref is not None
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"

    monkeypatch.setattr(corpus, "commit", original_commit)
    second = reap_orphaned_dirt(corpus, repo)
    assert second.failures == ()
    assert second.crashed_rollbacks_reaped == ()
    assert any("already settled" in item for item in second.legacy_mutations_skipped)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"


def test_cleanup_crash_after_spec_cas_is_preserved_on_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    original_commit = WorkspaceMutationJournal.commit

    def _die_before_marker(self, *, attached=False, corpus=None, **kwargs):
        if self.slice_id == "__legacy_reaper__" and attached:
            _unregister_live(self.transaction_id)
            return None
        return original_commit(
            self,
            attached=attached,
            corpus=corpus,
            **kwargs,
        )

    monkeypatch.setattr(WorkspaceMutationJournal, "commit", _die_before_marker)
    first = reap_orphaned_dirt(corpus, repo)

    assert first.failures == ()
    assert first.crashed_rollbacks_reaped == ("s0",)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"
    assert tuple(
        corpus.spec_dir.glob("runs/reaper/mutations/*/manifest.json")
    )

    monkeypatch.setattr(WorkspaceMutationJournal, "commit", original_commit)
    second = reap_orphaned_dirt(corpus, repo)

    assert second.failures == ()
    assert second.mutation_journals_preserved == ("__legacy_reaper__",)
    assert second.crashed_rollbacks_reaped == ()
    assert any("already settled" in item for item in second.legacy_mutations_skipped)
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "original\n"


def test_empty_reaper_settlement_cannot_suppress_dirty_failure(
    tmp_path: Path,
) -> None:
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import (
        _REAPER_EVENT_PREFIX,
        _REAPER_SETTLEMENT_SCHEMA,
        reap_orphaned_dirt,
    )

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    committed = corpus.commit(implementations=(failure,))
    payload_ref = corpus.put_artifact(
        json.dumps({
            "schema": _REAPER_SETTLEMENT_SCHEMA,
            "transaction_id": "a" * 32,
            "implementation_refs": [
                str(committed.refs.implementation_refs[0])
            ],
            "crashed_rollbacks_reaped": [],
            "scope_escapes_cleaned": [],
            "mutation_paths": [],
        }).encode("utf-8"),
        kind="reaper_settlements",
        name_hint="forged.json",
    )
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="reaped",
            reason=_REAPER_EVENT_PREFIX + str(payload_ref),
            affected_versions=(version,),
            execution=dataclasses.replace(
                failure.execution,
                recipe="reaper",
                run_id="forged",
            ),
        ),
    )

    report = reap_orphaned_dirt(corpus, repo)

    assert any("operations do not bind" in item for item in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "dirty\n"


def test_reaper_settlement_paths_are_stable_across_later_head(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    first = reap_orphaned_dirt(corpus, repo)
    assert first.failures == ()

    (repo / "foo.txt").write_text("new base\n", encoding="utf-8")
    _commit(repo, "new base")
    new_base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        project_context=dataclasses.replace(
            latest.project_context,
            base_commit=new_base,
        ),
    ))

    repeated = reap_orphaned_dirt(corpus, repo)

    assert repeated.failures == ()
    assert repeated.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "new base\n"


def test_unattached_legacy_record_is_diagnostic_only(tmp_path: Path) -> None:
    """No journal/Spec chronology means no diff or scope-escape mutation."""

    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, _tier, _slice, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("unattached dirt\n", encoding="utf-8")
    (repo / "rogue.txt").write_text("unattached escape\n", encoding="utf-8")
    orphan = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2099-01-01T00:00:00+00:00",
        scope_escapes=("rogue.txt",),
    )
    orphan_ref = corpus.put_record(orphan, kind="implementations")

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.unattached_legacy_skipped == (str(orphan_ref),)
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == ("unattached dirt\n")
    assert (repo / "rogue.txt").read_text(encoding="utf-8") == ("unattached escape\n")


def test_reused_slice_id_from_old_plan_cannot_authorize_cleanup(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("old-plan dirt\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    latest = corpus.latest_spec()
    replacement_slice = dataclasses.replace(
        slice_,
        directive=dataclasses.replace(
            slice_.directive,
            intent="plan B deliberately reuses the slice id",
        ),
    )
    replacement_tier = dataclasses.replace(tier, slices=(replacement_slice,))
    promoted = dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(replacement_tier,)),
    )
    corpus.put_spec(promoted)

    report = reap_orphaned_dirt(corpus, repo)

    assert any("not current-plan authority" in item for item in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "old-plan dirt\n"


def test_removed_slice_with_present_failure_blocks_startup(
    tmp_path: Path,
) -> None:
    """Removing a slice cannot hide its still-present failed mutation."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("removed-slice dirt\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))

    replacement_slice = dataclasses.replace(
        slice_,
        id="replacement",
        directive=dataclasses.replace(
            slice_.directive,
            slice_id="replacement",
            intent="replace the old slice instead of reusing its identity",
            files_in_scope=("replacement.txt",),
        ),
    )
    replacement_tier = dataclasses.replace(
        tier,
        slices=(replacement_slice,),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(replacement_tier,)),
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "removed-slice dirt\n"
    )


def test_non_current_scope_escape_parent_swap_cannot_prove_outside_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An intermediate symlink cannot turn outside ENOENT into clean proof."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="record an old scope escape",
            files_in_scope=("dir/anchor.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={"dir/anchor.txt": "anchor\n"},
    )
    failure = _real_impl(
        corpus,
        diff=b"",
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        scope_escapes=("dir/victim.txt",),
        slice_id="old",
        files_modified=(),
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        failure,
    )
    corpus.commit(implementations=(failure,))
    replacement = Slice(
        id="replacement",
        directive=ImplementationDirective(
            slice_id="replacement",
            intent="remove the old slice from authority",
            files_in_scope=("replacement.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(replacement,), depends_on=()),
        )),
    ))

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside survives\n", encoding="utf-8")
    parked = tmp_path / "parked-dir"
    armed = False
    swapped = False
    real_match = _reaper._worktree_paths_match_base_tree
    real_normalize = _reaper._normalize_bound_workspace_path

    def arm_after_base_proof(*args, **kwargs):
        nonlocal armed
        result = real_match(*args, **kwargs)
        armed = True
        return result

    def swap_after_escape_normalization(cwd, root_fd, rendered):
        nonlocal swapped
        normalized = real_normalize(cwd, root_fd, rendered)
        if armed and not swapped and normalized == "dir/victim.txt":
            (repo / "dir").rename(parked)
            (repo / "dir").symlink_to(outside, target_is_directory=True)
            swapped = True
        return normalized

    monkeypatch.setattr(
        _reaper,
        "_worktree_paths_match_base_tree",
        arm_after_base_proof,
    )
    monkeypatch.setattr(
        _reaper,
        "_normalize_bound_workspace_path",
        swap_after_escape_normalization,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped
    assert any(
        "cannot prove non-current scope escape absent" in failure
        for failure in report.failures
    ), report.failures
    assert sentinel.read_text(encoding="utf-8") == "outside survives\n"
    assert (parked / "anchor.txt").read_text(encoding="utf-8") == "anchor\n"


def test_non_current_scope_escape_parent_swap_after_first_absence_reproof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A swap between the two held-parent ENOENT checks fails closed."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="record an old scope escape",
            files_in_scope=("dir/anchor.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={"dir/anchor.txt": "anchor\n"},
    )
    failure = _real_impl(
        corpus,
        diff=b"",
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        scope_escapes=("dir/victim.txt",),
        slice_id="old",
        files_modified=(),
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        failure,
    )
    corpus.commit(implementations=(failure,))
    replacement = Slice(
        id="replacement",
        directive=ImplementationDirective(
            slice_id="replacement",
            intent="remove the old slice from authority",
            files_in_scope=("replacement.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(replacement,), depends_on=()),
        )),
    ))

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "victim.txt"
    sentinel.write_text("outside survives\n", encoding="utf-8")
    parked = tmp_path / "parked-dir"
    swapped = False
    binding_checks = 0
    real_assert_parent = _reaper._assert_workspace_parent_binding

    def swap_after_first_parent_reproof(root_fd, path, parent_fd):
        nonlocal binding_checks, swapped
        if path == "dir/victim.txt":
            binding_checks += 1
        result = real_assert_parent(root_fd, path, parent_fd)
        if path == "dir/victim.txt" and binding_checks == 1:
            (repo / "dir").rename(parked)
            (repo / "dir").symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(
        _reaper,
        "_assert_workspace_parent_binding",
        swap_after_first_parent_reproof,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped
    assert binding_checks >= 2
    assert any(
        "cannot prove non-current scope escape absent" in failure
        for failure in report.failures
    ), report.failures
    assert sentinel.read_text(encoding="utf-8") == "outside survives\n"
    assert (parked / "anchor.txt").read_text(encoding="utf-8") == "anchor\n"


def test_non_current_scope_escape_parent_swap_after_first_present_reproof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stable stats in a parked parent cannot prove the current path."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="record an old scope escape",
            files_in_scope=("dir/anchor.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={
            "dir/anchor.txt": "anchor\n",
            "dir/victim.txt": "original victim\n",
        },
    )
    failure = _real_impl(
        corpus,
        diff=b"",
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        scope_escapes=("dir/victim.txt",),
        slice_id="old",
        files_modified=(),
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        failure,
    )
    corpus.commit(implementations=(failure,))
    replacement = Slice(
        id="replacement",
        directive=ImplementationDirective(
            slice_id="replacement",
            intent="remove the old slice from authority",
            files_in_scope=("replacement.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(replacement,), depends_on=()),
        )),
    ))

    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "victim.txt"
    sentinel.write_text("outside survives\n", encoding="utf-8")
    parked = tmp_path / "parked-dir"
    swapped = False
    binding_checks = 0
    real_assert_parent = _reaper._assert_workspace_parent_binding

    def swap_after_first_parent_reproof(root_fd, path, parent_fd):
        nonlocal binding_checks, swapped
        if path == "dir/victim.txt":
            binding_checks += 1
        result = real_assert_parent(root_fd, path, parent_fd)
        if path == "dir/victim.txt" and binding_checks == 1:
            (repo / "dir").rename(parked)
            (repo / "dir").symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(
        _reaper,
        "_assert_workspace_parent_binding",
        swap_after_first_parent_reproof,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped
    assert binding_checks >= 2
    assert any(
        "cannot prove non-current scope escape absent" in failure
        for failure in report.failures
    ), report.failures
    assert sentinel.read_text(encoding="utf-8") == "outside survives\n"
    assert (parked / "victim.txt").read_text(encoding="utf-8") == (
        "original victim\n"
    )


def test_non_current_scope_escape_missing_frontier_creation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Creating a missing ancestor between absence probes is ambiguous."""

    from crucible.workflows import _reaper
    from crucible.workflows._partition import attach_resume_identity

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="record an old scope escape",
            files_in_scope=("dir/anchor.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={"dir/anchor.txt": "anchor\n"},
    )
    failure = _real_impl(
        corpus,
        diff=b"",
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        scope_escapes=("dir/missing/victim.txt",),
        slice_id="old",
        files_modified=(),
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        failure,
    )
    corpus.commit(implementations=(failure,))
    replacement = Slice(
        id="replacement",
        directive=ImplementationDirective(
            slice_id="replacement",
            intent="remove the old slice from authority",
            files_in_scope=("replacement.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            Tier(tier_id="t0", slices=(replacement,), depends_on=()),
        )),
    ))

    created = False
    assertion_calls = 0
    real_assert_missing = _reaper._assert_missing_ancestor_binding

    def create_after_first_absence(root_fd, path, binding):
        nonlocal assertion_calls, created
        result = real_assert_missing(root_fd, path, binding)
        if path == "dir/missing/victim.txt":
            assertion_calls += 1
            if assertion_calls == 1:
                missing = repo / "dir" / "missing"
                missing.mkdir()
                (missing / "victim.txt").write_text(
                    "created during audit\n",
                    encoding="utf-8",
                )
                created = True
        return result

    monkeypatch.setattr(
        _reaper,
        "_assert_missing_ancestor_binding",
        create_after_first_absence,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert created
    assert assertion_calls == 1
    assert any(
        "cannot prove non-current scope escape absent" in failure
        for failure in report.failures
    ), report.failures
    assert (repo / "dir" / "missing" / "victim.txt").read_text(
        encoding="utf-8"
    ) == "created during audit\n"
    assert (repo / "dir" / "anchor.txt").read_text(encoding="utf-8") == (
        "anchor\n"
    )


def test_removed_slice_partial_postimage_blocks_startup(
    tmp_path: Path,
) -> None:
    """Restoring one path cannot hide another path from the same old diff."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="mutate two files before the old plan is removed",
            files_in_scope=("a.txt", "b.txt"),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 2},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={"a.txt": "a-base\n", "b.txt": "b-base\n"},
    )
    (repo / "a.txt").write_text("a-stale\n", encoding="utf-8")
    (repo / "b.txt").write_text("b-stale\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="old",
        files_modified=("a.txt", "b.txt"),
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        failure,
    )
    corpus.commit(implementations=(failure,))

    # The cumulative patch no longer applies in reverse as a whole, but one
    # exact stale path remains. Absence must therefore be proved per path.
    (repo / "b.txt").write_text("b-base\n", encoding="utf-8")
    replacement_slice = dataclasses.replace(
        old_slice,
        id="replacement",
        directive=dataclasses.replace(
            old_slice.directive,
            slice_id="replacement",
            intent="new plan removes the old two-file slice",
            files_in_scope=("replacement.txt",),
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            dataclasses.replace(old_tier, slices=(replacement_slice,)),
        )),
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "a-stale\n"
    assert (repo / "b.txt").read_text(encoding="utf-8") == "b-base\n"


def test_base_tree_proof_rejects_checkout_root_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A held clean root cannot attest a replacement path with stale bytes."""
    from crucible.workflows import _reaper

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "tracked.txt").write_text("base bytes\n", encoding="utf-8")
    _commit(repo, "initial")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    displaced = tmp_path / "displaced-repo"
    real_tree_entry = _reaper._git_tree_entry
    swapped = False

    def replace_root_after_tree_lookup(root_fd, tree_oid, path):
        nonlocal swapped
        entry = real_tree_entry(root_fd, tree_oid, path)
        if not swapped:
            repo.rename(displaced)
            repo.mkdir()
            (repo / "tracked.txt").write_text(
                "attacker bytes\n",
                encoding="utf-8",
            )
            swapped = True
        return entry

    monkeypatch.setattr(
        _reaper,
        "_git_tree_entry",
        replace_root_after_tree_lookup,
    )

    with pytest.raises(
        _reaper.MutationJournalError,
        match="checkout .* changed identity during verification",
    ):
        _reaper._worktree_paths_match_base_tree(
            repo,
            base_commit=base,
            paths=("tracked.txt",),
        )

    assert (repo / "tracked.txt").read_text(encoding="utf-8") == (
        "attacker bytes\n"
    )
    assert (displaced / "tracked.txt").read_text(encoding="utf-8") == (
        "base bytes\n"
    )


def test_public_reaper_reverse_apply_is_bound_across_root_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A post-assert A/B swap cannot redirect destructive Git to B."""
    from crucible.workflows import _reaper

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "foo.txt").write_text("base\n", encoding="utf-8")
    _commit(repo, "base")
    (repo / "foo.txt").write_text("failed-a\n", encoding="utf-8")
    diff = _capture_diff(repo)
    corpus = _mock_corpus(
        [_mock_impl("s0", "hard_fail_max_turns", "sha256:failed")],
        {"sha256:failed": diff},
        cwd=repo,
    )

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _init_git_repo(replacement)
    (replacement / "foo.txt").write_text("base\n", encoding="utf-8")
    _commit(replacement, "base")
    (replacement / "foo.txt").write_text("attacker-b\n", encoding="utf-8")
    displaced = tmp_path / "displaced-a"
    real_reverse = _reaper._git_apply_reverse
    swapped = False

    def _swap_then_reverse(diff_bytes: bytes, root_fd: int) -> bool:
        nonlocal swapped
        if not swapped:
            repo.rename(displaced)
            replacement.rename(repo)
            swapped = True
        return real_reverse(diff_bytes, root_fd)

    monkeypatch.setattr(_reaper, "_git_apply_reverse", _swap_then_reverse)

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped is True
    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "workspace authority root changed identity" in failure
        for failure in report.failures
    ), report.failures
    # The descriptor-bound reverse touched A, then the forward compensation
    # restored it before the path-based outer recovery was attempted.
    assert (displaced / "foo.txt").read_text(encoding="utf-8") == "failed-a\n"
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "attacker-b\n"


def test_public_reaper_reports_workspace_recovery_admission_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A pre-body recovery admission failure remains a typed public report."""
    from crucible.workflows import _reaper

    class FailedAdmission:
        def __enter__(self):
            raise _reaper.MutationJournalError("synthetic admission failure")

        def retry_cleanup(self) -> None:
            return None

    def failed_admission(_corpus, _cwd):
        return FailedAdmission()

    monkeypatch.setattr(
        _reaper,
        "recovery_workspace_mutation_lock",
        failed_admission,
    )

    report = _reaper.reap_orphaned_dirt(object(), tmp_path)

    assert report.reaped_anything is False
    assert report.failures == (
        "workspace recovery coordination failed; journals retained: "
        "MutationJournalError: synthetic admission failure",
    )


def test_public_reaper_positive_proof_cannot_borrow_replacement_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Transient B bytes cannot authorize skipping failed dirt in A."""
    from crucible.workflows import _reaper

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "foo.txt").write_text("base\n", encoding="utf-8")
    _commit(repo, "base")
    replacement = tmp_path / "replacement"
    subprocess.run(
        ["git", "clone", "-q", "--local", str(repo), str(replacement)],
        check=True,
    )

    (repo / "foo.txt").write_text("failed-a\n", encoding="utf-8")
    failure_diff = _capture_diff(repo)
    (repo / "foo.txt").write_text("accepted-b\n", encoding="utf-8")
    positive_diff = _capture_diff(repo)
    (repo / "foo.txt").write_text("failed-a\n", encoding="utf-8")
    (replacement / "foo.txt").write_text("accepted-b\n", encoding="utf-8")
    corpus = _mock_corpus(
        [
            _mock_impl("failed", "hard_fail_max_turns", "sha256:failed"),
            _mock_impl("positive", "success", "sha256:positive"),
        ],
        {
            "sha256:failed": failure_diff,
            "sha256:positive": positive_diff,
        },
        cwd=repo,
    )
    displaced = tmp_path / "displaced-a"
    real_proof = _reaper._proven_positive_authorities
    swapped = False

    def _borrow_b_during_proof(corpus_arg, cwd_path, root_fd, occurrences):
        nonlocal swapped
        repo.rename(displaced)
        replacement.rename(repo)
        swapped = True
        try:
            return real_proof(
                corpus_arg,
                cwd_path,
                root_fd,
                occurrences,
            )
        finally:
            repo.rename(replacement)
            displaced.rename(repo)

    monkeypatch.setattr(
        _reaper,
        "_proven_positive_authorities",
        _borrow_b_during_proof,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped is True
    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "checkout" in failure and "changed identity" in failure
        for failure in report.failures
    ), report.failures
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "failed-a\n"
    assert (replacement / "foo.txt").read_text(encoding="utf-8") == (
        "accepted-b\n"
    )


def test_public_reaper_scope_cleanup_is_bound_across_root_replacement(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A post-assert A/B swap cannot redirect scope deletion to B."""
    from crucible.workflows import _reaper

    repo = tmp_path / "repo"
    repo.mkdir()
    _init_git_repo(repo)
    (repo / "anchor.txt").write_text("anchor\n", encoding="utf-8")
    _commit(repo, "base")
    (repo / "escape.tmp").write_text("escape-a\n", encoding="utf-8")
    corpus = _mock_corpus(
        [
            _mock_impl(
                "s0",
                "hard_fail_max_turns",
                "sha256:empty",
                scope_escapes=("escape.tmp",),
            )
        ],
        {"sha256:empty": b""},
        cwd=repo,
    )

    replacement = tmp_path / "replacement"
    replacement.mkdir()
    _init_git_repo(replacement)
    (replacement / "anchor.txt").write_text("anchor\n", encoding="utf-8")
    _commit(replacement, "base")
    (replacement / "escape.tmp").write_text("attacker-b\n", encoding="utf-8")
    displaced = tmp_path / "displaced-a"
    real_revert = _reaper._revert_scope_escapes_bound
    swapped = False

    def _swap_then_revert(impl, cwd_path, root_fd, *, strict):
        nonlocal swapped
        if not swapped:
            repo.rename(displaced)
            replacement.rename(repo)
            swapped = True
        return real_revert(
            impl,
            cwd_path,
            root_fd,
            strict=strict,
        )

    monkeypatch.setattr(
        _reaper,
        "_revert_scope_escapes_bound",
        _swap_then_revert,
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert swapped is True
    assert report.scope_escapes_deleted == ()
    assert any(
        "workspace authority root changed identity" in failure
        for failure in report.failures
    ), report.failures
    # Scope cleanup and its exact journal restore remain anchored to the
    # originally authorized checkout. The journal stays armed because the
    # canonical pathname cannot be rebound after the external replacement.
    assert (displaced / "escape.tmp").read_text(encoding="utf-8") == "escape-a\n"
    assert (repo / "escape.tmp").read_text(encoding="utf-8") == "attacker-b\n"
    assert tuple(
        corpus.spec_dir.glob("runs/reaper/mutations/*/manifest.json")
    )


def test_removed_slice_audits_failure_hidden_by_latest_empty_record(
    tmp_path: Path,
) -> None:
    """Every active ref is audited, not just one latest record per slice."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("older failure remains\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))

    empty_latest = _real_impl(
        corpus,
        diff=b"",
        classification="soft_fail_no_changes",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
        files_modified=(),
    )
    empty_latest = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        empty_latest,
    )
    corpus.commit(implementations=(empty_latest,))

    replacement_slice = dataclasses.replace(
        slice_,
        id="replacement",
        directive=dataclasses.replace(
            slice_.directive,
            slice_id="replacement",
            intent="new plan omits the old slice",
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            dataclasses.replace(tier, slices=(replacement_slice,)),
        )),
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "older failure remains\n"
    )


def test_identity_drifted_provisional_success_blocks_startup(
    tmp_path: Path,
) -> None:
    """A same-ID plan revision cannot silently inherit provisional bytes."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("stale provisional\n", encoding="utf-8")
    provisional = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    provisional = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        provisional,
    )
    corpus.commit(implementations=(provisional,))

    revised_slice = dataclasses.replace(
        slice_,
        directive=dataclasses.replace(
            slice_.directive,
            intent="same ID now carries revised execution authority",
        ),
    )
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(
            dataclasses.replace(tier, slices=(revised_slice,)),
        )),
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "stale provisional\n"
    )


def test_same_base_replanned_landing_without_current_reproof_blocks_startup(
    tmp_path: Path,
) -> None:
    """A stale Tier landing cannot authorize bytes under a revised plan."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("stale landed bytes\n", encoding="utf-8")
    landed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        landed,
    )
    corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    revised_slice = dataclasses.replace(
        slice_,
        directive=dataclasses.replace(
            slice_.directive,
            intent="same-base revised execution authority",
        ),
    )
    revised_tier = dataclasses.replace(tier, slices=(revised_slice,))
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(revised_tier,)),
    ))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "stale landed bytes\n"
    )


def test_current_handoff_reproof_supersedes_rejected_handoff_landing(
    tmp_path: Path,
) -> None:
    """A valid current Handoff may re-attest an old writer-bug landing."""
    from crucible.workflows import _build_verify
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("re-attested bytes\n", encoding="utf-8")
    shared_diff = _capture_diff(repo)
    old = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    old, old_ref = _land_with_rejected_handoff_identity(
        corpus,
        tier,
        slice_,
        old,
    )
    accepted, reason = _build_verify._tier_lifecycle_acceptance(
        corpus,
        corpus.latest_spec(),
        tier,
        {slice_.id: old},
    )
    assert accepted is False
    assert reason == "landed lifecycle Spec does not contain the complete tier"

    current = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
    )
    current = dataclasses.replace(
        current,
        execution=dataclasses.replace(current.execution, recipe="handoff"),
    )
    current = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        current,
    )
    current_commit = corpus.commit(implementations=(current,))
    current_ref = str(current_commit.refs.implementation_refs[0])

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert any(
        old_ref in diagnostic
        and "no unprotected exact postimage is present" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert any(
        current_ref in diagnostic and "is retained" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "re-attested bytes\n"
    )


def test_rejected_handoff_landing_without_reproof_still_blocks(
    tmp_path: Path,
) -> None:
    """Ignoring stale lifecycle authority never ignores live stale bytes."""
    from crucible.workflows import _build_verify
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("unproved bytes\n", encoding="utf-8")
    old = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    old, _old_ref = _land_with_rejected_handoff_identity(
        corpus,
        tier,
        slice_,
        old,
    )
    accepted, reason = _build_verify._tier_lifecycle_acceptance(
        corpus,
        corpus.latest_spec(),
        tier,
        {slice_.id: old},
    )
    assert accepted is False
    assert reason == "landed lifecycle Spec does not contain the complete tier"

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "unproved bytes\n"
    )


def test_current_reproof_supersedes_same_base_replanned_landing(
    tmp_path: Path,
) -> None:
    """A newer current-plan proof safely supersedes a stale Tier landing."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("re-attested bytes\n", encoding="utf-8")
    shared_diff = _capture_diff(repo)
    old = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    old = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        old,
    )
    old_commit = corpus.commit(
        implementations=(old,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=old.execution,
        ),
    )
    old_ref = str(old_commit.refs.implementation_refs[0])

    revised_slice = dataclasses.replace(
        slice_,
        directive=dataclasses.replace(
            slice_.directive,
            intent="same-base revised execution authority",
        ),
    )
    revised_tier = dataclasses.replace(tier, slices=(revised_slice,))
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(revised_tier,)),
    ))

    current = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
    )
    current = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        revised_tier,
        revised_slice,
        current,
    )
    current_commit = corpus.commit(implementations=(current,))
    current_ref = str(current_commit.refs.implementation_refs[0])

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert any(
        old_ref in diagnostic and "not current-plan authority" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert any(
        current_ref in diagnostic and "is retained" in diagnostic
        for diagnostic in report.legacy_mutations_skipped
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "re-attested bytes\n"
    )


def test_current_reproof_must_cover_every_stale_landing_path(
    tmp_path: Path,
) -> None:
    """A partial current proof cannot hide an uncovered stale postimage."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    old_slice = Slice(
        id="s0",
        directive=ImplementationDirective(
            slice_id="s0",
            intent="land two paths under the old plan",
            files_in_scope=("foo.txt", "bar.txt"),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 2},
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    repo, corpus, base_commit = _real_multi_setup(
        tmp_path,
        tiers=(old_tier,),
        files={"foo.txt": "foo base\n", "bar.txt": "bar base\n"},
    )
    (repo / "foo.txt").write_text("foo landed\n", encoding="utf-8")
    (repo / "bar.txt").write_text("bar landed\n", encoding="utf-8")
    old = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
        files_modified=("bar.txt", "foo.txt"),
    )
    old = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        old,
    )
    corpus.commit(
        implementations=(old,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=old.execution,
        ),
    )

    revised_slice = dataclasses.replace(
        old_slice,
        directive=dataclasses.replace(
            old_slice.directive,
            intent="re-attest only foo under the revised plan",
            files_in_scope=("foo.txt",),
            blast_radius={"max_files": 1},
        ),
    )
    revised_tier = dataclasses.replace(old_tier, slices=(revised_slice,))
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(revised_tier,)),
    ))
    foo_diff = subprocess.run(
        ["git", "diff", "HEAD", "--", "foo.txt"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    current = _real_impl(
        corpus,
        diff=foo_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
        files_modified=("foo.txt",),
    )
    current = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        revised_tier,
        revised_slice,
        current,
    )
    corpus.commit(implementations=(current,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "not current-plan authority" in failure
        and "mutation may still be present" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "foo landed\n"
    assert (repo / "bar.txt").read_text(encoding="utf-8") == "bar landed\n"


def test_tampered_current_reproof_cannot_supersede_stale_landing(
    tmp_path: Path,
) -> None:
    """Current identity is insufficient when its exact live postimage drifts."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("re-attested bytes\n", encoding="utf-8")
    shared_diff = _capture_diff(repo)
    old = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    old = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        old,
    )
    corpus.commit(
        implementations=(old,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t0: complete",
            affected_versions=(version,),
            execution=old.execution,
        ),
    )

    revised_slice = dataclasses.replace(
        slice_,
        directive=dataclasses.replace(
            slice_.directive,
            intent="same-base revised execution authority",
        ),
    )
    revised_tier = dataclasses.replace(tier, slices=(revised_slice,))
    latest = corpus.latest_spec()
    corpus.put_spec(dataclasses.replace(
        latest,
        version=latest.version + 1,
        parent=ContentHash(hash_record(latest)),
        tier_dag=TierDAG(tiers=(revised_tier,)),
    ))
    current = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
    )
    current = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        revised_tier,
        revised_slice,
        current,
    )
    corpus.commit(implementations=(current,))
    (repo / "foo.txt").write_text("tampered\n", encoding="utf-8")

    report = reap_orphaned_dirt(corpus, repo)

    assert report.crashed_rollbacks_reaped == ()
    assert any(
        "positive Implementation" in failure and "not live" in failure
        for failure in report.failures
    )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "tampered\n"


def test_current_pending_gates_postimage_blocks_provider_dispatch(
    tmp_path: Path,
) -> None:
    """A non-skip-worthy handoff is unproved and must never be cleaned."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("awaiting gates\n", encoding="utf-8")
    pending = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="pending_gates",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    pending = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        pending,
    )
    corpus.commit(implementations=(pending,))

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "mutation may still be present" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "awaiting gates\n"
    )


def test_older_unproved_postimage_is_covered_by_later_positive_authority(
    tmp_path: Path,
) -> None:
    """Exact shared bytes survive when a strictly later success owns them."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("shared postimage\n", encoding="utf-8")
    shared_diff = _capture_diff(repo)
    failure = _real_impl(
        corpus,
        diff=shared_diff,
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))

    success = _real_impl(
        corpus,
        diff=shared_diff,
        classification="success",
        base_commit=base_commit,
        timestamp="2026-08-06T00:01:00+00:00",
    )
    success = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        success,
    )
    corpus.commit(implementations=(success,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "shared postimage\n"
    )


def test_newer_positive_must_be_live_before_covering_old_failure(
    tmp_path: Path,
) -> None:
    """A landed ref cannot mask older dirt when its own bytes vanished."""
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="record the old failed image",
            files_in_scope=("foo.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    landed_slice = dataclasses.replace(
        old_slice,
        id="landed",
        directive=dataclasses.replace(
            old_slice.directive,
            slice_id="landed",
            intent="land a newer image of the same path",
        ),
    )
    old_tier = Tier(tier_id="t0", slices=(old_slice,), depends_on=())
    landed_tier = Tier(
        tier_id="t1",
        slices=(landed_slice,),
        depends_on=("t0",),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(old_tier, landed_tier),
        files={"foo.txt": "base\n"},
    )
    (repo / "foo.txt").write_text("old-failure\n", encoding="utf-8")
    old_failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="old",
    )
    old_failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        old_tier,
        old_slice,
        old_failure,
    )
    corpus.commit(implementations=(old_failure,))

    (repo / "foo.txt").write_text("new-landed\n", encoding="utf-8")
    landed = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="landed",
    )
    landed = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        landed_tier,
        landed_slice,
        landed,
    )
    corpus.commit(
        implementations=(landed,),
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier t1: complete",
            affected_versions=(version,),
            execution=landed.execution,
        ),
    )

    (repo / "foo.txt").write_text("old-failure\n", encoding="utf-8")
    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "positive Implementation" in failure and "not live" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "old-failure\n"


@pytest.mark.parametrize(
    "shape",
    ("create", "delete", "rename", "chmod", "symlink"),
)
def test_sole_positive_postimage_shape_must_be_live(
    tmp_path: Path,
    shape: str,
) -> None:
    """Every final positive existence/type/mode image is unconditional."""
    from assay.primitives.implement import capture_workspace_delta

    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    files = (
        {"anchor.txt": "anchor\n"}
        if shape == "create"
        else (
            {"old.txt": "base\n"}
            if shape == "rename"
            else {"target.txt": "base\n"}
        )
    )
    files_in_scope = {
        "create": (),
        "delete": ("target.txt",),
        "rename": ("old.txt",),
        "chmod": ("target.txt",),
        "symlink": ("target.txt",),
    }[shape]
    creates_paths = {
        "create": ("new.txt",),
        "delete": (),
        "rename": ("new.txt",),
        "chmod": (),
        "symlink": (),
    }[shape]
    slice_ = Slice(
        id="positive",
        directive=ImplementationDirective(
            slice_id="positive",
            intent=f"land a {shape} postimage",
            files_in_scope=files_in_scope,
            creates_paths=creates_paths,
            verify_mode="skip",
            blast_radius={"max_files": 2},
        ),
    )
    tier = Tier(tier_id="t0", slices=(slice_,), depends_on=())
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files=files,
    )

    if shape == "create":
        (repo / "new.txt").write_text("created\n", encoding="utf-8")
        modified = ("new.txt",)
    elif shape == "delete":
        (repo / "target.txt").unlink()
        modified = ("target.txt",)
    elif shape == "rename":
        (repo / "old.txt").rename(repo / "new.txt")
        modified = ("old.txt", "new.txt")
    elif shape == "chmod":
        (repo / "target.txt").chmod(0o755)
        modified = ("target.txt",)
    else:
        (repo / "target.txt").unlink()
        (repo / "target.txt").symlink_to("expected-target")
        modified = ("target.txt",)

    positive_diff, observed_modified, observed_base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=tuple((*files_in_scope, *creates_paths)),
    )
    assert observed_base == base
    assert set(observed_modified) == set(modified)
    positive = _real_impl(
        corpus,
        diff=positive_diff,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="positive",
        files_modified=observed_modified,
    )
    positive = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        positive,
    )
    corpus.commit(implementations=(positive,))

    # Remove/revert the accepted shape while leaving its positive record as
    # the only skip-worthy execution evidence.
    if shape == "create":
        (repo / "new.txt").unlink()
    elif shape == "delete":
        (repo / "target.txt").write_text("base\n", encoding="utf-8")
    elif shape == "rename":
        (repo / "new.txt").unlink()
        (repo / "old.txt").write_text("base\n", encoding="utf-8")
    elif shape == "chmod":
        (repo / "target.txt").chmod(0o644)
    else:
        (repo / "target.txt").unlink()
        (repo / "target.txt").write_text("base\n", encoding="utf-8")

    report = reap_orphaned_dirt(corpus, repo)

    assert any(
        "positive Implementation" in failure and "not live" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()


def test_declared_positive_scope_is_not_byte_authority(
    tmp_path: Path,
) -> None:
    """A directory reservation cannot mask an untouched older file."""
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="fail in a directory later reserved by a sibling",
            files_in_scope=("dir/a.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    positive_slice = Slice(
        id="positive",
        directive=ImplementationDirective(
            slice_id="positive",
            intent="reserve dir but modify only an unrelated exact file",
            files_in_scope=("dir", "other.txt"),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 2},
        ),
    )
    tier = Tier(
        tier_id="t0",
        slices=(old_slice, positive_slice),
        depends_on=(),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files={"dir/a.txt": "base\n", "other.txt": "other-base\n"},
    )
    (repo / "dir" / "a.txt").write_text("old dirt\n", encoding="utf-8")
    old_failure = _real_impl(
        corpus,
        diff=subprocess.run(
            ["git", "diff", "HEAD", "--", "dir/a.txt"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout,
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="old",
        files_modified=("dir/a.txt",),
    )
    old_failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        old_slice,
        old_failure,
    )
    corpus.commit(implementations=(old_failure,))

    (repo / "other.txt").write_text("positive bytes\n", encoding="utf-8")
    positive = _real_impl(
        corpus,
        diff=subprocess.run(
            ["git", "diff", "HEAD", "--", "other.txt"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="positive",
        files_modified=("other.txt",),
    )
    positive = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        positive_slice,
        positive,
    )
    corpus.commit(implementations=(positive,))

    report = reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.crashed_rollbacks_reaped == ("old",)
    assert (repo / "dir" / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert (repo / "other.txt").read_text(encoding="utf-8") == (
        "positive bytes\n"
    )


def test_preserved_journal_scope_is_not_byte_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A journal reservation protects only its bound Impl's exact diff."""
    from crucible.workflows import _reaper
    from crucible.workflows._mutation_journal import (
        RecoveryReport,
        RecoverySettlement,
    )
    from crucible.workflows._partition import attach_resume_identity

    old_slice = Slice(
        id="old",
        directive=ImplementationDirective(
            slice_id="old",
            intent="old dirt under a broad preserved reservation",
            files_in_scope=("dir/a.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    preserved_slice = Slice(
        id="preserved",
        directive=ImplementationDirective(
            slice_id="preserved",
            intent="preserve only an unrelated exact mutation",
            files_in_scope=("dir", "other.txt"),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 2},
        ),
    )
    tier = Tier(
        tier_id="t0",
        slices=(old_slice, preserved_slice),
        depends_on=(),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier,),
        files={"dir/a.txt": "base\n", "other.txt": "other-base\n"},
    )
    (repo / "dir" / "a.txt").write_text("old dirt\n", encoding="utf-8")
    old_failure = _real_impl(
        corpus,
        diff=subprocess.run(
            ["git", "diff", "HEAD", "--", "dir/a.txt"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout,
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="old",
        files_modified=("dir/a.txt",),
    )
    old_failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        old_slice,
        old_failure,
    )
    corpus.commit(implementations=(old_failure,))

    (repo / "other.txt").write_text("preserved bytes\n", encoding="utf-8")
    preserved = _real_impl(
        corpus,
        diff=subprocess.run(
            ["git", "diff", "HEAD", "--", "other.txt"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="preserved",
        files_modified=("other.txt",),
    )
    preserved = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        preserved_slice,
        preserved,
    )
    committed = corpus.commit(implementations=(preserved,))
    preserved_ref = str(committed.refs.implementation_refs[0])
    monkeypatch.setattr(
        _reaper,
        "_recover_mutation_journals_locked",
        lambda *args, **kwargs: RecoveryReport(
            recovered=(),
            preserved=("preserved",),
            failures=(),
            settlements=(RecoverySettlement(
                transaction_id="a" * 32,
                slice_id="preserved",
                disposition="preserved",
                implementation_ref=preserved_ref,
                lifecycle_ref=None,
                protected_scope=("dir",),
                baseline_spec_version=corpus.latest_spec().version,
                baseline_spec_hash=hash_record(corpus.latest_spec()),
            ),),
        ),
    )

    report = _reaper.reap_orphaned_dirt(corpus, repo)

    assert report.failures == ()
    assert report.mutation_journals_preserved == ("preserved",)
    assert report.crashed_rollbacks_reaped == ("old",)
    assert (repo / "dir" / "a.txt").read_text(encoding="utf-8") == "base\n"
    assert (repo / "other.txt").read_text(encoding="utf-8") == (
        "preserved bytes\n"
    )


def test_preserved_implementation_removed_from_plan_blocks_cleanup(
    tmp_path: Path,
) -> None:
    """Journal preservation alone cannot adopt Plan-A bytes into Plan B."""
    from assay.primitives.implement import capture_workspace_delta

    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    removed_slice = Slice(
        id="removed",
        directive=ImplementationDirective(
            slice_id="removed",
            intent="produce bytes accepted only by the original plan",
            files_in_scope=("removed.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    surviving_slice = Slice(
        id="surviving",
        directive=ImplementationDirective(
            slice_id="surviving",
            intent="remain in the revised plan",
            files_in_scope=("surviving.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    original_tier = Tier(
        tier_id="t0",
        slices=(removed_slice, surviving_slice),
        depends_on=(),
    )
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(original_tier,),
        files={"removed.txt": "base\n", "surviving.txt": "base\n"},
    )
    baseline_spec = corpus.latest_spec()
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "plan-a" / "mutations",
        slice_id="removed",
        scope=("removed.txt",),
        sequence=0,
        spec=baseline_spec,
    )
    (repo / "removed.txt").write_text("accepted plan-a bytes\n", encoding="utf-8")
    diff, files, observed_base = capture_workspace_delta(
        cwd=str(repo),
        path_scope=("removed.txt",),
    )
    assert observed_base == base
    implementation = _real_impl(
        corpus,
        diff=diff,
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="removed",
        files_modified=files,
    )
    implementation = attach_resume_identity(
        corpus,
        baseline_spec,
        original_tier,
        removed_slice,
        implementation,
    )
    journal.prepare(implementation, keep=True)
    committed = corpus.commit(implementations=(implementation,))
    preserved_ref = str(committed.refs.implementation_refs[0])

    plan_b_tier = Tier(
        tier_id="t0",
        slices=(surviving_slice,),
        depends_on=(),
    )
    corpus.put_spec(
        corpus.latest_spec().with_tier_dag(
            TierDAG(tiers=(plan_b_tier,)),
        ),
    )
    _unregister_live(journal.transaction_id)

    report = reap_orphaned_dirt(corpus, repo)

    assert report.mutation_journals_preserved == ("removed",)
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert len(report.failures) == 1
    assert "lacks independent current-retained or landed authority" in (
        report.failures[0]
    )
    assert preserved_ref in report.failures[0]
    assert (repo / "removed.txt").read_text(encoding="utf-8") == (
        "accepted plan-a bytes\n"
    )
    assert not journal.journal_dir.exists()


def test_later_lifecycle_relanding_wins_over_implementation_ref_order(
    tmp_path: Path,
) -> None:
    """Lifecycle occurrences, not original ref indices, own final overlap."""
    from assay import LifecycleEvent

    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    slice_a = Slice(
        id="a",
        directive=ImplementationDirective(
            slice_id="a",
            intent="write the A image",
            files_in_scope=("shared.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    slice_b = Slice(
        id="b",
        directive=ImplementationDirective(
            slice_id="b",
            intent="write the B image",
            files_in_scope=("shared.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    slice_c = Slice(
        id="c",
        directive=ImplementationDirective(
            slice_id="c",
            intent="record an intervening failed image",
            files_in_scope=("shared.txt",),
            creates_paths=(),
            verify_mode="skip",
            blast_radius={"max_files": 1},
        ),
    )
    tier_a = Tier(tier_id="ta", slices=(slice_a,), depends_on=())
    tier_b = Tier(tier_id="tb", slices=(slice_b,), depends_on=())
    tier_c = Tier(tier_id="tc", slices=(slice_c,), depends_on=())
    repo, corpus, base = _real_multi_setup(
        tmp_path,
        tiers=(tier_a, tier_b, tier_c),
        files={"shared.txt": "base\n"},
    )
    initial_spec = corpus.latest_spec()

    (repo / "shared.txt").write_text("A\n", encoding="utf-8")
    impl_a = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
        slice_id="a",
        files_modified=("shared.txt",),
    )
    impl_a = attach_resume_identity(
        corpus,
        initial_spec,
        tier_a,
        slice_a,
        impl_a,
    )
    (repo / "shared.txt").write_text("B\n", encoding="utf-8")
    impl_b = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="success",
        base_commit=base,
        timestamp="2026-08-06T00:01:00+00:00",
        slice_id="b",
        files_modified=("shared.txt",),
    )
    impl_b = attach_resume_identity(
        corpus,
        initial_spec,
        tier_b,
        slice_b,
        impl_b,
    )
    committed = corpus.commit(implementations=(impl_a, impl_b))
    ref_a, ref_b = (str(ref) for ref in committed.refs.implementation_refs)

    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier tb: complete",
            affected_versions=(version,),
            execution=impl_b.execution,
        ),
    )
    (repo / "shared.txt").write_text("C failed\n", encoding="utf-8")
    impl_c = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:02:00+00:00",
        slice_id="c",
        files_modified=("shared.txt",),
    )
    impl_c = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier_c,
        slice_c,
        impl_c,
    )
    corpus.commit(implementations=(impl_c,))
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason="tier ta: complete",
            affected_versions=(version,),
            execution=impl_a.execution,
        ),
    )

    # A has the lower Implementation ref index, but its later lifecycle event
    # is the final accepted occurrence for this overlapping path.
    (repo / "shared.txt").write_text("A\n", encoding="utf-8")
    accepted = reap_orphaned_dirt(corpus, repo)
    assert accepted.failures == ()
    assert accepted.crashed_rollbacks_reaped == ()
    assert any(
        "slice c: failure diff is superseded" in item
        for item in accepted.legacy_mutations_skipped
    )

    (repo / "shared.txt").write_text("B\n", encoding="utf-8")
    rejected = reap_orphaned_dirt(corpus, repo)
    assert rejected.crashed_rollbacks_reaped == ()
    assert any(
        ref_a in failure
        and "positive Implementation" in failure
        and "not live" in failure
        for failure in rejected.failures
    )
    assert all(ref_b not in failure for failure in rejected.failures)


def test_public_reaper_rejects_local_config_include_fifo_without_hanging(
    tmp_path: Path,
) -> None:
    """Lock discovery cannot follow a blocking repository-local include."""
    import os
    import signal
    import sys

    repo, corpus, _tier, _slice, _base = _real_setup(tmp_path)
    before = (repo / "foo.txt").read_bytes()
    fifo = tmp_path / "hostile-config-include"
    subprocess.run(
        ["git", "config", "--local", "include.path", str(fifo)],
        cwd=repo,
        check=True,
    )
    os.mkfifo(fifo)
    script = """
import json
import sys
from assay.corpus.store import Corpus
from crucible.workflows._reaper import reap_orphaned_dirt

try:
    report = reap_orphaned_dirt(Corpus(sys.argv[2]), sys.argv[1])
except Exception as exc:
    print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
else:
    print(json.dumps({"failures": list(report.failures)}))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(repo), str(corpus.spec_dir)],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail("public reaper hung while inspecting local config include")

    assert process.returncode == 0, stderr
    outcome = json.loads(stdout)
    diagnostic = str(outcome.get("error") or outcome.get("failures"))
    assert "include" in diagnostic.lower()
    assert (repo / "foo.txt").read_bytes() == before


def test_lifecycle_only_journal_barrier_survives_repeated_startup(
    tmp_path: Path,
) -> None:
    """No later startup may forget generic accepted mutation scope."""
    from assay import LifecycleEvent

    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("accepted bytes\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    (repo / "foo.txt").write_text("original\n", encoding="utf-8")

    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=corpus.spec_dir / "runs" / "generic-land" / "mutations",
        slice_id="s0",
        scope=("foo.txt",),
        sequence=1,
        spec=corpus.latest_spec(),
    )
    (repo / "foo.txt").write_text("accepted bytes\n", encoding="utf-8")
    reason = "generic accepted finalization"
    journal.prepare_lifecycle_acceptance(kind="landed", reason=reason)
    corpus.commit(
        lifecycle_event_factory=lambda version: LifecycleEvent(
            kind="landed",
            reason=reason,
            affected_versions=(version,),
            execution=failure.execution,
        ),
    )
    _unregister_live(journal.transaction_id)

    first = reap_orphaned_dirt(corpus, repo)
    second = reap_orphaned_dirt(corpus, repo)

    for report in (first, second):
        assert report.mutation_journals_preserved == ("s0",)
        assert report.crashed_rollbacks_reaped == ()
        assert any(
            "lifecycle-only prepared_keep" in failure
            and "journal retained" in failure
            for failure in report.failures
        )
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "accepted bytes\n"
    )
    assert journal.journal_dir.exists()


def test_current_tool_authority_digest_is_required_for_legacy_cleanup(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("stale-tool dirt\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    recorded_digest = "sha256:" + "a" * 64
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
        tool_authority_digest=recorded_digest,
    )
    corpus.commit(implementations=(failure,))

    report = reap_orphaned_dirt(
        corpus,
        repo,
        tool_authority_digests={"s0": "sha256:" + "b" * 64},
    )

    assert any("not current-plan authority" in item for item in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "stale-tool dirt\n"


def test_real_corrupt_attached_record_blocks_cleanup(tmp_path: Path) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    committed = corpus.commit(implementations=(failure,))
    ref = str(committed.refs.implementation_refs[0])
    artifact = (
        corpus.spec_dir
        / "artifacts"
        / "implementations"
        / f"{ref.split(':', 1)[1]}.bin"
    )
    artifact.write_bytes(b"corrupt")

    report = reap_orphaned_dirt(corpus, repo)

    assert any(ref in failure for failure in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "dirty\n"


def test_real_unreadable_journal_blocks_attached_legacy_cleanup(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    failure = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    failure = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        failure,
    )
    corpus.commit(implementations=(failure,))
    broken = corpus.spec_dir / "runs" / "broken" / "mutations" / "0000_bad"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not-json", encoding="utf-8")

    report = reap_orphaned_dirt(corpus, repo)

    assert any("unreadable" in failure for failure in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "dirty\n"


def test_unreadable_newest_attached_ref_blocks_older_cleanup(
    tmp_path: Path,
) -> None:
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("older dirt")
    old_diff = _capture_diff(tmp_path)
    older = _mock_impl("s0", "hard_fail_max_turns", "sha256:older")
    corpus = _mock_corpus(
        [older],
        {"sha256:older": old_diff},
        cwd=tmp_path,
        attached_refs=["sha256:older", "sha256:missing"],
        artifact_refs=["sha256:missing", "sha256:older"],
    )

    report = reap_orphaned_dirt(corpus, tmp_path)

    assert any("sha256:missing" in failure for failure in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (tmp_path / "foo.txt").read_text() == "older dirt"


def test_unreadable_attached_lifecycle_blocks_cleanup(tmp_path: Path) -> None:
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("dirty")
    impl = _mock_impl("s0", "hard_fail_max_turns", "sha256:dirty")
    corpus = _mock_corpus(
        [impl],
        {"sha256:dirty": _capture_diff(tmp_path)},
        cwd=tmp_path,
    )
    initial = corpus.latest_spec.return_value
    mutated = dataclasses.replace(
        initial,
        lifecycle_event_refs=("sha256:missing-lifecycle",),
    )
    corpus.latest_spec.return_value = mutated
    versions_dir = Path(corpus.spec_dir) / "versions"
    next(versions_dir.glob("v0_*.json")).unlink()
    digest = hash_record(mutated).split(":", 1)[-1]
    (versions_dir / f"v0_{digest[:12]}.json").write_bytes(serialize(mutated))

    report = reap_orphaned_dirt(corpus, tmp_path)

    assert any("sha256:missing-lifecycle" in failure for failure in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert (tmp_path / "foo.txt").read_text() == "dirty"


def test_journal_failure_blocks_all_legacy_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _reaper
    from crucible.workflows._mutation_journal import RecoveryReport

    _init_git_repo(tmp_path)
    (tmp_path / "foo.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "foo.txt").write_text("legacy dirt")
    (tmp_path / "rogue.txt").write_text("legacy escape")
    impl = _mock_impl(
        "s0",
        "hard_fail_max_turns",
        "sha256:legacy",
        scope_escapes=("rogue.txt",),
    )
    corpus = _mock_corpus(
        [impl],
        {"sha256:legacy": _capture_diff(tmp_path)},
        cwd=tmp_path,
    )
    monkeypatch.setattr(
        _reaper,
        "_recover_mutation_journals_locked",
        lambda *args, **kwargs: RecoveryReport(
            recovered=(),
            preserved=(),
            failures=("exact journal recovery failed",),
        ),
    )

    report = _reaper.reap_orphaned_dirt(corpus, tmp_path)

    assert report.failures == ("exact journal recovery failed",)
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert (tmp_path / "foo.txt").read_text() == "legacy dirt"
    assert (tmp_path / "rogue.txt").read_text() == "legacy escape"


def test_spec_advance_before_mutation_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from assay import LifecycleEvent
    from crucible.workflows._partition import attach_resume_identity
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, tier, slice_, base_commit = _real_setup(tmp_path)
    (repo / "foo.txt").write_text("dirty\n", encoding="utf-8")
    impl = _real_impl(
        corpus,
        diff=_capture_diff(repo),
        classification="hard_fail_max_turns",
        base_commit=base_commit,
        timestamp="2026-08-06T00:00:00+00:00",
    )
    impl = attach_resume_identity(
        corpus,
        corpus.latest_spec(),
        tier,
        slice_,
        impl,
    )
    corpus.commit(implementations=(impl,))
    expected = corpus.latest_spec()
    original_commit = corpus.commit
    raced = False

    def _racing_commit(*args, **kwargs):
        nonlocal raced
        if not raced:
            raced = True
            original_commit(
                lifecycle_events=(
                    LifecycleEvent(
                        kind="annotated",
                        reason="concurrent writer",
                        affected_versions=(expected.version + 1,),
                        execution=impl.execution,
                    ),
                )
            )
        return original_commit(*args, **kwargs)

    monkeypatch.setattr(corpus, "commit", _racing_commit)

    report = reap_orphaned_dirt(corpus, repo, expected_spec=expected)

    assert any("lost Spec CAS" in failure for failure in report.failures)
    assert report.crashed_rollbacks_reaped == ()
    assert report.cleanup_settlement_ref is None
    assert (repo / "foo.txt").read_text(encoding="utf-8") == "dirty\n"


def test_expected_spec_rejects_hydration_equivalent_raw_replacement(
    tmp_path: Path,
) -> None:
    """A parent workflow's pre-reaper authority includes manifest bytes."""
    from assay import PlanPolicy
    from assay.canonical import parse
    from crucible.workflows._mutation_journal import (
        WorkspaceMutationJournal,
        _unregister_live,
    )
    from crucible.workflows._reaper import reap_orphaned_dirt

    repo, corpus, _tier, _slice, _base_commit = _real_setup(tmp_path)
    root = dataclasses.replace(corpus.latest_spec(), plan_policy=PlanPolicy())
    original_manifest, = tuple(
        (Path(corpus.spec_dir) / "versions").glob("v0_*.json")
    )
    original_manifest.unlink()
    legacy_payload = parse(serialize(root))
    del legacy_payload["plan_policy"][
        "max_verification_only_empty_scope_slices"
    ]
    legacy_blob = serialize(legacy_payload)
    legacy_digest = "sha256:" + hashlib.sha256(legacy_blob).hexdigest()
    legacy_manifest = (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{legacy_digest.split(':', 1)[1][:12]}.json"
    )
    legacy_manifest.write_bytes(legacy_blob)
    expected, expected_digest = corpus.get_spec_with_digest(0)
    assert expected == root
    assert str(expected_digest) == legacy_digest
    journal = WorkspaceMutationJournal.begin(
        cwd=repo,
        journal_root=(
            corpus.spec_dir / "runs" / "raw_authority_swap" / "mutations"
        ),
        slice_id="raw_authority_swap",
        scope=("foo.txt",),
        sequence=1,
        spec=expected,
    )
    (repo / "foo.txt").write_text("interrupted bytes\n", encoding="utf-8")
    _unregister_live(journal.transaction_id)

    legacy_manifest.unlink()
    replacement_blob = serialize(root)
    replacement_digest = "sha256:" + hashlib.sha256(replacement_blob).hexdigest()
    assert replacement_digest != legacy_digest
    (
        Path(corpus.spec_dir)
        / "versions"
        / f"v0_{replacement_digest.split(':', 1)[1][:12]}.json"
    ).write_bytes(replacement_blob)

    report = reap_orphaned_dirt(
        corpus,
        repo,
        expected_spec=expected,
        expected_spec_digest=legacy_digest,
    )

    assert any(
        "raw Spec authority changed before recovery" in failure
        for failure in report.failures
    )
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert report.mutation_journals_recovered == ()
    assert (repo / "foo.txt").read_text(encoding="utf-8") == (
        "interrupted bytes\n"
    )
    assert journal.journal_dir.exists()


def test_mixed_slices(tmp_path: Path) -> None:
    """Two slices, one failing + one success → only the failing one reaped."""
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "fail_file.txt").write_text("original")
    (tmp_path / "success_file.txt").write_text("original")
    _commit(tmp_path)
    (tmp_path / "fail_file.txt").write_text("crash dirt")
    fail_diff = _capture_diff(tmp_path)
    (tmp_path / "fail_file.txt").write_text("original")  # capture only one slice's diff
    (tmp_path / "success_file.txt").write_text("success landed")
    # success_diff intentionally not captured into corpus — we only care that
    # the file stays on disk.
    fail_impl = _mock_impl("fail", "hard_fail_max_turns", "sha256:fail")
    succ_impl = _mock_impl("succ", "success", "sha256:succ")
    # Re-introduce the fail dirt:
    (tmp_path / "fail_file.txt").write_text("crash dirt")
    corpus = _mock_corpus(
        [fail_impl, succ_impl],
        {"sha256:fail": fail_diff, "sha256:succ": b""},
        cwd=tmp_path,
    )
    report = reap_orphaned_dirt(corpus, tmp_path)
    assert report.crashed_rollbacks_reaped == ("fail",)
    assert (tmp_path / "fail_file.txt").read_text() == "original"
    assert (tmp_path / "success_file.txt").read_text() == "success landed"


def test_multi_candidate_failure_restores_entire_legacy_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from crucible.workflows import _reaper

    _init_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a0\n")
    (tmp_path / "b.txt").write_text("b0\n")
    (tmp_path / "staged.txt").write_text("stage0\n")
    _commit(tmp_path, "base")
    (tmp_path / "a.txt").write_text("a-dirty\n")
    a_diff = _capture_diff(tmp_path)
    (tmp_path / "a.txt").write_text("a0\n")
    (tmp_path / "b.txt").write_text("b-dirty\n")
    b_diff = _capture_diff(tmp_path)
    (tmp_path / "b.txt").write_text("b0\n")
    (tmp_path / "a.txt").write_text("a-dirty\n")
    (tmp_path / "b.txt").write_text("b-dirty\n")
    (tmp_path / "staged.txt").write_text("stage1\n")
    subprocess.run(["git", "add", "staged.txt"], cwd=tmp_path, check=True)
    before_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout
    corpus = _mock_corpus(
        [
            _mock_impl("a", "hard_fail_max_turns", "sha256:a"),
            _mock_impl("b", "hard_fail_max_turns", "sha256:b"),
        ],
        {"sha256:a": a_diff, "sha256:b": b_diff},
        cwd=tmp_path,
    )
    real_apply = _reaper._git_apply_reverse
    calls = 0

    def _fail_second_apply(diff_bytes, cwd):
        nonlocal calls
        calls += 1
        return real_apply(diff_bytes, cwd) if calls == 1 else False

    monkeypatch.setattr(_reaper, "_git_apply_reverse", _fail_second_apply)

    report = _reaper.reap_orphaned_dirt(corpus, tmp_path)

    assert calls == 2
    assert report.crashed_rollbacks_reaped == ()
    assert report.scope_escapes_deleted == ()
    assert any("transaction failed and restored" in item for item in report.failures)
    assert (tmp_path / "a.txt").read_text() == "a-dirty\n"
    assert (tmp_path / "b.txt").read_text() == "b-dirty\n"
    assert subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    ).stdout == before_status


def test_has_signal_covers_all_four_fields(tmp_path: Path) -> None:
    """Drift-skipped + failure-only runs surface in summary.json
    via has_signal even though reaped_anything is False."""
    from crucible.workflows._reaper import ReaperReport

    empty = ReaperReport((), (), (), ())
    assert empty.has_signal is False
    assert empty.reaped_anything is False

    drift_only = ReaperReport((), (), ("s0",), ())
    assert drift_only.reaped_anything is False
    assert drift_only.has_signal is True

    failures_only = ReaperReport((), (), (), ("slice s1: err",))
    assert failures_only.reaped_anything is False
    assert failures_only.has_signal is True

    reaped = ReaperReport(("s0",), (), (), ())
    assert reaped.reaped_anything is True
    assert reaped.has_signal is True


def test_unresolvable_diff_ref_captured_in_failures(tmp_path: Path) -> None:
    from crucible.workflows._reaper import reap_orphaned_dirt

    _init_git_repo(tmp_path)
    (tmp_path / "anchor.txt").write_text("anchor")
    _commit(tmp_path)
    impl = _mock_impl("s0", "hard_fail_max_turns", "sha256:missing")
    corpus = _mock_corpus([impl], {}, cwd=tmp_path)
    original_get_artifact = corpus.get_artifact.side_effect

    def _missing_diff_only(ref):
        if str(ref) == "sha256:missing":
            raise FileNotFoundError("not in corpus")
        return original_get_artifact(ref)

    corpus.get_artifact.side_effect = _missing_diff_only
    report = reap_orphaned_dirt(corpus, tmp_path)
    # The reaper logged a failure but didn't crash.
    assert report.crashed_rollbacks_reaped == ()
    assert any("s0" in f for f in report.failures)
