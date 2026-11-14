# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Tests for lineage- and plan-bound tier resume partitioning."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from assay import Implementation, ProjectContext, Spec
from assay.corpus.store import Corpus
from assay.types.dag import ImplementationDirective, Slice, Tier, TierDAG


def _impl(slice_id: str, classification: str, suffix: str = "1") -> MagicMock:
    impl = MagicMock()
    impl.slice_id = slice_id
    impl.classification = classification
    impl.diff_ref = f"sha256:{slice_id}_{suffix}"
    impl.base_commit = "base"
    impl.files_modified = ()
    impl.artifacts = {}
    return impl


def _tier(
    slice_ids: tuple[str, ...],
    *,
    no_write: frozenset[str] = frozenset(),
    intent_suffix: str = "",
) -> Tier:
    slices = tuple(
        Slice(
            id=sid,
            directive=ImplementationDirective(
                slice_id=sid,
                intent=f"implement {sid}{intent_suffix}",
                files_in_scope=() if sid in no_write else (f"{sid}.py",),
                creates_paths=(),
                verify_mode="skip",
                blast_radius={},
            ),
        )
        for sid in slice_ids
    )
    return Tier(tier_id="t0", slices=slices, depends_on=())


def _spec(
    tier: Tier,
    refs: tuple[str, ...],
    *,
    base_commit: str = "base",
    runners: dict[str, str] | None = None,
    runner_parameters: dict[str, object] | None = None,
    constraints: tuple[object, ...] = (),
):
    return SimpleNamespace(
        tier_dag=TierDAG(tiers=(tier,)),
        project_context=SimpleNamespace(
            project="test",
            repo_url="https://example.invalid/test.git",
            base_commit=base_commit,
            branch="main",
            submodule_path=None,
            runners=runners or {},
            runner_parameters=runner_parameters or {},
        ),
        implementation_refs=refs,
        constraints=constraints,
        schema_version=1,
        intent="test",
        done_criteria=(),
        depends_on=(),
        plan_policy=None,
    )


def _corpus(
    tier: Tier,
    impls: list[MagicMock],
    *,
    attached: list[MagicMock] | None = None,
    bind_identity: bool = True,
    tool_authority_digest: str | None = None,
):
    """Build a corpus whose Spec attachment order is authoritative."""
    from crucible.workflows._partition import _resume_identity_payload

    attached = impls if attached is None else attached
    spec = _spec(tier, tuple(impl.diff_ref for impl in attached))
    records = {impl.diff_ref: impl for impl in impls}
    blobs: dict[str, bytes] = {}
    if bind_identity:
        by_slice = {slice_.id: slice_ for slice_ in tier.slices}
        for index, impl in enumerate(impls):
            slice_ = by_slice.get(impl.slice_id)
            if slice_ is None:
                continue
            identity_ref = f"sha256:identity_{index}"
            impl.artifacts = {"resume_identity": identity_ref}
            blobs[identity_ref] = json.dumps(
                _resume_identity_payload(
                    spec,
                    tier,
                    slice_,
                    impl,
                    tool_authority_digest=tool_authority_digest,
                ),
                sort_keys=True,
            ).encode("utf-8")

    corpus = MagicMock()
    corpus.latest_spec.return_value = spec
    corpus.get_record.side_effect = lambda ref, as_type: records[ref]
    corpus.get_artifact.side_effect = lambda ref: blobs[ref]

    def get_artifact_bounded(ref, *, max_bytes):
        payload = blobs[ref]
        if len(payload) > max_bytes:
            raise ValueError("artifact exceeds ceiling")
        return payload

    corpus.get_artifact_bounded.side_effect = get_artifact_bounded
    corpus.list_artifacts.return_value = iter(records)
    return corpus, spec


def _tier_manifest_fixture(
    tmp_path: Path,
    *,
    disposition: str = "passed",
    handoff_obligation: bool = False,
    scoped_obligation: bool = False,
) -> tuple[Corpus, Spec, Tier, Implementation, Implementation, str]:
    """Publish one real atomic manifest over no-surface A plus sibling B1."""

    from crucible.workflows import _gate_evidence, _partition
    from crucible.workflows._build_verify import (
        BuildVerifyOutcome,
        RunnerResult,
    )
    from crucible.workflows._run import exec_ctx_now

    assert not (handoff_obligation and scoped_obligation)
    tier = _tier(
        ("a", "b"),
        no_write=(frozenset() if scoped_obligation else frozenset({"a"})),
    )
    base = Spec(
        version=0,
        parent=None,
        root=None,
        schema_version=1,
        project_context=ProjectContext(
            project="tier-manifest-partition-test",
            repo_url="https://example.invalid/tier-manifest.git",
            base_commit="base",
            branch="main",
            submodule_path=None,
            runners={},
        ),
        execution=None,
        intent="Prove atomic Tier verification resume authority.",
        constraints=(),
        done_criteria=(),
        plan_policy=None,
    )
    corpus = Corpus(tmp_path / "corpus")
    corpus.put_spec(base)
    corpus.put_spec(base.with_tier_dag(TierDAG(tiers=(tier,))))
    spec = corpus.latest_spec()
    assert spec is not None

    candidate_ref = corpus.put_artifact(
        b"atomic tier candidate",
        kind="diffs",
        name_hint="t0_union.diff",
    )
    if disposition == "passed":
        outcome = BuildVerifyOutcome(
            "success",
            build=RunnerResult(
                role="build",
                runner="tests.fake_build",
                returncode=0,
                values={},
                raw_output=b"{}",
                stderr=b"",
                duration_s=0.1,
            ),
            verify=None,
        )
        skipped_by_policy = False
    elif disposition == "unavailable":
        outcome = BuildVerifyOutcome(
            "runner_missing",
            build=None,
            verify=None,
        )
        skipped_by_policy = False
    elif disposition == "waived":
        outcome = None
        skipped_by_policy = True
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported disposition {disposition!r}")
    candidate_files = (
        ("a.py", "b.py") if scoped_obligation else ("b.py",)
    )
    evidence_ref = _gate_evidence.persist_verification_evidence(
        corpus,
        tier_id=tier.tier_id,
        slice_id="a",
        spec_execution_authority_hash=(
            _partition.spec_execution_structure_hash(spec)
        ),
        candidate_diff_ref=candidate_ref,
        base_commit="base",
        files_modified=candidate_files,
        outcome=outcome,
        skipped_by_policy=skipped_by_policy,
        build_timeout_s=30.0,
        verify_timeout_s=30.0,
        name_hint="a_tier_evidence",
    )

    def implementation(
        slice_id: str,
        *,
        diff: bytes,
        files_modified: tuple[str, ...],
        classification: str,
        recipe: str = "tier",
        artifacts: dict[str, object] | None = None,
    ) -> Implementation:
        return Implementation(
            slice_id=slice_id,
            base_commit="base",
            result_commit=None,
            diff_ref=corpus.put_artifact(
                diff,
                kind="diffs",
                name_hint=f"{slice_id}.diff",
            ),
            transcript_ref=corpus.put_artifact(
                b"[]",
                kind="transcripts",
                name_hint=f"{slice_id}.json",
            ),
            files_modified=files_modified,
            classification=classification,
            execution=exec_ctx_now(recipe),
            scope_escapes=(),
            artifacts=artifacts or {},
        )

    a = implementation(
        "a",
        diff=b"a1" if scoped_obligation else b"",
        files_modified=("a.py",) if scoped_obligation else (),
        classification=(
            "success" if scoped_obligation else "soft_fail_no_changes"
        ),
        recipe=(
            "tier_final_workspace_binding"
            if scoped_obligation
            else "handoff" if handoff_obligation else "tier"
        ),
        artifacts={
            _gate_evidence.TIER_VERIFICATION_ARTIFACT: evidence_ref,
        },
    )
    b = implementation(
        "b",
        diff=b"b1",
        files_modified=candidate_files,
        classification="success",
    )
    by_id = {item.id: item for item in tier.slices}
    a = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        by_id["a"],
        a,
    )
    b = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        by_id["b"],
        b,
    )
    manifest_ref = _gate_evidence.persist_tier_verification_manifest(
        corpus,
        tier_id=tier.tier_id,
        spec_execution_authority_hash=(
            _partition.spec_execution_structure_hash(spec)
        ),
        project_base_commit=spec.project_context.base_commit,
        candidate_diff_ref=candidate_ref,
        base_commit="base",
        files_modified=candidate_files,
        gate="verify",
        obligation_slice_ids=("a",),
        current_implementations=(a, b),
        dependency_implementations=(),
        verification_evidence_refs=(evidence_ref,),
    )
    a = replace(
        a,
        artifacts={
            **(a.artifacts or {}),
            _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref,
        },
    )
    b = replace(
        b,
        artifacts={
            **(b.artifacts or {}),
            _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref,
        },
    )
    corpus.commit(implementations=(a, b))
    return corpus, spec, tier, a, b, str(manifest_ref)


def test_empty_corpus_all_pending():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a", "b", "c"))
    corpus, spec = _corpus(tier, [])
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a", "b", "c")


def test_bound_success_lands_and_preserves_pending_order():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("zeta", "alpha", "mu", "beta"))
    impls = [_impl("alpha", "success"), _impl("mu", "success")]
    corpus, spec = _corpus(tier, impls)
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert set(landed) == {"alpha", "mu"}
    assert tuple(s.id for s in pending) == ("zeta", "beta")


def test_resume_identity_default_reader_has_explicit_byte_ceiling():
    from crucible.workflows import _partition

    tier = _tier(("a",))
    corpus, spec = _corpus(tier, [_impl("a", "success")])
    observed_limits: list[int] = []

    def reject_oversized(_ref, *, max_bytes):
        observed_limits.append(max_bytes)
        raise ValueError("oversized resume identity")

    corpus.get_artifact_bounded.side_effect = reject_oversized

    landed, pending = _partition.partition_for_resume(corpus, tier, spec=spec)

    assert landed == {}
    assert tuple(item.id for item in pending) == ("a",)
    assert observed_limits == [_partition._RESUME_IDENTITY_MAX_BYTES]
    corpus.get_artifact.assert_not_called()


def test_required_verification_partition_separates_replay_from_implementation(
    monkeypatch,
):
    from crucible.workflows import _gate_evidence
    from crucible.workflows._partition import (
        partition_for_required_verification,
        spec_execution_structure_hash,
    )

    tier = _tier(("native", "verified", "replay", "implement"))
    native = _impl("native", "success")
    verified = _impl("verified", "success")
    replay = _impl("replay", "success")
    failed = _impl("implement", "verify_failed")
    verified.execution = {"recipe": "handoff"}
    replay.execution = SimpleNamespace(recipe="handoff")
    corpus, spec = _corpus(tier, [native, verified, replay, failed])
    calls = []

    def matches(
        actual_corpus,
        implementation,
        *,
        tier_id,
        slice_id,
        spec_execution_authority_hash,
        require_gate,
    ):
        calls.append((
            actual_corpus,
            implementation,
            tier_id,
            slice_id,
            spec_execution_authority_hash,
            require_gate,
        ))
        return slice_id == "verified"

    monkeypatch.setattr(
        _gate_evidence,
        "verification_matches_implementation",
        matches,
    )
    partition = partition_for_required_verification(
        corpus,
        tier,
        spec=spec,
    )

    assert set(partition.verified_retained) == {"native", "verified"}
    assert set(partition.verification_pending) == {"replay"}
    assert tuple(
        item.id for item in partition.implementation_pending
    ) == ("implement",)
    assert set(partition.retained) == {"native", "verified", "replay"}
    assert [call[3] for call in calls] == ["verified", "replay"]
    assert all(call[0] is corpus and call[2] == "t0" for call in calls)
    assert all(call[4] == spec_execution_structure_hash(spec) for call in calls)
    assert all(call[5] is True for call in calls)


def test_required_verification_partition_retains_waived_handoff_for_replay():
    from crucible.workflows._partition import (
        partition_for_required_verification,
        partition_for_resume,
    )

    tier = _tier(("a",))
    implementation = _impl("a", "success")
    implementation.execution = {"recipe": "handoff"}
    corpus, spec = _corpus(tier, [implementation])

    # The legacy structural API remains source-compatible for callers that
    # explicitly waive deterministic verification.
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert set(landed) == {"a"}
    assert pending == ()

    required = partition_for_required_verification(corpus, tier, spec=spec)
    assert required.verified_retained == {}
    assert set(required.verification_pending) == {"a"}
    assert required.implementation_pending == ()


@pytest.mark.parametrize(
    ("disposition", "manifest_verified"),
    (("passed", True), ("waived", False), ("unavailable", False)),
)
def test_required_verification_partition_consumes_exact_tier_manifest(
    tmp_path: Path,
    disposition: str,
    manifest_verified: bool,
) -> None:
    from crucible.workflows._partition import (
        partition_for_required_verification,
    )

    corpus, _spec, tier, _a, _b, manifest_ref = _tier_manifest_fixture(
        tmp_path,
        disposition=disposition,
    )
    current = corpus.latest_spec()
    assert current is not None
    partition = partition_for_required_verification(
        corpus,
        tier,
        spec=current,
    )

    assert partition.implementation_pending == ()
    assert partition.tier_manifest_obligations == ("a",)
    assert partition.tier_manifest_ref == manifest_ref
    assert partition.tier_manifest_verified is manifest_verified
    assert set(partition.verified_retained) == (
        {"a", "b"} if manifest_verified else {"b"}
    )
    assert set(partition.verification_pending) == (
        set() if manifest_verified else {"a"}
    )


@pytest.mark.parametrize("invalidity", ("missing", "corrupt", "differing"))
def test_required_verification_partition_fails_closed_on_manifest_invalidity(
    tmp_path: Path,
    invalidity: str,
) -> None:
    from crucible.workflows import _gate_evidence
    from crucible.workflows._partition import (
        partition_for_required_verification,
    )

    corpus, _spec, tier, a, b, _manifest_ref = _tier_manifest_fixture(
        tmp_path,
    )
    if invalidity == "missing":
        changed = tuple(
            replace(
                implementation,
                artifacts={
                    key: value
                    for key, value in (implementation.artifacts or {}).items()
                    if key
                    != _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT
                },
            )
            for implementation in (a, b)
        )
    else:
        bad_ref = corpus.put_artifact(
            b"not a Tier verification manifest",
            kind="tier_verification_manifests",
            name_hint=f"{invalidity}.json",
        )
        targets = (a, b) if invalidity == "corrupt" else (b,)
        replacements = {
            implementation.slice_id: replace(
                implementation,
                artifacts={
                    **(implementation.artifacts or {}),
                    _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT: bad_ref,
                },
            )
            for implementation in targets
        }
        changed = tuple(
            replacements.get(implementation.slice_id, implementation)
            for implementation in (a, b)
        )
    corpus.commit(implementations=changed)

    current = corpus.latest_spec()
    assert current is not None
    partition = partition_for_required_verification(
        corpus,
        tier,
        spec=current,
    )

    assert partition.implementation_pending == ()
    assert partition.tier_manifest_obligations == ("a",)
    assert partition.tier_manifest_verified is False
    assert set(partition.verified_retained) == {"b"}
    assert set(partition.verification_pending) == {"a"}
    if invalidity == "differing":
        assert partition.tier_manifest_ref is None


@pytest.mark.parametrize(
    "scoped_obligation",
    (False, True),
    ids=("no-surface", "scoped-tier-terminal"),
)
def test_required_verification_partition_rejects_superseded_sibling_b2(
    tmp_path: Path,
    scoped_obligation: bool,
) -> None:
    from crucible.workflows import _gate_evidence
    from crucible.workflows._partition import (
        attach_resume_identity,
        partition_for_required_verification,
    )

    corpus, _spec, tier, a, b1, manifest_ref = _tier_manifest_fixture(
        tmp_path,
        scoped_obligation=scoped_obligation,
    )
    current = corpus.latest_spec()
    assert current is not None
    initial = partition_for_required_verification(
        corpus,
        tier,
        spec=current,
    )
    assert initial.tier_manifest_obligations == ("a",)
    assert initial.tier_manifest_verified is True
    assert set(initial.verified_retained) == {"a", "b"}
    assert initial.verification_pending == {}
    if scoped_obligation:
        assert a.execution.recipe == "tier_final_workspace_binding"
        assert a.artifacts is not None
        assert _gate_evidence.TIER_VERIFICATION_ARTIFACT in a.artifacts

    b_slice = next(item for item in tier.slices if item.id == "b")
    b2 = replace(
        b1,
        diff_ref=corpus.put_artifact(
            b"b2",
            kind="diffs",
            name_hint="b2.diff",
        ),
        artifacts={
            **(b1.artifacts or {}),
            _gate_evidence.TIER_VERIFICATION_MANIFEST_ARTIFACT: manifest_ref,
        },
    )
    b2 = attach_resume_identity(corpus, current, tier, b_slice, b2)
    corpus.commit(implementations=(b2,))

    latest = corpus.latest_spec()
    assert latest is not None
    partition = partition_for_required_verification(
        corpus,
        tier,
        spec=latest,
    )

    assert partition.implementation_pending == ()
    assert partition.tier_manifest_ref == manifest_ref
    assert partition.tier_manifest_verified is False
    assert set(partition.verified_retained) == {"b"}
    assert set(partition.verification_pending) == {"a"}


def test_tier_manifest_authoritatively_promotes_handoff_pending_slice(
    tmp_path: Path,
) -> None:
    from crucible.workflows._partition import (
        partition_for_required_verification,
    )

    corpus, _spec, tier, _a, _b, manifest_ref = _tier_manifest_fixture(
        tmp_path,
        handoff_obligation=True,
    )
    current = corpus.latest_spec()
    assert current is not None
    partition = partition_for_required_verification(
        corpus,
        tier,
        spec=current,
    )

    assert partition.implementation_pending == ()
    assert partition.tier_manifest_ref == manifest_ref
    assert partition.tier_manifest_verified is True
    assert set(partition.verified_retained) == {"a", "b"}
    assert partition.verification_pending == {}


def test_manifest_dependency_bindings_follow_producer_dfs_and_slice_order(
    monkeypatch,
) -> None:
    from crucible.workflows import _partition

    def tier(
        tier_id: str,
        slice_ids: tuple[str, ...],
        depends_on: tuple[str, ...] = (),
    ) -> Tier:
        return Tier(
            tier_id=tier_id,
            depends_on=depends_on,
            slices=tuple(
                Slice(
                    id=slice_id,
                    directive=ImplementationDirective(
                        slice_id=slice_id,
                        intent=f"implement {slice_id}",
                        files_in_scope=(f"{slice_id}.py",),
                        creates_paths=(),
                        verify_mode="skip",
                        blast_radius={},
                    ),
                )
                for slice_id in slice_ids
            ),
        )

    root = tier("root", ("root-a", "root-b"))
    left = tier("left", ("left-a",), ("root",))
    right = tier("right", ("right-a", "right-b"), ("root",))
    current = tier("current", ("current-a",), ("right", "left"))
    spec = SimpleNamespace(
        tier_dag=TierDAG(tiers=(root, left, right, current)),
    )
    retained = {
        candidate.tier_id: {
            slice_.id: _impl(slice_.id, "success")
            for slice_ in candidate.slices
        }
        for candidate in (root, left, right)
    }

    def partition(_corpus, candidate, *, spec):
        assert spec is not None
        return _partition.VerificationResumePartition(
            verified_retained=retained[candidate.tier_id],
            verification_pending={},
            implementation_pending=(),
        )

    monkeypatch.setattr(
        _partition,
        "partition_for_required_verification",
        partition,
    )
    bindings = _partition._manifest_dependency_implementations(
        MagicMock(),
        spec,
        current,
    )

    assert bindings is not None
    assert [
        (tier_id, implementation.slice_id)
        for tier_id, implementation in bindings
    ] == [
        ("root", "root-a"),
        ("root", "root-b"),
        ("right", "right-a"),
        ("right", "right-b"),
        ("left", "left-a"),
    ]


def test_legacy_success_without_resume_identity_retries():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    corpus, spec = _corpus(tier, [_impl("a", "success")], bind_identity=False)
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_soft_fail_with_declared_write_surface_retries():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    corpus, spec = _corpus(tier, [_impl("a", "soft_fail_no_changes")])
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_operator_waived_soft_fail_with_declared_write_surface_lands():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    implementation = _impl("a", "soft_fail_no_changes")
    implementation.execution = {"recipe": "handoff"}
    corpus, spec = _corpus(tier, [implementation])

    landed, pending = partition_for_resume(corpus, tier, spec=spec)

    assert set(landed) == {"a"}
    assert pending == ()


def test_soft_fail_for_intentionally_no_write_slice_lands():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",), no_write=frozenset({"a"}))
    corpus, spec = _corpus(tier, [_impl("a", "soft_fail_no_changes")])
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert set(landed) == {"a"}
    assert pending == ()


def test_failure_classifications_retry():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a", "b", "c"))
    impls = [
        _impl("a", "hard_fail_max_turns"),
        _impl("b", "build_failed"),
        _impl("c", "verify_failed"),
    ]
    corpus, spec = _corpus(tier, impls)
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a", "b", "c")


def test_passing_declared_test_diagnostic_never_satisfies_resume():
    from crucible.workflows._gate_evidence import (
        DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT,
    )
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    implementation = _impl(
        "a",
        "hard_fail_missing_declared_test_mutation",
    )
    implementation.artifacts = {
        DECLARED_TEST_EVIDENCE_PENDING_ARTIFACT: "sha256:" + "e" * 64,
    }
    corpus, spec = _corpus(tier, [implementation])

    landed, pending = partition_for_resume(corpus, tier, spec=spec)

    assert landed == {}
    assert tuple(item.id for item in pending) == ("a",)


def test_latest_attached_ref_wins_without_timestamp_guessing():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    first = _impl("a", "success", "first")
    retry = _impl("a", "hard_fail_max_turns", "retry")
    corpus, spec = _corpus(tier, [first, retry], attached=[first, retry])
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_orphan_success_not_attached_to_latest_spec_is_ignored():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    orphan = _impl("a", "success")
    corpus, spec = _corpus(tier, [orphan], attached=[])
    landed, pending = partition_for_resume(corpus, tier, spec=spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_resume_identity_rejects_changed_directive():
    from crucible.workflows._partition import partition_for_resume

    old_tier = _tier(("a",))
    impl = _impl("a", "success")
    corpus, _old_spec = _corpus(old_tier, [impl])
    changed_tier = _tier(("a",), intent_suffix=" differently")
    changed_spec = _spec(changed_tier, (impl.diff_ref,))
    corpus.latest_spec.return_value = changed_spec

    landed, pending = partition_for_resume(corpus, changed_tier, spec=changed_spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_resume_identity_rejects_changed_project_base():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    impl = _impl("a", "success")
    corpus, _old_spec = _corpus(tier, [impl])
    changed_spec = _spec(tier, (impl.diff_ref,), base_commit="new-base")
    corpus.latest_spec.return_value = changed_spec

    landed, pending = partition_for_resume(corpus, tier, spec=changed_spec)
    assert landed == {}
    assert tuple(s.id for s in pending) == ("a",)


def test_resume_identity_binds_current_tool_authority_digest():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    impl = _impl("a", "success")
    original_digest = "sha256:" + "a" * 64
    corpus, spec = _corpus(
        tier,
        [impl],
        tool_authority_digest=original_digest,
    )

    landed, pending = partition_for_resume(
        corpus,
        tier,
        spec=spec,
        tool_authority_digests={"a": original_digest},
    )
    assert set(landed) == {"a"}
    assert pending == ()

    landed, pending = partition_for_resume(
        corpus,
        tier,
        spec=spec,
        tool_authority_digests={"a": "sha256:" + "b" * 64},
    )
    assert landed == {}
    assert tuple(item.id for item in pending) == ("a",)


def test_resume_identity_preserves_operator_handoff_authority():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    impl = _impl("a", "success")
    impl.execution = SimpleNamespace(recipe="handoff")
    corpus, spec = _corpus(tier, [impl], tool_authority_digest=None)

    landed, pending = partition_for_resume(
        corpus,
        tier,
        spec=spec,
        tool_authority_digests={"a": "sha256:" + "b" * 64},
    )

    assert set(landed) == {"a"}
    assert pending == ()


def test_resume_identity_writer_preserves_operator_handoff_authority(tmp_path):
    from assay import ExecutionContext
    from crucible.workflows import _partition

    tier = _tier(("a",), no_write=frozenset({"a"}))
    spec = _spec(tier, ())
    corpus = Corpus(tmp_path / "corpus")
    implementation = Implementation(
        slice_id="a",
        base_commit="base",
        result_commit=None,
        diff_ref=corpus.put_artifact(b"", kind="diffs"),
        transcript_ref=corpus.put_artifact(b"[]", kind="transcripts"),
        files_modified=(),
        classification="success",
        execution=ExecutionContext(
            timestamp="2026-09-06T00:00:00+00:00",
            machine="test",
            cwd="/test",
            toolchain={},
            backend=None,
            model=None,
            recipe="handoff",
            recipe_commit=None,
            gpu_model=None,
            run_id=None,
        ),
        scope_escapes=(),
    )

    bound = _partition.attach_resume_identity(
        corpus,
        spec,
        tier,
        tier.slices[0],
        implementation,
        tool_authority_digest="sha256:" + "b" * 64,
    )
    identity = json.loads(
        corpus.get_artifact(bound.artifacts["resume_identity"])
    )

    assert identity["tool_authority_digest"] is None


def test_resume_identity_rejects_missing_current_tool_authority():
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    impl = _impl("a", "success")
    corpus, spec = _corpus(
        tier,
        [impl],
        tool_authority_digest="sha256:" + "a" * 64,
    )

    landed, pending = partition_for_resume(
        corpus,
        tier,
        spec=spec,
        tool_authority_digests={},
    )

    assert landed == {}
    assert tuple(item.id for item in pending) == ("a",)


@pytest.mark.parametrize(
    "changed",
    [
        {"runners": {"verify": "different.runner"}},
        {
            "runner_parameters": {
                "per_language": {"rust": {"crate": "different-crate"}},
            },
        },
        {"constraints": ({"kind": "languages", "languages": ["python"]},)},
        {
            "runner_parameters": {
                "per_language": {"rust": {"cargo_features": ["gpu"]}},
            },
        },
        {
            "runner_parameters": {
                "per_language": {
                    "rust": {"cargo_test_args": ["--", "--test-threads=1"]},
                },
            },
        },
    ],
)
def test_resume_identity_rejects_changed_execution_authority(changed):
    from crucible.workflows._partition import partition_for_resume

    tier = _tier(("a",))
    impl = _impl("a", "success")
    corpus, old_spec = _corpus(tier, [impl])
    changed_spec = _spec(
        tier,
        (impl.diff_ref,),
        runners=changed.get("runners", old_spec.project_context.runners),
        runner_parameters=changed.get(
            "runner_parameters",
            old_spec.project_context.runner_parameters,
        ),
        constraints=changed.get("constraints", old_spec.constraints),
    )
    corpus.latest_spec.return_value = changed_spec

    landed, pending = partition_for_resume(corpus, tier, spec=changed_spec)
    assert landed == {}
    assert tuple(slice_.id for slice_ in pending) == ("a",)
