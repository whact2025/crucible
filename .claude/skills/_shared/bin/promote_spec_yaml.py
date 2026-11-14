# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Promote a hand-edited spec.yaml into the Corpus as the new latest manifest.

Workflows (goal / plan / validate / architect / ...) read the spec
they critique via ``Corpus.latest_spec()``, which scans
``<spec_dir>/versions/v<N>_*.json`` and returns the **highest N**. A
manual edit to ``<spec_dir>/spec.yaml`` is INVISIBLE to those
workflows until the new YAML is parsed and stored at a version >=
the current latest.

This helper does exactly that:

1. Strictly load the on-disk ``spec.yaml`` via ``Spec.load_v0`` (which
   resolves Crucible constraint kinds via the registry), rejecting generated
   lifecycle/evidence fields in operator-authored input.
2. Rebase the authored fields onto the authoritative latest Corpus
   lineage, preserving every append-only evidence reference. Generated
   ``tier_dag`` authority is cleared by default. An operator may retain
   exactly one existing DAG only by supplying its canonical digest via
   ``--retain-plan-digest``.
3. Call ``corpus.put_spec(spec)``, which writes
   ``versions/v<bumped_N>_<new_hash>.json``.

Usage::

    uv run --project <crucible> python promote_spec_yaml.py <spec_dir>

The new manifest's hash + path are printed on stdout. Prior
version manifests stay on disk as provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace as _dc_replace
from pathlib import Path


def _assert_manifest_filename_matches_content(spec_dir: Path, spec) -> None:
    """Fail closed when the latest content-addressed manifest was edited.

    Assay names a Spec manifest with a prefix of its canonical byte digest.
    Recompute that binding before trusting authored-field or append-only-ref
    state; parsing valid JSON alone cannot detect an in-place edit.
    """

    versions_dir = spec_dir / "versions"
    prefix = f"v{spec.version}_"
    candidates = sorted(versions_dir.glob(f"{prefix}*.json"))
    if len(candidates) != 1:
        raise ValueError(
            f"canonical Spec v{spec.version} has {len(candidates)} manifests"
        )
    manifest = candidates[0]
    observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected_name = f"{prefix}{observed[:12]}.json"
    if manifest.name != expected_name:
        raise ValueError(
            "canonical Spec manifest content hash does not match its filename: "
            f"{manifest.name}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_dir", type=Path,
                        help="Canonical spec dir containing spec.yaml")
    parser.add_argument(
        "--retain-plan-digest",
        default=None,
        help=(
            "Retain the latest canonical tier_dag only when its exact "
            "sha256:<64-hex> digest matches this value. Omit to clear the "
            "generated plan before replanning."
        ),
    )
    args = parser.parse_args(argv)

    spec_yaml = args.spec_dir / "spec.yaml"
    if not spec_yaml.is_file():
        print(f"ERROR: no spec.yaml at {spec_yaml}", file=sys.stderr)
        return 2

    # Importing crucible.workflows registers the LanguagesConstraint
    # kind (and any future crucible-specific kinds) into
    # CONSTRAINT_TYPES so Spec.load_v0 can resolve them.
    import crucible.workflows  # noqa: F401
    from assay.canonical import hash_record
    from assay.corpus.store import Corpus
    from assay.types.identity import ContentHash
    from assay.types.spec import Spec

    try:
        spec = Spec.load_v0(spec_yaml)
    except Exception as exc:
        print(f"ERROR: spec.yaml failed to parse: {exc}", file=sys.stderr)
        return 1

    try:
        corpus = Corpus(args.spec_dir)
        latest = corpus.latest_spec()
        retained_plan_digest: str | None = None
        if latest is None:
            if args.retain_plan_digest is not None:
                raise ValueError(
                    "--retain-plan-digest requires an existing canonical Spec"
                )
            target_version = spec.version
            before = sorted(
                p.name
                for p in (args.spec_dir / "versions").iterdir()
                if p.name.startswith(f"v{target_version}_") and p.name.endswith(".json")
            )
            digest = corpus.put_spec(spec)
        else:
            _assert_manifest_filename_matches_content(args.spec_dir, latest)
            expected_parent = ContentHash(hash_record(latest))
            if args.retain_plan_digest is not None:
                if latest.tier_dag is None:
                    raise ValueError("latest canonical Spec has no tier_dag to retain")
                retained_plan_digest = hash_record(latest.tier_dag)
                if args.retain_plan_digest != retained_plan_digest:
                    raise ValueError(
                        "--retain-plan-digest does not match the latest "
                        "canonical tier_dag: expected "
                        f"{retained_plan_digest}, got "
                        f"{args.retain_plan_digest}"
                    )

            target_version = latest.version + 1
            before = sorted(
                p.name
                for p in (args.spec_dir / "versions").iterdir()
                if p.name.startswith(f"v{target_version}_") and p.name.endswith(".json")
            )

            def promote(candidate: Spec) -> Spec:
                locked_latest = corpus.get_spec(candidate.version - 1)
                _assert_manifest_filename_matches_content(
                    args.spec_dir,
                    locked_latest,
                )
                if candidate.parent != expected_parent:
                    raise RuntimeError(
                        "canonical Spec changed during promotion; reload and "
                        "retry against the new latest version"
                    )
                retained_tier_dag = None
                if args.retain_plan_digest is not None:
                    if candidate.tier_dag is None:
                        raise RuntimeError(
                            "canonical tier_dag disappeared during promotion"
                        )
                    observed_digest = hash_record(candidate.tier_dag)
                    if observed_digest != args.retain_plan_digest:
                        raise RuntimeError(
                            "canonical tier_dag changed during promotion: "
                            f"expected {args.retain_plan_digest}, observed "
                            f"{observed_digest}"
                        )
                    retained_tier_dag = candidate.tier_dag
                return _dc_replace(
                    candidate,
                    schema_version=spec.schema_version,
                    project_context=spec.project_context,
                    intent=spec.intent,
                    constraints=spec.constraints,
                    done_criteria=spec.done_criteria,
                    plan_policy=spec.plan_policy,
                    depends_on=spec.depends_on,
                    tier_dag=retained_tier_dag,
                )

            committed = corpus.commit(transform=promote)
            spec = committed.spec
            digest = ContentHash(hash_record(spec))
    except Exception as exc:
        print(f"ERROR: spec promotion failed: {exc}", file=sys.stderr)
        return 1

    after = sorted(
        p.name
        for p in (args.spec_dir / "versions").iterdir()
        if p.name.startswith(f"v{spec.version}_") and p.name.endswith(".json")
    )
    new_manifests = sorted(set(after) - set(before))

    print(f"promoted spec.yaml -> Corpus")
    print(f"  digest: {digest}")
    print(f"  version: v{spec.version}")
    if retained_plan_digest is None:
        print("  tier_dag: cleared (replanning required)")
    else:
        print(f"  tier_dag: retained {retained_plan_digest}")
    if new_manifests:
        for n in new_manifests:
            print(f"  new manifest: versions/{n}")
    else:
        print(f"  manifest: versions/{after[-1]} (already present — no-op)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
