# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Load + validate SWE-bench Verified instances from a local file.

The official dataset lives at ``princeton-nlp/SWE-bench_Verified`` on
HuggingFace (500 human-verified instances). Fetch it out-of-band:

    huggingface-cli download princeton-nlp/SWE-bench_Verified \\
        --repo-type dataset --local-dir <dest>

or convert the parquet to JSONL via the ``datasets`` library. This
module loads from whatever path the workflow's ``--dataset`` flag
points to, validates the canonical SWE-bench shape, and returns
typed records.

Pure: stdlib only (json), no network, no HF deps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# Canonical fields the SWE-bench Verified dataset carries per instance.
# Source: SWE-bench paper + HuggingFace dataset card. The adapter
# requires the seven below; ``hints_text`` / ``created_at`` /
# ``version`` / ``environment_setup_commit`` are accepted but optional.
_REQUIRED_FIELDS: tuple[str, ...] = (
    "instance_id",
    "repo",
    "base_commit",
    "patch",
    "test_patch",
    "problem_statement",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


class SwebenchInstanceError(ValueError):
    """Raised when an instance record fails canonical-shape validation."""


@dataclass(frozen=True)
class SwebenchInstance:
    """One SWE-bench Verified instance.

    ``patch`` is the GOLD diff (the human-authored fix); the adapter
    never feeds this to the agent — it stays in the record for
    parity-check / oracle baselines / debugging only.

    ``test_patch`` is applied at evaluation time alongside the
    agent's patch; it carries the hidden tests that
    ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` reference.

    ``problem_statement`` is the issue text the agent sees.
    """

    instance_id: str
    repo: str
    base_commit: str
    # NOTE: ``patch`` and ``test_patch`` are kept on the record for
    # AUDIT ONLY — neither is fed to the agent during ``--implement``.
    # ``patch`` is the gold human-authored fix; ``test_patch`` carries
    # the hidden FAIL_TO_PASS / PASS_TO_PASS tests applied at eval
    # time by the swebench harness. We persist them so postmortem can
    # compare agent output to the gold patch and inspect which tests
    # the eval will run, but they MUST NOT leak into the implementer
    # prompt under any circumstances.
    patch: str
    test_patch: str
    problem_statement: str
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    hints_text: str = ""
    created_at: str = ""
    version: str = ""
    environment_setup_commit: str = ""

    @property
    def repo_org(self) -> str:
        return self.repo.split("/")[0]

    @property
    def repo_name(self) -> str:
        return self.repo.split("/")[-1]


def _coerce_list(raw: object, *, field: str, instance_id: str) -> tuple[str, ...]:
    """SWE-bench dataset variants serialize FAIL_TO_PASS / PASS_TO_PASS
    sometimes as a JSON-encoded string, sometimes as a native list.
    Coerce to ``tuple[str, ...]`` either way; raise on anything else."""
    if isinstance(raw, list):
        return tuple(str(x) for x in raw)
    if isinstance(raw, str):
        # Best-effort: try JSON decode first, then fall back to a
        # single-element tuple containing the literal string.
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(decoded, list):
                    return tuple(str(x) for x in decoded)
        return (raw,) if raw else ()
    if raw is None:
        return ()
    raise SwebenchInstanceError(
        f"instance {instance_id!r}: field {field!r} must be a list or "
        f"JSON-encoded list string; got {type(raw).__name__}"
    )


def _validate_record(raw: dict) -> SwebenchInstance:
    """Validate one raw record and return a typed SwebenchInstance.

    Raises SwebenchInstanceError on missing required fields, malformed
    types, or empty mandatory strings.
    """
    if not isinstance(raw, dict):
        raise SwebenchInstanceError(
            f"instance record must be a JSON object, got {type(raw).__name__}"
        )
    missing = [f for f in _REQUIRED_FIELDS if f not in raw]
    if missing:
        raise SwebenchInstanceError(
            f"instance record missing required field(s): {missing}"
        )
    iid = str(raw["instance_id"])
    if not iid:
        raise SwebenchInstanceError("instance_id must be non-empty")
    base = str(raw["base_commit"])
    if not base:
        raise SwebenchInstanceError(f"instance {iid!r}: base_commit must be non-empty")
    repo_value = str(raw["repo"])
    if "/" not in repo_value:
        # SWE-bench's canonical shape is ``<org>/<name>`` (e.g.
        # ``django/django``). Downstream consumers split on ``/`` to
        # derive repo_org / repo_name + the GitHub URL; a record
        # missing the slash would silently degrade until it hit
        # ``_cached_worktree``'s mkdir or the URL constructor. Fail
        # at load time with a clear message instead.
        raise SwebenchInstanceError(
            f"instance {iid!r}: repo {repo_value!r} must be in "
            f"``<org>/<name>`` shape (e.g. ``django/django``)"
        )
    return SwebenchInstance(
        instance_id=iid,
        repo=str(raw["repo"]),
        base_commit=base,
        patch=str(raw.get("patch") or ""),
        test_patch=str(raw.get("test_patch") or ""),
        problem_statement=str(raw.get("problem_statement") or ""),
        fail_to_pass=_coerce_list(
            raw.get("FAIL_TO_PASS"),
            field="FAIL_TO_PASS",
            instance_id=iid,
        ),
        pass_to_pass=_coerce_list(
            raw.get("PASS_TO_PASS"),
            field="PASS_TO_PASS",
            instance_id=iid,
        ),
        hints_text=str(raw.get("hints_text") or ""),
        created_at=str(raw.get("created_at") or ""),
        version=str(raw.get("version") or ""),
        environment_setup_commit=str(raw.get("environment_setup_commit") or ""),
    )


def _load_dataset_text(
    text: str,
    *,
    source: Path | str,
) -> tuple[SwebenchInstance, ...]:
    """Parse already-decoded JSON/JSONL dataset text."""

    if not text.strip():
        return ()

    first_nonws = next((c for c in text if not c.isspace()), "")
    records: list[dict]
    if first_nonws == "[":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise SwebenchInstanceError(
                f"dataset {source}: top-level JSON parse failed: {e}"
            ) from e
        if not isinstance(parsed, list):
            raise SwebenchInstanceError(
                f"dataset {source}: top-level JSON must be a list, got "
                f"{type(parsed).__name__}"
            )
        records = parsed
    elif first_nonws == "{":
        # First non-whitespace is ``{`` — either JSONL (one record per
        # line) or a single top-level object (a parquet-row dump that's
        # not our shape). Disambiguate by parsing the whole file as
        # JSON: success → reject (single object, wrong shape); failure
        # → fall through to line-by-line JSONL parsing.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Multi-object file → JSONL path.
            records = []
        else:
            raise SwebenchInstanceError(
                f"dataset {source}: top-level JSON must be a list of "
                f"instance records, got a single object. (For JSONL, "
                f"place one object per line without enclosing array.)"
            )
        if not records:
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SwebenchInstanceError(
                        f"dataset {source}:{lineno}: JSONL parse failed: {e}"
                    ) from e
        return tuple(_validate_record(r) for r in records)
    else:
        records = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SwebenchInstanceError(
                    f"dataset {source}:{lineno}: JSONL parse failed: {e}"
                ) from e

    return tuple(_validate_record(r) for r in records)


def load_dataset_bytes(
    payload: bytes,
    *,
    source: Path | str = "<authenticated-dataset-bytes>",
) -> tuple[SwebenchInstance, ...]:
    """Validate one already-measured dataset payload without reopening a path."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SwebenchInstanceError(
            f"dataset {source}: UTF-8 decode failed: {exc}"
        ) from exc
    return _load_dataset_text(text, source=source)


def load_dataset(path: Path) -> tuple[SwebenchInstance, ...]:
    """Load + validate a SWE-bench dataset from a JSON or JSONL file.

    JSON is a top-level list; JSONL is one record per line. The payload is
    read once, then the measured bytes are parsed without reopening the path.
    """

    if not path.exists():
        raise FileNotFoundError(f"dataset path does not exist: {path}")
    return load_dataset_bytes(path.read_bytes(), source=path)


def filter_instances(
    instances: Iterable[SwebenchInstance],
    *,
    instance_ids: Iterable[str] | None = None,
    repos: Iterable[str] | None = None,
    limit: int | None = None,
) -> tuple[SwebenchInstance, ...]:
    """Apply optional filters in declaration order: instance_ids → repos → limit.

    All filters are inclusive (whitelists). Empty / None means "no filter."
    """
    out = list(instances)
    if instance_ids is not None:
        wanted = set(instance_ids)
        if wanted:
            out = [i for i in out if i.instance_id in wanted]
    if repos is not None:
        wanted_repos = set(repos)
        if wanted_repos:
            out = [i for i in out if i.repo in wanted_repos]
    if limit is not None and limit > 0:
        out = out[:limit]
    return tuple(out)


__all__ = [
    "SwebenchInstance",
    "SwebenchInstanceError",
    "load_dataset",
    "load_dataset_bytes",
    "filter_instances",
]
