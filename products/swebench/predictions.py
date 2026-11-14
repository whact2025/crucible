# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Emit predictions in the SWE-bench leaderboard contract.

The official ``swebench`` evaluation harness consumes a predictions
file shaped as JSONL with one object per line:

    {"instance_id": "django__django-15347",
     "model_patch": "<unified diff>",
     "model_name_or_path": "gpt-5.6-sol"}

This module renders + persists that file. Pure: stdlib only.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Prediction:
    """One agent prediction for one SWE-bench instance.

    ``model_patch`` is the unified diff the agent produced. An empty
    string is permitted (and required by the leaderboard contract for
    instances where the agent declined / errored) — the eval harness
    treats it as "no fix attempted".
    """

    instance_id: str
    model_patch: str
    model_name_or_path: str


def to_record(p: Prediction) -> dict:
    """Render one Prediction to the leaderboard JSON shape."""
    return {
        "instance_id": p.instance_id,
        "model_patch": p.model_patch,
        "model_name_or_path": p.model_name_or_path,
    }


def write_predictions_jsonl(
    predictions: Iterable[Prediction], path: Path,
) -> int:
    """Atomically and durably write predictions as JSONL to ``path``.

    The complete replacement is written and fsynced in the destination
    directory before ``os.replace`` publishes it.  A crash or serialization
    failure therefore leaves the prior incremental checkpoint intact rather
    than truncating it to a partial final consolidation.  Each line is one
    canonical JSON object; the harness tolerates the trailing newline.
    """
    with _prediction_lock(path):
        return _write_predictions_jsonl_unlocked(predictions, path)


def _write_predictions_jsonl_unlocked(
    predictions: Iterable[Prediction], path: Path,
) -> int:
    """Publish a complete replacement while the sidecar lock is held."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".tmp_{path.name}_{os.getpid()}_{secrets.token_hex(4)}"
    )
    count = 0
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(
                    json.dumps(to_record(prediction), ensure_ascii=False)
                    + "\n"
                )
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return count
    finally:
        temporary.unlink(missing_ok=True)


def append_prediction_jsonl(prediction: Prediction, path: Path) -> None:
    """Durably append one prediction to a JSONL checkpoint.

    Used by the live ``--implement`` loop to persist each prediction
    as soon as the instance finishes — so a Ctrl-C / OOM / crash
    after instance N doesn't lose instances 0..N-1. Multi-hour
    runs of 500 Verified instances at $0.50 each can no longer be
    incinerated by a single interrupted run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(to_record(prediction), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with _prediction_lock(path):
        fd = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            remaining = memoryview(line)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("failed to append SWE-bench prediction")
                remaining = remaining[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
    _fsync_directory(path.parent)


def merge_predictions_jsonl(
    predictions: Iterable[Prediction],
    path: Path,
) -> int:
    """Durably upsert predictions without clobbering sibling processes.

    Existing rows are read and repaired under the same sidecar lock used by
    append and full replacement.  Duplicate instance ids collapse to the last
    complete record; a torn final line is discarded.  Updated ids retain their
    original position and new ids append in caller order.
    """

    updates = tuple(predictions)
    with _prediction_lock(path):
        existing = _read_predictions_unlocked(path)
        for prediction in updates:
            existing[prediction.instance_id] = prediction
        return _write_predictions_jsonl_unlocked(existing.values(), path)


def _read_predictions_unlocked(path: Path) -> dict[str, Prediction]:
    if not path.exists():
        return {}
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        boundary = payload.rfind(b"\n")
        payload = payload[:boundary + 1] if boundary >= 0 else b""
    records: dict[str, Prediction] = {}
    for line_number, raw in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"predictions JSONL line {line_number} is invalid: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"predictions JSONL line {line_number} must be an object"
            )
        fields = ("instance_id", "model_patch", "model_name_or_path")
        if any(not isinstance(value.get(field), str) for field in fields):
            raise ValueError(
                f"predictions JSONL line {line_number} has invalid fields"
            )
        prediction = Prediction(
            instance_id=value["instance_id"],
            model_patch=value["model_patch"],
            model_name_or_path=value["model_name_or_path"],
        )
        records[prediction.instance_id] = prediction
    return records


@contextmanager
def _prediction_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Best-effort persistence barrier for a directory entry update."""

    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems do not support fsync on directory handles.
        pass


def write_predictions_json(
    predictions: Iterable[Prediction], path: Path,
) -> int:
    """Write predictions as a single JSON array to ``path``.

    Some downstream tools prefer the array form over JSONL; expose
    both. Returns the count.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [to_record(p) for p in predictions]
    path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(records)


__all__ = [
    "Prediction",
    "to_record",
    "write_predictions_jsonl",
    "write_predictions_json",
    "append_prediction_jsonl",
    "merge_predictions_jsonl",
]
