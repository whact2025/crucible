# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Absolute byte limits for DeepSWE candidate patches."""

from __future__ import annotations

import os


PATCH_LIMIT_ENV = "CRUCIBLE_DEEPSWE_MAX_PATCH_BYTES"
DEFAULT_MAX_PATCH_BYTES = 64 * 1024 * 1024
HARD_MAX_PATCH_BYTES = 512 * 1024 * 1024


def validate_patch_byte_limit(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("patch byte limit must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("patch byte limit must be an integer") from exc
    if parsed <= 0 or parsed > HARD_MAX_PATCH_BYTES:
        raise ValueError(
            "patch byte limit must be between 1 and "
            f"{HARD_MAX_PATCH_BYTES} bytes"
        )
    return parsed


def configured_patch_byte_limit() -> int:
    raw = os.environ.get(PATCH_LIMIT_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_PATCH_BYTES
    return validate_patch_byte_limit(raw.strip())


__all__ = [
    "DEFAULT_MAX_PATCH_BYTES",
    "HARD_MAX_PATCH_BYTES",
    "PATCH_LIMIT_ENV",
    "configured_patch_byte_limit",
    "validate_patch_byte_limit",
]
