# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Drivers: one per backend kind, plus the toy the oracle is proved on."""

from __future__ import annotations

from .base import (
    Bound,
    Driver,
    Lens,
    MatrixSetupError,
    get,
    register,
    registered_kinds,
)


def load() -> frozenset[str]:
    """Import every driver module, which registers its driver; return the kinds."""

    from . import toy  # noqa: F401,PLC0415 - registration on import

    return registered_kinds()


__all__ = [
    "Bound",
    "Driver",
    "Lens",
    "MatrixSetupError",
    "get",
    "load",
    "register",
    "registered_kinds",
]
