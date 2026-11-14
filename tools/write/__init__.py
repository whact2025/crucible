# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Write ToolSet — scoped source mutation and artifact publication.

All tools mutate the workspace; the recipe
layer typically gates this set behind a workflow-specific allowlist
(e.g. the Implementer phase gets it; the Critic gets read-only).

Schemas + handlers live in ``handlers.py`` (decorator-builder pattern).
"""

from __future__ import annotations

from .exact_create import build_exact_create_tools
from .handlers import (
    apply_patch,
    build_write_tools,
    builder,
    delete_source,
    materialize_artifact,
    write_source,
)

__all__ = [
    "build_write_tools",
    "builder",
    "write_source",
    "apply_patch",
    "delete_source",
    "materialize_artifact",
    "build_exact_create_tools",
]
