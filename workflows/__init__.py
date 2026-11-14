# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Filesystem-convention workflow layer for crucible.

Each module in this package declares a ``main()`` callable that
composes assay primitives + a backend adapter into a runnable
workflow. The dispatcher at ``crucible/workflows/__main__.py``
discovers them by filesystem convention:

    uv run python -m crucible.workflows <name> [args...]
    uv run python -m crucible.workflows --list

Equivalent direct form:

    uv run python -m crucible.workflows.<name>

Importing this package transitively registers crucible's
domain-specific Constraint kinds (``crucible.constraints`` → languages and
planner-visible authoring instructions) so workflows can resolve them when
loading authored Specs.

See ``assay/docs/ASSAY_ARCHITECTURE.md`` for the workflow model.
"""

from .. import constraints as _constraints  # noqa: F401
