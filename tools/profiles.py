# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Pre-built ``CompositeToolSet`` profiles for common workflow needs.

Each profile is a factory ``build_<name>_profile(context)`` that
returns a composed ``ToolSet`` covering one workflow archetype. Pick
the profile closest to your need; if you want a custom mix, compose
the per-domain factories yourself via ``crucible.tools.composite``.

Profiles:

  ``build_implementer_profile`` — source + write + build (host), source + write
    (container). Retained for explicitly trusted/manual use and negative
    qualification probes.

  ``build_autonomous_implementer_profile`` — source + process-hostable,
    directive-scoped source mutations only. For autonomous workflow authoring;
    workflow-owned gates compile, test, and materialize artifacts.

  ``build_exact_create_recovery_profile`` — source + one content-only writer whose path
    is bound by workflow code. For a missing declared output.

  ``build_breach_defender_profile`` — source + directive-scoped text writes.
    The workflow, rather than the model tool set, executes breach tests.

"""

from __future__ import annotations

from dataclasses import replace

from ._composite import CompositeToolSet, composite
from ._context import ToolContext
from .build import build_build_tools
from .source import build_source_tools
from .write import build_exact_create_tools, build_write_tools


def build_implementer_profile(context: ToolContext) -> CompositeToolSet:
    """source + write + build — for the implementer phase.

    When ``context.container_id`` is set, both the host build tool and an
    arbitrary in-container shell are absent.  The workflow owns deterministic
    build/test execution; model roles can only use source navigation and the
    directive-scoped write handlers.
    """
    domains = [
        build_source_tools(context),
        build_write_tools(context),
    ]
    if context.container_id is None:
        domains.append(build_build_tools(context))
    return composite(*domains)


def build_autonomous_implementer_profile(
    context: ToolContext,
) -> CompositeToolSet:
    """Source navigation plus crash-settleable source mutation only.

    Every mutating adapter shares the process-isolated host contract. That
    host deliberately accepts only the canonical ``write_source``,
    ``apply_patch``, and exact-scope ``delete_source`` handlers. Build/test
    execution and binary artifact materialization remain workflow-owned so a
    timeout or killed model turn cannot leave an untracked child mutation
    behind.
    """

    return composite(
        build_source_tools(context),
        build_write_tools(context).filter_to({
            "write_source",
            "apply_patch",
            "delete_source",
        }),
    )


def build_exact_create_recovery_profile(
    context: ToolContext,
    *,
    target_path: str,
) -> CompositeToolSet:
    """Source navigation plus one content-only, exact-bound writer.

    A fresh context keeps recovery authority independent from the ordinary
    implementer profile retained by the workflow.  The runtime path checks and
    the model-facing schema therefore agree: only ``target_path`` can be
    written, and no write call accepts a path argument.
    """

    exact_context = replace(
        context,
        write_scope=(target_path,),
        write_exact_scope=(target_path,),
    )
    return composite(
        build_source_tools(exact_context),
        build_exact_create_tools(
            exact_context,
            target_path=target_path,
        ),
    )


def build_breach_defender_profile(context: ToolContext) -> CompositeToolSet:
    """Source + bounded writes for breach author/repair roles.

    Model-authored attack descriptions and diagnostic output are untrusted
    data. These roles therefore receive no shell, build, or test-execution
    capability and no binary-artifact materialization capability; the workflow
    executes tests deterministically after each text-write phase and supplies
    a structured result envelope.
    """
    return composite(
        build_source_tools(context),
        build_write_tools(context).filter_to({"write_source", "apply_patch"}),
    )


__all__ = [
    "build_autonomous_implementer_profile",
    "build_implementer_profile",
    "build_exact_create_recovery_profile",
    "build_breach_defender_profile",
]
