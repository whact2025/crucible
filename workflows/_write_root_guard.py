# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Refuse a host-routed write root that holds a directory of its own name.

Codex, Claude, DeepSeek and Kimi hand their source mutators to the isolated
mutation host. While ``ToolContext.workspace_relative_paths`` is set, that
host drops a leading path component equal to the name of its resolved root
without looking at the disk. The in-process handlers of the OpenAI-compatible
and local adapters keep the component when the path or its parent exists. In
a root that holds a directory of its own name the host therefore names the
root-level twin of the file a directive declares, past a scope root, an exact
list, and a deny entry.

Implement, Tier, Breach and Architect ask
:func:`host_route_write_root_refusal` before their run opens and end as
``setup_error`` when it answers with a message. Which adapters write through
the host is stated here and nowhere else: this module, the four refusals, and
their tests go together when both write rules read a leading root name alike.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from crucible.backends.config import load_routing
from crucible.tools._context import ToolContext


ISOLATED_HOST_WRITER_BACKENDS: frozenset[str] = frozenset({
    "claude",
    "codex",
    "deepseek",
    "kimi",
})
WRITE_ROOT_REFUSAL_ERROR_KIND = "write_root_name_collision"


def host_route_write_root_refusal(
    args: argparse.Namespace,
    root: Path | str,
    write_roles: tuple[str, ...],
) -> str | None:
    """Return the setup-error message for ``root``, or None to proceed.

    ``root`` is the directory the workflow's write context will use, after
    any worktree rebinding. ``write_roles`` are the roles the workflow will
    hand a write-capable ToolSet. Their adapters are read from the routing
    ``args`` selects, discovered exactly as ``backend_pool(args, ...)``
    discovers it, and only when the root can collide at all.
    """

    write_root = Path(root)
    # No workflow passes ``workspace_relative_paths``, so a context built for
    # the root as the workflows build theirs carries what their write
    # contexts carry; the host drops the root name only while it is set.
    if not ToolContext(
        workspace_root=write_root, repo_root=write_root,
    ).workspace_relative_paths:
        return None
    # The host names its root after resolving links. ``is_dir`` follows a
    # link, so a link to a directory counts and a regular file does not.
    resolved = write_root.resolve()
    if not resolved.name or not (resolved / resolved.name).is_dir():
        return None
    routing = load_routing(config_path=args.backend_config, spec_dir=args.spec)
    host_routed: list[str] = []
    for role in write_roles:
        backend = routing.for_role(role).backend
        if backend in ISOLATED_HOST_WRITER_BACKENDS:
            host_routed.append(f"{role} ({backend})")
    if not host_routed:
        return None
    name = resolved.name
    return (
        f"write root {resolved} holds a directory named like itself "
        f"({name!r}). Write roles {', '.join(host_routed)} reach it through "
        f"the isolated mutation host, which reads a leading {name!r} path "
        f"component as the root itself and would write {name}/<path> to "
        "<path>. Run tier or architect with --worktree, or give the "
        "checkout directory another name."
    )


__all__ = [
    "ISOLATED_HOST_WRITER_BACKENDS",
    "WRITE_ROOT_REFUSAL_ERROR_KIND",
    "host_route_write_root_refusal",
]
