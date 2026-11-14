# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""``runner_path_prefix`` — runner paths must be sane.

Every ``project_context.runners[role]`` value must be:
- a non-empty string
- relative (no leading ``/``)
- not contain ``..`` traversal segments
- not contain absolute Windows drives (``C:\\``)

Findings are High severity — a runner path that escapes the project
root either silently fails to import or executes arbitrary code from
outside the project tree.
"""

from __future__ import annotations

from assay import Evidence, Finding, Spec


def runner_path_prefix(spec: Spec) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    runners = spec.project_context.runners or {}
    for role, raw in runners.items():
        evidence = (
            Evidence(
                kind="runner_value",
                location=f"project_context.runners[{role!r}]",
                excerpt=str(raw),
            ),
        )
        if not isinstance(raw, str) or not raw:
            findings.append(
                Finding(
                    id=f"runner_empty_{role}",
                    severity="High",
                    category="runner_path",
                    evidence=evidence,
                    suggested_action=f"set runners[{role!r}] to a project-relative path",
                    source="deterministic",
                )
            )
            continue
        if raw.startswith("/"):
            findings.append(
                Finding(
                    id=f"runner_absolute_{role}",
                    severity="High",
                    category="runner_path",
                    evidence=evidence,
                    suggested_action=(
                        "use a project-relative path; absolute paths "
                        "are not portable across hosts"
                    ),
                    source="deterministic",
                )
            )
            continue
        if ".." in raw.replace("\\", "/").split("/"):
            findings.append(
                Finding(
                    id=f"runner_traversal_{role}",
                    severity="High",
                    category="runner_path",
                    evidence=evidence,
                    suggested_action="remove `..` traversal segments",
                    source="deterministic",
                )
            )
            continue
        # Windows-drive prefix (C:\, D:\, etc.).
        if len(raw) >= 3 and raw[1] == ":" and raw[2] in ("/", "\\"):
            findings.append(
                Finding(
                    id=f"runner_windows_drive_{role}",
                    severity="High",
                    category="runner_path",
                    evidence=evidence,
                    suggested_action=(
                        "use a forward-slash project-relative path"
                    ),
                    source="deterministic",
                )
            )
    return tuple(findings)
