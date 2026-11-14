# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Print a Corpus state report — used by the skill preflight.

Invocation:

    uv run --project <crucible_path> python <this_file> <spec_dir>

Prints `KEY=value` lines that callers can parse:

    spec_dir=<absolute path>
    spec_version=<int or "none">
    manifest_path=<relative to spec_dir or "none">
    workspace_root=<absolute path or "none">
    target=<target dir name under .crucible/specs/, or "">
    submodule_path=<from spec.yaml's project_context, or "">
    runs_total=<int>
    runs_design=<int>
    runs_build=<int>
    runs_benchmark=<int>
    runs_recovery=<int>
    last_run=<dirname or "none">

Strictly read-only — does NOT call ``Corpus(spec_dir)`` (its
constructor auto-creates the six subdirs as a side effect).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_WORKFLOWS_BY_STAGE = {
    "design": {"goal", "plan", "review", "validate"},
    "build": {"architect", "tier", "implement", "handoff", "breach"},
    "benchmark": {"swebench", "deepswe"},
    "recovery": {"reconcile", "journal_recover"},
}


def _git(spec_dir: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(spec_dir), *args],
            capture_output=True, text=True, check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _resolve_paths(spec_dir: Path) -> dict[str, str]:
    """Derive workspace_root + target + submodule_path.

    Canonical spec layout puts everything at the workspace root under
    `.crucible/specs/<target>/<spec_name>/`, so:

    - ``workspace_root`` comes from git (the outer repo / superproject).
    - ``target`` is the parent directory name of the spec_dir when it
      sits under `.crucible/specs/`; informational only.
    - ``submodule_path`` is read out of `spec.yaml` (the spec author
      sets `project_context.submodule_path` to the submodule the
      workflows should chdir into). The git-resolved repo_root is
      always equal to workspace_root for canonical specs, so we don't
      derive submodule_path from git.
    """
    workspace = _git(spec_dir, "rev-parse", "--show-superproject-working-tree") \
                 or _git(spec_dir, "rev-parse", "--show-toplevel") \
                 or ""

    target = _derive_target(spec_dir)

    project = _read_yaml_scalar(spec_dir / "spec.yaml", "project")
    submodule = _read_yaml_scalar(spec_dir / "spec.yaml", "submodule_path")

    return {
        "workspace_root": workspace or "none",
        "target": target,
        "project": project,
        "submodule_path": submodule,
    }


def _derive_target(spec_dir: Path) -> str:
    """Extract `<target>` from a path shaped `.../.crucible/specs/<target>/<spec>/`.

    Returns ``""`` if the path doesn't match the canonical layout.
    """
    try:
        parts = spec_dir.resolve().parts
    except OSError:
        return ""
    # Scan for the `<...>/.crucible/specs/<target>/<spec>` shape: we want
    # the segment two slots after `.crucible`. Walk left so the closest
    # `.crucible` wins (a spec_dir might contain `.crucible` further down,
    # though we don't expect it).
    for i in range(len(parts) - 1, -1, -1):
        if (
            parts[i] == ".crucible"
            and i + 2 < len(parts)
            and parts[i + 1] == "specs"
        ):
            return parts[i + 2]
    return ""


def _read_yaml_scalar(spec_yaml: Path, key: str) -> str:
    """Best-effort scan of spec.yaml for a top-level / nested scalar.

    Avoids a YAML-parser dependency by reading line-by-line. Returns
    "" when the file is missing, the key isn't present, or the value
    is the literal ``null`` / ``~`` / empty (the user-visible "this
    field is unset" forms).
    """
    if not spec_yaml.is_file():
        return ""
    try:
        text = spec_yaml.read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(
        rf"(?m)^\s*{re.escape(key)}:\s*[\"']?([^\"'\s#]+)[\"']?\s*(?:#.*)?$",
        text,
    )
    if not m:
        return ""
    value = m.group(1)
    if value.lower() in ("null", "~", "none"):
        return ""
    return value


# Spec manifest filenames are `v<N>_<short>.json` (assay/corpus/store.py).
_VERSION_FILE_RE = re.compile(r"^v(\d+)_([0-9a-f]+)\.json$")

# Run dirs are stamped by assay.Corpus.open_run as
# `<YYYYMMDDTHHMMSSZ>_<recipe>_<tag>`. The trailing Z on the timestamp is
# mandatory; older or alternate stampings without it are tolerated.
_RUN_DIR_RE = re.compile(r"^\d{8}T\d{6}Z?_([a-z_]+?)(?:_[0-9a-f]+)?$")


def _spec_state(spec_dir: Path) -> dict[str, str]:
    versions_dir = spec_dir / "versions"
    runs_dir = spec_dir / "runs"

    state: dict[str, str] = {
        "spec_dir": str(spec_dir.resolve()),
        "spec_version": "none",
        "manifest_path": "none",
        "runs_total": "0",
        "last_run": "none",
    }
    for stage in _WORKFLOWS_BY_STAGE:
        state[f"runs_{stage}"] = "0"

    # Strictly read-only — we do NOT call assay.Corpus(spec_dir) because
    # the constructor auto-creates the six subdirs as a side effect.
    # Direct filename scan matches the published Corpus filename scheme
    # (versions/v<N>_<short>.json); when that scheme evolves, update
    # _VERSION_FILE_RE accordingly.
    if versions_dir.is_dir():
        best: int | None = None
        best_path: Path | None = None
        for p in versions_dir.iterdir():
            m = _VERSION_FILE_RE.match(p.name)
            if not m:
                continue
            n = int(m.group(1))
            if best is None or n > best:
                best = n
                best_path = p
        if best is not None:
            state["spec_version"] = str(best)
            state["manifest_path"] = str(best_path.relative_to(spec_dir))

    # Run counts by stage.
    if runs_dir.is_dir():
        total = 0
        last_dir: str | None = None
        last_mtime = -1.0
        for run_path in runs_dir.iterdir():
            if not run_path.is_dir():
                continue
            total += 1
            m = _RUN_DIR_RE.match(run_path.name)
            workflow = m.group(1) if m else ""
            for stage, members in _WORKFLOWS_BY_STAGE.items():
                if workflow in members:
                    state[f"runs_{stage}"] = str(int(state[f"runs_{stage}"]) + 1)
                    break
            mtime = run_path.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                last_dir = run_path.name
        state["runs_total"] = str(total)
        if last_dir:
            state["last_run"] = last_dir

    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec_dir", type=Path,
                        help="Path to the Corpus / spec directory")
    parser.add_argument("--stage", default=None,
                        choices=list(_WORKFLOWS_BY_STAGE.keys()),
                        help="Requested stage — emits a stage-ordering hint")
    args = parser.parse_args(argv)

    spec_dir = args.spec_dir
    if not spec_dir.is_dir():
        print(f"error=spec_dir_missing\nspec_dir={spec_dir}", file=sys.stderr)
        return 2

    paths = _resolve_paths(spec_dir)
    state = _spec_state(spec_dir)

    out = {**paths, **state}
    for key in (
        "spec_dir", "spec_version", "manifest_path",
        "workspace_root", "target", "project", "submodule_path",
        "runs_total", "runs_design", "runs_build", "runs_benchmark",
        "runs_recovery", "last_run",
    ):
        print(f"{key}={out.get(key, '')}")

    if args.stage:
        hint = _ordering_hint(args.stage, out)
        if hint:
            print(f"warn_out_of_order={hint}")

    return 0


def _ordering_hint(stage: str, state: dict[str, str]) -> str | None:
    """Return a one-line warning if the requested stage is out-of-order."""
    has_spec = state["spec_version"] != "none"
    if stage == "build" and not has_spec:
        return "no spec yet — run /crucible-design first"
    return None


if __name__ == "__main__":
    sys.exit(main())
