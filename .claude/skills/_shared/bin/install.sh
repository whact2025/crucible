#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

# Link the same operator skills into Claude Code and Codex discovery paths.
set -euo pipefail

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
INSTALL_ROOT="$HOME"
DRY_RUN=0

usage() {
  cat <<EOF
usage: $0 [--project <directory>] [--dry-run] [--help]

Install crucible-status, crucible-design, crucible-build, and crucible-handoff
for Claude Code (.claude/skills) and Codex (.agents/skills).

By default, links are installed in the current user's home so the skills are
available in any working folder. --project installs only in the selected
existing directory; it does not need to be a Git repository.

Links refer to this checkout. Existing files and directories are preserved;
conflicts make the installer exit 1. --dry-run previews without writing.
EOF
}

while (( $# )); do
  case "$1" in
    --project)
      if (( $# < 2 )) || [[ ! -d "$2" ]]; then
        echo "--project requires an existing directory" >&2
        exit 2
      fi
      INSTALL_ROOT="$(cd "$2" && pwd -P)"
      shift 2
      ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

link_one() {
  local target="$1" linkpath="$2"
  if [[ "$linkpath" == "$target" ]]; then
    echo "OK (source directory): $linkpath"
  elif [[ -L "$linkpath" ]]; then
    if [[ "$(readlink "$linkpath")" == "$target" ]]; then
      echo "OK (already linked): $linkpath"
    elif (( DRY_RUN )); then
      echo "WOULD relink: $linkpath -> $target"
    else
      ln -sfn "$target" "$linkpath" || return 1
      echo "RELINK: $linkpath -> $target"
    fi
  elif [[ -e "$linkpath" ]]; then
    echo "CONFLICT (preserved): $linkpath" >&2
    return 1
  elif (( DRY_RUN )); then
    echo "WOULD link: $linkpath -> $target"
  else
    mkdir -p "$(dirname "$linkpath")" || return 1
    ln -s "$target" "$linkpath" || return 1
    echo "LINK: $linkpath -> $target"
  fi
}

SKILLS=(crucible-status crucible-design crucible-build crucible-handoff)
CONFLICTS=0
for client in .claude .agents; do
  for skill in "${SKILLS[@]}"; do
    if ! link_one "$SKILLS_DIR/$skill" "$INSTALL_ROOT/$client/skills/$skill"; then
      CONFLICTS=1
    fi
  done
done

if (( CONFLICTS )); then
  echo "Installation incomplete: resolve the reported errors and run again." >&2
  exit 1
fi
if ! (( DRY_RUN )); then
  echo "Done. Start a new agent session to discover the skills."
fi
