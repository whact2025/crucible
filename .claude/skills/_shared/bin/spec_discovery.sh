#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

# Spec discovery helpers — short name → absolute path.
#
# Source this file from a skill, then call:
#   resolve_spec   <name>            # exit 0 success / 1 not-found / 2 ambiguous / 3 setup
#   list_all_specs                   # one path per line, grouped by target
#
# Requires $WORKSPACE_ROOT to be set, or falls back to
# `git rev-parse --show-superproject-working-tree` then `--show-toplevel`.
#
# The only accepted layout is
# $WORKSPACE_ROOT/.crucible/specs/<target>/<spec>/.

resolve_spec() {
  local name="$1"
  local workspace_root="${2:-${WORKSPACE_ROOT:-}}"
  if [ -z "$workspace_root" ]; then
    # --show-superproject-working-tree returns empty stdout with exit 0
    # when there's no superproject; `||` wouldn't trigger the fallback.
    workspace_root="$(git rev-parse --show-superproject-working-tree 2>/dev/null || true)"
    if [ -z "$workspace_root" ]; then
      workspace_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    fi
  fi
  if [ -z "$workspace_root" ]; then
    echo "ERROR: cannot resolve workspace_root" >&2
    return 3
  fi

  local specs_root="$workspace_root/.crucible/specs"
  if [ ! -d "$specs_root" ]; then
    echo "ERROR: no canonical specs root at $specs_root" >&2
    return 3
  fi

  # Path-style arguments are accepted only when they resolve to an exact
  # canonical <target>/<spec> directory. There is no compatibility
  # passthrough for alternate layouts.
  local candidate=""
  case "$name" in
    /*)
      [ -d "$name" ] && candidate="$(realpath "$name")"
      ;;
    */*)
      if [ -d "$name" ]; then
        candidate="$(realpath "$name")"
      elif [ -d "$workspace_root/$name" ]; then
        candidate="$(realpath "$workspace_root/$name")"
      fi
      ;;
  esac
  if [ -n "$candidate" ]; then
    if is_canonical_spec_path "$candidate" "$workspace_root"; then
      echo "$candidate"
      return 0
    fi
    echo "ERROR: spec path is outside the canonical tree: $candidate" >&2
    echo "       expected $specs_root/<target>/<spec>/" >&2
    return 1
  fi

  # Pass 1: exact-basename match across every <target>/ under specs/.
  local exact=()
  while IFS= read -r -d '' d; do
    [ "$(basename "$d")" = "$name" ] && exact+=("$d")
  done < <(find "$specs_root" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null)

  if [ ${#exact[@]} -eq 1 ]; then
    echo "${exact[0]}"
    return 0
  fi
  if [ ${#exact[@]} -gt 1 ]; then
    echo "AMBIGUOUS: '$name' matches multiple specs exactly. Pick one:" >&2
    printf '  - %s\n' "${exact[@]}" >&2
    return 2
  fi

  # Pass 2: case-insensitive substring across every spec dir.
  local pat_lc; pat_lc="$(echo "$name" | tr '[:upper:]' '[:lower:]')"
  local fuzzy=()
  while IFS= read -r -d '' d; do
    local base_lc; base_lc="$(basename "$d" | tr '[:upper:]' '[:lower:]')"
    case "$base_lc" in
      *"$pat_lc"*) fuzzy+=("$d") ;;
    esac
  done < <(find "$specs_root" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null)

  if [ ${#fuzzy[@]} -eq 0 ]; then
    echo "ERROR: no spec found matching '$name' under $specs_root/<target>/" >&2
    echo "       run \`list_all_specs\` to see the full set." >&2
    return 1
  fi
  if [ ${#fuzzy[@]} -eq 1 ]; then
    # Even one substring match could be a happy accident — surface it
    # as a question, not as an answer.
    echo "FUZZY: '$name' is a substring match for exactly one spec:" >&2
    echo "  - ${fuzzy[0]}" >&2
    echo "       confirm with the user before using this path." >&2
    return 2
  fi
  echo "AMBIGUOUS: '$name' substring-matches ${#fuzzy[@]} specs. Pick one:" >&2
  printf '  - %s\n' "${fuzzy[@]}" >&2
  return 2
}


list_all_specs() {
  local workspace_root="${1:-${WORKSPACE_ROOT:-}}"
  if [ -z "$workspace_root" ]; then
    # --show-superproject-working-tree returns empty stdout with exit 0
    # when there's no superproject; `||` wouldn't trigger the fallback.
    workspace_root="$(git rev-parse --show-superproject-working-tree 2>/dev/null || true)"
    if [ -z "$workspace_root" ]; then
      workspace_root="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    fi
  fi
  local specs_root="$workspace_root/.crucible/specs"
  [ -d "$specs_root" ] || { echo "no specs root at $specs_root" >&2; return 1; }

  # Group by target. Output:
  #   == kit ==
  #     <spec_dir>
  #     ...
  local prev_target=""
  while IFS= read -r -d '' d; do
    local target; target="$(basename "$(dirname "$d")")"
    if [ "$target" != "$prev_target" ]; then
      [ -n "$prev_target" ] && echo ""
      echo "== $target =="
      prev_target="$target"
    fi
    echo "  $d"
  done < <(find "$specs_root" -mindepth 2 -maxdepth 2 -type d -print0 2>/dev/null \
            | sort -z)
}


is_canonical_spec_path() {
  # 0 only for an existing, exact
  # $WORKSPACE_ROOT/.crucible/specs/<target>/<spec>/ directory.
  local p="$1"
  local workspace_root="${2:-${WORKSPACE_ROOT:-}}"
  [ -z "$workspace_root" ] && return 1
  [ -d "$p" ] || return 1
  local specs_root canonical rel
  specs_root="$(realpath "$workspace_root/.crucible/specs" 2>/dev/null)" || return 1
  canonical="$(realpath "$p" 2>/dev/null)" || return 1
  case "$canonical" in
    "$specs_root"/*) rel="${canonical#"$specs_root"/}" ;;
    *) return 1 ;;
  esac
  case "$rel" in
    */*/*|/*|*/|"") return 1 ;;
    */*) return 0 ;;
    *) return 1 ;;
  esac
}
