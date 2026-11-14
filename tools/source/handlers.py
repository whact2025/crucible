# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Source-navigation tool handlers.

Four tools, all read-only:
- ``read_source``      — read a line range from a file
- ``grep_symbol``      — ripgrep / grep for a pattern in a directory
- ``find_callers``     — grep_symbol with a default code-file glob
- ``find_definition``  — resolve a symbol to its declaration site(s)

Path resolution rejects ``..`` escapes from ``ctx.repo_root`` and applies the
shared VCS-metadata visibility policy.

Current output limits:
- ``_MAX_GREP_RESULTS = 30``
- ``_MAX_READ_LINES = 300``
- ``_MAX_OUTPUT_CHARS = 15_000`` (final truncation)
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .._builder import ToolSetBuilder
from .._context import ToolContext
from .._paths import resolve_for_read

logger = logging.getLogger(__name__)

_MAX_GREP_RESULTS = 30
_MAX_READ_LINES = 300
_MAX_OUTPUT_CHARS = 15_000


# ----------------------------------------------------------------------
# Path resolution helpers (private)
# ----------------------------------------------------------------------


def _resolve_within_repo(ctx: ToolContext, relative: str) -> Path | None:
    """Use the shared containment and VCS-metadata visibility policy."""

    return resolve_for_read(ctx, relative)


def _grep_cmd(
    pattern: str,
    search_dir: str,
    max_count: int,
    globs: list[str],
    *,
    max_filesize: str = "",
) -> list[str]:
    """Build a ripgrep command; fall back to grep -E if rg is unavailable.

    The grep fallback uses ``-E`` (extended regex) so patterns containing
    ``|`` alternation or unescaped ``(`` grouping are interpreted as the
    callers expect. Without ``-E`` GNU grep would treat ``|`` as a literal
    pipe character (BRE), silently matching nothing for the typical
    alternation patterns the agent emits.
    """
    if shutil.which("rg"):
        cmd = [
            "rg", "--no-heading", "--with-filename", "-n",
            "--max-count", str(max_count),
        ]
        for metadata in (".git", ".hg", ".svn", ".bzr", "_darcs"):
            cmd.extend(["-g", f"!**/{metadata}/**"])
        if max_filesize:
            cmd.extend(["--max-filesize", max_filesize])
        for g in globs:
            cmd.extend(["-g", g])
        cmd.extend(["--", pattern, search_dir])
        return cmd
    cmd = ["grep", "-rEnH", "--max-count", str(max_count)]
    for metadata in (".git", ".hg", ".svn", ".bzr", "_darcs"):
        cmd.append(f"--exclude-dir={metadata}")
    for g in globs:
        cmd.extend(["--include", g])
    cmd.extend(["--", pattern, search_dir])
    return cmd


# ----------------------------------------------------------------------
# Builder + registrations
# ----------------------------------------------------------------------


builder = ToolSetBuilder("source")


@builder.tool(
    description=(
        "Read source code lines from a file in the repository. "
        "Returns numbered lines. Use to inspect implementation details "
        "of functions or classes."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path relative to the repo root (e.g. '<project>/source/foo.cpp')",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed, default 1)",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (inclusive, default start_line+200)",
            },
        },
        "required": ["file_path"],
    },
)
def read_source(
    ctx: ToolContext,
    *,
    file_path: str,
    start_line: int = 1,
    end_line: int | None = None,
    **_: Any,
) -> str:
    """Read a line range from a source file (tolerant path resolution)."""
    path = _resolve_within_repo(ctx, file_path)
    # Tolerant fallback: if the agent included the repo-root directory
    # name as a redundant prefix, strip it and retry.
    if path is None or not path.is_file():
        repo_name = ctx.repo_root.name
        if file_path.startswith(repo_name + "/"):
            stripped = file_path[len(repo_name) + 1:]
            alt = _resolve_within_repo(ctx, stripped)
            if alt is not None and alt.is_file():
                path = alt
    if path is None:
        return f"Error: path '{file_path}' is outside the repository"
    if not path.is_file():
        return f"Error: file not found: {file_path}"

    start_line = max(1, start_line)
    if end_line is None:
        end_line = start_line + 200
    end_line = min(end_line, start_line + _MAX_READ_LINES)

    try:
        raw = path.read_bytes()
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except Exception as exc:  # noqa: BLE001
        return f"Error reading {file_path}: {exc}"

    selected = lines[start_line - 1: end_line]
    numbered = [
        f"{start_line + i:5d} | {line}"
        for i, line in enumerate(selected)
    ]
    covers_whole_file = start_line == 1 and end_line >= len(lines)
    body = "\n".join(numbered)
    if covers_whole_file:
        digest = hashlib.sha256(raw).hexdigest()
        header = f"# file_sha256: {digest}"
    else:
        header = (
            f"# partial_file: returned lines {start_line}-"
            f"{min(end_line, len(lines))} of {len(lines)}; "
            "whole-file hash withheld"
        )
    result = header + "\n" + body
    if covers_whole_file and len(result) > _MAX_OUTPUT_CHARS:
        header = (
            f"# partial_file: output capped at {_MAX_OUTPUT_CHARS} "
            "characters; whole-file hash withheld"
        )
        result = header + "\n" + body
    return result[:_MAX_OUTPUT_CHARS]


def _expand_globs(file_glob: str) -> list[str]:
    """Expand shell-style brace syntax (``*.{cpp,h,py}``) into multiple globs.

    Ripgrep/grep don't honor brace expansion when invoked directly
    (no shell); they want multiple ``-g`` / ``--include`` flags. A bare
    ``*.cpp`` returns ``["*.cpp"]``; ``*.{cpp,h,py}`` returns
    ``["*.cpp", "*.h", "*.py"]``.
    """
    if not file_glob:
        return []
    if "{" not in file_glob or "}" not in file_glob:
        return [file_glob]
    prefix, brace_and_after = file_glob.split("{", 1)
    body, suffix = brace_and_after.split("}", 1)
    parts = [p.strip() for p in body.split(",") if p.strip()]
    return [f"{prefix}{p}{suffix}" for p in parts]


@builder.tool(
    description=(
        "Search for a pattern (regex) in the repository using ripgrep. "
        "Returns matching file:line:content strings. Use to find where "
        "a function is defined or called."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "directory": {
                "type": "string",
                "description": (
                    "Directory or file to search, relative to repo root "
                    "(default: entire repo)"
                ),
            },
            "file_glob": {
                "type": "string",
                "description": "File glob filter (e.g. '*.cpp', '*.py')",
            },
        },
        "required": ["pattern"],
    },
)
def grep_symbol(
    ctx: ToolContext,
    *,
    pattern: str,
    directory: str = "",
    file_glob: str = "",
    **_: Any,
) -> str:
    """ripgrep / grep for *pattern* scoped to *directory*."""
    if not pattern or len(pattern) < 2:
        return "Error: pattern must be at least 2 characters"

    search_path = ctx.repo_root
    # Models commonly spell the repository root as ``.``. Treat only the
    # canonical dot aliases as the already-authorized root; all other values
    # still flow through the shared containment/VCS-metadata resolver.
    if directory and directory not in {".", "./"}:
        resolved = _resolve_within_repo(ctx, directory)
        if resolved is None:
            return f"Error: directory '{directory}' is outside the repository"
        if not (resolved.is_dir() or resolved.is_file()):
            return f"Error: search path not found: {directory}"
        search_path = resolved

    globs = _expand_globs(file_glob)
    cmd = _grep_cmd(
        pattern, str(search_path), _MAX_GREP_RESULTS, globs, max_filesize="1M",
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        output = proc.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: search timed out (>15s)"
    except FileNotFoundError:
        return "Error: neither ripgrep (rg) nor grep found"

    if proc.returncode not in {0, 1}:
        diagnostic = (proc.stderr or "search command failed").strip()
        return f"Error: search failed: {diagnostic[:1000]}"
    if not output:
        return f"No matches for pattern '{pattern}'"

    repo_prefix = str(ctx.repo_root.resolve()) + "/"
    lines = [
        line[len(repo_prefix):] if line.startswith(repo_prefix) else line
        for line in output.splitlines()[:_MAX_GREP_RESULTS]
    ]
    return "\n".join(lines)[:_MAX_OUTPUT_CHARS]


# Per-language file extension registry. Add entries here when a new
# language's grep should be supported. Each value is the list of glob
# patterns ripgrep / grep should include for that language.
_LANGUAGE_GLOBS: dict[str, list[str]] = {
    "python":     ["*.py", "*.pyi"],
    "rust":       ["*.rs"],
    "cpp":        ["*.cpp", "*.cc", "*.cxx", "*.h", "*.hpp", "*.inl"],
    "c":          ["*.c", "*.h"],
    "cuda":       ["*.cu", "*.cuh"],
    "typescript": ["*.ts", "*.tsx"],
    "javascript": ["*.js", "*.jsx", "*.mjs", "*.cjs"],
    "go":         ["*.go"],
    "java":       ["*.java"],
    "kotlin":     ["*.kt", "*.kts"],
    "swift":      ["*.swift"],
    "ruby":       ["*.rb"],
    "shell":      ["*.sh", "*.bash"],
}


def _languages_to_glob(languages: list[str] | None) -> str:
    """Combine *languages* (or all known) into a single brace-expansion glob."""
    if not languages:
        langs = list(_LANGUAGE_GLOBS.keys())
    else:
        langs = languages
    exts: set[str] = set()
    for lang in langs:
        for pattern in _LANGUAGE_GLOBS.get(lang, ()):
            # Strip the leading "*." so we can brace-expand.
            if pattern.startswith("*."):
                exts.add(pattern[2:])
    if not exts:
        return ""
    return "*.{" + ",".join(sorted(exts)) + "}"


@builder.tool(
    description=(
        "Find call sites of a function across the codebase. "
        "Returns file:line:content for lines that reference the function name. "
        "Search is scoped to the file extensions of the requested languages; "
        "default scope is every known language."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "function_name": {
                "type": "string",
                "description": "Function or method name to search for",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in, relative to repo root (default: entire repo)",
            },
            "languages": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(_LANGUAGE_GLOBS.keys()),
                },
                "description": (
                    "Languages to search; default = all known. Each entry "
                    "expands to that language's file globs. Available: "
                    + ", ".join(sorted(_LANGUAGE_GLOBS.keys())) + "."
                ),
            },
        },
        "required": ["function_name"],
    },
)
def find_callers(
    ctx: ToolContext,
    *,
    function_name: str,
    directory: str = "",
    languages: list[str] | None = None,
    **_: Any,
) -> str:
    """Grep for *function_name* across the requested languages' file globs."""
    if not function_name or len(function_name) < 3:
        return "Error: function_name must be at least 3 characters"

    file_glob = _languages_to_glob(languages)
    return grep_symbol(
        ctx,
        pattern=function_name,
        directory=directory,
        file_glob=file_glob,
    )


# ----------------------------------------------------------------------
# find_definition: locate where a symbol is DECLARED (class/struct/enum/
# trait/function/type alias). Solves the planner's failure mode of
# citing a namespaced symbol (e.g. `omni::usd::hydra::IHydraEngine`) and
# guessing the header path from the namespace structure — the
# filesystem layout is often independent of the namespace.
#
# Schema-shape note: ``find_definition`` takes a single ``language``
# string (not ``languages: list[str]`` like ``find_callers``). The
# definition patterns are per-language and don't generalize across
# language boundaries — a C++ "class X { …" can't validly be searched
# with the Rust "struct X" pattern, so collapsing to a single language
# per call keeps the regex tight and the false-positive rate low.
# ----------------------------------------------------------------------


# Per-language definition patterns. Each entry is a list of regex
# fragments (POSIX ERE) that match a declaration of <symbol>. The
# handler joins them with `|` and substitutes the symbol name.
#
# Use POSIX bracket expressions (`[[:alnum:]_]`, `[[:space:]]`) rather
# than Perl shortcuts (`\w`, `\s`) INSIDE character classes — GNU grep
# in ERE mode doesn't honor `\w` / `\s` inside `[...]`. Outside char
# classes both work, but POSIX is portable across grep/ripgrep.
_DEFINITION_PATTERNS: dict[str, list[str]] = {
    "cpp": [
        # class / struct / union / enum declarations
        r"\b(class|struct|union|enum([[:space:]]+class)?)[[:space:]]+{sym}\b",
        # template specialization / forward decl
        r"\btemplate[[:space:]]*<[^>]*>[[:space:]]*(class|struct)[[:space:]]+{sym}\b",
        # using-alias and old-style typedef
        r"\busing[[:space:]]+{sym}[[:space:]]*=",
        r"\btypedef\b.*\b{sym}[[:space:]]*;",
        # function declaration / definition: REQUIRED return-type group,
        # then symbol name, then opening paren. Requiring the return type
        # eliminates the call-site false positive (`foo()` on its own
        # would match a "no-return-type" capture and conflate calls with
        # declarations). Constructors (which have no return type) are
        # discovered via the class/struct pattern above; their declared
        # name matches the class identifier.
        r"^[[:space:]]*[[:alnum:]_:<>,&*][[:alnum:]_:<>,&*[:space:]]*[[:space:]]+{sym}[[:space:]]*\(",
    ],
    "c": [
        r"\b(struct|union|enum)[[:space:]]+{sym}\b",
        r"\btypedef\b.*\b{sym}[[:space:]]*;",
        # Required return-type — see cpp comment above on call-site false
        # positives.
        r"^[[:space:]]*[[:alnum:]_*][[:alnum:]_*[:space:]]*[[:space:]]+{sym}[[:space:]]*\(",
    ],
    "rust": [
        r"\b(struct|enum|trait|union)[[:space:]]+{sym}\b",
        r"\bfn[[:space:]]+{sym}[[:space:]]*[<(]",
        r"\btype[[:space:]]+{sym}[[:space:]]*=",
        r"\bimpl([[:space:]]*<[^>]*>)?[[:space:]]+{sym}\b",
    ],
    "python": [
        r"^[[:space:]]*(class|def)[[:space:]]+{sym}[[:space:]]*[\(:\[]",
        # Module-level assignment only — leading column-0 match excludes
        # local assignments inside functions / methods (which would
        # otherwise turn this pattern into a noisy false-positive
        # generator).
        r"^{sym}[[:space:]]*=",
    ],
    "go": [
        r"\bfunc[[:space:]]+(\([[:space:]]*[[:alnum:]_]+[[:space:]]+\*?[[:alnum:]_]+[[:space:]]*\)[[:space:]]+)?{sym}[[:space:]]*\(",
        r"\btype[[:space:]]+{sym}[[:space:]]+(struct|interface|=)",
    ],
    "java": [
        r"\b(class|interface|enum)[[:space:]]+{sym}\b",
        # Required return-type — see cpp comment above on call-site false
        # positives.
        r"^[[:space:]]*[[:alnum:]_<>,?\[\]][[:alnum:]_<>,?\[\][:space:]]*[[:space:]]+{sym}[[:space:]]*\(",
    ],
}


@builder.tool(
    description=(
        "Resolve a symbol name to the file(s) and line(s) where it is "
        "DECLARED (class / struct / enum / trait / function / type "
        "alias). Use this BEFORE citing a file path in a plan: many "
        "C++ headers sit in filesystem locations that do not match "
        "their declared namespace, so guessing the path from the "
        "namespace is unsafe. Returns file:line:content for matching "
        "declarations across the requested language's file globs."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": (
                    "Symbol name to locate (e.g. 'IHydraEngine', "
                    "'FrameRenderStatePublisher', 'publishFrameRenderState'). "
                    "Plain identifier — do not include namespace "
                    "qualifiers (the search is namespace-agnostic by "
                    "design). Minimum 3 characters."
                ),
            },
            "language": {
                "type": "string",
                "enum": sorted(_DEFINITION_PATTERNS.keys()),
                "description": (
                    "Language whose declaration patterns to use. "
                    "Available: " + ", ".join(sorted(_DEFINITION_PATTERNS.keys())) + "."
                ),
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in, relative to repo root (default: entire repo)",
            },
        },
        "required": ["symbol", "language"],
    },
)
def find_definition(
    ctx: ToolContext,
    *,
    symbol: str,
    language: str,
    directory: str = "",
    **_: Any,
) -> str:
    """Locate declaration sites of *symbol* across *language*'s globs."""
    if not symbol or len(symbol) < 3:
        return "Error: symbol must be at least 3 characters"
    if language not in _DEFINITION_PATTERNS:
        return (
            f"Error: unsupported language '{language}'. "
            f"Choose from: {sorted(_DEFINITION_PATTERNS.keys())}"
        )

    # Strip any accidental namespace qualifier the caller included.
    plain = symbol.split("::")[-1].split(".")[-1]

    # Build combined regex.
    patterns = _DEFINITION_PATTERNS[language]
    combined = "|".join(p.format(sym=plain) for p in patterns)

    file_glob = _languages_to_glob([language])
    return grep_symbol(
        ctx,
        pattern=combined,
        directory=directory,
        file_glob=file_glob,
    )


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_source_tools(context: ToolContext):
    """Build the source ToolSet bound to *context*."""
    return builder.build(context=context)


__all__ = [
    "build_source_tools",
    "builder",
    "read_source",
    "grep_symbol",
    "find_callers",
    "find_definition",
]
