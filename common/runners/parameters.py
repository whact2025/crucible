# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Language-adapter view of a generic runner parameter envelope."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from crucible.common.runners.declared_verification import (
    ACCEPTED_VERIFICATION_REFERENCE_KEY,
    DECLARED_VERIFICATION_SURFACE_KEY,
)
from crucible.common.runners.breach_observation import (
    BREACH_TEST_COMMAND_KEY,
    validate_breach_test_command,
)

WORKFLOW_CONTROL_KEYS = frozenset({
    "breach_attack_ids",
    "breach_observations",
    "breach_test_file",
    "deepswe_metadata",
    ACCEPTED_VERIFICATION_REFERENCE_KEY,
    DECLARED_VERIFICATION_SURFACE_KEY,
    "language",
    "languages",
    "role",
    "runner_mode",
    "runner_timeout_s",
    "swebench_metadata",
    "working_dir",
})


def validate_static(parameters: object) -> None:
    """Reject workflow-owned controls from canonical runner selection."""

    if not isinstance(parameters, Mapping):
        raise TypeError("runner parameters must be a mapping")
    _reject_misplaced_breach_test_command(parameters)
    forbidden = sorted(set(parameters).intersection(WORKFLOW_CONTROL_KEYS))
    if forbidden:
        raise ValueError(
            "canonical runner parameters contain workflow-owned controls: "
            f"{forbidden!r}"
        )
    if BREACH_TEST_COMMAND_KEY in parameters:
        raise ValueError(
            "canonical runner parameter 'breach_test_command' is accepted "
            "only under per_language.python"
        )
    _validate_per_language(parameters.get("per_language"), path="per_language")
    per_slice = parameters.get("per_slice")
    if per_slice is None:
        return
    if not isinstance(per_slice, Mapping):
        raise ValueError("runner parameter 'per_slice' must be a mapping")
    for slice_id, overlay in per_slice.items():
        if not isinstance(slice_id, str) or not slice_id.strip():
            raise ValueError(
                "runner parameter 'per_slice' requires non-empty string keys"
            )
        if not isinstance(overlay, Mapping):
            raise ValueError(
                f"runner parameter per_slice.{slice_id} must be a mapping"
            )
        if "per_slice" in overlay:
            raise ValueError(
                f"canonical per_slice.{slice_id} may not nest another "
                "'per_slice' envelope"
            )
        forbidden = sorted(set(overlay).intersection(WORKFLOW_CONTROL_KEYS))
        if forbidden:
            raise ValueError(
                f"canonical per_slice.{slice_id} contains workflow-owned "
                f"controls: {forbidden!r}"
            )
        if BREACH_TEST_COMMAND_KEY in overlay:
            raise ValueError(
                f"canonical per_slice.{slice_id}.{BREACH_TEST_COMMAND_KEY} "
                "is accepted only under per_language.python"
            )
        _validate_per_language(
            overlay.get("per_language"),
            path=f"per_slice.{slice_id}.per_language",
        )


def _validate_per_language(per_language: object, *, path: str) -> None:
    """Validate one base or slice-local language overlay."""

    if per_language is None:
        return
    if not isinstance(per_language, Mapping):
        raise ValueError(f"runner parameter '{path}' must be a mapping")
    for language, overlay in per_language.items():
        if not isinstance(language, str) or not isinstance(overlay, Mapping):
            raise ValueError(
                f"runner parameter '{path}' requires string keys and "
                "mapping values"
            )
        if "per_language" in overlay:
            raise ValueError(
                f"canonical {path}.{language} may not nest another "
                "'per_language' envelope"
            )
        if "per_slice" in overlay:
            raise ValueError(
                f"canonical {path}.{language} may not nest a 'per_slice' "
                "envelope"
            )
        forbidden = sorted(set(overlay).intersection(WORKFLOW_CONTROL_KEYS))
        if forbidden:
            raise ValueError(
                f"canonical {path}.{language} contains workflow-owned "
                f"controls: {forbidden!r}"
            )
        if BREACH_TEST_COMMAND_KEY in overlay:
            if language != "python":
                raise ValueError(
                    f"canonical {path}.{language}.{BREACH_TEST_COMMAND_KEY} "
                    "is Python-only"
                )
            validate_breach_test_command(
                overlay[BREACH_TEST_COMMAND_KEY],
                path=f"{path}.{language}.{BREACH_TEST_COMMAND_KEY}",
            )


def _reject_misplaced_breach_test_command(
    parameters: Mapping[object, object],
) -> None:
    """Reserve the binding key at the two exact canonical paths only."""

    def walk(value: object, path: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key)
            child_path = (*path, key)
            if key == BREACH_TEST_COMMAND_KEY:
                allowed = (
                    len(child_path) == 3
                    and child_path[:2] == ("per_language", "python")
                ) or (
                    len(child_path) == 5
                    and child_path[0] == "per_slice"
                    and child_path[2:4] == ("per_language", "python")
                )
                if not allowed:
                    rendered = ".".join(child_path)
                    raise ValueError(
                        f"canonical runner parameter {rendered!r} is "
                        "accepted only under per_language.python"
                    )
            walk(child, child_path)

    walk(parameters, ())


def for_slice(parameters: object, slice_id: str | None) -> dict:
    """Overlay canonical ``per_slice.<slice_id>`` runner selection.

    Slice overlays are applied before language projection. Their
    ``per_language`` map deep-merges by language so a slice can replace only
    its Rust ``project_subdir`` without discarding canonical Python/C++
    selection. The ``per_slice`` routing envelope itself never reaches a
    runner subprocess.
    """

    if not isinstance(parameters, Mapping):
        return {}
    validate_static(parameters)
    merged = {
        key: value for key, value in parameters.items() if key != "per_slice"
    }
    if slice_id is None:
        return merged
    if not isinstance(slice_id, str) or not slice_id.strip():
        raise ValueError("slice_id must be a non-empty string when supplied")
    per_slice = parameters.get("per_slice")
    overlay = per_slice.get(slice_id) if isinstance(per_slice, Mapping) else None
    if overlay is None:
        return merged

    for key, value in overlay.items():
        if key != "per_language":
            merged[key] = value
    overlay_languages = overlay.get("per_language")
    if isinstance(overlay_languages, Mapping):
        base_languages = merged.get("per_language")
        selected_languages = (
            dict(base_languages) if isinstance(base_languages, Mapping) else {}
        )
        for language, language_overlay in overlay_languages.items():
            prior = selected_languages.get(language)
            selected = dict(prior) if isinstance(prior, Mapping) else {}
            selected.update(language_overlay)
            selected_languages[language] = selected
        merged["per_language"] = selected_languages
    return merged


def for_language(parameters: object, language: str) -> dict:
    """Return shared parameters overlaid by ``per_language.<language>``.

    Polyglot dispatch and directly selected language runners must interpret
    the same canonical Spec envelope. Keeping this projection in one helper
    prevents direct adapters from silently ignoring language-scoped selection.
    """

    if not isinstance(parameters, Mapping):
        return {}
    merged = {
        key: value
        for key, value in parameters.items()
        if key not in {"languages", "per_language", "per_slice"}
    }
    per_language = parameters.get("per_language")
    if isinstance(per_language, Mapping):
        overlay = per_language.get(language)
        if isinstance(overlay, Mapping):
            if "per_language" in overlay:
                raise ValueError(
                    f"per_language.{language} may not nest another "
                    "'per_language' envelope"
                )
            forbidden = sorted(
                set(overlay).intersection(WORKFLOW_CONTROL_KEYS)
            )
            if forbidden:
                raise ValueError(
                    f"per_language.{language} may not override workflow-owned "
                    f"controls: {forbidden!r}"
                )
            merged.update(overlay)
    return merged


def resolve_project_working_dir(
    parameters: object,
    *,
    process_cwd: str | Path,
) -> Path:
    """Resolve a canonical project subdirectory beneath the workflow root.

    ``working_dir`` remains workflow-owned runtime metadata (for example a
    container checkout). ``project_subdir`` is canonical Spec selection and
    may only narrow execution beneath that root; absolute paths, traversal,
    symlink escape, and non-directories fail closed.
    """

    projected = parameters if isinstance(parameters, Mapping) else {}
    root = Path(projected.get("working_dir") or process_cwd).resolve()
    raw_subdir = projected.get("project_subdir")
    if raw_subdir is None:
        return root
    if not isinstance(raw_subdir, str) or not raw_subdir.strip():
        raise ValueError("runner parameter 'project_subdir' must be a string")
    subdir = Path(raw_subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError(
            "runner parameter 'project_subdir' must be a confined relative path"
        )
    candidate = (root / subdir).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "runner parameter 'project_subdir' escapes the workflow root"
        ) from exc
    if not candidate.is_dir():
        raise ValueError(
            "runner parameter 'project_subdir' does not resolve to a directory: "
            f"{raw_subdir!r}"
        )
    return candidate


__all__ = [
    "WORKFLOW_CONTROL_KEYS",
    "for_language",
    "for_slice",
    "resolve_project_working_dir",
    "validate_static",
]
