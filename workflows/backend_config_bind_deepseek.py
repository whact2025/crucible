# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Materialize one run-local, worker-bound DeepSeek routing document.

This provider-free operator command captures the project-locked worker once
and adds that predeclared identity to an otherwise complete DeepSeek-only
diagnostic template.  It never constructs an adapter or reads provider
credentials.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml  # type: ignore[import-untyped]

from crucible.backends.config import BackendRouting, lint_routing_config
from crucible.backends.deepseek import (
    DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
    DeepSeekSDKQualificationError,
    DeepSeekSDKUnavailable,
    capture_project_locked_deepseek_runtime_identity,
)
from crucible.common.authority_io import (
    AuthorityIOError,
    read_exact_regular,
    read_exact_regular_at,
    stat_identity,
)


_RECEIPT_SCHEMA = "crucible.deepseek-backend-config-binding-receipt.v1"
_ERROR_SCHEMA = "crucible.deepseek-backend-config-binding-error.v1"
_RUNTIME_FIELD = "expected_worker_runtime_fingerprint"
_MAX_ROUTING_BYTES = 4 * 1024 * 1024
_DIGEST_PREFIX = "sha256:"


class DeepSeekConfigBindingError(ValueError):
    """The requested provider-free route materialization was not admissible."""


@dataclass(frozen=True, slots=True)
class DeepSeekConfigBindingReceipt:
    """Secret-free identity of one exact derived routing file."""

    source_path: str
    output_path: str
    source_sha256: str
    output_sha256: str
    worker_identity_schema: str
    worker_runtime_fingerprint: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "schema_version": _RECEIPT_SCHEMA,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "worker_identity_schema": self.worker_identity_schema,
            "worker_runtime_fingerprint": self.worker_runtime_fingerprint,
        }


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that also rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found a duplicate mapping key",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(payload: bytes) -> str:
    return _DIGEST_PREFIX + hashlib.sha256(payload).hexdigest()


def _load_document(payload: bytes) -> dict[str, Any]:
    try:
        decoded = payload.decode("utf-8", errors="strict")
        document = yaml.load(decoded, Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DeepSeekConfigBindingError(
            "DeepSeek routing template is not unambiguous UTF-8 YAML"
        ) from exc
    if not isinstance(document, dict):
        raise DeepSeekConfigBindingError(
            "DeepSeek routing template root must be a mapping"
        )
    return document


def _raw_route_entries(
    document: Mapping[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    default = document.get("default")
    roles = document.get("roles")
    if not isinstance(default, dict) or not isinstance(roles, dict):
        raise DeepSeekConfigBindingError(
            "DeepSeek routing template requires default and roles mappings"
        )
    yield "default", default
    for role, value in roles.items():
        if not isinstance(role, str) or not isinstance(value, dict):
            raise DeepSeekConfigBindingError(
                "DeepSeek routing template contains a malformed role entry"
            )
        yield f"roles.{role}", value


def _require_deepseek_only(routing: BackendRouting) -> None:
    entries = [("default", routing.default)]
    entries.extend(
        (f"roles.{role}", routing.for_role(role)) for role in sorted(routing.roles)
    )
    for location, config in entries:
        if config.backend != "deepseek":
            raise DeepSeekConfigBindingError(
                f"{location} is not an exact DeepSeek route; mixed templates are refused"
            )
        if config.feature_admission != "diagnostic":
            raise DeepSeekConfigBindingError(
                f"{location} must retain feature_admission='diagnostic'"
            )


def _lint_bytes(payload: bytes) -> BackendRouting:
    try:
        routing = lint_routing_config(payload)
    except (TypeError, ValueError) as exc:
        raise DeepSeekConfigBindingError(
            "DeepSeek routing template is malformed or outside the closed routing schema"
        ) from exc
    _require_deepseek_only(routing)
    return routing


def _admit_unbound_template(payload: bytes) -> tuple[dict[str, Any], tuple[str, ...]]:
    document = _load_document(payload)
    routing = _lint_bytes(payload)
    for location, raw in _raw_route_entries(document):
        extra = raw.get("extra", {})
        if not isinstance(extra, dict):  # also enforced by lint; keep mutation local.
            raise DeepSeekConfigBindingError(f"{location}.extra must be a mapping")
        if _RUNTIME_FIELD in extra:
            raise DeepSeekConfigBindingError(
                f"{location}.extra.{_RUNTIME_FIELD} is already bound; "
                "derive only from an unbound construction template"
            )
    return document, tuple(sorted(routing.roles))


def _inject_runtime_fingerprint(
    source_document: dict[str, Any],
    runtime_fingerprint: str,
) -> bytes:
    document = copy.deepcopy(source_document)
    for _location, raw in _raw_route_entries(document):
        extra = raw.setdefault("extra", {})
        assert isinstance(extra, dict)
        extra[_RUNTIME_FIELD] = runtime_fingerprint
    return yaml.safe_dump(
        document,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")


def _verify_bound_bytes(
    payload: bytes,
    *,
    source_document: Mapping[str, Any],
    expected_roles: tuple[str, ...],
    runtime_fingerprint: str,
) -> None:
    document = _load_document(payload)
    routing = _lint_bytes(payload)
    if tuple(sorted(routing.roles)) != expected_roles:
        raise DeepSeekConfigBindingError("derived routing role inventory changed")

    stripped = copy.deepcopy(document)
    for location, raw in _raw_route_entries(stripped):
        extra = raw.get("extra")
        if (
            not isinstance(extra, dict)
            or extra.get(_RUNTIME_FIELD) != runtime_fingerprint
        ):
            raise DeepSeekConfigBindingError(
                f"{location} does not carry the captured worker fingerprint"
            )
        del extra[_RUNTIME_FIELD]
    if stripped != source_document:
        raise DeepSeekConfigBindingError(
            "derived routing changed fields other than the worker fingerprint"
        )


def _validate_output_path(path: Path) -> None:
    if (
        not path.is_absolute()
        or ".." in path.parts
        or "\x00" in str(path)
        or path.name in {"", ".", ".."}
    ):
        raise DeepSeekConfigBindingError(
            "--output must be an absolute normalized file path"
        )


def _directory_identity(
    metadata: os.stat_result,
) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


@dataclass(frozen=True, slots=True)
class _PinnedOutputParent:
    path: Path
    name: str
    descriptor: int
    identity: tuple[int, int, int, int, int, int]


def _require_pinned_parent(parent: _PinnedOutputParent) -> None:
    try:
        lexical = parent.path.lstat()
        canonical = parent.path.resolve(strict=True)
        held = os.fstat(parent.descriptor)
    except OSError as exc:
        raise DeepSeekConfigBindingError(
            "output parent cannot be rebound to its pinned directory"
        ) from exc
    if (
        canonical != parent.path
        or not stat.S_ISDIR(lexical.st_mode)
        or _directory_identity(lexical) != parent.identity
        or _directory_identity(held) != parent.identity
    ):
        raise DeepSeekConfigBindingError(
            "output parent path changed after it was pinned"
        )


def _open_output_parent(path: Path) -> _PinnedOutputParent:
    _validate_output_path(path)
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        lexical = path.parent.lstat()
        canonical = path.parent.resolve(strict=True)
        if (
            canonical != path.parent
            or stat.S_ISLNK(lexical.st_mode)
            or not stat.S_ISDIR(lexical.st_mode)
        ):
            raise DeepSeekConfigBindingError(
                "output parent must be an exact, non-symlinked directory"
            )
        descriptor = os.open(path.parent, parent_flags)
        identity = _directory_identity(lexical)
        parent = _PinnedOutputParent(
            path=path.parent,
            name=path.name,
            descriptor=descriptor,
            identity=identity,
        )
        _require_pinned_parent(parent)
        try:
            os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise DeepSeekConfigBindingError("exclusive output path already exists")
        _require_pinned_parent(parent)
        descriptor = None
        return parent
    except DeepSeekConfigBindingError:
        raise
    except OSError as exc:
        raise DeepSeekConfigBindingError(
            "output parent must already exist as an exact directory"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


@dataclass(frozen=True, slots=True)
class _PublishedOutputIdentity:
    device: int
    inode: int


def _unlink_published_at(
    parent_fd: int,
    name: str,
    identity: _PublishedOutputIdentity,
) -> None:
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != identity.device
        or current.st_ino != identity.inode
    ):
        raise DeepSeekConfigBindingError("refusing to clean up a replaced output path")
    os.unlink(name, dir_fd=parent_fd)
    os.fsync(parent_fd)
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise DeepSeekConfigBindingError("exclusive output cleanup was not exact")


def _unlink_published_if_same_at(
    parent_fd: int,
    name: str,
    identity: _PublishedOutputIdentity,
) -> bool:
    """Remove an ambiguously published leaf only when it is exactly ours."""

    try:
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_dev != identity.device
        or current.st_ino != identity.inode
    ):
        return False
    _unlink_published_at(parent_fd, name, identity)
    return True


def _cleanup_staging_at(
    parent_fd: int,
    name: str,
    identity: _PublishedOutputIdentity | None,
) -> None:
    if identity is None:
        raise DeepSeekConfigBindingError("uncertain staging cleanup; do not dispatch")
    try:
        _unlink_published_at(parent_fd, name, identity)
    except FileNotFoundError:
        return
    except BaseException as exc:
        raise DeepSeekConfigBindingError(
            "uncertain staging cleanup; do not dispatch"
        ) from exc


def _verify_final_published_path(
    parent: _PinnedOutputParent,
    *,
    expected_bytes: bytes,
    expected_metadata: os.stat_result,
    identity: _PublishedOutputIdentity,
) -> None:
    """Make lexical leaf and pinned-parent proof the final success boundary."""

    try:
        lexical_bytes, lexical_metadata = read_exact_regular(
            parent.path / parent.name,
            maximum=_MAX_ROUTING_BYTES,
            expected_mode=0o600,
            allow_empty=False,
        )
    except AuthorityIOError as exc:
        raise DeepSeekConfigBindingError(
            "published routing path cannot be rebound to the pinned output"
        ) from exc
    if (
        lexical_bytes != expected_bytes
        or stat_identity(lexical_metadata) != stat_identity(expected_metadata)
        or lexical_metadata.st_dev != identity.device
        or lexical_metadata.st_ino != identity.inode
        or lexical_metadata.st_nlink != 1
        or stat.S_IMODE(lexical_metadata.st_mode) != 0o600
    ):
        raise DeepSeekConfigBindingError(
            "published routing path changed before receipt"
        )
    _require_pinned_parent(parent)


def _write_atomic_exclusive(
    parent: _PinnedOutputParent,
    payload: bytes,
) -> _PublishedOutputIdentity:
    """Publish complete bytes atomically through a no-replace hard link."""

    stage_flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    stage_fd: int | None = None
    stage_name: str | None = None
    stage_identity: _PublishedOutputIdentity | None = None
    link_attempted = False
    published = False
    published_identity: _PublishedOutputIdentity | None = None
    try:
        _require_pinned_parent(parent)
        try:
            os.stat(parent.name, dir_fd=parent.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise DeepSeekConfigBindingError(
                "exclusive output path appeared during preparation"
            )
        for _attempt in range(32):
            candidate = f".crucible-deepseek-bind-{secrets.token_hex(16)}.tmp"
            try:
                stage_fd = os.open(
                    candidate,
                    stage_flags,
                    0o600,
                    dir_fd=parent.descriptor,
                )
            except FileExistsError:
                continue
            stage_name = candidate
            created = os.fstat(stage_fd)
            stage_identity = _PublishedOutputIdentity(created.st_dev, created.st_ino)
            break
        if stage_fd is None or stage_name is None or stage_identity is None:
            raise DeepSeekConfigBindingError(
                "could not allocate an output staging file"
            )
        view = memoryview(payload)
        while view:
            written = os.write(stage_fd, view)
            if written <= 0:
                raise DeepSeekConfigBindingError(
                    "output staging write made no progress"
                )
            view = view[written:]
        os.fsync(stage_fd)
        staged = os.fstat(stage_fd)
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_dev != stage_identity.device
            or staged.st_ino != stage_identity.inode
            or staged.st_nlink != 1
            or staged.st_size != len(payload)
            or stat.S_IMODE(staged.st_mode) != 0o600
        ):
            raise DeepSeekConfigBindingError("staged output identity is not exact")
        published_identity = stage_identity
        os.close(stage_fd)
        stage_fd = None
        try:
            link_attempted = True
            os.link(
                stage_name,
                parent.name,
                src_dir_fd=parent.descriptor,
                dst_dir_fd=parent.descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise DeepSeekConfigBindingError(
                "exclusive output path appeared during publication"
            ) from exc
        published = True
        _unlink_published_at(
            parent.descriptor,
            stage_name,
            published_identity,
        )
        stage_name = None
        published_metadata = os.stat(
            parent.name,
            dir_fd=parent.descriptor,
            follow_symlinks=False,
        )
        if (
            published_metadata.st_dev != published_identity.device
            or published_metadata.st_ino != published_identity.inode
            or published_metadata.st_nlink != 1
            or stat.S_IMODE(published_metadata.st_mode) != 0o600
        ):
            raise DeepSeekConfigBindingError("published output identity is not exact")
        _require_pinned_parent(parent)
        return published_identity
    except BaseException as exc:
        if published and published_identity is not None:
            try:
                _unlink_published_at(
                    parent.descriptor,
                    parent.name,
                    published_identity,
                )
                published = False
            except BaseException as cleanup_exc:
                raise DeepSeekConfigBindingError(
                    "failed publication left uncertain output cleanup; do not dispatch"
                ) from cleanup_exc
        elif link_attempted and published_identity is not None:
            try:
                _unlink_published_if_same_at(
                    parent.descriptor,
                    parent.name,
                    published_identity,
                )
            except BaseException as cleanup_exc:
                raise DeepSeekConfigBindingError(
                    "ambiguous publication left uncertain output cleanup; do not dispatch"
                ) from cleanup_exc
        if isinstance(exc, DeepSeekConfigBindingError):
            raise
        if isinstance(exc, OSError):
            raise DeepSeekConfigBindingError(
                "cannot durably publish exclusive output"
            ) from exc
        raise
    finally:
        if stage_fd is not None:
            try:
                os.close(stage_fd)
            except OSError:
                pass
        if stage_name is not None:
            _cleanup_staging_at(
                parent.descriptor,
                stage_name,
                stage_identity,
            )


def bind_deepseek_backend_config(
    template: Path | str,
    output: Path | str,
) -> DeepSeekConfigBindingReceipt:
    """Create one exact, run-local routing file without provider authority."""

    source_path = Path(os.path.abspath(os.fspath(template)))
    output_path = Path(output)
    _validate_output_path(output_path)
    if source_path == output_path:
        raise DeepSeekConfigBindingError("template and output paths must differ")
    parent = _open_output_parent(output_path)
    published_identity: _PublishedOutputIdentity | None = None
    try:
        try:
            source_bytes, _metadata = read_exact_regular(
                source_path,
                maximum=_MAX_ROUTING_BYTES,
                allow_empty=False,
            )
        except AuthorityIOError as exc:
            raise DeepSeekConfigBindingError(
                "template must be one exact, non-symlinked regular file"
            ) from exc
        source_document, expected_roles = _admit_unbound_template(source_bytes)

        try:
            identity = capture_project_locked_deepseek_runtime_identity()
        except (DeepSeekSDKQualificationError, DeepSeekSDKUnavailable) as exc:
            raise DeepSeekConfigBindingError(
                "provider-free project worker identity capture failed"
            ) from exc
        runtime_fingerprint = identity.runtime_fingerprint
        if (
            not isinstance(runtime_fingerprint, str)
            or not runtime_fingerprint.startswith(_DIGEST_PREFIX)
            or len(runtime_fingerprint) != 71
            or any(char not in "0123456789abcdef" for char in runtime_fingerprint[7:])
        ):
            raise DeepSeekConfigBindingError(
                "captured worker runtime fingerprint is malformed"
            )

        output_bytes = _inject_runtime_fingerprint(
            source_document,
            runtime_fingerprint,
        )
        if len(output_bytes) > _MAX_ROUTING_BYTES:
            raise DeepSeekConfigBindingError("derived routing exceeds the byte bound")
        _verify_bound_bytes(
            output_bytes,
            source_document=source_document,
            expected_roles=expected_roles,
            runtime_fingerprint=runtime_fingerprint,
        )
        published_identity = _write_atomic_exclusive(parent, output_bytes)
        try:
            published_bytes, published_metadata = read_exact_regular_at(
                parent.descriptor,
                parent.name,
                maximum=_MAX_ROUTING_BYTES,
                expected_mode=0o600,
                allow_empty=False,
            )
        except AuthorityIOError as exc:
            raise DeepSeekConfigBindingError(
                "published routing cannot be reread as exact private bytes"
            ) from exc
        if (
            published_metadata.st_dev != published_identity.device
            or published_metadata.st_ino != published_identity.inode
        ):
            raise DeepSeekConfigBindingError("published routing inode changed")
        if published_bytes != output_bytes:
            raise DeepSeekConfigBindingError("published routing bytes changed")
        _verify_bound_bytes(
            published_bytes,
            source_document=source_document,
            expected_roles=expected_roles,
            runtime_fingerprint=runtime_fingerprint,
        )

        receipt = DeepSeekConfigBindingReceipt(
            source_path=str(source_path),
            output_path=str(output_path),
            source_sha256=_sha256(source_bytes),
            output_sha256=_sha256(published_bytes),
            worker_identity_schema=DEEPSEEK_PROJECT_LOCKED_RUNTIME_IDENTITY_SCHEMA,
            worker_runtime_fingerprint=runtime_fingerprint,
        )

        try:
            final_held_bytes, final_held_metadata = read_exact_regular_at(
                parent.descriptor,
                parent.name,
                maximum=_MAX_ROUTING_BYTES,
                expected_mode=0o600,
                allow_empty=False,
            )
        except AuthorityIOError as exc:
            raise DeepSeekConfigBindingError(
                "published routing leaf changed before receipt"
            ) from exc
        if (
            final_held_bytes != published_bytes
            or stat_identity(final_held_metadata) != stat_identity(published_metadata)
            or final_held_metadata.st_dev != published_identity.device
            or final_held_metadata.st_ino != published_identity.inode
            or final_held_metadata.st_nlink != 1
            or stat.S_IMODE(final_held_metadata.st_mode) != 0o600
        ):
            raise DeepSeekConfigBindingError(
                "published routing leaf changed before receipt"
            )
        _verify_final_published_path(
            parent,
            expected_bytes=published_bytes,
            expected_metadata=published_metadata,
            identity=published_identity,
        )
        return receipt
    except BaseException:
        if published_identity is not None:
            try:
                _unlink_published_at(
                    parent.descriptor,
                    parent.name,
                    published_identity,
                )
            except BaseException as cleanup_exc:
                raise DeepSeekConfigBindingError(
                    "failed validation left uncertain output cleanup; do not dispatch"
                ) from cleanup_exc
        raise
    finally:
        try:
            os.close(parent.descriptor)
        except OSError:
            pass


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="crucible backend_config_bind_deepseek")
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="unbound DeepSeek-only diagnostic routing template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new absolute run-local routing path (never overwritten)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = bind_deepseek_backend_config(args.template, args.output)
    except DeepSeekConfigBindingError as exc:
        print(
            json.dumps(
                {
                    "schema_version": _ERROR_SCHEMA,
                    "status": "error",
                    "error": str(exc),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(receipt.to_mapping(), sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "DeepSeekConfigBindingError",
    "DeepSeekConfigBindingReceipt",
    "bind_deepseek_backend_config",
    "main",
]
