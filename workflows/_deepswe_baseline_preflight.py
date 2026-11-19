# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Provider-free development-verifier preflight for DeepSWE workspaces.

This module owns the closed parameter projection and the mutation-journal
transaction used to qualify the candidate and clean-baseline checkouts before
any model backend is constructed.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, cast

from assay import Spec

from crucible.common.benchmark_runtime import canonical_fingerprint
from crucible.products.deepswe.dataset import DeepSWEInstance
from crucible.products.deepswe.dev_verify_overrides import (
    resolve_dev_verify_command,
)

if TYPE_CHECKING:
    from crucible.workflows._build_verify import BuildVerifyOutcome


DEVELOPMENT_PREFLIGHT_RESTORATION_SCHEMA = (
    "crucible.deepswe-development-preflight-restoration.v2"
)
_JOURNAL_TRANSACTION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_RESTORATION_LABELS = ("candidate", "baseline")
_WORKSPACE_RESTORATION_KEYS = frozenset({
    "label",
    "workspace",
    "workspace_device",
    "workspace_inode",
    "namespace",
    "split_namespaces",
    "initialized_gitlinks",
    "journal_transaction_id",
    "journal_family",
    "journal_state",
    "journal_cleanup_pending",
    "gate_created_paths",
})
_SPLIT_NAMESPACE_KEYS = frozenset({
    "relative",
    "namespace",
    "device",
    "inode",
})
_INITIALIZED_GITLINK_KEYS = frozenset({
    "relative",
    "device",
    "inode",
    "guard_transaction_id",
    "guard_slice_id",
})
_JOURNAL_FAMILY_MEMBER_KEYS = frozenset({
    "transaction_id",
    "parent_transaction_id",
})


def _exact_nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"development preflight {field} must be an exact integer")
    return value


def _exact_dict(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"development preflight {field} must be an exact object")
    return cast(dict[str, object], value)


def _exact_list(value: object, *, field: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"development preflight {field} must be an exact list")
    return cast(list[object], value)


def _exact_namespace_name(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\x00" in value
    ):
        raise ValueError(f"development preflight {field} is invalid")
    return value


def _exact_relative_path(value: object, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"development preflight {field} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"development preflight {field} is not canonical")
    return value


def _exact_canonical_workspace(value: object) -> str:
    """Validate a retained path lexically, without consulting its filesystem."""

    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("development preflight workspace path is invalid")
    path = Path(value)
    components = value[1:].split("/") if value.startswith("/") else []
    if (
        not path.is_absolute()
        or not components
        or any(component in {"", ".", ".."} for component in components)
        or str(path) != value
    ):
        raise ValueError("development preflight workspace path is not canonical")
    return value


@dataclass(frozen=True)
class DevelopmentPreflightWorkspaceRestoration:
    """Exact post-rollback authority for one preflight checkout.

    ``gate_created_paths`` names what the gate run created in this checkout
    and the restoration removed again, limited to what a later slice gate
    would charge to its slice: untracked files Git does not ignore, and new
    directories inside the mutation journal's directory authority (not
    ignored, or ignored below a directory that existed before the gate run).
    The entries are checkout-relative roots, sorted, none below another.  A
    new top-level directory whose whole content Git ignores, and an ignored
    file, are not recorded: a slice gate accepts them.

    Two effects of a gate run are restored here without being recorded,
    although a slice gate charges them: a path added below a directory that
    was already untracked before the gate run (Git lists such a directory as
    one root, however empty), and a changed or deleted file the checkout
    already had, tracked or untracked.
    """

    label: str
    workspace: str
    workspace_device: int
    workspace_inode: int
    namespace: tuple[str, ...]
    split_namespaces: tuple[tuple[str, tuple[str, ...], int, int], ...]
    initialized_gitlinks: tuple[tuple[str, int, int, str, str], ...]
    journal_transaction_id: str
    journal_family: tuple[tuple[str, str | None], ...]
    gate_created_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self) is not DevelopmentPreflightWorkspaceRestoration:
            raise ValueError("development preflight workspace subclasses are invalid")
        if type(self.label) is not str or self.label not in _RESTORATION_LABELS:
            raise ValueError("development preflight workspace label is invalid")
        _exact_canonical_workspace(self.workspace)
        _exact_nonnegative_int(self.workspace_device, field="workspace device")
        _exact_nonnegative_int(self.workspace_inode, field="workspace inode")
        if type(self.namespace) is not tuple:
            raise ValueError("development preflight workspace namespace is invalid")
        for name in self.namespace:
            _exact_namespace_name(name, field="workspace namespace entry")
        if tuple(sorted(set(self.namespace))) != self.namespace:
            raise ValueError("development preflight workspace namespace is not canonical")
        if type(self.split_namespaces) is not tuple:
            raise ValueError("development preflight split namespaces are invalid")
        split_paths: list[str] = []
        for split_item in self.split_namespaces:
            if type(split_item) is not tuple or len(split_item) != 4:
                raise ValueError("development preflight split namespace is invalid")
            relative, namespace, device, inode = split_item
            relative = _exact_relative_path(
                relative,
                field="split namespace path",
            )
            if type(namespace) is not tuple:
                raise ValueError("development preflight split namespace is invalid")
            for name in namespace:
                _exact_namespace_name(name, field="split namespace entry")
            if tuple(sorted(set(namespace))) != namespace:
                raise ValueError("development preflight split namespace is not canonical")
            _exact_nonnegative_int(device, field="split namespace device")
            _exact_nonnegative_int(inode, field="split namespace inode")
            split_paths.append(relative)
        if (
            tuple(split_paths) != tuple(sorted(split_paths))
            or len(split_paths) != len(set(split_paths))
        ):
            raise ValueError("development preflight split paths are not canonical")
        if type(self.initialized_gitlinks) is not tuple:
            raise ValueError("development preflight Git-link authority is invalid")
        gitlink_paths: list[str] = []
        gitlink_guard_ids: dict[str, str] = {}
        gitlink_guard_slices: dict[str, str] = {}
        for gitlink_item in self.initialized_gitlinks:
            if type(gitlink_item) is not tuple or len(gitlink_item) != 5:
                raise ValueError("development preflight Git-link path is invalid")
            relative, device, inode, guard_transaction_id, guard_slice_id = (
                gitlink_item
            )
            relative = _exact_relative_path(relative, field="Git-link path")
            _exact_nonnegative_int(device, field="Git-link device")
            _exact_nonnegative_int(inode, field="Git-link inode")
            if (
                type(guard_transaction_id) is not str
                or _JOURNAL_TRANSACTION_ID_RE.fullmatch(guard_transaction_id)
                is None
                or type(guard_slice_id) is not str
                or not guard_slice_id
                or "\x00" in guard_slice_id
            ):
                raise ValueError(
                    "development preflight Git-link journal binding is invalid"
                )
            gitlink_paths.append(relative)
            gitlink_guard_ids[relative] = guard_transaction_id
            gitlink_guard_slices[relative] = guard_slice_id
        if (
            tuple(gitlink_paths) != tuple(sorted(gitlink_paths))
            or len(gitlink_paths) != len(set(gitlink_paths))
            or any(
                split_path == gitlink_path
                or split_path.startswith(gitlink_path + "/")
                for split_path in split_paths
                for gitlink_path in gitlink_paths
            )
        ):
            raise ValueError("development preflight Git-link paths are not canonical")
        nested_namespaces = {
            relative: namespace
            for relative, namespace, _device, _inode in self.split_namespaces
        }
        for relative in (*split_paths, *gitlink_paths):
            parts = PurePosixPath(relative).parts
            for offset, name in enumerate(parts):
                parent = "/".join(parts[:offset])
                if any(
                    parent == gitlink_path
                    or parent.startswith(gitlink_path + "/")
                    for gitlink_path in gitlink_paths
                ):
                    continue
                parent_namespace = (
                    self.namespace
                    if not parent
                    else nested_namespaces.get(parent)
                )
                if parent_namespace is None or name not in parent_namespace:
                    raise ValueError(
                        "development preflight nested path is outside its "
                        "recorded parent namespace"
                    )
        if type(self.gate_created_paths) is not tuple:
            raise ValueError("development preflight gate-created paths are invalid")
        for created in self.gate_created_paths:
            _exact_relative_path(created, field="gate-created path")
        gate_created = frozenset(self.gate_created_paths)
        if tuple(sorted(gate_created)) != self.gate_created_paths or any(
            parent.as_posix() in gate_created
            for created in self.gate_created_paths
            for parent in PurePosixPath(created).parents
        ):
            raise ValueError(
                "development preflight gate-created paths are not canonical"
            )
        for created in self.gate_created_paths:
            if any(
                created == gitlink_path
                or created.startswith(gitlink_path + "/")
                for gitlink_path in gitlink_paths
            ):
                raise ValueError(
                    "development preflight gate-created path lies in an "
                    "initialized Git link"
                )
            # The recorded namespaces are the names that existed before the
            # gate ran, so none of them can be one the gate run created.
            parent, _separator, name = created.rpartition("/")
            parent_namespace = (
                self.namespace if not parent else nested_namespaces.get(parent)
            )
            if parent_namespace is not None and name in parent_namespace:
                raise ValueError(
                    "development preflight gate-created path existed before "
                    "the gate run"
                )
        identities = [
            (self.workspace_device, self.workspace_inode),
            *(
                (device, inode)
                for _relative, _namespace, device, inode in self.split_namespaces
            ),
            *(
                (device, inode)
                for _relative, device, inode, _transaction, _slice in (
                    self.initialized_gitlinks
                )
            ),
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("development preflight workspace identities overlap")
        if (
            type(self.journal_transaction_id) is not str
            or _JOURNAL_TRANSACTION_ID_RE.fullmatch(
                self.journal_transaction_id
            ) is None
        ):
            raise ValueError("development preflight journal transaction is invalid")
        if type(self.journal_family) is not tuple or not self.journal_family:
            raise ValueError("development preflight journal family is invalid")
        family_ids: list[str] = []
        family_parents: dict[str, str | None] = {}
        for family_item in self.journal_family:
            if type(family_item) is not tuple or len(family_item) != 2:
                raise ValueError("development preflight journal family is invalid")
            transaction_id, parent_transaction_id = family_item
            if (
                type(transaction_id) is not str
                or _JOURNAL_TRANSACTION_ID_RE.fullmatch(transaction_id) is None
                or (
                    parent_transaction_id is not None
                    and (
                        type(parent_transaction_id) is not str
                        or _JOURNAL_TRANSACTION_ID_RE.fullmatch(
                            parent_transaction_id
                        ) is None
                    )
                )
            ):
                raise ValueError("development preflight journal family is invalid")
            family_ids.append(transaction_id)
            family_parents[transaction_id] = parent_transaction_id
        if (
            len(family_ids) != len(set(family_ids))
            or set(family_parents) != set(family_ids)
            or len(family_ids) < 1 + len(self.initialized_gitlinks)
            or self.journal_transaction_id not in family_parents
            or family_parents[self.journal_transaction_id] is not None
        ):
            raise ValueError("development preflight journal family is incomplete")
        depths = {self.journal_transaction_id: 0}
        unresolved = set(family_ids).difference(depths)
        while unresolved:
            progressed = False
            for transaction_id in sorted(unresolved):
                parent_transaction_id = family_parents[transaction_id]
                if parent_transaction_id in depths:
                    depths[transaction_id] = depths[cast(str, parent_transaction_id)] + 1
                    unresolved.remove(transaction_id)
                    progressed = True
                    break
            if not progressed:
                raise ValueError(
                    "development preflight journal family is disconnected or cyclic"
                )
        expected_family_order = tuple(
            sorted(family_ids, key=lambda item: (depths[item], item))
        )
        if tuple(family_ids) != expected_family_order:
            raise ValueError("development preflight journal family is not canonical")
        if (
            len(set(gitlink_guard_ids.values())) != len(gitlink_guard_ids)
            or not set(gitlink_guard_ids.values()).issubset(family_parents)
            or self.journal_transaction_id in gitlink_guard_ids.values()
        ):
            raise ValueError(
                "development preflight Git-link journal binding is incomplete"
            )
        for relative in gitlink_paths:
            ancestors = [
                candidate
                for candidate in gitlink_paths
                if relative.startswith(candidate + "/")
            ]
            nearest = max(ancestors, key=len, default=None)
            if nearest is None:
                expected_parent = self.journal_transaction_id
                expected_slice = (
                    f"dev_verify_preflight__{self.label}::gitlink::{relative}"
                )
            else:
                expected_parent = gitlink_guard_ids[nearest]
                nested_relative = relative[len(nearest) + 1 :]
                expected_slice = (
                    f"{gitlink_guard_slices[nearest]}::gitlink::{nested_relative}"
                )
            guard_transaction_id = gitlink_guard_ids[relative]
            if (
                family_parents[guard_transaction_id] != expected_parent
                or gitlink_guard_slices[relative] != expected_slice
            ):
                raise ValueError(
                    "development preflight Git-link journal topology is invalid"
                )

    @classmethod
    def from_payload(
        cls,
        payload: object,
    ) -> "DevelopmentPreflightWorkspaceRestoration":
        """Decode one exact canonical workspace-restoration preimage."""

        if cls is not DevelopmentPreflightWorkspaceRestoration:
            raise ValueError("development preflight workspace subclasses are invalid")
        record = _exact_dict(payload, field="workspace restoration")
        if set(record) != _WORKSPACE_RESTORATION_KEYS:
            raise ValueError("development preflight workspace fields are invalid")
        if type(record["journal_state"]) is not str or record["journal_state"] != (
            "rolled_back"
        ):
            raise ValueError("development preflight journal state is invalid")
        if record["journal_cleanup_pending"] is not False:
            raise ValueError("development preflight journal cleanup state is invalid")

        namespace = tuple(
            _exact_namespace_name(item, field="workspace namespace entry")
            for item in _exact_list(
                record["namespace"],
                field="workspace namespace",
            )
        )
        split_namespaces: list[tuple[str, tuple[str, ...], int, int]] = []
        for raw_split in _exact_list(
            record["split_namespaces"],
            field="split namespaces",
        ):
            split = _exact_dict(raw_split, field="split namespace")
            if set(split) != _SPLIT_NAMESPACE_KEYS:
                raise ValueError("development preflight split fields are invalid")
            split_namespaces.append(
                (
                    _exact_relative_path(
                        split["relative"],
                        field="split namespace path",
                    ),
                    tuple(
                        _exact_namespace_name(
                            item,
                            field="split namespace entry",
                        )
                        for item in _exact_list(
                            split["namespace"],
                            field="split namespace entries",
                        )
                    ),
                    _exact_nonnegative_int(
                        split["device"],
                        field="split namespace device",
                    ),
                    _exact_nonnegative_int(
                        split["inode"],
                        field="split namespace inode",
                    ),
                )
            )

        initialized_gitlinks: list[tuple[str, int, int, str, str]] = []
        for raw_gitlink in _exact_list(
            record["initialized_gitlinks"],
            field="initialized Git links",
        ):
            gitlink = _exact_dict(raw_gitlink, field="initialized Git link")
            if set(gitlink) != _INITIALIZED_GITLINK_KEYS:
                raise ValueError("development preflight Git-link fields are invalid")
            initialized_gitlinks.append(
                (
                    _exact_relative_path(
                        gitlink["relative"],
                        field="Git-link path",
                    ),
                    _exact_nonnegative_int(
                        gitlink["device"],
                        field="Git-link device",
                    ),
                    _exact_nonnegative_int(
                        gitlink["inode"],
                        field="Git-link inode",
                    ),
                    cast(str, gitlink["guard_transaction_id"]),
                    cast(str, gitlink["guard_slice_id"]),
                )
            )

        journal_family: list[tuple[str, str | None]] = []
        for raw_member in _exact_list(
            record["journal_family"],
            field="journal family",
        ):
            member = _exact_dict(raw_member, field="journal family member")
            if set(member) != _JOURNAL_FAMILY_MEMBER_KEYS:
                raise ValueError("development preflight journal family fields are invalid")
            transaction_id = member["transaction_id"]
            parent_transaction_id = member["parent_transaction_id"]
            if type(transaction_id) is not str or (
                parent_transaction_id is not None
                and type(parent_transaction_id) is not str
            ):
                raise ValueError("development preflight journal family is invalid")
            journal_family.append((transaction_id, parent_transaction_id))

        gate_created_paths = tuple(
            _exact_relative_path(item, field="gate-created path")
            for item in _exact_list(
                record["gate_created_paths"],
                field="gate-created paths",
            )
        )

        result = DevelopmentPreflightWorkspaceRestoration(
            label=cast(str, record["label"]),
            workspace=cast(str, record["workspace"]),
            workspace_device=_exact_nonnegative_int(
                record["workspace_device"],
                field="workspace device",
            ),
            workspace_inode=_exact_nonnegative_int(
                record["workspace_inode"],
                field="workspace inode",
            ),
            namespace=namespace,
            split_namespaces=tuple(split_namespaces),
            initialized_gitlinks=tuple(initialized_gitlinks),
            journal_transaction_id=cast(str, record["journal_transaction_id"]),
            journal_family=tuple(journal_family),
            gate_created_paths=gate_created_paths,
        )
        if result.to_payload() != record:
            raise ValueError("development preflight workspace payload is not canonical")
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "label": self.label,
            "workspace": self.workspace,
            "workspace_device": self.workspace_device,
            "workspace_inode": self.workspace_inode,
            "namespace": list(self.namespace),
            "split_namespaces": [
                {
                    "relative": relative,
                    "namespace": list(namespace),
                    "device": device,
                    "inode": inode,
                }
                for relative, namespace, device, inode in self.split_namespaces
            ],
            "initialized_gitlinks": [
                {
                    "relative": relative,
                    "device": device,
                    "inode": inode,
                    "guard_transaction_id": guard_transaction_id,
                    "guard_slice_id": guard_slice_id,
                }
                for (
                    relative,
                    device,
                    inode,
                    guard_transaction_id,
                    guard_slice_id,
                ) in self.initialized_gitlinks
            ],
            "journal_transaction_id": self.journal_transaction_id,
            "journal_family": [
                {
                    "transaction_id": transaction_id,
                    "parent_transaction_id": parent_transaction_id,
                }
                for transaction_id, parent_transaction_id in self.journal_family
            ],
            "journal_state": "rolled_back",
            "journal_cleanup_pending": False,
            "gate_created_paths": list(self.gate_created_paths),
        }


@dataclass(frozen=True)
class DevelopmentPreflightRestorationReceipt:
    """Typed restoration-only evidence emitted after both checkouts restore.

    This receipt does not bind the verifier result or process settlement.  A
    future executor must compose those independent authorities explicitly.
    It is emitted whatever the gate run returned or raised, and names the
    paths that run created in each checkout.
    """

    workspaces: tuple[DevelopmentPreflightWorkspaceRestoration, ...]

    def __post_init__(self) -> None:
        if type(self) is not DevelopmentPreflightRestorationReceipt:
            raise ValueError("development preflight receipt subclasses are invalid")
        if (
            type(self.workspaces) is not tuple
            or len(self.workspaces) != 2
            or any(
                type(item) is not DevelopmentPreflightWorkspaceRestoration
                for item in self.workspaces
            )
            or tuple(item.label for item in self.workspaces)
            != _RESTORATION_LABELS
            or self.workspaces[0].workspace == self.workspaces[1].workspace
            or (
                self.workspaces[0].workspace_device,
                self.workspaces[0].workspace_inode,
            )
            == (
                self.workspaces[1].workspace_device,
                self.workspaces[1].workspace_inode,
            )
            or self.workspaces[0].journal_transaction_id
            == self.workspaces[1].journal_transaction_id
        ):
            raise ValueError("development preflight receipt workspace set is invalid")
        workspace_identities = [
            identity
            for workspace in self.workspaces
            for identity in (
                (workspace.workspace_device, workspace.workspace_inode),
                *(
                    (device, inode)
                    for _relative, _namespace, device, inode in (
                        workspace.split_namespaces
                    )
                ),
                *(
                    (device, inode)
                    for _relative, device, inode, _transaction, _slice in (
                        workspace.initialized_gitlinks
                    )
                ),
            )
        ]
        family_ids = [
            transaction_id
            for workspace in self.workspaces
            for transaction_id, _parent in workspace.journal_family
        ]
        if (
            len(workspace_identities) != len(set(workspace_identities))
            or len(family_ids) != len(set(family_ids))
        ):
            raise ValueError("development preflight receipt authorities overlap")

    @classmethod
    def from_payload(cls, payload: object) -> "DevelopmentPreflightRestorationReceipt":
        """Decode an exact canonical restoration receipt preimage."""

        if cls is not DevelopmentPreflightRestorationReceipt:
            raise ValueError("development preflight receipt subclasses are invalid")
        record = _exact_dict(payload, field="restoration receipt")
        if set(record) != {"schema_version", "workspaces"}:
            raise ValueError("development preflight receipt fields are invalid")
        if (
            type(record["schema_version"]) is not str
            or record["schema_version"] != DEVELOPMENT_PREFLIGHT_RESTORATION_SCHEMA
        ):
            raise ValueError("development preflight receipt schema is invalid")
        raw_workspaces = _exact_list(
            record["workspaces"],
            field="receipt workspaces",
        )
        result = DevelopmentPreflightRestorationReceipt(
            workspaces=tuple(
                DevelopmentPreflightWorkspaceRestoration.from_payload(item)
                for item in raw_workspaces
            )
        )
        if result.to_payload() != record:
            raise ValueError("development preflight receipt payload is not canonical")
        return result

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": DEVELOPMENT_PREFLIGHT_RESTORATION_SCHEMA,
            "workspaces": [item.to_payload() for item in self.workspaces],
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.to_payload())


def build_development_preflight_parameters(
    *,
    instance: DeepSWEInstance,
    candidate_container_id: str,
    baseline_container_id: str,
    logs_dir: Path,
    max_patch_bytes: int,
) -> dict[str, Any]:
    """Project one task into the closed development-preflight parameters."""

    metadata: dict[str, Any] = {
        "container_id": candidate_container_id,
        "logs_dir": str(logs_dir),
        "language": instance.language,
        "task_id": instance.task_id,
        "ext_id": instance.ext_id,
        "max_patch_bytes": max_patch_bytes,
        "baseline_container_id": baseline_container_id,
        "image_digest": instance.docker_image,
        "base_commit": instance.base_commit_hash,
    }
    dev_verify_command = resolve_dev_verify_command(
        task_id=instance.task_id,
        repo=instance.repo,
    )
    if dev_verify_command:
        metadata["dev_verify_command"] = dev_verify_command
    return {"deepswe_metadata": metadata}


def _preflight_journal_scope(
    workspace: Path,
    *,
    gitlinks: tuple[str, ...],
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, frozenset[str], int, int], ...],
]:
    """Cover a checkout without claiming an initialized nested repository.

    ``WorkspaceMutationJournal`` automatically creates guard journals for
    initialized Git links, but its owning transaction must not declare a
    parent directory that contains one.  Image checkouts can place a gitlink
    below an otherwise ordinary top-level tree (for example,
    ``benchmarks/release_data``).  Recursively split only those ancestors and
    keep every sibling subtree in the owning transaction.
    """

    scope: list[str] = []
    split_namespaces: list[tuple[str, frozenset[str], int, int]] = []

    def _visit(relative: str = "") -> None:
        directory = workspace / relative if relative else workspace
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            rendered = child.relative_to(workspace).as_posix()
            if not relative and child.name == ".git":
                continue
            related = tuple(
                gitlink
                for gitlink in gitlinks
                if gitlink == rendered or gitlink.startswith(rendered + "/")
            )
            if not related:
                scope.append(rendered)
                continue
            if rendered in gitlinks:
                # The mutation journal creates a child guard for an
                # initialized gitlink.  It must remain outside this scope.
                continue
            if child.is_symlink() or not child.is_dir():
                raise RuntimeError(
                    "Git-link ancestor is not a real directory during "
                    f"development-verifier preflight: {rendered}"
                )
            metadata = child.lstat()
            split_namespaces.append(
                (
                    rendered,
                    frozenset(descendant.name for descendant in child.iterdir()),
                    metadata.st_dev,
                    metadata.st_ino,
                )
            )
            _visit(rendered)

    _visit()
    return tuple(scope), tuple(split_namespaces)


def _preflight_untracked_roots(worktree: Path) -> tuple[str, ...]:
    """Return every untracked leaf or collapsed untracked directory.

    Development-verifier workspaces are freshly populated from the immutable
    task image. Requiring initialized submodules to start without any untracked
    residue lets preflight safely remove every untracked artifact it creates,
    including ignored build caches, without taking an unbounded snapshot.
    """

    from crucible.workflows import _mutation_journal  # noqa: PLC0415

    raw = _mutation_journal._git(
        worktree,
        "ls-files",
        "-z",
        "--others",
        "--directory",
    )
    return _mutation_journal._dedupe_roots(
        _mutation_journal._normalize_relative(
            worktree,
            os.fsdecode(encoded).rstrip("/"),
        )
        for encoded in raw.split(b"\0")
        if encoded
    )


def _preflight_gate_residue(
    worktree: Path,
) -> tuple[frozenset[str], frozenset[str]]:
    """List what the two slice-gate checks treat as untracked in a checkout.

    Returns the collapsed roots Git reports as untracked and not ignored
    (files, and directories however empty), and the collapsed ignored
    directory roots from which the mutation journal derives its directory
    authority.  Neither listing descends into a collapsed directory.
    """

    from crucible.workflows import _mutation_journal  # noqa: PLC0415

    raw = _mutation_journal._git(
        worktree,
        "ls-files",
        "-z",
        "--others",
        "--exclude-standard",
        "--directory",
    )
    return (
        frozenset(
            os.fsdecode(encoded).rstrip("/")
            for encoded in raw.split(b"\0")
            if encoded
        ),
        frozenset(_mutation_journal._ignored_directory_roots(worktree)),
    )


def _preflight_gate_created_roots(
    before: tuple[frozenset[str], frozenset[str]],
    after: tuple[frozenset[str], frozenset[str]],
    *,
    namespace: frozenset[str],
    split_namespaces: tuple[tuple[str, frozenset[str], int, int], ...],
    initialized_gitlinks: tuple[tuple[str, int, int], ...],
) -> tuple[str, ...]:
    """Name the roots a gate run created that a slice gate would charge.

    A later slice gate fails its slice for an untracked file Git reports and
    for a new directory inside the mutation journal's directory authority.
    The journal keeps an ignored directory root opaque only when it is
    disjoint from every directory that existed before: a new ignored root at
    the top level is, one below an existing directory is not.  So the record
    is every new untracked root Git does not ignore, without the new
    top-level directories the journal treats as ignored roots, plus every
    new ignored directory root below the top level.

    Names that existed before the gate ran are left out where the preflight
    knows them (the top level and each split ancestor): a tracked path the
    gate run merely stopped tracking was not created by it.

    Both listings are collapsed, so the record is a difference of roots.  A
    directory that was untracked before the gate ran is the same one root
    afterwards: what the gate run added below it is not in the record, and
    a slice gate, which lists every untracked file, still charges it.
    """

    from crucible.workflows import _mutation_journal  # noqa: PLC0415

    visible_before, ignored_before = before
    visible_after, ignored_after = after
    created = {
        root
        for root in visible_after - visible_before
        if "/" in root or root not in ignored_after
    }
    created.update(
        root for root in ignored_after - ignored_before if "/" in root
    )
    existing_names = {
        "": namespace,
        **{
            relative: ancestor_namespace
            for relative, ancestor_namespace, _device, _inode in split_namespaces
        },
    }

    def _created_by_gate(root: str) -> bool:
        if any(
            root == gitlink or root.startswith(gitlink + "/")
            for gitlink, _device, _inode in initialized_gitlinks
        ):
            # Residue in an initialized submodule is a restoration error.
            return False
        parent, _separator, name = root.rpartition("/")
        return name not in existing_names.get(parent, frozenset())

    return tuple(
        sorted(
            _mutation_journal._dedupe_roots(
                root for root in created if _created_by_gate(root)
            )
        )
    )


def _preflight_initialized_gitlinks(
    workspace: Path,
    *,
    gitlinks: tuple[str, ...],
) -> tuple[tuple[str, int, int], ...]:
    """Recursively bind every initialized gitlink worktree by identity."""

    from crucible.workflows import _mutation_journal  # noqa: PLC0415

    initialized: list[tuple[str, int, int]] = []

    def _visit(repository: Path, repository_gitlinks: tuple[str, ...]) -> None:
        for gitlink in repository_gitlinks:
            nested = repository / gitlink
            if not (nested / ".git").exists():
                continue
            metadata = nested.lstat()
            if nested.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
                raise _mutation_journal.MutationJournalError(
                    "initialized Git submodule is missing or aliased during "
                    f"development-verifier preflight: {nested}"
                )
            relative = nested.relative_to(workspace).as_posix()
            initialized.append(
                (
                    relative,
                    metadata.st_dev,
                    metadata.st_ino,
                )
            )
            _visit(
                nested,
                tuple(_mutation_journal._gitlink_paths(nested)),
            )

    _visit(workspace, gitlinks)
    return tuple(initialized)


def _preflight_gitlink_journal_bindings(
    workspace: Path,
    *,
    initialized_gitlinks: tuple[tuple[str, int, int], ...],
    journal: Any,
) -> tuple[tuple[str, str, str], ...]:
    """Bind every initialized Git link to its exact guard transaction."""

    from crucible.workflows import _mutation_journal  # noqa: PLC0415

    expected = {relative for relative, _device, _inode in initialized_gitlinks}
    observed: dict[str, tuple[str, str]] = {}
    seen_transactions = {journal.transaction_id}

    def _visit(owner: Any) -> None:
        for child in owner._nested_journals():
            if child.transaction_id in seen_transactions:
                raise _mutation_journal.MutationJournalError(
                    "development-verifier preflight journal family repeats a "
                    "transaction"
                )
            seen_transactions.add(child.transaction_id)
            raw_cwd = child._manifest.get("cwd")
            raw_slice = child._manifest.get("slice_id")
            if type(raw_cwd) is not str or type(raw_slice) is not str:
                raise _mutation_journal.MutationJournalError(
                    "development-verifier preflight Git-link journal binding "
                    "is malformed"
                )
            child_workspace = Path(raw_cwd)
            try:
                relative = child_workspace.relative_to(workspace).as_posix()
            except ValueError:
                relative = None
            if relative in expected:
                if (
                    relative in observed
                    or child.parent_transaction_id != owner.transaction_id
                    or child._manifest.get("nested_gitlink_guard") is not True
                    or child._manifest.get("external_worktree_guard") is not False
                ):
                    raise _mutation_journal.MutationJournalError(
                        "development-verifier preflight Git-link journal binding "
                        "is ambiguous"
                    )
                observed[relative] = (child.transaction_id, raw_slice)
            _visit(child)

    _visit(journal)
    if set(observed) != expected:
        raise _mutation_journal.MutationJournalError(
            "development-verifier preflight Git-link journal binding is incomplete"
        )
    return tuple(
        (relative, *observed[relative])
        for relative in sorted(observed)
    )


def _preflight_workspace_lock_order(
    workspace: Path,
    *,
    mutation_journal: Any,
) -> tuple[int, int, str]:
    """Return the journal's canonical lock component plus an exact root tie.

    A top-level checkout and each of its initialized submodules share the
    topmost superproject lock component, but the journal still re-enters that
    component through every exact submodule worktree while creating and
    restoring guard journals.  Ordering first by the shared component and
    then by exact worktree path lets preflight hold all of those authorities
    before moving to the next independent checkout.
    """

    common = mutation_journal._workspace_authority_common_dir(workspace)
    metadata = common.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise mutation_journal.MutationJournalError(
            "development-verifier preflight lock root is not a real "
            f"directory: {common}"
        )
    component = mutation_journal._workspace_lock_order_key(
        (
            common,
            metadata.st_dev,
            metadata.st_ino,
        )
    )
    return (*component, str(workspace))


def run_development_verifier_preflight(
    *,
    spec: Spec,
    candidate_workspace: Path,
    baseline_workspace: Path,
    log_dir: Path,
    journal_root: Path,
    extra_parameters: Mapping[str, Any],
    restoration_callback: (
        Callable[[DevelopmentPreflightRestorationReceipt], None] | None
    ) = None,
) -> BuildVerifyOutcome:
    """Qualify and exactly restore both writable public-test checkouts."""

    from crucible.workflows import (  # noqa: PLC0415
        _build_verify,
        _mutation_journal,
    )

    resolved = (
        candidate_workspace.resolve(),
        baseline_workspace.resolve(),
    )
    if resolved[0] == resolved[1]:
        raise ValueError(
            "development-verifier candidate and baseline checkouts must differ"
        )
    preflight_parameters = dict(extra_parameters)
    raw_metadata = preflight_parameters.get("deepswe_metadata")
    if isinstance(raw_metadata, dict):
        preflight_parameters["deepswe_metadata"] = {
            **raw_metadata,
            "dev_verify_clean_probe": True,
        }

    discovered_gitlinks: dict[Path, tuple[str, ...]] = {}
    discovered_initialized: dict[
        Path,
        tuple[tuple[str, int, int], ...],
    ] = {}
    lock_workspaces = set(resolved)
    for workspace in resolved:
        if not workspace.is_dir() or workspace.is_symlink():
            raise _mutation_journal.MutationJournalError(
                f"preflight checkout is missing or aliased: {workspace}"
            )
        gitlinks = tuple(_mutation_journal._gitlink_paths(workspace))
        initialized = _preflight_initialized_gitlinks(
            workspace,
            gitlinks=gitlinks,
        )
        discovered_gitlinks[workspace] = gitlinks
        discovered_initialized[workspace] = initialized
        lock_workspaces.update(
            workspace / relative for relative, _device, _inode in initialized
        )

    sealed: list[
        tuple[
            str,
            Path,
            frozenset[str],
            int,
            int,
            tuple[tuple[str, frozenset[str], int, int], ...],
            tuple[tuple[str, int, int], ...],
            tuple[tuple[str, str, str], ...],
            _mutation_journal.WorkspaceMutationJournal,
        ]
    ] = []
    # The record of gate-created paths travels in the restoration receipt, so
    # the checkouts are listed only when a caller consumes that receipt.
    residue_before: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    gate_created: dict[str, tuple[str, ...]] = {}
    record_errors: list[BaseException] = []
    preflight_ready = False
    with ExitStack() as locks:
        for workspace in sorted(
            lock_workspaces,
            key=lambda item: _preflight_workspace_lock_order(
                item,
                mutation_journal=_mutation_journal,
            ),
        ):
            locks.enter_context(_mutation_journal.workspace_mutation_lock(workspace))
        try:
            for sequence, (label, workspace) in enumerate(
                (("candidate", resolved[0]), ("baseline", resolved[1])),
                start=1,
            ):
                if not workspace.is_dir() or workspace.is_symlink():
                    raise _mutation_journal.MutationJournalError(
                        f"preflight checkout is missing or aliased: {workspace}"
                    )
                workspace_metadata = workspace.lstat()
                namespace = frozenset(path.name for path in workspace.iterdir())
                gitlinks = tuple(_mutation_journal._gitlink_paths(workspace))
                if gitlinks != discovered_gitlinks[workspace]:
                    raise _mutation_journal.MutationJournalError(
                        "development-verifier preflight Git-link topology "
                        f"changed during lock acquisition: {workspace}"
                    )
                scope, split_namespaces = _preflight_journal_scope(
                    workspace,
                    gitlinks=gitlinks,
                )
                initialized_gitlinks = _preflight_initialized_gitlinks(
                    workspace,
                    gitlinks=gitlinks,
                )
                if initialized_gitlinks != discovered_initialized[workspace]:
                    raise _mutation_journal.MutationJournalError(
                        "development-verifier preflight initialized Git-link "
                        "authority changed during lock acquisition: "
                        f"{workspace}"
                    )
                untracked_by_gitlink = {
                    relative: _preflight_untracked_roots(workspace / relative)
                    for relative, _device, _inode in initialized_gitlinks
                }
                dirty_untracked = {
                    gitlink: untracked
                    for gitlink, untracked in untracked_by_gitlink.items()
                    if untracked
                }
                if dirty_untracked:
                    raise _mutation_journal.MutationJournalError(
                        "development-verifier preflight requires fresh "
                        "initialized submodules without untracked residue: "
                        f"{dirty_untracked!r}"
                    )
                journal = _mutation_journal.WorkspaceMutationJournal.begin(
                    cwd=workspace,
                    journal_root=journal_root,
                    slice_id=f"dev_verify_preflight__{label}",
                    scope=scope,
                    sequence=sequence,
                    spec=spec,
                )
                sealed.append(
                    (
                        label,
                        workspace,
                        namespace,
                        workspace_metadata.st_dev,
                        workspace_metadata.st_ino,
                        split_namespaces,
                        initialized_gitlinks,
                        (),
                        journal,
                    )
                )
                gitlink_journal_bindings = _preflight_gitlink_journal_bindings(
                    workspace,
                    initialized_gitlinks=initialized_gitlinks,
                    journal=journal,
                )
                sealed[-1] = (
                    label,
                    workspace,
                    namespace,
                    workspace_metadata.st_dev,
                    workspace_metadata.st_ino,
                    split_namespaces,
                    initialized_gitlinks,
                    gitlink_journal_bindings,
                    journal,
                )
                if restoration_callback is not None:
                    residue_before[label] = _preflight_gate_residue(workspace)

            preflight_ready = True
            return _build_verify.build_and_verify(
                spec,
                slice_id="__preflight__",
                cwd=str(candidate_workspace),
                log_dir=log_dir,
                extra_parameters=preflight_parameters,
                gate="dev_verify",
            )
        finally:
            primary_error = sys.exception()
            restore_errors: list[BaseException] = []
            restored_material = []
            failed_workspaces = 0
            for (
                label,
                workspace,
                namespace,
                workspace_device,
                workspace_inode,
                split_namespaces,
                initialized_gitlinks,
                gitlink_journal_bindings,
                journal,
            ) in reversed(sealed):
                workspace_errors: list[BaseException] = []
                observed_nested_untracked: dict[str, tuple[str, ...]] = {}
                if preflight_ready and label in residue_before:
                    # Name what the gate run created before the rollback
                    # below removes it.  A listing failure never stops that
                    # rollback and is not a restoration failure.
                    try:
                        gate_created[label] = _preflight_gate_created_roots(
                            residue_before[label],
                            _preflight_gate_residue(workspace),
                            namespace=namespace,
                            split_namespaces=split_namespaces,
                            initialized_gitlinks=initialized_gitlinks,
                        )
                    except BaseException as exc:
                        record_errors.append(exc)
                try:
                    workspace_metadata = workspace.lstat()
                    if (
                        workspace.is_symlink()
                        or not stat.S_ISDIR(workspace_metadata.st_mode)
                        or workspace_metadata.st_dev != workspace_device
                        or workspace_metadata.st_ino != workspace_inode
                    ):
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight checkout changed "
                            f"identity before rollback: {workspace}"
                        )
                    for path in workspace.iterdir():
                        if path.name not in namespace:
                            _mutation_journal._remove_path(path)
                    for (
                        relative,
                        ancestor_namespace,
                        ancestor_device,
                        ancestor_inode,
                    ) in split_namespaces:
                        ancestor = workspace / relative
                        metadata = ancestor.lstat()
                        if (
                            ancestor.is_symlink()
                            or not stat.S_ISDIR(metadata.st_mode)
                            or metadata.st_dev != ancestor_device
                            or metadata.st_ino != ancestor_inode
                        ):
                            raise _mutation_journal.MutationJournalError(
                                "development-verifier preflight split "
                                "ancestor changed identity before rollback: "
                                f"{ancestor}"
                            )
                        for path in ancestor.iterdir():
                            if path.name not in ancestor_namespace:
                                _mutation_journal._remove_path(path)
                    for (
                        gitlink,
                        gitlink_device,
                        gitlink_inode,
                    ) in sorted(
                        initialized_gitlinks,
                        key=lambda item: item[0].count("/"),
                        reverse=True,
                    ):
                        nested_root = workspace / gitlink
                        metadata = nested_root.lstat()
                        if (
                            nested_root.is_symlink()
                            or not stat.S_ISDIR(metadata.st_mode)
                            or metadata.st_dev != gitlink_device
                            or metadata.st_ino != gitlink_inode
                        ):
                            raise _mutation_journal.MutationJournalError(
                                "development-verifier preflight initialized "
                                "submodule changed identity before rollback: "
                                f"{nested_root}"
                            )
                        current_untracked = _preflight_untracked_roots(nested_root)
                        if current_untracked:
                            observed_nested_untracked[gitlink] = current_untracked
                        for relative in current_untracked:
                            _mutation_journal._remove_path(nested_root / relative)
                except BaseException as exc:
                    workspace_errors.append(exc)
                if observed_nested_untracked:
                    workspace_errors.append(
                        _mutation_journal.MutationJournalError(
                            "development-verifier preflight observed "
                            "baseline-absent artifacts in initialized "
                            f"submodules: {observed_nested_untracked!r}"
                        )
                    )
                try:
                    if not journal.restore(full=True):
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight rollback failed "
                            f"for {workspace}; durable journal retained"
                        )
                except BaseException as exc:
                    workspace_errors.append(exc)
                try:
                    workspace_metadata = workspace.lstat()
                    restored_namespace = frozenset(
                        path.name for path in workspace.iterdir()
                    )
                    if (
                        workspace.is_symlink()
                        or not stat.S_ISDIR(workspace_metadata.st_mode)
                        or workspace_metadata.st_dev != workspace_device
                        or workspace_metadata.st_ino != workspace_inode
                        or restored_namespace != namespace
                    ):
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight changed checkout "
                            f"namespace after rollback: {workspace}"
                        )
                    for (
                        relative,
                        ancestor_namespace,
                        ancestor_device,
                        ancestor_inode,
                    ) in split_namespaces:
                        ancestor = workspace / relative
                        metadata = ancestor.lstat()
                        restored_ancestor_namespace = frozenset(
                            path.name for path in ancestor.iterdir()
                        )
                        if (
                            ancestor.is_symlink()
                            or not stat.S_ISDIR(metadata.st_mode)
                            or metadata.st_dev != ancestor_device
                            or metadata.st_ino != ancestor_inode
                            or restored_ancestor_namespace != ancestor_namespace
                        ):
                            raise _mutation_journal.MutationJournalError(
                                "development-verifier preflight changed split "
                                "ancestor namespace after rollback: "
                                f"{ancestor}"
                            )
                    untracked_after: dict[str, tuple[str, ...]] = {}
                    for (
                        gitlink,
                        gitlink_device,
                        gitlink_inode,
                    ) in initialized_gitlinks:
                        nested_root = workspace / gitlink
                        metadata = nested_root.lstat()
                        if (
                            nested_root.is_symlink()
                            or not stat.S_ISDIR(metadata.st_mode)
                            or metadata.st_dev != gitlink_device
                            or metadata.st_ino != gitlink_inode
                        ):
                            raise _mutation_journal.MutationJournalError(
                                "development-verifier preflight initialized "
                                "submodule changed identity after rollback: "
                                f"{nested_root}"
                            )
                        untracked_after[gitlink] = _preflight_untracked_roots(
                            nested_root
                        )
                    if any(untracked_after.values()):
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight left untracked "
                            "submodule artifacts after rollback: "
                            f"{untracked_after!r}"
                        )
                except BaseException as exc:
                    workspace_errors.append(exc)
                try:
                    if journal.state != "rolled_back" or journal.cleanup_pending:
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight journal did not "
                            f"reach clean rollback terminality: {workspace}"
                        )
                except BaseException as exc:
                    workspace_errors.append(exc)
                if workspace_errors:
                    failed_workspaces += 1
                else:
                    restored_material.append(
                        (
                            label,
                            workspace,
                            namespace,
                            workspace_device,
                            workspace_inode,
                            split_namespaces,
                            initialized_gitlinks,
                            gitlink_journal_bindings,
                            journal,
                        )
                    )
                restore_errors.extend(workspace_errors)
            if restore_errors:
                restoration_error = _mutation_journal.MutationJournalError(
                    "development-verifier preflight did not exactly restore "
                    f"{failed_workspaces} writable checkout(s)"
                )
                control_error = next(
                    (
                        error
                        for error in (*restore_errors, *record_errors)
                        if not isinstance(error, Exception)
                    ),
                    None,
                )
                if control_error is None and primary_error is not None and not (
                    isinstance(primary_error, Exception)
                ):
                    control_error = primary_error
                if control_error is not None:
                    retained_cause = next(
                        (
                            error
                            for error in restore_errors
                            if error is not control_error
                        ),
                        primary_error
                        if primary_error is not control_error
                        else None,
                    )
                    restoration_error.__cause__ = retained_cause
                    restoration_error.__suppress_context__ = True
                    raise control_error.with_traceback(
                        control_error.__traceback__
                    ) from restoration_error
                raise restoration_error from restore_errors[0]
            record_control_error = next(
                (
                    error
                    for error in record_errors
                    if not isinstance(error, Exception)
                ),
                None,
            )
            if record_control_error is not None:
                raise record_control_error.with_traceback(
                    record_control_error.__traceback__
                )
            if restoration_callback is not None and preflight_ready:
                try:
                    if record_errors:
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight restored both "
                            "writable checkouts but could not list the paths "
                            "its gate run created"
                        ) from record_errors[0]
                    restored = []
                    for (
                        label,
                        workspace,
                        namespace,
                        workspace_device,
                        workspace_inode,
                        split_namespaces,
                        initialized_gitlinks,
                        gitlink_journal_bindings,
                        journal,
                    ) in restored_material:
                        bindings_by_gitlink = {
                            relative: (transaction_id, slice_id)
                            for relative, transaction_id, slice_id in (
                                gitlink_journal_bindings
                            )
                        }
                        restored.append(
                            DevelopmentPreflightWorkspaceRestoration(
                                label=label,
                                workspace=str(workspace),
                                workspace_device=workspace_device,
                                workspace_inode=workspace_inode,
                                namespace=tuple(sorted(namespace)),
                                split_namespaces=tuple(
                                    (
                                        relative,
                                        tuple(sorted(ancestor_namespace)),
                                        ancestor_device,
                                        ancestor_inode,
                                    )
                                    for (
                                        relative,
                                        ancestor_namespace,
                                        ancestor_device,
                                        ancestor_inode,
                                    ) in sorted(
                                        split_namespaces,
                                        key=lambda item: item[0],
                                    )
                                ),
                                initialized_gitlinks=tuple(
                                    (
                                        relative,
                                        device,
                                        inode,
                                        *bindings_by_gitlink[relative],
                                    )
                                    for relative, device, inode in sorted(
                                        initialized_gitlinks
                                    )
                                ),
                                journal_transaction_id=journal.transaction_id,
                                journal_family=(
                                    _mutation_journal._generation_zero_writer_epoch_family(
                                        journal._manifest
                                    )
                                ),
                                gate_created_paths=gate_created[label],
                            )
                        )
                    by_label = {item.label: item for item in restored}
                    if set(by_label) != {"candidate", "baseline"}:
                        raise _mutation_journal.MutationJournalError(
                            "development-verifier preflight restoration evidence "
                            "is incomplete"
                        )
                    receipt = DevelopmentPreflightRestorationReceipt(
                        workspaces=(
                            by_label["candidate"],
                            by_label["baseline"],
                        )
                    )
                except BaseException as evidence_error:
                    if primary_error is not None and not isinstance(
                        primary_error,
                        Exception,
                    ):
                        raise primary_error.with_traceback(
                            primary_error.__traceback__
                        ) from evidence_error
                    raise
                try:
                    restoration_callback(receipt)
                except BaseException as callback_error:
                    if primary_error is None:
                        raise
                    if isinstance(primary_error, Exception) and not isinstance(
                        callback_error,
                        Exception,
                    ):
                        raise callback_error.with_traceback(
                            callback_error.__traceback__
                        ) from primary_error
                    raise primary_error.with_traceback(
                        primary_error.__traceback__
                    ) from callback_error
