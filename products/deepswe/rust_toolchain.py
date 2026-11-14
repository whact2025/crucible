# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Non-root Rust toolchain paths for isolated DeepSWE containers.

DeepSWE's Rust images install rustup under ``/root`` while Crucible runs
benchmark commands as the host uid.  The container bootstrap copies only the
Rust toolchain and Cargo state into its private ``/tmp`` tmpfs.  Executable
toolchain bytes remain root-owned; mutable Cargo registry/cache state belongs
to the runtime uid.  These fixed paths are shared by container setup and every
runner environment so visible and hidden verification cannot drift.
"""

from __future__ import annotations

RUST_IMMUTABLE_ROOT = "/tmp/crucible-rust-immutable"
RUST_CARGO_BIN = f"{RUST_IMMUTABLE_ROOT}/cargo-bin"
RUSTUP_HOME = f"{RUST_IMMUTABLE_ROOT}/rustup"
RUST_CARGO_HOME = "/tmp/crucible-cargo-home"
# Keep compilation output on the quota-governed workspace disk.  A Rust image
# can carry nearly 2 GiB of toolchain/cache state and /tmp is deliberately
# capped, so placing target/ there can exhaust the isolation tmpfs.
RUST_CARGO_TARGET_DIR = "/app/target"

_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def runtime_environment() -> dict[str, str]:
    """Return the exact staged Rust environment used by all runner lanes."""

    return {
        "CARGO_HOME": RUST_CARGO_HOME,
        "RUSTUP_HOME": RUSTUP_HOME,
        "CARGO_TARGET_DIR": RUST_CARGO_TARGET_DIR,
        "PATH": f"{RUST_CARGO_BIN}:{_SYSTEM_PATH}",
    }


def environment_args() -> tuple[str, ...]:
    """Render deterministic ``docker run/exec --env`` arguments."""

    return tuple(
        argument
        for key, value in runtime_environment().items()
        for argument in ("--env", f"{key}={value}")
    )


__all__ = [
    "RUST_CARGO_BIN",
    "RUST_CARGO_HOME",
    "RUST_CARGO_TARGET_DIR",
    "RUST_IMMUTABLE_ROOT",
    "RUSTUP_HOME",
    "environment_args",
    "runtime_environment",
]
