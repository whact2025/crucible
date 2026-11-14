# SPDX-FileCopyrightText: 2026 Whact Inc.
# SPDX-License-Identifier: MIT

"""Versioned capabilities for Crucible-owned runner modules.

Runner roles are intentionally coarse: a ``dev_verify`` adapter may run a
project-visible suite without implementing the higher-authority protocol that
publishes durable, exact-path verification checkpoints.  Keep those meanings
separate and default every unregistered runner to no optional capabilities.

The registry contains only bundled module paths whose implementations and
protocol tests ship with Crucible.  A canonical Spec may select another runner,
but it cannot grant that runner additional workflow authority merely by naming
it.
"""

from __future__ import annotations


DECLARED_VERIFICATION_CHECKPOINT_V1 = (
    "crucible.runner-capability.declared-verification-checkpoint.v1"
)
EXACT_PYTEST_OBSERVATION_V1 = (
    "crucible.runner-capability.exact-pytest-observation.v1"
)

_TRUSTED_RUNNER_CAPABILITIES: dict[str, frozenset[str]] = {
    "crucible.common.runners.lang.python": frozenset({
        EXACT_PYTEST_OBSERVATION_V1,
    }),
    "crucible.common.runners.polyglot": frozenset({
        EXACT_PYTEST_OBSERVATION_V1,
    }),
    "crucible.products.deepswe.runners.deepswe_dev_verify": frozenset({
        DECLARED_VERIFICATION_CHECKPOINT_V1,
    }),
}


def trusted_runner_capabilities(runner: object) -> frozenset[str]:
    """Return optional capabilities for one exact bundled runner module."""

    if not isinstance(runner, str):
        return frozenset()
    return _TRUSTED_RUNNER_CAPABILITIES.get(runner, frozenset())


def trusted_runner_has_capability(runner: object, capability: str) -> bool:
    """Return whether the trusted registry grants ``capability`` to runner."""

    return capability in trusted_runner_capabilities(runner)


__all__ = [
    "DECLARED_VERIFICATION_CHECKPOINT_V1",
    "EXACT_PYTEST_OBSERVATION_V1",
    "trusted_runner_capabilities",
    "trusted_runner_has_capability",
]
