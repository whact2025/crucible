<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# Isolated official Codex SDK worker

This independently locked project supplies the official-SDK transport selected
by the Codex route's `extra.transport: official_sdk` option. The app-server
transport is selected explicitly or by its configured default; transports are
never substituted automatically.

```bash
uv sync --project backends/codex_worker --locked
```

The worker accepts one bounded JSON request and emits bounded JSONL thread
identity and terminal events. Thread IDs support explicitly authorized resume;
they grant no tool authority. Credentials stay in a parent-created private
`CODEX_HOME` and never enter requests, arguments, result provenance, or debug
records. Confined requests suppress ambient configuration and native executors.
The parent owns deadline, cancellation, child-process settlement, and cleanup.

SDK, CLI, worker source, lock, installed runtime, route, session behavior, and
feature admission are explicit identities. Changing any of them requires
conformance and failure-path validation. A diagnostic route remains diagnostic
until its complete route-specific evidence supports promotion.
