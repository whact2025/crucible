<!--
SPDX-FileCopyrightText: 2026 Whact Inc.
SPDX-License-Identifier: MIT
-->

# `backend_config_bind_deepseek` — Bind a DeepSeek diagnostic route

This provider-free command materializes an exact worker-bound routing input
from a checked-in DeepSeek construction template. It does not read credentials,
construct a paid adapter, create a benchmark Corpus, or qualify the route.

```bash
uv sync --project backends/deepseek_worker --frozen --reinstall --link-mode copy
uv run crucible backend_config_bind_deepseek \
  --template products/swebench/backends_deepseek_diagnostic.yaml \
  --output /absolute/path/to/new-run-inputs/backends_deepseek_bound.yaml
```

The command authenticates the installed worker once, pins that same runtime
fingerprint into every route, publishes exact bytes exclusively, and writes a
secret-free receipt. The output must be a new absolute path. Supply that file
to the suite with `--backend-config` and credentials only at authorized dispatch.
The resulting route remains diagnostic and non-promotable. Its runs record
DeepSeek cost at the published peak rate, an upper bound: it equals the charge
at peak and is twice the charge off-peak.

## Options

`uv run crucible backend_config_bind_deepseek --help` reports accepted values, defaults, mutual
exclusions, and the precise invocation scope.

| Option | Purpose |
|---|---|
| `--output` | new absolute run-local routing path (never overwritten) |
| `--template` | unbound DeepSeek-only diagnostic routing template |

## Evidence

Read the workflow’s typed terminal output and retained records before claiming
success. This command follows its explicit check/apply or materialization contract
rather than creating the normal Corpus run envelope. Source:
[workflows/backend_config_bind_deepseek.py](../../workflows/backend_config_bind_deepseek.py).

## Exit codes

`0` means the exact routing input and receipt were published. Invalid inputs,
existing output, or failed worker authentication terminate nonzero without
publishing a usable route. Argument parsing errors use argparse’s status.
